use std::{
    ffi::{CStr, CString},
    ptr::{addr_of, null},
};

use anyhow::Result;
use ash::{
    prelude::*,
    vk::{
        make_api_version, ApplicationInfo, DeviceCreateFlags, DeviceCreateInfo,
        DeviceQueueCreateFlags, DeviceQueueCreateInfo, ExtensionProperties, InstanceCreateFlags,
        InstanceCreateInfo, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceProperties,
        PhysicalDeviceType, QueueFlags, StructureType, API_VERSION_1_3,
    },
    *,
};

use log::{error, trace};
use raw_window_handle::RawDisplayHandle;
use thiserror::Error;

const KHRONOS_VALIDATION_LAYER: &'static str = "VK_LAYER_KHRONOS_validation";

pub struct GpuDescription {
    name: String,
}
impl GpuDescription {
    fn new(physical_device: &SelectedPhysicalDevice) -> Self {
        let name =
            unsafe { CStr::from_ptr(physical_device.device_properties.device_name.as_ptr()) };
        let name = name.to_str().expect("Invalid device name");
        let name = String::from(name);

        Self { name }
    }
}

pub struct Gpu {
    instance: Instance,
    device: Device,
    description: GpuDescription,
}

pub struct GpuConfiguration<'a> {
    pub raw_display_handle: Option<RawDisplayHandle>,
    pub app_name: &'a str,
    pub engine_name: &'a str,
    pub enable_validation_layer: bool,
}

#[derive(Error, Debug, Clone, Copy)]
pub enum GpuError {
    #[error("No physical devices were found on this machine")]
    NoPhysicalDevices,

    #[error("A suitable device with the requested capabilities was not found")]
    NoSuitableDevice,

    #[error("Could not find a queue family with these flags")]
    NoQueueFamilyFound(QueueFlags),
}

#[derive(Clone)]
struct QueueFamily {
    index: u32,
    count: u32,
}

#[derive(Clone)]
struct QueueFamilies {
    graphics_family: QueueFamily,
    async_compute_family: QueueFamily,
    transfer_family: QueueFamily,
}

struct SelectedPhysicalDevice {
    physical_device: PhysicalDevice,
    device_properties: PhysicalDeviceProperties,
    device_features: PhysicalDeviceFeatures,
}

impl QueueFamilies {
    pub fn is_valid(&self) -> bool {
        self.graphics_family.count > 0
            && self.async_compute_family.count > 0
            && self.transfer_family.count > 0
    }
}

impl Gpu {
    pub fn new(configuration: GpuConfiguration) -> Result<Self> {
        let entry = Entry::linked();

        let all_extensions = entry.enumerate_instance_extension_properties(None)?;
        let required_extensions = if let Some(raw_display_handle) = configuration.raw_display_handle
        {
            ash_window::enumerate_required_extensions(raw_display_handle)?
        } else {
            &[]
        };

        Self::ensure_required_extensions_are_available(required_extensions, &all_extensions);

        trace!("Created instance");
        let instance = Self::create_instance(&entry, &configuration, required_extensions)?;

        trace!("Created physical device");
        let physical_device = Self::select_discrete_physical_device(&instance)?;
        let description = GpuDescription::new(&physical_device);

        let queue_indices = Self::get_queue_families_indices(&physical_device, &instance)?;
        if !queue_indices.is_valid() {
            log::error!("Queue configurations are invalid!");
        }
        let device =
            Self::create_device(&configuration, &instance, physical_device, queue_indices)?;

        log::info!(
            "Created a GPU from a device with name '{}'",
            description.name
        );
        Ok(Gpu {
            instance,
            device,
            description,
        })
    }

    fn create_instance(
        entry: &Entry,
        configuration: &GpuConfiguration,
        required_extensions: &[*const i8],
    ) -> VkResult<Instance> {
        let vk_layer_khronos_validation = CString::new(KHRONOS_VALIDATION_LAYER).unwrap();
        let vk_layer_khronos_validation = vk_layer_khronos_validation.as_ptr();

        let app_name =
            CString::new(configuration.app_name).expect("Failed to create valid App name");
        let engine_name =
            CString::new(configuration.engine_name).expect("Failed to create valid Engine Engine");

        let app_info = ApplicationInfo {
            s_type: StructureType::APPLICATION_INFO,
            p_next: null(),
            p_application_name: app_name.as_ptr(),
            application_version: make_api_version(0, 0, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: make_api_version(0, 0, 0, 0),
            api_version: API_VERSION_1_3,
        };
        let create_info = InstanceCreateInfo {
            s_type: StructureType::INSTANCE_CREATE_INFO,
            p_next: null(),
            flags: InstanceCreateFlags::empty(),
            p_application_info: addr_of!(app_info),
            enabled_layer_count: if configuration.enable_validation_layer {
                1
            } else {
                0
            },
            pp_enabled_layer_names: if configuration.enable_validation_layer {
                addr_of!(vk_layer_khronos_validation)
            } else {
                null()
            },
            enabled_extension_count: required_extensions.len() as u32,
            pp_enabled_extension_names: required_extensions.as_ptr(),
        };

        unsafe { entry.create_instance(&create_info, None) }
    }

    fn select_discrete_physical_device(
        instance: &Instance,
    ) -> Result<SelectedPhysicalDevice, GpuError> {
        unsafe {
            let devices = instance
                .enumerate_physical_devices()
                .map_err(|_| GpuError::NoPhysicalDevices)?;

            for physical_device in devices {
                let device_properties = instance.get_physical_device_properties(physical_device);
                let device_features = instance.get_physical_device_features(physical_device);

                if device_properties.device_type == PhysicalDeviceType::DISCRETE_GPU {
                    return Ok(SelectedPhysicalDevice {
                        physical_device,
                        device_properties,
                        device_features,
                    });
                }
            }
        }
        Err(GpuError::NoSuitableDevice)
    }

    fn get_queue_families_indices(
        device: &SelectedPhysicalDevice,
        instance: &Instance,
    ) -> Result<QueueFamilies, GpuError> {
        let all_queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device.physical_device) };
        let graphics_family = Self::find_queue_family(&all_queue_families, QueueFlags::GRAPHICS)?;
        let async_compute_family =
            Self::find_queue_family(&all_queue_families, QueueFlags::COMPUTE)?;

        let transfer_family = Self::find_queue_family(
            &all_queue_families,
            QueueFlags::TRANSFER | QueueFlags::SPARSE_BINDING,
        )?;

        Ok(QueueFamilies {
            graphics_family,
            async_compute_family,
            transfer_family,
        })
    }

    fn create_device(
        configuration: &GpuConfiguration,
        instance: &Instance,
        selected_device: SelectedPhysicalDevice,
        queue_indices: QueueFamilies,
    ) -> VkResult<Device> {
        let priority_one: f32 = 1.0;
        let vk_layer_khronos_validation = CString::new(KHRONOS_VALIDATION_LAYER).unwrap();
        let vk_layer_khronos_validation = vk_layer_khronos_validation.as_ptr();

        let make_queue_create_info = |index| DeviceQueueCreateInfo {
            s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: null(),
            flags: DeviceQueueCreateFlags::empty(),
            queue_family_index: index,
            queue_count: 1,
            p_queue_priorities: addr_of!(priority_one),
        };
        let queue_create_infos = [
            make_queue_create_info(queue_indices.graphics_family.index),
            make_queue_create_info(queue_indices.async_compute_family.index),
            make_queue_create_info(queue_indices.transfer_family.index),
        ];

        let create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            p_next: null(),
            flags: DeviceCreateFlags::empty(),
            queue_create_info_count: 3,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: if configuration.enable_validation_layer {
                1
            } else {
                0
            },
            pp_enabled_layer_names: if configuration.enable_validation_layer {
                addr_of!(vk_layer_khronos_validation)
            } else {
                null()
            },
            enabled_extension_count: 0,
            pp_enabled_extension_names: null(),
            p_enabled_features: null(),
        };

        let device =
            unsafe { instance.create_device(selected_device.physical_device, &create_info, None) };
        device
    }

    fn ensure_required_extensions_are_available(
        required_extensions: &[*const i8],
        all_extensions: &[ExtensionProperties],
    ) {
        let mut all_extensions_c_names = all_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) });
        let required_extensions_c_names = required_extensions
            .iter()
            .map(|name| unsafe { CStr::from_ptr(*name) });

        for required_c_name in required_extensions_c_names {
            if all_extensions_c_names
                .find(|name| *name == required_c_name)
                .is_none()
            {
                error!("Extension {:?} is not supported", required_c_name);
            }
        }
    }

    fn find_queue_family(
        all_queue_families: &Vec<vk::QueueFamilyProperties>,
        requested_family: QueueFlags,
    ) -> Result<QueueFamily, GpuError> {
        for (index, queue_family) in all_queue_families.iter().enumerate() {
            if queue_family.queue_flags.contains(requested_family) && queue_family.queue_count > 0 {
                return Ok({
                    QueueFamily {
                        index: index as u32,
                        count: queue_family.queue_count,
                    }
                });
            }
        }

        Err(GpuError::NoQueueFamilyFound(requested_family))
    }
}
