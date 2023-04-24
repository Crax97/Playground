use std::{
    ffi::{CStr, CString},
    ops::{Deref, DerefMut},
    ptr::{addr_of, addr_of_mut, null},
    sync::Arc,
};

use anyhow::Result;
use ash::{
    extensions::khr::Surface,
    prelude::*,
    vk::{
        make_api_version, ApplicationInfo, Bool32, DeviceCreateFlags, DeviceCreateInfo,
        DeviceQueueCreateFlags, DeviceQueueCreateInfo, ExtensionProperties, InstanceCreateFlags,
        InstanceCreateInfo, KhrSurfaceFn, PFN_vkGetPhysicalDeviceSurfaceSupportKHR, PhysicalDevice,
        PhysicalDeviceFeatures, PhysicalDeviceProperties, PhysicalDeviceType, Queue, QueueFlags,
        StructureType, SurfaceKHR, API_VERSION_1_3,
    },
    *,
};

use log::{error, trace};
use once_cell::sync::OnceCell;
use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use thiserror::Error;
use winit::window::Window;

use crate::gpu_extension::{GpuExtension, GpuParameters};

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

pub struct GpuInfo {
    pub entry: Entry,
    pub instance: Instance,
    pub logical_device: Device,
    pub physical_device: PhysicalDevice,
    pub graphics_queue: Queue,
    pub async_compute_queue: Queue,
    pub transfer_queue: Queue,
    pub description: GpuDescription,
}

pub type SharedGpuInfo = Arc<GpuInfo>;

pub struct Gpu<E: GpuExtension> {
    extension: E,
    gpu_info: Arc<GpuInfo>,
}

pub struct GpuConfiguration<'a> {
    pub app_name: &'a str,
    pub engine_name: &'a str,
    pub enable_validation_layer: bool,
}

impl<'a> Default for GpuConfiguration<'a> {
    fn default() -> Self {
        Self {
            app_name: Default::default(),
            engine_name: Default::default(),
            enable_validation_layer: false,
        }
    }
}

#[derive(Error, Debug, Clone)]
pub enum GpuError {
    #[error("No physical devices were found on this machine")]
    NoPhysicalDevices,

    #[error("A suitable device with the requested capabilities was not found")]
    NoSuitableDevice,

    #[error("One or more queue families aren't supported")]
    NoQueueFamilyFound(
        Option<(u32, vk::QueueFamilyProperties)>,
        Option<(u32, vk::QueueFamilyProperties)>,
        Option<(u32, vk::QueueFamilyProperties)>,
    ),

    #[error("Generic")]
    GenericGpuError(String),
}

#[derive(Clone, Copy, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct QueueFamilies {
    pub graphics_family: QueueFamily,
    pub async_compute_family: QueueFamily,
    pub transfer_family: QueueFamily,
}

#[derive(Clone, Copy, Debug)]
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

impl<E: GpuExtension> Gpu<E> {
    pub fn new(
        configuration: GpuConfiguration,
        extension_params: E::SetupParameters,
    ) -> Result<Self> {
        let entry = Entry::linked();

        let instance_extensions = E::get_instance_extensions(&extension_params);

        Self::ensure_required_instance_extensions_are_available(&instance_extensions, &entry)?;

        let instance = Self::create_instance(&entry, &configuration, &instance_extensions)?;
        trace!("Created instance");
        let gpu_parameters = GpuParameters {
            entry: &entry,
            instance: &instance,
        };
        let device_extensions = E::get_device_extensions(&extension_params, &gpu_parameters);

        let physical_device = Self::select_discrete_physical_device(&instance)?;
        trace!("Created physical device");

        let description = GpuDescription::new(&physical_device);

        trace!("Created presentation surface");

        let queue_indices = Self::select_queue_families_indices(&physical_device, &instance)?;
        if !queue_indices.is_valid() {
            log::error!("Queue configurations are invalid!");
        }

        Self::ensure_required_device_extensions_are_available(
            &device_extensions,
            &instance,
            &physical_device,
        )?;

        let logical_device = Self::create_device(
            &configuration,
            &device_extensions,
            &instance,
            physical_device,
            &queue_indices,
        )?;
        trace!("Created logical device");

        let (graphics_queue, async_compute_queue, transfer_queue) =
            Self::get_device_queues(&logical_device, &queue_indices)?;
        trace!("Created queues");

        let gpu_info = Arc::new(GpuInfo {
            entry,
            instance,
            logical_device,
            physical_device: physical_device.physical_device,
            graphics_queue,
            async_compute_queue,
            transfer_queue,
            description,
        });

        let extension: E = E::new(extension_params, gpu_info.clone())?;

        if !extension.accepts_queue_families(queue_indices, physical_device.physical_device)? {
            error!("Extension did not accept the selected queues!");
        } else {
            trace!("Extension accepted the selected queue families");
        }

        trace!(
            "Created a GPU from a device with name '{}'",
            &gpu_info.description.name
        );
        Ok(Gpu {
            gpu_info,
            extension,
        })
    }

    fn create_instance(
        entry: &Entry,
        configuration: &GpuConfiguration,
        instance_extensions: &Vec<String>,
    ) -> VkResult<Instance> {
        let vk_layer_khronos_validation = CString::new(KHRONOS_VALIDATION_LAYER).unwrap();
        let vk_layer_khronos_validation = vk_layer_khronos_validation.as_ptr();

        let required_extensions: Vec<CString> = instance_extensions
            .iter()
            .map(|str| CString::new(str.clone()).unwrap())
            .collect();
        let required_extensions: Vec<*const i8> =
            required_extensions.iter().map(|ext| ext.as_ptr()).collect();

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

    fn select_queue_families_indices(
        device: &SelectedPhysicalDevice,
        instance: &Instance,
    ) -> Result<QueueFamilies, GpuError> {
        let all_queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device.physical_device) };

        let mut graphics_queue = None;
        let mut async_compute_queue = None;
        let mut transfer_queue = None;

        for (index, queue_family) in all_queue_families.iter().enumerate() {
            if queue_family.queue_count == 0 {
                continue;
            }

            if queue_family.queue_flags.intersects(QueueFlags::GRAPHICS) {
                graphics_queue = Some((index as u32, *queue_family));
            } else if queue_family.queue_flags.intersects(QueueFlags::COMPUTE) {
                async_compute_queue = Some((index as u32, *queue_family));
            } else if queue_family.queue_flags.intersects(QueueFlags::TRANSFER) {
                transfer_queue = Some((index as u32, *queue_family));
            }
        }

        match (graphics_queue, async_compute_queue, transfer_queue) {
            (Some(g), Some(a), Some(t)) => Ok(QueueFamilies {
                graphics_family: QueueFamily {
                    index: g.0,
                    count: g.1.queue_count,
                },
                async_compute_family: QueueFamily {
                    index: a.0,
                    count: a.1.queue_count,
                },
                transfer_family: QueueFamily {
                    index: t.0,
                    count: t.1.queue_count,
                },
            }),
            _ => Err(GpuError::NoQueueFamilyFound(
                graphics_queue,
                async_compute_queue,
                transfer_queue,
            )),
        }
    }

    fn create_device(
        configuration: &GpuConfiguration,
        device_extensions: &Vec<String>,
        instance: &Instance,
        selected_device: SelectedPhysicalDevice,
        queue_indices: &QueueFamilies,
    ) -> VkResult<Device> {
        let priority_one: f32 = 1.0;
        let vk_layer_khronos_validation = CString::new(KHRONOS_VALIDATION_LAYER).unwrap();
        let vk_layer_khronos_validation = vk_layer_khronos_validation.as_ptr();

        let c_string_device_extensions: Vec<CString> = device_extensions
            .iter()
            .map(|e| CString::new(e.as_str()).unwrap())
            .collect();

        let c_ptr_device_extensions: Vec<*const i8> = c_string_device_extensions
            .iter()
            .map(|cstr| cstr.as_ptr())
            .collect();

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
            enabled_extension_count: c_ptr_device_extensions.len() as u32,
            pp_enabled_extension_names: c_ptr_device_extensions.as_ptr(),
            p_enabled_features: null(),
        };

        let device =
            unsafe { instance.create_device(selected_device.physical_device, &create_info, None) };
        device
    }

    fn ensure_required_instance_extensions_are_available(
        requested_extensions: &[String],
        entry: &Entry,
    ) -> VkResult<()> {
        let all_extensions = entry.enumerate_instance_extension_properties(None)?;
        trace!(
            "Requested instance extensions: {}",
            requested_extensions.join(",")
        );
        let mut all_extensions_c_names = all_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) });

        for requested_extension in requested_extensions {
            let required_c_name =
                unsafe { CString::from_vec_unchecked(requested_extension.clone().into_bytes()) };
            if all_extensions_c_names
                .find(|name| *name == required_c_name.as_c_str())
                .is_none()
            {
                error!("Instance extension {:?} is not supported", required_c_name);
            }
        }

        Ok(())
    }

    fn get_device_queues(
        device: &Device,
        queues: &QueueFamilies,
    ) -> Result<(Queue, Queue, Queue), GpuError> {
        let graphics_queue = unsafe { device.get_device_queue(queues.graphics_family.index, 0) };
        let async_compute_queue =
            unsafe { device.get_device_queue(queues.async_compute_family.index, 0) };
        let transfer_queue = unsafe { device.get_device_queue(queues.transfer_family.index, 0) };
        Ok((graphics_queue, async_compute_queue, transfer_queue))
    }

    fn ensure_required_device_extensions_are_available(
        device_extensions: &[String],
        instance: &Instance,
        physical_device: &SelectedPhysicalDevice,
    ) -> VkResult<()> {
        trace!(
            "Requested device extensions: {}",
            device_extensions.join(",")
        );
        let all_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device.physical_device)
        }?;
        let mut all_extensions_c_names = all_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) });

        for requested_extension in device_extensions {
            let required_c_name =
                unsafe { CString::from_vec_unchecked(requested_extension.clone().into_bytes()) };

            if all_extensions_c_names
                .find(|name| *name == required_c_name.as_c_str())
                .is_none()
            {
                error!("Device extension {:?} is not supported", required_c_name);
            }
        }

        Ok(())
    }
}

impl<E: GpuExtension> Deref for Gpu<E> {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        &self.extension
    }
}

impl<E: GpuExtension> DerefMut for Gpu<E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.extension
    }
}

impl<T: GpuExtension> Drop for Gpu<T> {
    fn drop(&mut self) {
        unsafe {
            self.gpu_info.logical_device.destroy_device(None);
            self.gpu_info.instance.destroy_instance(None);
        }
    }
}
