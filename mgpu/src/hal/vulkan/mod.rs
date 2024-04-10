use crate::hal::Hal;
use crate::{DeviceConfiguration, DeviceFeatures, DevicePreference, MgpuError, MgpuResult};
use ash::{vk, Entry, Instance};
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;

pub struct VulkanHal {
    entry: Entry,
    instance: Instance,
    physical_device: VulkanPhysicalDevice,
}

pub struct VulkanPhysicalDevice {
    device_handle: vk::PhysicalDevice,
    name: String,
    limits: vk::PhysicalDeviceLimits,
    device_id: u32,
    features: VulkanDeviceFeatures,
}

pub struct VulkanDeviceFeatures {
    swapchain_support: bool,
}

pub enum VulkanHalError {
    NoSuitableDevice(Option<DevicePreference>),
    ApiError(vk::Result),
    LayerNotAvailable(std::borrow::Cow<'static, str>),
    ExtensionNotAvailable(std::borrow::Cow<'static, str>),
}

pub type VulkanHalResult<T> = Result<T, VulkanHalError>;

impl Hal for VulkanHal {
    fn device_info(&self) -> crate::DeviceInfo {
        let major = vk::api_version_major(Self::VULKAN_API_VERSION);
        let minor = vk::api_version_minor(Self::VULKAN_API_VERSION);
        let patch = vk::api_version_patch(Self::VULKAN_API_VERSION);
        crate::DeviceInfo {
            name: self.physical_device.name.clone(),
            api_description: format!("Vulkan {}.{}.{}", major, minor, patch),
            swapchain_support: self.physical_device.features.swapchain_support,
        }
    }
}

impl VulkanHal {
    const VULKAN_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
    pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry, configuration)?;
        let physical_device = Self::pick_physical_device(&instance, configuration)?;

        let hal = Self {
            entry,
            instance,
            physical_device,
        };

        Ok(Arc::new(hal))
    }

    fn create_instance(
        entry: &Entry,
        configuration: &DeviceConfiguration,
    ) -> VulkanHalResult<ash::Instance> {
        const LAYER_KHRONOS_VALIDATION: &CStr =
            unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

        let application_name = configuration.app_name.unwrap_or("mgpu application");
        let application_name =
            CString::new(application_name).expect("Failed to convert application name to CString");
        let application_name = application_name.as_c_str();
        let engine_name = CString::new("mgpu").expect("Failed to convert engine name to CString");
        let engine_name = engine_name.as_c_str();
        let application_info = vk::ApplicationInfo::default()
            .application_name(application_name)
            .engine_name(engine_name)
            .api_version(Self::VULKAN_API_VERSION);
        let mut requested_layers = vec![];
        if configuration
            .features
            .contains(DeviceFeatures::DEBUG_LAYERS)
        {
            requested_layers.push(LAYER_KHRONOS_VALIDATION.as_ptr());
        }

        let mut requested_instance_extensions: Vec<*const c_char> = vec![];

        if cfg!(feature = "swapchain") {
            let extensions =
                ash_window::enumerate_required_extensions(configuration.display_handle)?;
            requested_instance_extensions.extend(extensions);
        }

        Self::ensure_requested_layers_are_avaliable(entry, &requested_layers)?;
        Self::ensure_requested_instance_extensions_are_available(
            entry,
            &requested_instance_extensions,
        )?;
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&requested_layers);
        let instance =
            unsafe { entry.create_instance(&instance_info, get_allocation_callbacks()) }?;
        Ok(instance)
    }

    fn ensure_requested_layers_are_avaliable(
        entry: &Entry,
        requested_layers: &[*const std::ffi::c_char],
    ) -> VulkanHalResult<()> {
        let available_instance_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        for requested in requested_layers {
            let requested = unsafe { CStr::from_ptr(*requested) };
            if !available_instance_layers.iter().any(|layer| {
                let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                layer_name == requested
            }) {
                return Err(VulkanHalError::LayerNotAvailable(
                    requested.to_string_lossy(),
                ));
            }
        }
        Ok(())
    }

    fn ensure_requested_instance_extensions_are_available(
        entry: &Entry,
        requested_instance_extensions: &[*const c_char],
    ) -> VulkanHalResult<()> {
        let available_instance_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None)? };
        for requested in requested_instance_extensions {
            let requested = unsafe { CStr::from_ptr(*requested) };
            if !available_instance_extensions.iter().any(|extension| {
                let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                extension_name == requested
            }) {
                return Err(VulkanHalError::ExtensionNotAvailable(
                    requested.to_string_lossy(),
                ));
            }
        }
        Ok(())
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        configuration: &DeviceConfiguration,
    ) -> VulkanHalResult<VulkanPhysicalDevice> {
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let device = if let Some(preference) = configuration.device_preference {
            let filter_fn: Box<dyn Fn(&&vk::PhysicalDevice) -> bool> = match preference {
                DevicePreference::HighPerformance => Box::new(|dev| {
                    let properties = unsafe { instance.get_physical_device_properties(**dev) };
                    properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                }),
                DevicePreference::Software => Box::new(|dev| {
                    let properties = unsafe { instance.get_physical_device_properties(**dev) };
                    properties.device_type == vk::PhysicalDeviceType::CPU
                }),
                DevicePreference::AnyDevice => Box::new(|_dev| true),
            };

            devices.iter().find(filter_fn).copied()
        } else {
            devices.first().copied()
        };
        if let Some(device) = device {
            let device_props = unsafe { instance.get_physical_device_properties(device) };
            let device_name = unsafe { CStr::from_ptr(device_props.device_name.as_ptr()) };
            let device_name = device_name
                .to_str()
                .expect("Failed to convert device name to UTF-8")
                .to_owned();

            Ok(VulkanPhysicalDevice {
                device_handle: device,
                name: device_name,
                limits: device_props.limits,
                device_id: device_props.device_id,
                features: VulkanDeviceFeatures {
                    swapchain_support: cfg!(feature = "swapchain"),
                },
            })
        } else {
            Err(VulkanHalError::NoSuitableDevice(
                configuration.device_preference,
            ))
        }
    }
}

fn get_allocation_callbacks() -> Option<&'static vk::AllocationCallbacks<'static>> {
    None
}

impl From<VulkanHalError> for MgpuError {
    fn from(value: VulkanHalError) -> Self {
        let message = match value {
            VulkanHalError::NoSuitableDevice(device_pref) => {
                format!("No suitable device of type {:?}", device_pref)
            }
            VulkanHalError::ApiError(code) => format!("A vulkan api call failed: {}", code),
            VulkanHalError::LayerNotAvailable(layer) => {
                format!("Vulkan Instance layer not available: {layer}")
            }
            VulkanHalError::ExtensionNotAvailable(extension) => {
                format!("Vulkan Instance Extension not available: {extension}")
            }
        };
        MgpuError::Dynamic(format!("Vulkan Hal error: {}", message))
    }
}

impl From<vk::Result> for VulkanHalError {
    fn from(value: vk::Result) -> Self {
        VulkanHalError::ApiError(value)
    }
}

impl From<ash::LoadingError> for MgpuError {
    fn from(value: ash::LoadingError) -> Self {
        MgpuError::Dynamic(value.to_string())
    }
}
