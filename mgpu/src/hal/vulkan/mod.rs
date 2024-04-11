use crate::hal::Hal;
use crate::{DeviceConfiguration, DeviceFeatures, DevicePreference, MgpuError, MgpuResult};
use ash::vk::{DebugUtilsMessageSeverityFlagsEXT, QueueFlags};
use ash::{vk, Entry, Instance};
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::{self, c_char, CStr, CString};
use std::sync::Arc;

#[cfg(feature = "swapchain")]
mod swapchain;

pub struct VulkanHal {
    entry: Entry,
    instance: Instance,
    physical_device: VulkanPhysicalDevice,
    logical_device: VulkanLogicalDevice,
    debug_utilities: Option<VulkanDebugUtilities>,
}

pub struct VulkanPhysicalDevice {
    handle: vk::PhysicalDevice,
    name: String,
    limits: vk::PhysicalDeviceLimits,
    device_id: u32,
    features: VulkanDeviceFeatures,
}

pub struct VulkanQueueFamily {
    pub index: u32,
    pub requested_flags: QueueFlags,
}

pub struct VulkanQueueFamilies {
    pub families: Vec<VulkanQueueFamily>,
}

pub struct VulkanLogicalDevice {
    handle: ash::Device,
    graphics_queue: vk::Queue,
    compute_queue: vk::Queue,
    transfer_queue: vk::Queue,
}

pub(crate) struct VulkanDebugUtilities {
    debug_messenger: vk::DebugUtilsMessengerEXT,
    debug_instance: ash::ext::debug_utils::Instance,
}

pub struct VulkanDeviceFeatures {
    swapchain_support: bool,
}

pub enum VulkanHalError {
    NoSuitableDevice(Option<DevicePreference>),
    ApiError(vk::Result),
    LayerNotAvailable(std::borrow::Cow<'static, str>),
    ExtensionNotAvailable(std::borrow::Cow<'static, str>),
    NoSuitableQueueFamily(vk::QueueFlags),
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

    #[cfg(feature = "swapchain")]
    fn create_swapchain_impl(
        &self,
        swapchain_info: &crate::SwapchainCreationInfo,
    ) -> MgpuResult<Box<dyn crate::SwapchainImpl>> {
        use self::swapchain::VulkanSwapchain;

        Ok(Box::new(VulkanSwapchain::create(self, swapchain_info)?))
    }
}

impl VulkanHal {
    const VULKAN_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
    pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry, configuration)?;
        let physical_device = Self::pick_physical_device(&instance, configuration)?;

        let debug_utilities = if configuration
            .features
            .contains(DeviceFeatures::DEBUG_LAYERS)
        {
            Some(Self::create_debug_utilities(&entry, &instance)?)
        } else {
            None
        };
        let queue_families = Self::pick_queue_families(&instance, physical_device.handle)?;
        let logical_device =
            Self::create_device(&instance, physical_device.handle, &queue_families)?;
        let hal = Self {
            entry,
            instance,
            physical_device,
            debug_utilities,
            logical_device,
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
                handle: device,
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

    fn create_debug_utilities(
        entry: &Entry,
        instance: &Instance,
    ) -> VulkanHalResult<VulkanDebugUtilities> {
        let debug_utils_ext = ash::ext::debug_utils::Instance::new(entry, instance);

        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));
        let messenger = unsafe {
            debug_utils_ext
                .create_debug_utils_messenger(&messenger_info, get_allocation_callbacks())?
        };
        Ok(VulkanDebugUtilities {
            debug_messenger: messenger,
            debug_instance: debug_utils_ext,
        })
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_families: &VulkanQueueFamilies,
    ) -> VulkanHalResult<VulkanLogicalDevice> {
        #[derive(Default)]
        struct QueueInfo {
            count: u32,
            priorities: Vec<f32>,
        };
        let mut num_queues = HashMap::<u32, QueueInfo>::default();
        for family in &queue_families.families {
            num_queues.entry(family.index).or_default().count += 1;
        }
        let mut queue_create_infos = vec![];
        for (&fam, info) in &mut num_queues {
            let priorities = std::iter::repeat(1.0)
                .take(info.count as usize)
                .collect::<Vec<_>>();
            info.priorities = priorities;

            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(fam)
                    .queue_priorities(&info.priorities),
            )
        }
        let supported_device_features =
            unsafe { instance.get_physical_device_features(physical_device) };
        let device_features = Self::get_physical_device_features(supported_device_features);

        let required_extensions = vec![];
        Self::ensure_requested_device_extensions_are_available(
            instance,
            physical_device,
            &required_extensions,
        )?;
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&required_extensions);

        let logical_device = unsafe {
            instance.create_device(
                physical_device,
                &device_create_info,
                get_allocation_callbacks(),
            )
        }?;

        let mut find_fam_and_index = |flags| {
            let family = queue_families
                .families
                .iter()
                .find(|fam| fam.requested_flags.intersects(flags))
                .expect("Family not found");
            let index = num_queues
                .get_mut(&family.index)
                .expect("Failed to find family index");
            index.count -= 1;
            (family.index, index.count)
        };

        let (g_f, g_i) = find_fam_and_index(QueueFlags::GRAPHICS);
        let (c_f, c_i) = find_fam_and_index(QueueFlags::COMPUTE);
        let (t_f, t_i) = find_fam_and_index(QueueFlags::TRANSFER);
        let graphics_queue = unsafe { logical_device.get_device_queue(g_f, g_i) };

        let compute_queue = unsafe { logical_device.get_device_queue(c_f, c_i) };

        let transfer_queue = unsafe { logical_device.get_device_queue(t_f, t_i) };

        Ok(VulkanLogicalDevice {
            handle: logical_device,
            graphics_queue,
            compute_queue,
            transfer_queue,
        })
    }

    fn pick_queue_families(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> VulkanHalResult<VulkanQueueFamilies> {
        let families_len =
            unsafe { instance.get_physical_device_queue_family_properties2_len(physical_device) };
        let mut families = Vec::with_capacity(families_len);
        families.resize(families_len, vk::QueueFamilyProperties2::default());
        unsafe {
            instance.get_physical_device_queue_family_properties2(physical_device, &mut families)
        };

        let mut pick_queue_family = |queue_family_flags| {
            for (index, family) in families.iter_mut().enumerate() {
                if family
                    .queue_family_properties
                    .queue_flags
                    .intersects(queue_family_flags)
                    && family.queue_family_properties.queue_count > 0
                {
                    family.queue_family_properties.queue_count -= 1;
                    return Ok(VulkanQueueFamily {
                        index: index as u32,
                        requested_flags: queue_family_flags,
                    });
                }
            }
            Err(VulkanHalError::NoSuitableQueueFamily(queue_family_flags))
        };
        let graphics_queue_family = pick_queue_family(QueueFlags::GRAPHICS)?;
        let compute_queue_family = pick_queue_family(QueueFlags::COMPUTE)?;
        let transfer_queue_family = pick_queue_family(QueueFlags::TRANSFER)?;
        Ok(VulkanQueueFamilies {
            families: vec![
                graphics_queue_family,
                compute_queue_family,
                transfer_queue_family,
            ],
        })
    }

    fn get_physical_device_features(
        supported_device_features: vk::PhysicalDeviceFeatures,
    ) -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::default()
    }

    fn ensure_requested_device_extensions_are_available(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        required_extensions: &[*const c_char],
    ) -> VulkanHalResult<()> {
        let supported_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };

        for requested in required_extensions {
            let requested = unsafe { CStr::from_ptr(*requested) };
            if !supported_extensions.iter().any(|prop| {
                let ext_name = unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) };
                ext_name == requested
            }) {
                return Err(VulkanHalError::LayerNotAvailable(
                    requested.to_string_lossy(),
                ));
            }
        }

        Ok(())
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
            VulkanHalError::NoSuitableQueueFamily(flags) => {
                format!("No queue family found with the following properties: {flags:?}")
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

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        panic!("Invalid vulkan state: check error log above.");
    }

    vk::FALSE
}
