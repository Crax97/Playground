use std::ptr::addr_of_mut;
use std::{
    cell::RefCell,
    ffi::{c_void, CStr, CString},
    ptr::{addr_of, null},
    sync::Arc,
};

use anyhow::{bail, Result};
use ash::extensions::khr::DynamicRendering;
use ash::vk::{PhysicalDeviceDynamicRenderingFeaturesKHR, PhysicalDeviceFeatures2KHR};
use ash::{
    extensions::ext::DebugUtils,
    prelude::*,
    vk::{
        make_api_version, ApplicationInfo, BufferCreateFlags, CommandBufferAllocateInfo,
        CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags,
        CommandPoolCreateFlags, CommandPoolCreateInfo, DebugUtilsMessageSeverityFlagsEXT,
        DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCreateFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DebugUtilsObjectNameInfoEXT, DescriptorBufferInfo,
        DescriptorImageInfo, DeviceCreateFlags, DeviceCreateInfo, DeviceQueueCreateFlags,
        DeviceQueueCreateInfo, Extent3D, Fence, FormatFeatureFlags, FramebufferCreateFlags, Handle,
        ImageCreateFlags, ImageSubresourceLayers, ImageTiling, ImageType, ImageViewCreateFlags,
        InstanceCreateFlags, InstanceCreateInfo, MemoryHeap, MemoryHeapFlags, Offset3D,
        PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceProperties, PhysicalDeviceType,
        PipelineCache, PipelineCacheCreateFlags, PipelineCacheCreateInfo, Queue, QueueFlags,
        SampleCountFlags, ShaderModuleCreateFlags, SharingMode, StructureType, SubmitInfo,
        WriteDescriptorSet, API_VERSION_1_3,
    },
    *,
};

use log::{error, trace, warn};
use raw_window_handle::HasRawDisplayHandle;
use thiserror::Error;
use winit::window::Window;

use crate::{
    get_allocation_callbacks, CommandBuffer, ComponentMapping, ComputePipeline,
    ComputePipelineDescription, Extent2D, GPUFence, GpuFramebuffer, GpuImageView, GpuShaderModule,
    GraphicsPipeline, GraphicsPipelineDescription, ImageAspectFlags, ImageFormat, ImageLayout,
    ImageMemoryBarrier, ImageSubresourceRange, ImageUsageFlags, ImageViewType, PipelineBarrierInfo,
    PipelineStageFlags, QueueType, RenderPass, RenderPassDescription, SamplerCreateInfo, ToVk,
};

use super::descriptor_set::PooledDescriptorSetAllocator;

use super::{
    allocator::{GpuAllocator, PasstroughAllocator},
    descriptor_set::DescriptorSetAllocator,
    AccessFlags, AllocationRequirements, BufferUsageFlags, DescriptorSetInfo, GpuBuffer,
    GpuDescriptorSet, GpuImage, GpuSampler, MemoryDomain,
};

const KHRONOS_VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

pub struct GpuDescription {
    name: String,
}
impl GpuDescription {
    fn new(physical_device: &SelectedPhysicalDevice) -> Self {
        let name =
            unsafe { CStr::from_ptr(physical_device.device_properties.device_name.as_ptr()) };
        let name = name.to_str().expect("Invalid device name");
        let name = name.to_owned();

        Self { name }
    }
}

#[derive(Default, Clone, Copy)]
struct SupportedFeatures {
    supports_rgb_images: bool,
}

pub struct GpuState {
    pub entry: Entry,
    pub instance: Instance,
    pub logical_device: Device,
    pub physical_device: SelectedPhysicalDevice,
    pub graphics_queue: Queue,
    pub async_compute_queue: Queue,
    pub transfer_queue: Queue,
    pub queue_families: QueueFamilies,
    pub description: GpuDescription,
    pub gpu_memory_allocator: Arc<RefCell<dyn GpuAllocator>>,
    pub descriptor_set_allocator: Arc<RefCell<dyn DescriptorSetAllocator>>,
    pub debug_utilities: Option<DebugUtils>,
    pub(crate) pipeline_cache: PipelineCache,
    features: SupportedFeatures,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub dynamic_rendering: DynamicRendering,
}

impl Drop for GpuState {
    fn drop(&mut self) {
        if let (Some(messenger), Some(debug_utils)) = (&self.messenger, &self.debug_utilities) {
            unsafe {
                self.logical_device
                    .destroy_pipeline_cache(self.pipeline_cache, get_allocation_callbacks());
                debug_utils.destroy_debug_utils_messenger(*messenger, get_allocation_callbacks())
            };
        }
    }
}

pub struct GpuThreadLocalState {
    pub graphics_command_pool: vk::CommandPool,
    pub compute_command_pool: vk::CommandPool,
    pub transfer_command_pool: vk::CommandPool,

    shared_state: Arc<GpuState>,
}

impl GpuThreadLocalState {
    pub fn new(shared_state: Arc<GpuState>) -> VkResult<Self> {
        let graphics_command_pool = unsafe {
            shared_state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: null(),
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: shared_state.queue_families.graphics_family.index,
                },
                None,
            )
        }?;
        let compute_command_pool = unsafe {
            shared_state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: null(),
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: shared_state.queue_families.async_compute_family.index,
                },
                None,
            )
        }?;
        let transfer_command_pool = unsafe {
            shared_state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: null(),
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: shared_state.queue_families.transfer_family.index,
                },
                None,
            )
        }?;

        Ok(Self {
            graphics_command_pool,
            compute_command_pool,
            transfer_command_pool,
            shared_state,
        })
    }
}

impl Drop for GpuThreadLocalState {
    fn drop(&mut self) {
        let device = &self.shared_state.logical_device;
        unsafe {
            device
                .device_wait_idle()
                .expect("Failed to wait for device while dropping thread local state");
            device.destroy_command_pool(self.graphics_command_pool, None);
            device.destroy_command_pool(self.compute_command_pool, None);
            device.destroy_command_pool(self.transfer_command_pool, None);
        }
    }
}

pub struct Gpu {
    pub(crate) state: Arc<GpuState>,
    pub(crate) thread_local_state: GpuThreadLocalState,
    pub(crate) staging_buffer: GpuBuffer,
}

pub struct GpuConfiguration<'a> {
    pub app_name: &'a str,
    pub engine_name: &'a str,
    pub pipeline_cache_path: Option<&'a str>,
    pub enable_debug_utilities: bool,
    pub window: Option<&'a Window>,
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

    #[error("Invalid queue family")]
    InvalidQueueFamilies(QueueFamilies),
}

#[derive(Clone, Copy, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct QueueFamilies {
    pub graphics_family: QueueFamily,
    pub async_compute_family: QueueFamily,
    pub transfer_family: QueueFamily,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct SelectedPhysicalDevice {
    pub physical_device: PhysicalDevice,
    pub device_properties: PhysicalDeviceProperties,
    pub device_features: PhysicalDeviceFeatures,
}

unsafe extern "system" fn on_message(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    _message_types: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> u32 {
    let cb_data: vk::DebugUtilsMessengerCallbackDataEXT = *p_callback_data;
    let message = CStr::from_ptr(cb_data.p_message);
    if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("VULKAN ERROR: {:?}", message);
        std::process::abort();
    } else if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::INFO) {
        log::info!("Vulkan - : {:?}", message);
    } else if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("Vulkan - : {:?}", message);
    }

    0
}
impl QueueFamilies {
    pub fn is_valid(&self) -> bool {
        self.graphics_family.count > 0
            && self.async_compute_family.count > 0
            && self.transfer_family.count > 0
            && self.graphics_family.index != self.async_compute_family.index
            && self.graphics_family.index != self.transfer_family.index
    }
}

impl Gpu {
    pub fn new(configuration: GpuConfiguration) -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let mut instance_extensions = if let Some(window) = configuration.window {
            ash_window::enumerate_required_extensions(window.raw_display_handle())?
                .iter()
                .map(|c_ext| unsafe { CStr::from_ptr(*c_ext) })
                .map(|c_str| c_str.to_string_lossy().to_string())
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        if configuration.enable_debug_utilities {
            instance_extensions.push("VK_EXT_debug_utils".into());
        }

        Self::ensure_required_instance_extensions_are_available(&instance_extensions, &entry)?;

        let instance = Self::create_instance(&entry, &configuration, &instance_extensions)?;
        trace!("Created instance");

        let mut device_extensions = vec!["VK_KHR_dynamic_rendering".into()];

        if configuration.window.is_some() {
            device_extensions.push("VK_KHR_swapchain".into());
        }

        let physical_device = Self::select_discrete_physical_device(&instance)?;
        trace!("Created physical device");

        Self::log_physical_device_memory(&physical_device, instance.clone());

        let description = GpuDescription::new(&physical_device);

        trace!("Created presentation surface");

        let queue_families = Self::select_queue_families_indices(&physical_device, &instance)?;
        if !queue_families.is_valid() {
            log::error!("Queue configurations are invalid!");
            bail!(GpuError::InvalidQueueFamilies(queue_families));
        }

        Self::ensure_required_device_extensions_are_available(
            &device_extensions,
            &instance,
            &physical_device,
        )?;

        let supported_features = find_supported_features(&instance, physical_device);

        let logical_device = Self::create_device(
            &configuration,
            &device_extensions,
            &instance,
            physical_device,
            &queue_families,
        )?;
        trace!("Created logical device");

        let (graphics_queue, async_compute_queue, transfer_queue) =
            Self::get_device_queues(&logical_device, &queue_families)?;
        trace!("Created queues");

        trace!(
            "Created a GPU from a device with name '{}'",
            &description.name
        );

        let gpu_memory_allocator =
            PasstroughAllocator::new(&instance, physical_device.physical_device, &logical_device)?;

        let descriptor_set_allocator = PooledDescriptorSetAllocator::new(logical_device.clone())?;

        let debug_utilities = if configuration.enable_debug_utilities {
            let utilities = DebugUtils::new(&entry, &instance);

            Some(utilities)
        } else {
            None
        };

        let messenger = if let Some(utils) = &debug_utilities {
            Some(unsafe {
                utils.create_debug_utils_messenger(
                    &DebugUtilsMessengerCreateInfoEXT {
                        s_type: StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                        p_next: std::ptr::null(),
                        flags: DebugUtilsMessengerCreateFlagsEXT::empty(),
                        message_severity: DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | DebugUtilsMessageSeverityFlagsEXT::INFO
                            | DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                            | DebugUtilsMessageSeverityFlagsEXT::WARNING,
                        message_type: DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                            | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                            | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                        pfn_user_callback: Some(on_message),
                        p_user_data: std::ptr::null_mut(),
                    },
                    get_allocation_callbacks(),
                )
            }?)
        } else {
            None
        };

        let pipeline_cache =
            Self::create_pipeline_cache(&logical_device, configuration.pipeline_cache_path)?;

        let dynamic_rendering = Self::create_dynamic_rendering(&instance, &logical_device)?;

        let state = Arc::new(GpuState {
            entry,
            instance,
            logical_device,
            physical_device,
            graphics_queue,
            async_compute_queue,
            transfer_queue,
            description,
            queue_families,
            debug_utilities,
            features: supported_features,
            pipeline_cache,
            gpu_memory_allocator: Arc::new(RefCell::new(gpu_memory_allocator)),
            descriptor_set_allocator: Arc::new(RefCell::new(descriptor_set_allocator)),
            messenger,
            dynamic_rendering,
        });

        let thread_local_state = GpuThreadLocalState::new(state.clone())?;

        let staging_buffer = create_staging_buffer(&state)?;
        Ok(Gpu {
            state,
            thread_local_state,
            staging_buffer,
        })
    }

    fn create_instance(
        entry: &Entry,
        configuration: &GpuConfiguration,
        instance_extensions: &[String],
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
            enabled_layer_count: if configuration.enable_debug_utilities {
                1
            } else {
                0
            },
            pp_enabled_layer_names: if configuration.enable_debug_utilities {
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
                indices: vec![g.0, a.0, t.0],
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
        device_extensions: &[String],
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

        let device_features = PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,

            ..Default::default()
        };

        let mut dynamic_state_features = PhysicalDeviceDynamicRenderingFeaturesKHR {
            p_next: std::ptr::null_mut(),
            s_type: StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
            dynamic_rendering: vk::TRUE,
        };

        let device_features_2 = PhysicalDeviceFeatures2KHR {
            p_next: addr_of_mut!(dynamic_state_features).cast(),
            s_type: StructureType::PHYSICAL_DEVICE_FEATURES_2_KHR,
            features: device_features,
        };

        let create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            p_next: addr_of!(device_features_2).cast(),
            flags: DeviceCreateFlags::empty(),
            queue_create_info_count: 3,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: if configuration.enable_debug_utilities {
                1
            } else {
                0
            },
            pp_enabled_layer_names: if configuration.enable_debug_utilities {
                addr_of!(vk_layer_khronos_validation)
            } else {
                null()
            },
            enabled_extension_count: c_ptr_device_extensions.len() as u32,
            pp_enabled_extension_names: c_ptr_device_extensions.as_ptr(),
            p_enabled_features: std::ptr::null(),
        };

        unsafe { instance.create_device(selected_device.physical_device, &create_info, None) }
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
            if !all_extensions_c_names.any(|name| name == required_c_name.as_c_str()) {
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
        let all_supported_extensions: Vec<_> = all_extensions
            .iter()
            .map(|ext| {
                unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }
                    .to_str()
                    .expect("Failed to get extension name")
                    .to_owned()
            })
            .collect();

        for requested_extension in device_extensions {
            if !all_supported_extensions.contains(requested_extension) {
                error!(
                    "Device extension {:?} is not supported",
                    requested_extension
                );
            }
        }

        Ok(())
    }
    pub fn instance(&self) -> Instance {
        self.state.instance.clone()
    }

    pub fn dynamic_rendering(&self) -> DynamicRendering {
        self.state.dynamic_rendering.clone()
    }

    pub fn vk_logical_device(&self) -> Device {
        self.state.logical_device.clone()
    }

    pub fn vk_physical_device(&self) -> vk::PhysicalDevice {
        self.state.physical_device.physical_device
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.thread_local_state.graphics_command_pool
    }
    pub fn queue_families(&self) -> QueueFamilies {
        self.state.queue_families.clone()
    }

    pub fn graphics_queue_family_index(&self) -> u32 {
        self.state.queue_families.graphics_family.index
    }

    pub fn graphics_queue(&self) -> Queue {
        self.state.graphics_queue
    }

    pub fn state(&self) -> &GpuState {
        &self.state
    }

    fn log_physical_device_memory(physical_device: &SelectedPhysicalDevice, instance: Instance) {
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device.physical_device)
        };

        let stringify_memory_heap = |heap: MemoryHeap| {
            let flags_str = {
                let mut s = "{ ".to_owned();
                if heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL) {
                    s += "DEVICE_LOCAL | ";
                }
                if heap.flags.contains(MemoryHeapFlags::MULTI_INSTANCE) {
                    s += "MULTI_INSTANCE | ";
                }
                if heap.flags.contains(MemoryHeapFlags::MULTI_INSTANCE_KHR) {
                    s += "MULTI_INSTANCE_KHR | ";
                }
                if heap.flags.contains(MemoryHeapFlags::RESERVED_2_KHR) {
                    s += "RESERVED_2_KHR ";
                }

                s += "}";
                s
            };
            format!("size: {} flags {}", heap.size, flags_str)
        };

        trace!(
            "Device has {} memory types:",
            memory_properties.memory_type_count
        );
        let mut s = String::new();
        for i in 0..memory_properties.memory_type_count {
            let memory_type = memory_properties.memory_types[i as usize];
            let memory_heap = memory_properties.memory_heaps[memory_type.heap_index as usize];
            s += format!(
                "\n\t{}) Memory type {:?} Heap info: {}",
                i,
                memory_type,
                stringify_memory_heap(memory_heap)
            )
            .as_str();
        }
        trace!("{}", s);
    }

    fn initialize_descriptor_set(
        &self,
        descriptor_set: &vk::DescriptorSet,
        info: &DescriptorSetInfo,
    ) -> VkResult<()> {
        let mut buffer_descriptors = vec![];
        let mut image_descriptors = vec![];
        info.descriptors.iter().for_each(|i| match &i.element_type {
            super::DescriptorType::UniformBuffer(buf) => buffer_descriptors.push((
                i.binding,
                DescriptorBufferInfo {
                    buffer: buf.handle.inner,
                    offset: buf.offset,
                    range: if buf.size == crate::WHOLE_SIZE {
                        vk::WHOLE_SIZE
                    } else {
                        buf.size
                    },
                },
                vk::DescriptorType::UNIFORM_BUFFER,
            )),
            super::DescriptorType::StorageBuffer(buf) => buffer_descriptors.push((
                i.binding,
                DescriptorBufferInfo {
                    buffer: buf.handle.inner,
                    offset: buf.offset,
                    range: if buf.size == crate::WHOLE_SIZE {
                        vk::WHOLE_SIZE
                    } else {
                        buf.size
                    },
                },
                vk::DescriptorType::STORAGE_BUFFER,
            )),
            super::DescriptorType::Sampler(sam) => image_descriptors.push((
                i.binding,
                DescriptorImageInfo {
                    sampler: sam.sampler.inner,
                    image_view: sam.image_view.inner,
                    image_layout: sam.image_layout.to_vk(),
                },
                vk::DescriptorType::SAMPLER,
            )),
            super::DescriptorType::CombinedImageSampler(sam) => image_descriptors.push((
                i.binding,
                DescriptorImageInfo {
                    sampler: sam.sampler.inner,
                    image_view: sam.image_view.inner,
                    image_layout: sam.image_layout.to_vk(),
                },
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            )),
        });

        let mut write_descriptor_sets = vec![];

        for (bind, desc, ty) in &buffer_descriptors {
            write_descriptor_sets.push(WriteDescriptorSet {
                s_type: StructureType::WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: *descriptor_set,
                dst_binding: *bind,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: *ty,
                p_image_info: std::ptr::null(),
                p_buffer_info: addr_of!(*desc),
                p_texel_buffer_view: std::ptr::null(),
            });
        }
        for (bind, desc, ty) in &image_descriptors {
            write_descriptor_sets.push(WriteDescriptorSet {
                s_type: StructureType::WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: *descriptor_set,
                dst_binding: *bind,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: *ty,
                p_image_info: addr_of!(*desc),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            });
        }
        unsafe {
            self.vk_logical_device()
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }
        Ok(())
    }

    pub fn wait_device_idle(&self) -> VkResult<()> {
        unsafe { self.vk_logical_device().device_wait_idle() }
    }
    pub fn wait_queue_idle(&self, queue_type: QueueType) -> VkResult<()> {
        unsafe {
            self.vk_logical_device().queue_wait_idle(match queue_type {
                QueueType::Graphics => self.state.graphics_queue,
                QueueType::AsyncCompute => self.state.async_compute_queue,
                QueueType::Transfer => self.state.transfer_queue,
            })
        }
    }

    pub fn physical_device_properties(&self) -> PhysicalDeviceProperties {
        self.state.physical_device.device_properties
    }

    pub fn create_shader_module(
        &self,
        create_info: &ShaderModuleCreateInfo,
    ) -> VkResult<GpuShaderModule> {
        let code = bytemuck::cast_slice(create_info.code);
        let p_code = code.as_ptr();

        assert_eq!(
            p_code as u32 % 4,
            0,
            "Pointers to shader modules code must be 4 byte aligned"
        );

        let create_info = vk::ShaderModuleCreateInfo {
            s_type: StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: ShaderModuleCreateFlags::empty(),
            code_size: create_info.code.len() as _,
            p_code,
        };

        let shader = GpuShaderModule::create(self.vk_logical_device(), &create_info)?;

        Ok(shader)
    }

    fn create_pipeline_cache(
        logical_device: &Device,
        filename: Option<&str>,
    ) -> VkResult<PipelineCache> {
        let mut data_ptr = std::ptr::null();
        let mut len = 0;

        if let Some(filename) = filename {
            let file: std::result::Result<Vec<u8>, std::io::Error> = std::fs::read(filename);
            match file {
                Ok(data) => {
                    len = data.len();
                    data_ptr = data.as_ptr();
                }
                Err(e) => {
                    println!("Failed to read pipeline cache because {}", e);
                }
            };
        }
        unsafe {
            logical_device.create_pipeline_cache(
                &PipelineCacheCreateInfo {
                    s_type: StructureType::PIPELINE_CACHE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: PipelineCacheCreateFlags::empty(),
                    initial_data_size: len as _,
                    p_initial_data: data_ptr as *const c_void,
                },
                get_allocation_callbacks(),
            )
        }
    }

    fn create_dynamic_rendering(
        instance: &Instance,
        device: &Device,
    ) -> VkResult<DynamicRendering> {
        let dynamic_rendering = DynamicRendering::new(instance, device);
        Ok(dynamic_rendering)
    }
}

fn find_supported_features(
    instance: &Instance,
    physical_device: SelectedPhysicalDevice,
) -> SupportedFeatures {
    let mut supported_features = SupportedFeatures::default();

    let rgb_format_properties = unsafe {
        instance.get_physical_device_format_properties(
            physical_device.physical_device,
            vk::Format::R8G8B8_UNORM,
        )
    };

    if (rgb_format_properties.linear_tiling_features
        & rgb_format_properties.optimal_tiling_features)
        .contains(
            FormatFeatureFlags::COLOR_ATTACHMENT
                | FormatFeatureFlags::SAMPLED_IMAGE
                | FormatFeatureFlags::TRANSFER_DST
                | FormatFeatureFlags::TRANSFER_SRC,
        )
    {
        supported_features.supports_rgb_images = true;
        trace!("Selected physical device supports RGB Images");
    }
    supported_features
}

fn create_staging_buffer(state: &Arc<GpuState>) -> VkResult<GpuBuffer> {
    let mb_64 = 1024 * 1024 * 64;
    let create_info: vk::BufferCreateInfo = vk::BufferCreateInfo {
        s_type: StructureType::BUFFER_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: BufferCreateFlags::empty(),
        size: mb_64 as u64,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        sharing_mode: SharingMode::CONCURRENT,
        queue_family_index_count: state.queue_families.indices.len() as _,
        p_queue_family_indices: state.queue_families.indices.as_ptr(),
    };

    let buffer = unsafe { state.logical_device.create_buffer(&create_info, None) }?;
    let memory_requirements =
        unsafe { state.logical_device.get_buffer_memory_requirements(buffer) };
    let allocation_requirements = AllocationRequirements {
        memory_requirements,
        memory_domain: MemoryDomain::HostVisible,
    };
    let allocation = state
        .gpu_memory_allocator
        .borrow_mut()
        .allocate(allocation_requirements)?;
    unsafe {
        state
            .logical_device
            .bind_buffer_memory(buffer, allocation.device_memory, 0)
    }?;

    let buffer = GpuBuffer::create(
        state.logical_device.clone(),
        buffer,
        MemoryDomain::HostVisible,
        allocation,
        state.gpu_memory_allocator.clone(),
    )?;
    Ok(buffer)
}

pub struct ImageCreateInfo<'a> {
    pub label: Option<&'a str>,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub usage: ImageUsageFlags,
}

pub struct ImageViewCreateInfo<'a> {
    pub image: &'a GpuImage,
    pub view_type: ImageViewType,
    pub format: ImageFormat,
    pub components: ComponentMapping,
    pub subresource_range: ImageSubresourceRange,
}
pub struct BufferCreateInfo<'a> {
    pub label: Option<&'a str>,
    pub size: usize,
    pub usage: BufferUsageFlags,
}

#[derive(Clone, Copy)]
pub struct TransitionInfo {
    pub layout: ImageLayout,
    pub access_mask: AccessFlags,
    pub stage_mask: PipelineStageFlags,
}

#[derive(Clone, Copy)]
pub struct FramebufferCreateInfo<'a> {
    pub render_pass: &'a RenderPass,
    pub attachments: &'a [&'a GpuImageView],
    pub width: u32,
    pub height: u32,
}

pub struct ShaderModuleCreateInfo<'a> {
    pub code: &'a [u8],
}

impl Gpu {
    pub fn create_buffer(
        &self,
        create_info: &BufferCreateInfo,
        memory_domain: MemoryDomain,
    ) -> VkResult<GpuBuffer> {
        let size = create_info.size as u64;
        assert_ne!(size, 0, "Can't create a buffer with size 0!");

        let vk_usage = create_info.usage.to_vk();

        let create_info_vk = vk::BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: BufferCreateFlags::empty(),
            size,
            usage: vk_usage
                | if memory_domain.contains(MemoryDomain::HostVisible) {
                    vk::BufferUsageFlags::empty()
                } else {
                    vk::BufferUsageFlags::TRANSFER_DST
                },
            sharing_mode: SharingMode::CONCURRENT,
            queue_family_index_count: self.state.queue_families.indices.len() as _,
            p_queue_family_indices: self.state.queue_families.indices.as_ptr(),
        };
        let buffer = unsafe {
            self.state
                .logical_device
                .create_buffer(&create_info_vk, None)
        }?;
        let memory_requirements = unsafe {
            self.state
                .logical_device
                .get_buffer_memory_requirements(buffer)
        };

        let allocation_requirements = AllocationRequirements {
            memory_requirements,
            memory_domain,
        };

        let allocation = self
            .state
            .gpu_memory_allocator
            .borrow_mut()
            .allocate(allocation_requirements)?;
        unsafe {
            self.state
                .logical_device
                .bind_buffer_memory(buffer, allocation.device_memory, 0)
        }?;

        self.set_object_debug_name(create_info.label, buffer)?;

        GpuBuffer::create(
            self.vk_logical_device(),
            buffer,
            memory_domain,
            allocation,
            self.state.gpu_memory_allocator.clone(),
        )
    }

    fn set_object_debug_name<T: Handle>(
        &self,
        label: Option<&str>,
        object: T,
    ) -> Result<(), vk::Result> {
        if let (Some(label), Some(debug)) = (label, &self.state.debug_utilities) {
            unsafe {
                let c_label = CString::new(label).unwrap();
                debug.set_debug_utils_object_name(
                    self.vk_logical_device().handle(),
                    &DebugUtilsObjectNameInfoEXT {
                        s_type: StructureType::DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                        p_next: std::ptr::null(),
                        object_type: T::TYPE,
                        object_handle: object.as_raw(),
                        p_object_name: c_label.as_ptr(),
                    },
                )?
            }
        };
        Ok(())
    }

    pub fn write_buffer_data<T: Copy>(&self, buffer: &GpuBuffer, data: &[T]) -> VkResult<()> {
        self.write_buffer_data_with_offset(buffer, 0, data)
    }

    pub fn write_buffer_data_with_offset<T: Copy>(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        data: &[T],
    ) -> VkResult<()> {
        if data.is_empty() {
            return Ok(());
        }

        if buffer.memory_domain.contains(MemoryDomain::HostVisible) {
            buffer.write_data(offset, data);
        } else {
            self.staging_buffer.write_data(0, data);
            self.copy_buffer(
                &self.staging_buffer,
                buffer,
                offset,
                std::mem::size_of_val(data),
            )?;
        }
        Ok(())
    }

    pub fn write_image_data(&self, image: &GpuImage, data: &[u8]) -> VkResult<()> {
        self.staging_buffer.write_data(0, data);

        self.transition_image_layout(
            image,
            TransitionInfo {
                layout: ImageLayout::Undefined,
                access_mask: AccessFlags::empty(),
                stage_mask: PipelineStageFlags::TOP_OF_PIPE,
            },
            TransitionInfo {
                layout: ImageLayout::TransferDst,
                access_mask: AccessFlags::TRANSFER_WRITE,
                stage_mask: PipelineStageFlags::TRANSFER,
            },
            ImageAspectFlags::COLOR,
        )?;

        self.copy_buffer_to_image(
            &self.staging_buffer,
            image,
            image.extents.width,
            image.extents.height,
        )?;
        self.transition_image_layout(
            image,
            TransitionInfo {
                layout: ImageLayout::TransferDst,
                access_mask: AccessFlags::TRANSFER_WRITE,
                stage_mask: PipelineStageFlags::TRANSFER,
            },
            TransitionInfo {
                layout: ImageLayout::ShaderReadOnly,
                access_mask: AccessFlags::SHADER_READ,
                stage_mask: PipelineStageFlags::FRAGMENT_SHADER | PipelineStageFlags::VERTEX_SHADER,
            },
            ImageAspectFlags::COLOR,
        )?;
        Ok(())
    }

    pub fn create_image(
        &self,
        create_info: &ImageCreateInfo,
        memory_domain: MemoryDomain,
        data: Option<&[u8]>,
    ) -> VkResult<GpuImage> {
        let mut format = create_info.format;
        if format == ImageFormat::Rgb8 && !self.state.features.supports_rgb_images {
            format = ImageFormat::Rgba8;
        }

        let image = unsafe {
            let create_info = vk::ImageCreateInfo {
                s_type: StructureType::IMAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ImageCreateFlags::empty(),
                image_type: ImageType::TYPE_2D,
                format: format.to_vk(),
                extent: Extent3D {
                    width: create_info.width,
                    height: create_info.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCountFlags::TYPE_1,
                tiling: if memory_domain.contains(MemoryDomain::HostVisible) {
                    ImageTiling::LINEAR
                } else {
                    ImageTiling::OPTIMAL
                },
                usage: create_info.usage.to_vk(),
                sharing_mode: SharingMode::CONCURRENT,
                queue_family_index_count: self.state.queue_families.indices.len() as _,
                p_queue_family_indices: self.state.queue_families.indices.as_ptr(),
                initial_layout: ImageLayout::Undefined.to_vk(),
            };
            self.state.logical_device.create_image(&create_info, None)?
        };
        let memory_requirements = unsafe {
            self.state
                .logical_device
                .get_image_memory_requirements(image)
        };
        let allocation_requirements = AllocationRequirements {
            memory_requirements,
            memory_domain,
        };
        let allocation = self
            .state
            .gpu_memory_allocator
            .borrow_mut()
            .allocate(allocation_requirements)?;
        unsafe {
            self.state
                .logical_device
                .bind_image_memory(image, allocation.device_memory, 0)
        }?;
        self.set_object_debug_name(create_info.label, image)?;

        let image = GpuImage::create(
            self,
            image,
            allocation,
            self.state.gpu_memory_allocator.clone(),
            Extent2D {
                width: create_info.width,
                height: create_info.height,
            },
            format.into(),
        )?;

        if let Some(data) = data {
            if create_info.format == ImageFormat::Rgb8 && !self.state.features.supports_rgb_images {
                let mut rgba_data = vec![];
                let rgba_size = create_info.width * create_info.height * 4;
                rgba_data.reserve(rgba_size as _);
                for chunk in data.chunks(3) {
                    rgba_data.push(chunk[0]);
                    rgba_data.push(chunk[1]);
                    rgba_data.push(chunk[2]);
                    rgba_data.push(255);
                }

                self.write_image_data(&image, &rgba_data)?
            } else {
                self.write_image_data(&image, data)?
            }
        }

        Ok(image)
    }

    pub fn create_image_view(&self, create_info: &ImageViewCreateInfo) -> VkResult<GpuImageView> {
        let image = create_info.image.inner;

        let gpu_view_format: ImageFormat = create_info.format.into();
        let format = if gpu_view_format == create_info.image.format {
            create_info.format
        } else {
            warn!(
                "Creating an image view of an image with a different format: Requested {:?} but image uses {:?}! Using Image format",
            gpu_view_format,
            create_info.image.format);
            create_info.image.format
        };

        // TODO: implement ToVk for ImageViewCreateInfo
        let vk_create_info = vk::ImageViewCreateInfo {
            s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: ImageViewCreateFlags::empty(),
            image,
            view_type: create_info.view_type.to_vk(),
            format: format.to_vk(),
            components: create_info.components.to_vk(),
            subresource_range: create_info.subresource_range.to_vk(),
        };
        GpuImageView::create(
            self.vk_logical_device(),
            &vk_create_info,
            gpu_view_format,
            image,
            create_info.image.extents,
        )
    }
    pub fn create_sampler(&self, create_info: &SamplerCreateInfo) -> VkResult<GpuSampler> {
        GpuSampler::create(self.vk_logical_device(), create_info)
    }

    pub fn create_framebuffer(
        &self,
        create_info: &FramebufferCreateInfo,
    ) -> VkResult<GpuFramebuffer> {
        let attachments: Vec<_> = create_info.attachments.iter().map(|a| a.inner).collect();
        let create_info = vk::FramebufferCreateInfo {
            s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: FramebufferCreateFlags::empty(),
            render_pass: create_info.render_pass.inner,

            attachment_count: attachments.len() as _,
            p_attachments: attachments.as_ptr(),
            width: create_info.width,
            height: create_info.height,

            // We only support one single framebuffer
            layers: 1,
        };

        GpuFramebuffer::create(self.vk_logical_device(), &create_info)
    }

    pub fn copy_buffer(
        &self,
        source_buffer: &GpuBuffer,
        dest_buffer: &GpuBuffer,
        dest_offset: u64,
        size: usize,
    ) -> VkResult<()> {
        unsafe {
            let command_pool = self.state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandPoolCreateFlags::empty(),
                    queue_family_index: self.graphics_queue_family_index(),
                },
                None,
            )?;

            let command_buffer =
                self.state
                    .logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })?[0];

            self.state.logical_device.begin_command_buffer(
                command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )?;

            let src_buffer = source_buffer.inner;
            let dst_buffer = dest_buffer.inner;

            self.state.logical_device.cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: dest_offset as _,
                    size: size as u64,
                }],
            );
            self.state
                .logical_device
                .end_command_buffer(command_buffer)?;
            self.state.logical_device.queue_submit(
                self.state.graphics_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 0,
                    p_wait_semaphores: std::ptr::null(),
                    p_wait_dst_stage_mask: std::ptr::null(),
                    command_buffer_count: 1,
                    p_command_buffers: addr_of!(command_buffer),
                    signal_semaphore_count: 0,
                    p_signal_semaphores: std::ptr::null(),
                }],
                Fence::null(),
            )?;
            self.state
                .logical_device
                .queue_wait_idle(self.graphics_queue())?;

            self.state
                .logical_device
                .free_command_buffers(command_pool, &[command_buffer]);
            self.state
                .logical_device
                .destroy_command_pool(command_pool, None);
        }

        Ok(())
    }

    pub fn transition_image_layout(
        &self,
        image: &GpuImage,
        old_layout: TransitionInfo,
        new_layout: TransitionInfo,
        aspect_mask: ImageAspectFlags,
    ) -> VkResult<()> {
        let mut command_buffer = super::CommandBuffer::new(self, crate::QueueType::Graphics)?;

        self.transition_image_layout_in_command_buffer(
            image,
            &mut command_buffer,
            old_layout,
            new_layout,
            aspect_mask,
        );
        command_buffer.submit(&crate::CommandBufferSubmitInfo::default())?;
        self.wait_queue_idle(QueueType::Graphics)
    }

    pub fn transition_image_layout_in_command_buffer(
        &self,
        image: &GpuImage,
        command_buffer: &mut crate::CommandBuffer,
        old_layout: TransitionInfo,
        new_layout: TransitionInfo,
        aspect_mask: ImageAspectFlags,
    ) {
        let memory_barrier = ImageMemoryBarrier {
            src_access_mask: old_layout.access_mask,
            dst_access_mask: new_layout.access_mask,
            old_layout: old_layout.layout,
            new_layout: new_layout.layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };
        command_buffer.pipeline_barrier(&PipelineBarrierInfo {
            src_stage_mask: old_layout.stage_mask,
            dst_stage_mask: new_layout.stage_mask,
            image_memory_barriers: &[memory_barrier],
            ..Default::default()
        });
    }

    pub fn copy_buffer_to_image(
        &self,
        source_buffer: &GpuBuffer,
        dest_image: &GpuImage,
        width: u32,
        height: u32,
    ) -> VkResult<()> {
        unsafe {
            let command_pool = self.state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandPoolCreateFlags::empty(),
                    queue_family_index: self.graphics_queue_family_index(),
                },
                None,
            )?;

            let command_buffer =
                self.state
                    .logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })?[0];

            self.state.logical_device.begin_command_buffer(
                command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )?;

            let src_buffer = source_buffer.inner;
            let dst_image = dest_image.inner;

            self.state.logical_device.cmd_copy_buffer_to_image(
                command_buffer,
                src_buffer,
                dst_image,
                ImageLayout::TransferDst.to_vk(),
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: ImageSubresourceLayers {
                        aspect_mask: ImageAspectFlags::COLOR.to_vk(),
                        mip_level: 0,
                        layer_count: 1,
                        base_array_layer: 0,
                    },
                    image_offset: Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                }],
            );
            self.state
                .logical_device
                .end_command_buffer(command_buffer)?;
            self.state.logical_device.queue_submit(
                self.state.graphics_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 0,
                    p_wait_semaphores: std::ptr::null(),
                    p_wait_dst_stage_mask: std::ptr::null(),
                    command_buffer_count: 1,
                    p_command_buffers: addr_of!(command_buffer),
                    signal_semaphore_count: 0,
                    p_signal_semaphores: std::ptr::null(),
                }],
                Fence::null(),
            )?;
            self.state
                .logical_device
                .queue_wait_idle(self.graphics_queue())?;

            self.state
                .logical_device
                .free_command_buffers(command_pool, &[command_buffer]);
            self.state
                .logical_device
                .destroy_command_pool(command_pool, None);
        }

        Ok(())
    }

    pub fn create_descriptor_set(&self, info: &DescriptorSetInfo) -> VkResult<GpuDescriptorSet> {
        let allocated_descriptor_set = self
            .state
            .descriptor_set_allocator
            .borrow_mut()
            .allocate(info)?;
        self.initialize_descriptor_set(&allocated_descriptor_set.descriptor_set, info)?;
        GpuDescriptorSet::create(
            allocated_descriptor_set,
            self.state.descriptor_set_allocator.clone(),
        )
    }

    pub fn wait_for_fences(
        &self,
        fences: &[&GPUFence],
        wait_all: bool,
        timeout_ns: u64,
    ) -> VkResult<()> {
        let fences: Vec<_> = fences.iter().map(|f| f.inner).collect();
        unsafe {
            self.vk_logical_device()
                .wait_for_fences(&fences, wait_all, timeout_ns)
        }
    }

    pub fn reset_fences(&self, fences: &[&GPUFence]) -> VkResult<()> {
        let fences: Vec<_> = fences.iter().map(|f| f.inner).collect();
        unsafe { self.vk_logical_device().reset_fences(&fences) }
    }

    pub fn create_render_pass(&self, description: &RenderPassDescription) -> VkResult<RenderPass> {
        RenderPass::new(self, description)
    }

    pub fn create_graphics_pipeline(
        &self,
        description: &GraphicsPipelineDescription,
    ) -> VkResult<GraphicsPipeline> {
        GraphicsPipeline::new(self, description)
    }

    pub fn create_compute_pipeline(
        &self,
        description: &ComputePipelineDescription,
    ) -> VkResult<ComputePipeline> {
        ComputePipeline::new(self, description)
    }

    pub fn create_command_buffer(&self, queue_type: QueueType) -> VkResult<CommandBuffer> {
        CommandBuffer::new(self, queue_type)
    }

    pub fn save_pipeline_cache(&self, path: &str) -> VkResult<()> {
        let cache_data = unsafe {
            self.vk_logical_device()
                .get_pipeline_cache_data(self.state.pipeline_cache)
        }?;

        match std::fs::write(path, cache_data) {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to write pipeline cache: {e}");
                Err(vk::Result::ERROR_UNKNOWN)
            }
        }
    }

    pub fn allocator(&self) -> Arc<RefCell<dyn GpuAllocator>> {
        self.state.gpu_memory_allocator.clone()
    }
}
