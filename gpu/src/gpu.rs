use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ptr::addr_of_mut;
use std::{
    cell::RefCell,
    ffi::{c_void, CStr, CString},
    ptr::{addr_of, null},
    sync::Arc,
};

use anyhow::{bail, Result};
use ash::extensions::khr::DynamicRendering;
use ash::vk::{
    DescriptorType, PhysicalDeviceDynamicRenderingFeaturesKHR, PhysicalDeviceFeatures2KHR,
};
use ash::{
    extensions::ext::DebugUtils,
    prelude::*,
    vk::{
        make_api_version, ApplicationInfo, BufferCreateFlags, DebugUtilsMessageSeverityFlagsEXT,
        DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCreateFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DebugUtilsObjectNameInfoEXT, DescriptorBufferInfo,
        DescriptorImageInfo, DeviceCreateFlags, DeviceCreateInfo, DeviceQueueCreateFlags,
        DeviceQueueCreateInfo, Extent3D, FormatFeatureFlags, FramebufferCreateFlags, Handle,
        ImageCreateFlags, ImageTiling, ImageType, ImageViewCreateFlags, InstanceCreateFlags,
        InstanceCreateInfo, MemoryHeap, MemoryHeapFlags, PhysicalDevice, PhysicalDeviceFeatures,
        PhysicalDeviceProperties, PhysicalDeviceType, PipelineCacheCreateFlags,
        PipelineCacheCreateInfo, Queue, QueueFlags, ShaderModuleCreateFlags, SharingMode,
        StructureType, WriteDescriptorSet, API_VERSION_1_3,
    },
    *,
};

use log::{error, trace, warn};
use raw_window_handle::HasRawDisplayHandle;
use thiserror::Error;

use crate::{
    get_allocation_callbacks, BufferCreateInfo, BufferHandle, BufferImageCopyInfo,
    CommandBufferSubmitInfo, CommandPoolCreateFlags, CommandPoolCreateInfo,
    ComputePipelineDescription, DescriptorBindings, DescriptorInfo, DescriptorSetState, Extent2D,
    FramebufferCreateInfo, GPUFence, Gpu, GpuConfiguration, GraphicsPipelineDescription,
    Handle as GpuHandle, ImageCreateInfo, ImageFormat, ImageHandle, ImageLayout,
    ImageMemoryBarrier, ImageSubresourceRange, ImageViewCreateInfo, ImageViewCreateInfo2,
    ImageViewHandle, LogicOp, Offset2D, Offset3D, PipelineBarrierInfo, PipelineStageFlags,
    PipelineState, QueueType, Rect2D, RenderPassDescription, SamplerCreateInfo, SamplerHandle,
    SamplerState, ShaderModuleCreateInfo, ShaderModuleHandle, ToVk, TransitionInfo,
    VkCommandBuffer, VkCommandPool, VkComputePipeline, VkFramebuffer, VkGraphicsPipeline,
    VkImageView, VkRenderPass, VkShaderModule,
};

use super::descriptor_set::PooledDescriptorSetAllocator;

use super::{
    allocator::{GpuAllocator, PasstroughAllocator},
    descriptor_set::DescriptorSetAllocator,
    AccessFlags, AllocationRequirements, DescriptorSetInfo, MemoryDomain, VkBuffer,
    VkDescriptorSet, VkImage, VkSampler,
};

const KHRONOS_VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

pub(crate) struct GpuResourceMap<T>(HashMap<u64, T>);

impl<T> GpuResourceMap<T> {
    fn resolve(&self, id: &u64) -> &T {
        let res = self.0.get(id).expect("Failed to resolve resource");
        res
    }

    fn insert(&mut self, id: u64, res: T) {
        self.0.insert(id, res);
    }

    fn new() -> GpuResourceMap<T> {
        Self(HashMap::new())
    }
}

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

pub(crate) struct PipelineCache {
    device: ash::Device,
    pipelines: RefCell<HashMap<u64, vk::Pipeline>>,
}

impl PipelineCache {
    pub(crate) fn get_pipeline(
        &self,
        pipeline_state: &PipelineState,
        layout: vk::PipelineLayout,
        shader_cache: &GpuResourceMap<VkShaderModule>,
    ) -> vk::Pipeline {
        let mut pipelines = self.pipelines.borrow_mut();
        let mut hasher = DefaultHasher::new();
        pipeline_state.hash(&mut hasher);
        let pipeline_hash = hasher.finish();
        if pipelines.contains_key(&pipeline_hash) {
            return pipelines[&pipeline_hash];
        }

        let pipeline = self.create_pipeline(pipeline_state, layout, shader_cache);
        pipelines.insert(pipeline_hash, pipeline);
        pipeline
    }

    fn new(device: ash::Device) -> Self {
        Self {
            device,
            pipelines: RefCell::new(HashMap::new()),
        }
    }

    fn create_pipeline(
        &self,
        pipeline_state: &PipelineState,
        layout: vk::PipelineLayout,
        shader_cache: &GpuResourceMap<VkShaderModule>,
    ) -> vk::Pipeline {
        let mut stages = vec![];
        let main_name = std::ffi::CString::new("main").unwrap();
        let vertex_shader = shader_cache.resolve(&pipeline_state.vertex_shader.id);
        stages.push(vk::PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::VERTEX,
            module: vertex_shader.inner,
            p_name: main_name.as_ptr(),
            p_specialization_info: std::ptr::null(),
        });
        if !pipeline_state.fragment_shader.is_null() {
            let fragment_shader = shader_cache.resolve(&pipeline_state.fragment_shader.id);
            stages.push(vk::PipelineShaderStageCreateInfo {
                s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fragment_shader.inner,
                p_name: main_name.as_ptr(),
                p_specialization_info: std::ptr::null(),
            });
        }

        let color_attachments = pipeline_state
            .color_blend_states
            .iter()
            .map(|c| c.to_vk())
            .collect::<Vec<_>>();

        let (vertex_input_bindings, vertex_attribute_descriptions) =
            pipeline_state.get_vertex_inputs_description();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            s_type: StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: vertex_input_bindings.len() as _,
            p_vertex_binding_descriptions: vertex_input_bindings.as_ptr() as *const _,
            vertex_attribute_description_count: vertex_attribute_descriptions.len() as _,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions.as_ptr() as *const _,
        };

        let input_assembly_state = pipeline_state.input_assembly_state();
        let rasterization_state = pipeline_state.rasterization_state();
        let multisample_state = pipeline_state.multisample_state();
        let depth_stencil_state = pipeline_state.depth_stencil_state();
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            s_type: StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: false.to_vk(),
            logic_op: LogicOp::NoOp.to_vk(),
            attachment_count: color_attachments.len() as _,
            p_attachments: color_attachments.as_ptr() as *const _,
            blend_constants: [1.0, 1.0, 1.0, 1.0],
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: 1,
            p_viewports: std::ptr::null(),
            scissor_count: 1,
            p_scissors: std::ptr::null(),
        };
        let dynamic_state = pipeline_state.dynamic_state();
        let create_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: stages.len() as _,
            p_stages: stages.as_ptr() as *const _,
            p_vertex_input_state: addr_of!(vertex_input_state),
            p_input_assembly_state: addr_of!(input_assembly_state),
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: addr_of!(viewport_state), // It's part of the dynamic state
            p_rasterization_state: addr_of!(rasterization_state),
            p_multisample_state: addr_of!(multisample_state),
            p_depth_stencil_state: addr_of!(depth_stencil_state),
            p_color_blend_state: addr_of!(color_blend_state),
            p_dynamic_state: addr_of!(dynamic_state),
            layout,
            render_pass: vk::RenderPass::null(),
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
        };
        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
        }
        .expect("Failed to create pipelines");
        pipeline[0]
    }
}

pub(crate) struct DescriptorSetLayoutCache {
    device: ash::Device,
    descriptor_set_layouts: RefCell<HashMap<u64, vk::DescriptorSetLayout>>,
}

impl DescriptorSetLayoutCache {
    fn new(device: ash::Device) -> Self {
        Self {
            device,
            descriptor_set_layouts: RefCell::new(HashMap::new()),
        }
    }
    pub(crate) fn get_descriptor_set_layout(
        &self,
        info: &DescriptorSetState,
    ) -> vk::DescriptorSetLayout {
        let mut descriptor_set_layouts = self.descriptor_set_layouts.borrow_mut();
        let mut hasher = DefaultHasher::new();
        info.hash(&mut hasher);
        let descriptor_set_hash = hasher.finish();
        if descriptor_set_layouts.contains_key(&descriptor_set_hash) {
            return descriptor_set_layouts[&descriptor_set_hash];
        }

        let descriptor_set_layout = self.create_descriptor_set_layout(info);
        descriptor_set_layouts.insert(descriptor_set_hash, descriptor_set_layout);
        descriptor_set_layout
    }

    pub(crate) fn create_descriptor_set_layout(
        &self,
        info: &DescriptorSetState,
    ) -> vk::DescriptorSetLayout {
        let mut descriptor_set_bindings = vec![];
        for (binding_idx, descriptor_binding) in info.bindings.iter().enumerate() {
            for binding_location in &descriptor_binding.locations {
                let stage_flags = binding_location.binding_stage.to_vk();
                let descriptor_type = match binding_location.ty {
                    crate::DescriptorBindingType::BufferRange { .. } => {
                        DescriptorType::UNIFORM_BUFFER
                    }
                    crate::DescriptorBindingType::ImageView { .. } => {
                        DescriptorType::COMBINED_IMAGE_SAMPLER
                    } //                    super::DescriptorType::StorageBuffer(_) => DescriptorType::STORAGE_BUFFER,
                      //                    super::DescriptorType::Sampler(_) => DescriptorType::SAMPLER,
                      //                    super::DescriptorType::CombinedImageSampler(_) => {
                      //                        DescriptorType::COMBINED_IMAGE_SAMPLER
                      //                    }
                };
                let binding = vk::DescriptorSetLayoutBinding {
                    binding: binding_idx as _,
                    descriptor_type,
                    descriptor_count: 1,
                    stage_flags,
                    p_immutable_samplers: std::ptr::null(),
                };
                descriptor_set_bindings.push(binding);
            }
        }
        unsafe {
            self.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo {
                        s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                        binding_count: descriptor_set_bindings.len() as _,
                        p_bindings: descriptor_set_bindings.as_ptr(),
                    },
                    None,
                )
                .expect("Failure in descriptor set layout creation")
        }
    }
}

pub(crate) struct PipelineLayoutCache {
    device: ash::Device,
    cache: RefCell<HashMap<u64, vk::PipelineLayout>>,
}

impl PipelineLayoutCache {
    fn new(device: ash::Device) -> Self {
        Self {
            device,
            cache: RefCell::new(HashMap::new()),
        }
    }
    pub(crate) fn get_pipeline_layout(
        &self,
        info: &DescriptorSetState,
        descriptor_set_layout_cache: &DescriptorSetLayoutCache,
    ) -> vk::PipelineLayout {
        let mut cache = self.cache.borrow_mut();
        let mut hasher = DefaultHasher::new();
        info.hash(&mut hasher);
        let hash = hasher.finish();
        if cache.contains_key(&hash) {
            return cache[&hash];
        }

        let layout = self.create_pipeline_layout(info, descriptor_set_layout_cache);
        cache.insert(hash, layout);
        layout
    }

    pub(crate) fn create_pipeline_layout(
        &self,
        info: &DescriptorSetState,
        descriptor_set_layout_cache: &DescriptorSetLayoutCache,
    ) -> vk::PipelineLayout {
        let descriptor_set_layout = descriptor_set_layout_cache.get_descriptor_set_layout(info);

        let constant_ranges = info
            .push_constant_range
            .iter()
            .map(|c| vk::PushConstantRange {
                stage_flags: c.stage_flags.to_vk(),
                offset: c.offset,
                size: c.size,
            })
            .collect::<Vec<_>>();

        let create_info = vk::PipelineLayoutCreateInfo {
            s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 1,
            p_set_layouts: addr_of!(descriptor_set_layout),
            push_constant_range_count: constant_ranges.len() as _,
            p_push_constant_ranges: constant_ranges.as_ptr() as *const _,
        };

        unsafe {
            self.device
                .create_pipeline_layout(&create_info, get_allocation_callbacks())
                .expect("Failed to create pipeline layout")
        }
    }
}
pub(crate) struct DescriptorSetCache {
    device: ash::Device,
    cache: RefCell<HashMap<u64, vk::DescriptorSet>>,
}

impl DescriptorSetCache {
    fn new(device: ash::Device) -> Self {
        Self {
            device,
            cache: RefCell::new(HashMap::new()),
        }
    }
    pub(crate) fn get(&self, info: &DescriptorBindings) -> vk::DescriptorSet {
        let mut cache = self.cache.borrow_mut();
        let mut hasher = DefaultHasher::new();
        info.hash(&mut hasher);
        let hash = hasher.finish();
        if cache.contains_key(&hash) {
            return cache[&hash];
        }

        let layout = self.create_descriptor_set(info);
        cache.insert(hash, layout);
        layout
    }

    pub(crate) fn create_descriptor_set(&self, info: &DescriptorBindings) -> vk::DescriptorSet {
        todo!()
    }
}

/*
 * All the state that can be shared across threads
 * */
pub struct GpuThreadSharedState {
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
    pub(crate) vk_pipeline_cache: vk::PipelineCache,
    pub(crate) pipeline_cache: PipelineCache,
    pub(crate) descriptor_set_layout_cache: DescriptorSetLayoutCache,
    pub(crate) pipeline_layout_cache: PipelineLayoutCache,
    pub(crate) descriptor_set_cache: DescriptorSetCache,
    features: SupportedFeatures,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub dynamic_rendering: DynamicRendering,
}

impl Drop for GpuThreadSharedState {
    fn drop(&mut self) {
        if let (Some(messenger), Some(debug_utils)) = (&self.messenger, &self.debug_utilities) {
            unsafe {
                self.logical_device
                    .destroy_pipeline_cache(self.vk_pipeline_cache, get_allocation_callbacks());
                debug_utils.destroy_debug_utils_messenger(*messenger, get_allocation_callbacks())
            };
        }
    }
}

/*
 * All the stuff that should be specific for a thread should go here
 * e.g since command pools cannot be shared across threads, each thread should have its own command
 * pool
 * */
pub struct GpuThreadLocalState {
    pub graphics_command_pool: VkCommandPool,
    pub async_compute_command_pool: VkCommandPool,
    pub transfer_command_pool: VkCommandPool,
}

impl GpuThreadLocalState {
    pub fn new(shared_state: Arc<GpuThreadSharedState>) -> VkResult<Self> {
        let graphics_command_pool = VkCommandPool::new(
            shared_state.logical_device.clone(),
            &shared_state.queue_families,
            &CommandPoolCreateInfo {
                queue_type: QueueType::Graphics,
                flags: CommandPoolCreateFlags::empty(),
            },
        )?;

        let async_compute_command_pool = VkCommandPool::new(
            shared_state.logical_device.clone(),
            &shared_state.queue_families,
            &CommandPoolCreateInfo {
                queue_type: QueueType::AsyncCompute,
                flags: CommandPoolCreateFlags::empty(),
            },
        )?;

        let transfer_command_pool = VkCommandPool::new(
            shared_state.logical_device.clone(),
            &shared_state.queue_families,
            &CommandPoolCreateInfo {
                queue_type: QueueType::Transfer,
                flags: CommandPoolCreateFlags::empty(),
            },
        )?;

        Ok(Self {
            graphics_command_pool,
            async_compute_command_pool,
            transfer_command_pool,
        })
    }
}

pub struct VkGpu {
    pub(crate) state: Arc<GpuThreadSharedState>,
    pub(crate) thread_local_state: GpuThreadLocalState,
    pub(crate) staging_buffer: VkBuffer,

    pub(crate) allocated_buffers: RefCell<GpuResourceMap<VkBuffer>>,
    pub(crate) allocated_shader_modules: RefCell<GpuResourceMap<VkShaderModule>>,
    pub(crate) allocated_images: RefCell<GpuResourceMap<VkImage>>,
    pub(crate) allocated_image_views: RefCell<GpuResourceMap<VkImageView>>,
    pub(crate) allocated_samplers: RefCell<GpuResourceMap<VkSampler>>,
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

impl VkGpu {
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

        let state = Arc::new(GpuThreadSharedState {
            entry,
            instance,
            logical_device: logical_device.clone(),
            physical_device,
            graphics_queue,
            async_compute_queue,
            transfer_queue,
            description,
            queue_families,
            debug_utilities,
            features: supported_features,
            vk_pipeline_cache: pipeline_cache,
            gpu_memory_allocator: Arc::new(RefCell::new(gpu_memory_allocator)),
            descriptor_set_allocator: Arc::new(RefCell::new(descriptor_set_allocator)),
            messenger,
            pipeline_cache: PipelineCache::new(logical_device.clone()),
            pipeline_layout_cache: PipelineLayoutCache::new(logical_device.clone()),
            descriptor_set_layout_cache: DescriptorSetLayoutCache::new(logical_device.clone()),
            descriptor_set_cache: DescriptorSetCache::new(logical_device),
            dynamic_rendering,
        });

        let thread_local_state = GpuThreadLocalState::new(state.clone())?;

        let staging_buffer = create_staging_buffer(&state)?;
        Ok(VkGpu {
            state,
            thread_local_state,
            staging_buffer,
            allocated_buffers: RefCell::new(GpuResourceMap::new()),
            allocated_shader_modules: RefCell::new(GpuResourceMap::new()),
            allocated_images: RefCell::new(GpuResourceMap::new()),
            allocated_image_views: RefCell::new(GpuResourceMap::new()),
            allocated_samplers: RefCell::new(GpuResourceMap::new()),
        })
    }

    fn create_instance(
        entry: &Entry,
        configuration: &GpuConfiguration,
        instance_extensions: &[String],
    ) -> VkResult<Instance> {
        const ENGINE_NAME: &'static str = "PlaygroundEngine";
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
        let engine_name = CString::new(ENGINE_NAME).expect("Failed to create valid Engine Engine");

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

    pub fn graphics_command_pool(&self) -> &VkCommandPool {
        &self.thread_local_state.graphics_command_pool
    }

    pub fn async_compute_command_pool(&self) -> &VkCommandPool {
        &self.thread_local_state.async_compute_command_pool
    }

    pub fn transfer_command_pool(&self) -> &VkCommandPool {
        &self.thread_local_state.transfer_command_pool
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

    pub fn state(&self) -> &GpuThreadSharedState {
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
    ) -> VkResult<VkShaderModule> {
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

        let shader = VkShaderModule::create(self.vk_logical_device(), &create_info)?;

        Ok(shader)
    }

    fn create_pipeline_cache(
        logical_device: &Device,
        filename: Option<&str>,
    ) -> VkResult<vk::PipelineCache> {
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

    pub(crate) fn get_pipeline(
        &self,
        pipeline_state: &PipelineState,
        layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        self.state.pipeline_cache.get_pipeline(
            &pipeline_state,
            layout,
            &self.allocated_shader_modules.borrow(),
        )
    }

    pub(crate) fn get_pipeline_layout(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> vk::PipelineLayout {
        self.state
            .pipeline_layout_cache
            .get_pipeline_layout(descriptor_state, &self.state.descriptor_set_layout_cache)
    }

    pub(crate) fn get_descriptor_sets(
        &self,
        bindings: &[DescriptorBindings],
    ) -> Vec<vk::DescriptorSet> {
        let mut descriptors = vec![];

        for binding in bindings {
            self.state.descriptor_set_cache.get(&binding);
        }

        descriptors
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

fn create_staging_buffer(state: &Arc<GpuThreadSharedState>) -> VkResult<VkBuffer> {
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

    let buffer = VkBuffer::create(
        state.logical_device.clone(),
        buffer,
        MemoryDomain::HostVisible,
        allocation,
        state.gpu_memory_allocator.clone(),
    )?;
    Ok(buffer)
}

impl VkGpu {
    pub fn create_buffer(
        &self,
        create_info: &BufferCreateInfo,
        memory_domain: MemoryDomain,
    ) -> VkResult<VkBuffer> {
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

        VkBuffer::create(
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

    pub fn write_buffer_data<T: Copy>(&self, buffer: &VkBuffer, data: &[T]) -> VkResult<()> {
        self.write_buffer_data_with_offset(buffer, 0, data)
    }

    pub fn write_buffer_data_with_offset<T: Copy>(
        &self,
        buffer: &VkBuffer,
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

    pub fn write_image_data(
        &self,
        image: &VkImage,
        data: &[u8],
        offset: Offset2D,
        extent: Extent2D,
        layer: u32,
    ) -> VkResult<()> {
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
            ImageSubresourceRange {
                aspect_mask: image.format().aspect_mask(),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            },
        )?;

        if data.len() < self.staging_buffer.size() {
            self.staging_buffer.write_data(0, &data);
            self.copy_buffer_to_image(&BufferImageCopyInfo {
                source: &self.staging_buffer,
                dest: &image,
                dest_layout: ImageLayout::TransferDst,
                image_offset: Offset3D {
                    x: offset.x,
                    y: offset.y,
                    z: 0,
                },
                image_extent: crate::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                mip_level: 0,
                base_layer: layer as u32,
                num_layers: 1,
            })?;
        } else {
            let intermediary_buffer = self.create_buffer(
                &BufferCreateInfo {
                    label: None,
                    size: data.len(),
                    usage: crate::BufferUsageFlags::TRANSFER_DST
                        | crate::BufferUsageFlags::TRANSFER_SRC,
                },
                MemoryDomain::DeviceLocal,
            )?;
            let mut written = 0;
            while written < data.len() {
                let remain = data.len() - written;
                let written_this_iteration = if remain > self.staging_buffer.size() {
                    self.staging_buffer.size()
                } else {
                    remain
                };
                self.staging_buffer
                    .write_data(0, &data[written..written + written_this_iteration]);
                self.copy_buffer(
                    &self.staging_buffer,
                    &intermediary_buffer,
                    written as _,
                    written_this_iteration,
                )?;

                written += written_this_iteration;
            }
            self.copy_buffer_to_image(&BufferImageCopyInfo {
                source: &self.staging_buffer,
                dest: &image,
                dest_layout: ImageLayout::TransferDst,
                image_offset: Offset3D {
                    x: offset.x,
                    y: offset.y,
                    z: 0,
                },
                image_extent: crate::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                mip_level: 0,
                base_layer: layer as u32,
                num_layers: 1,
            })?;
        }
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
            ImageSubresourceRange {
                aspect_mask: image.format().aspect_mask(),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            },
        )?;
        Ok(())
    }

    pub fn create_image(
        &self,
        create_info: &ImageCreateInfo,
        memory_domain: MemoryDomain,
        data: Option<&[u8]>,
    ) -> VkResult<VkImage> {
        let mut format = create_info.format;
        if format == ImageFormat::Rgb8 && !self.state.features.supports_rgb_images {
            format = ImageFormat::Rgba8;
        }

        let image = unsafe {
            let create_info = vk::ImageCreateInfo {
                s_type: StructureType::IMAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: if create_info.layers == 6 {
                    ImageCreateFlags::CUBE_COMPATIBLE
                } else {
                    ImageCreateFlags::empty()
                },
                image_type: ImageType::TYPE_2D,
                format: format.to_vk(),
                extent: Extent3D {
                    width: create_info.width,
                    height: create_info.height,
                    depth: create_info.depth,
                },
                mip_levels: create_info.mips,
                array_layers: create_info.layers,
                samples: create_info.samples.to_vk(),
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

        let image = VkImage::create(
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

        let write_layered_image = |data: &[u8]| -> VkResult<()> {
            let texel_size = format.texel_size_bytes();
            for i in 0..create_info.layers {
                let layer_size = (create_info.width * create_info.height * texel_size) as usize;
                let offset = i as usize * layer_size;

                self.write_image_data(
                    &image,
                    &data[offset..offset + layer_size],
                    Offset2D { x: 0, y: 0 },
                    Extent2D {
                        width: create_info.width,
                        height: create_info.height,
                    },
                    i,
                )?;
            }
            Ok(())
        };

        if let Some(data) = data {
            assert!(data.len() > 0);
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

                write_layered_image(&rgba_data)?;
            } else {
                write_layered_image(&data)?;
            }
        }

        Ok(image)
    }

    pub fn create_image_view(&self, create_info: &ImageViewCreateInfo) -> VkResult<VkImageView> {
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
        VkImageView::create(
            self.vk_logical_device(),
            &vk_create_info,
            gpu_view_format,
            image,
            create_info.image.extents,
        )
    }
    pub fn create_sampler(&self, create_info: &SamplerCreateInfo) -> VkResult<VkSampler> {
        VkSampler::create(self.vk_logical_device(), create_info)
    }

    pub fn create_framebuffer(
        &self,
        create_info: &FramebufferCreateInfo,
    ) -> VkResult<VkFramebuffer> {
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

        VkFramebuffer::create(self.vk_logical_device(), &create_info)
    }

    pub fn copy_buffer(
        &self,
        source_buffer: &VkBuffer,
        dest_buffer: &VkBuffer,
        dest_offset: u64,
        size: usize,
    ) -> VkResult<()> {
        let mut command_buffer = self.create_command_buffer(QueueType::Transfer)?;

        command_buffer.copy_buffer(source_buffer, dest_buffer, dest_offset, size)?;
        command_buffer.submit(&CommandBufferSubmitInfo::default())?;
        self.wait_queue_idle(QueueType::Transfer)?;

        Ok(())
    }

    pub fn transition_image_layout(
        &self,
        image: &VkImage,
        old_layout: TransitionInfo,
        new_layout: TransitionInfo,
        subresource_range: ImageSubresourceRange,
    ) -> VkResult<()> {
        let mut command_buffer = self.create_command_buffer(QueueType::Graphics)?;

        self.transition_image_layout_in_command_buffer(
            image,
            &mut command_buffer,
            old_layout,
            new_layout,
            subresource_range,
        );
        command_buffer.submit(&crate::CommandBufferSubmitInfo::default())?;
        self.wait_queue_idle(QueueType::Graphics)
    }

    pub fn transition_image_layout_in_command_buffer(
        &self,
        image: &VkImage,
        command_buffer: &mut crate::VkCommandBuffer,
        old_layout: TransitionInfo,
        new_layout: TransitionInfo,
        subresource_range: ImageSubresourceRange,
    ) {
        let memory_barrier = ImageMemoryBarrier {
            src_access_mask: old_layout.access_mask,
            dst_access_mask: new_layout.access_mask,
            old_layout: old_layout.layout,
            new_layout: new_layout.layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range,
        };
        command_buffer.pipeline_barrier(&PipelineBarrierInfo {
            src_stage_mask: old_layout.stage_mask,
            dst_stage_mask: new_layout.stage_mask,
            image_memory_barriers: &[memory_barrier],
            ..Default::default()
        });
    }

    pub fn copy_buffer_to_image(&self, info: &BufferImageCopyInfo) -> VkResult<()> {
        let mut command_buffer = self.create_command_buffer(QueueType::Transfer)?;
        command_buffer.copy_buffer_to_image(info)?;
        command_buffer.submit(&CommandBufferSubmitInfo::default())?;
        self.wait_queue_idle(QueueType::Transfer)?;

        Ok(())
    }

    pub fn create_descriptor_set(&self, info: &DescriptorSetInfo) -> VkResult<VkDescriptorSet> {
        let allocated_descriptor_set = self
            .state
            .descriptor_set_allocator
            .borrow_mut()
            .allocate(info)?;
        self.initialize_descriptor_set(&allocated_descriptor_set.descriptor_set, info)?;
        VkDescriptorSet::create(
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

    pub fn create_render_pass(
        &self,
        description: &RenderPassDescription,
    ) -> VkResult<VkRenderPass> {
        VkRenderPass::new(self, description)
    }

    pub fn create_graphics_pipeline(
        &self,
        description: &GraphicsPipelineDescription,
    ) -> VkResult<VkGraphicsPipeline> {
        VkGraphicsPipeline::new(self, description)
    }

    pub fn create_compute_pipeline(
        &self,
        description: &ComputePipelineDescription,
    ) -> VkResult<VkComputePipeline> {
        VkComputePipeline::new(self, description)
    }

    pub fn create_command_pool(
        &self,
        create_info: &CommandPoolCreateInfo,
    ) -> VkResult<VkCommandPool> {
        VkCommandPool::new(
            self.vk_logical_device(),
            &self.queue_families(),
            create_info,
        )
    }

    pub fn create_command_buffer(&self, queue_type: QueueType) -> VkResult<VkCommandBuffer> {
        let command_pool = match queue_type {
            QueueType::Graphics => self.graphics_command_pool(),
            QueueType::AsyncCompute => self.async_compute_command_pool(),
            QueueType::Transfer => self.transfer_command_pool(),
        };
        VkCommandBuffer::new(self, command_pool, queue_type)
    }

    pub fn save_pipeline_cache(&self, path: &str) -> VkResult<()> {
        let cache_data = unsafe {
            self.vk_logical_device()
                .get_pipeline_cache_data(self.state.vk_pipeline_cache)
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

impl VkGpu {
    fn buffer_map(&self) -> *const GpuResourceMap<VkBuffer> {
        self.allocated_buffers.as_ptr() as *const _
    }
    fn image_map(&self) -> *const GpuResourceMap<VkImage> {
        self.allocated_images.as_ptr() as *const _
    }
    fn image_view_map(&self) -> *const GpuResourceMap<VkImageView> {
        self.allocated_image_views.as_ptr() as *const _
    }
    fn sampler_map(&self) -> *const GpuResourceMap<VkSampler> {
        self.allocated_samplers.as_ptr() as *const _
    }
    pub(crate) fn resolve_buffer(&self, buffer: BufferHandle) -> &VkBuffer {
        let ptr = self.buffer_map();
        let map = unsafe { &*ptr };
        map.resolve(&buffer.id)
    }

    pub(crate) fn resolve_image(&self, image: ImageHandle) -> &VkImage {
        let ptr = self.image_map();
        let map = unsafe { &*ptr };
        map.resolve(&image.id)
    }

    pub(crate) fn resolve_image_view(&self, view: ImageViewHandle) -> &VkImageView {
        let ptr = self.image_view_map();
        let map = unsafe { &*ptr };
        map.resolve(&view.id)
    }

    pub(crate) fn resolve_sampler(&self, handle: SamplerHandle) -> &VkSampler {
        let ptr = self.sampler_map();
        let map = unsafe { &*ptr };
        map.resolve(&handle.id)
    }
}

impl Gpu for VkGpu {
    fn make_shader_module(
        &self,
        info: &ShaderModuleCreateInfo,
    ) -> anyhow::Result<crate::ShaderModuleHandle> {
        let buffer = self.create_shader_module(info)?;
        let handle = ShaderModuleHandle::new(buffer.inner.as_raw());
        self.allocated_shader_modules
            .borrow_mut()
            .insert(handle.id, buffer);

        Ok(handle)
    }

    fn make_buffer(
        &self,
        buffer_info: &BufferCreateInfo,
        memory_domain: MemoryDomain,
    ) -> anyhow::Result<crate::BufferHandle> {
        let buffer = self.create_buffer(buffer_info, memory_domain)?;
        let handle = BufferHandle::new(buffer.inner.as_raw());
        self.allocated_buffers
            .borrow_mut()
            .insert(handle.id, buffer);

        Ok(handle)
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> anyhow::Result<()> {
        let buffer = self.resolve_buffer(buffer);
        self.write_buffer_data_with_offset(buffer, offset, data)?;
        Ok(())
    }

    fn make_image(
        &self,
        info: &ImageCreateInfo,
        memory_domain: MemoryDomain,
    ) -> anyhow::Result<ImageHandle> {
        let image = self.create_image(info, memory_domain, None)?;
        let handle = ImageHandle::new(image.inner.as_raw());

        self.transition_image_layout(
            &image,
            TransitionInfo {
                layout: ImageLayout::Undefined,
                access_mask: AccessFlags::empty(),
                stage_mask: PipelineStageFlags::TOP_OF_PIPE,
            },
            TransitionInfo {
                layout: ImageLayout::ShaderReadOnly,
                access_mask: AccessFlags::empty(),
                stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
            },
            ImageSubresourceRange {
                aspect_mask: crate::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        )?;

        self.allocated_images.borrow_mut().insert(handle.id, image);

        Ok(handle)
    }

    fn write_image(
        &self,
        handle: ImageHandle,
        data: &[u8],
        region: Rect2D,
        layer: u32,
    ) -> anyhow::Result<()> {
        let image = self.resolve_image(handle);
        self.write_image_data(image, data, region.offset, region.extent, layer)?;
        Ok(())
    }

    fn make_image_view(
        &self,
        info: &ImageViewCreateInfo2,
    ) -> anyhow::Result<crate::ImageViewHandle> {
        let image = self.resolve_image(info.image);
        let info = ImageViewCreateInfo {
            image,
            view_type: info.view_type,
            format: info.format,
            components: info.components,
            subresource_range: info.subresource_range,
        };
        let view = self.create_image_view(&info)?;
        let handle = ImageViewHandle::new(view.inner_image_view().as_raw());
        self.allocated_image_views
            .borrow_mut()
            .insert(handle.id, view);
        Ok(handle)
    }

    fn make_sampler(&self, info: &SamplerCreateInfo) -> anyhow::Result<crate::SamplerHandle> {
        let sampler = self.create_sampler(info)?;
        let handle = SamplerHandle::new(sampler.inner.as_raw());
        self.allocated_samplers
            .borrow_mut()
            .insert(handle.id, sampler);

        Ok(handle)
    }
}
