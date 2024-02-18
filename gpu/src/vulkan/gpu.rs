use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CStr, CString},
    ptr::{addr_of, addr_of_mut, null},
    sync::{
        atomic::{AtomicBool, AtomicUsize},
        Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use anyhow::{bail, Result};
use ash::extensions::khr::DynamicRendering;
use ash::{
    extensions::ext::{self, DebugUtils},
    prelude::*,
    vk::{
        make_api_version, ApplicationInfo, BufferCreateFlags, CommandBufferAllocateInfo,
        CommandBufferLevel, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCreateFlagsEXT, DebugUtilsMessengerCreateInfoEXT,
        DebugUtilsObjectNameInfoEXT, DependencyInfo, DescriptorBufferInfo, DescriptorImageInfo,
        DeviceCreateFlags, DeviceCreateInfo, DeviceQueueCreateFlags, DeviceQueueCreateInfo,
        Extent3D, FormatFeatureFlags, ImageCreateFlags, ImageTiling, ImageType,
        ImageViewCreateFlags, InstanceCreateFlags, InstanceCreateInfo, MemoryHeap, MemoryHeapFlags,
        PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceProperties, PhysicalDeviceType,
        PipelineCacheCreateFlags, Queue, QueueFlags, ShaderModuleCreateFlags, SharingMode,
        StructureType, SubmitInfo, WriteDescriptorSet, API_VERSION_1_3,
    },
    *,
};
use ash::{
    extensions::khr,
    vk::{
        CommandBufferBeginInfo, CommandBufferResetFlags, FenceCreateFlags, ObjectType,
        PhysicalDeviceDynamicRenderingFeaturesKHR, PhysicalDeviceFeatures2KHR,
        PhysicalDeviceSynchronization2Features, SemaphoreCreateFlags, TaggedStructure,
    },
};

use crate::{
    gpu_resource_manager::{lifetime_cache_constants, quick_hash, GpuResourceMap},
    vulkan::*,
    AccessFlags, BeginRenderPassInfo, BlendMode, BlendOp, BufferCreateInfo, BufferHandle,
    ColorComponentFlags, CommandBuffer, CommandPoolCreateFlags, CommandPoolCreateInfo,
    DescriptorBindingInfo, DescriptorSetDescription, DescriptorSetInfo2, DescriptorSetState,
    Extent2D, FenceHandle, Gpu, GpuConfiguration, HandleType, ImageCreateInfo, ImageFormat,
    ImageHandle, ImageLayout, ImageMemoryBarrier, ImageSubresourceRange, ImageViewCreateInfo,
    ImageViewHandle, LogicOp, MemoryDomain, Offset2D, PipelineBarrierInfo,
    PipelineColorBlendAttachmentState, PipelineStageFlags, PushConstantBlockDescription, QueueType,
    Rect2D, SamplerCreateInfo, SamplerHandle, SemaphoreHandle, ShaderAttribute,
    ShaderModuleCreateInfo, ShaderModuleHandle, Swapchain, TransitionInfo,
    UniformVariableDescription,
};

use crate::Handle;

use log::{debug, error, info, trace, warn};
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use thiserror::Error;
use winit::window::Window;

use crate::{
    gpu_resource_manager::{
        utils::associate_to_handle, AllocatedResourceMap, HasAssociatedHandle, LifetimedCache,
    },
    ShaderInfo,
};

const KHRONOS_VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

associate_to_handle!(VkBuffer, BufferHandle);
associate_to_handle!(VkImage, ImageHandle);
associate_to_handle!(VkImageView, ImageViewHandle);
associate_to_handle!(VkSampler, SamplerHandle);
associate_to_handle!(VkShaderModule, ShaderModuleHandle);
associate_to_handle!(VkSemaphore, SemaphoreHandle);
associate_to_handle!(VkFence, FenceHandle);

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
impl QueueType {
    pub(crate) fn get_vk_queue_index(&self, families: &QueueFamilies) -> u32 {
        match self {
            QueueType::Graphics => families.graphics_family.index,
            QueueType::AsyncCompute => families.graphics_family.index,
            QueueType::Transfer => families.graphics_family.index,
        }
    }
}

#[derive(Default, Clone, Copy)]
struct SupportedFeatures {
    supports_rgb_images: bool,
}

#[derive(Clone)]
struct DescriptorPoolAllocation {
    pool: vk::DescriptorPool,
    max_descriptors: u32,
    allocated_descriptors: u32,
}

pub(crate) struct DescriptorSetAllocation {
    pub(crate) set: vk::DescriptorSet,
    pub(crate) pool_index: usize,
    pub(crate) pool_hash: u64,
}

impl DescriptorPoolAllocation {
    fn new(pool: vk::DescriptorPool, max_descriptors: u32) -> Self {
        Self {
            pool,
            max_descriptors,
            allocated_descriptors: 0,
        }
    }
}

struct DestroyedResource<T: std::fmt::Debug> {
    resource: T,
    destroyed_frame_counter: u64,
    handle_type: HandleType,
}
impl<T: std::fmt::Debug> DestroyedResource<T> {
    const DESTROYED_FRAME_COUNTER: u64 = 3;
    fn new(resource: T, handle_type: HandleType) -> Self {
        Self {
            resource,
            destroyed_frame_counter: Self::DESTROYED_FRAME_COUNTER,
            handle_type,
        }
    }
}

#[derive(Default)]
pub struct DestroyedResources {
    destroyed_shader_modules: RwLock<Vec<DestroyedResource<VkShaderModule>>>,
    destroyed_buffers: RwLock<Vec<DestroyedResource<VkBuffer>>>,
    destroyed_images: RwLock<Vec<DestroyedResource<VkImage>>>,
    destroyed_image_views: RwLock<Vec<DestroyedResource<VkImageView>>>,
    destroyed_samplers: RwLock<Vec<DestroyedResource<VkSampler>>>,
    destroyed_semaphores: RwLock<Vec<DestroyedResource<VkSemaphore>>>,
    destroyed_fences: RwLock<Vec<DestroyedResource<VkFence>>>,
}

pub struct PrimaryCommandBufferInfo {
    pub command_pool: VkCommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub was_started: AtomicBool,
    pub work_done_semaphore: vk::Semaphore,
    pub work_done_fence: vk::Fence,
}

pub struct InFlightFrame {
    pub graphics_command_buffer: PrimaryCommandBufferInfo,
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

    pub queue_families: QueueFamilies,
    pub description: GpuDescription,
    pub gpu_memory_allocator: Arc<RwLock<dyn GpuAllocator>>,
    pub debug_utilities: Option<DebugUtils>,
    descriptor_pool_cache: LifetimedCache<Vec<DescriptorPoolAllocation>>,
    pub(crate) vk_pipeline_cache: vk::PipelineCache,
    pub(crate) graphics_pipeline_cache: LifetimedCache<vk::Pipeline>,
    pub(crate) compute_pipeline_cache: LifetimedCache<vk::Pipeline>,
    pub(crate) descriptor_set_layout_cache: LifetimedCache<vk::DescriptorSetLayout>,
    pub(crate) pipeline_layout_cache: LifetimedCache<vk::PipelineLayout>,
    pub(crate) descriptor_set_cache: LifetimedCache<DescriptorSetAllocation>,
    pub(crate) framebuffer_cache: LifetimedCache<vk::Framebuffer>,

    features: SupportedFeatures,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub dynamic_rendering: DynamicRendering,
    pub allocated_resources: Arc<RwLock<GpuResourceMap>>,
    pub destroyed_resources: DestroyedResources,

    command_pools: Vec<InFlightFrame>,
    current_command_pools: AtomicUsize,

    pub render_graph: RwLock<RenderGraph>,
    pub(crate) image_layouts: RwLock<HashMap<vk::Image, ImageLayoutInfo>>,
    pub(crate) buffer_layouts: RwLock<HashMap<vk::Buffer, BufferLayoutInfo>>,
}
pub struct VkGpu {
    pub(crate) state: Arc<GpuThreadSharedState>,
    pub(crate) staging_buffer: RwLock<VkStagingBuffer>,
}

#[derive(Error, Debug, Clone)]
pub enum GpuError {
    #[error("No physical devices were found on this machine")]
    NoPhysicalDevices,

    #[error("A suitable device with the requested capabilities was not found")]
    NoSuitableDevice,

    #[error("One or more queue families aren't supported")]
    NoQueueFamilyFound(Option<(u32, vk::QueueFamilyProperties)>),

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
        panic!("Invalid vulkan state: check log above");
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
    }
}

impl GpuThreadSharedState {
    pub fn current_frame(&self) -> &InFlightFrame {
        &self.command_pools[self
            .current_command_pools
            .load(std::sync::atomic::Ordering::Relaxed)]
    }

    pub(crate) fn allocated_resources(&self) -> RwLockReadGuard<'_, GpuResourceMap> {
        self.allocated_resources
            .read()
            .expect("Failed to RWLock read Gpu Resource Map")
    }

    pub(crate) fn write_resource_map(&self) -> RwLockWriteGuard<GpuResourceMap> {
        self.allocated_resources
            .write()
            .expect("Failed to lock resource map")
    }

    pub(crate) fn resolve_resource<T: HasAssociatedHandle + Clone + 'static>(
        &self,
        source: &T::AssociatedHandle,
    ) -> T {
        self.allocated_resources().resolve(source)
    }
    pub(crate) fn get_graphics_pipeline_dynamic(
        &self,
        pipeline_state: &GraphicsPipelineState2,
        layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        self.graphics_pipeline_cache.get(&pipeline_state, || {
            self.create_graphics_pipeline_dynamic(pipeline_state, layout)
        })
    }
    fn create_graphics_pipeline_dynamic(
        &self,
        pipeline_state: &GraphicsPipelineState2,
        layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        assert!(pipeline_state.vertex_shader.is_valid());
        let mut stages = vec![];
        let main_name = std::ffi::CString::new("main").unwrap();
        let vertex_shader = self.resolve_resource::<VkShaderModule>(&pipeline_state.vertex_shader);
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
            let fragment_shader =
                self.resolve_resource::<VkShaderModule>(&pipeline_state.fragment_shader);
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

        let color_attachments = pipeline_state
            .color_attachments
            .iter()
            .enumerate()
            .map(|(i, _)| {
                pipeline_state
                    .blend_states
                    .get(i)
                    .cloned()
                    .unwrap_or(PipelineColorBlendAttachmentState {
                        blend_enable: true,
                        src_color_blend_factor: crate::BlendMode::One,
                        dst_color_blend_factor: BlendMode::OneMinusSrcAlpha,
                        color_blend_op: BlendOp::Add,
                        src_alpha_blend_factor: BlendMode::One,
                        dst_alpha_blend_factor: BlendMode::OneMinusSrcAlpha,
                        alpha_blend_op: BlendOp::Add,
                        color_write_mask: ColorComponentFlags::RGBA,
                    })
                    .to_vk()
            })
            .collect::<Vec<_>>();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            s_type: StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: false.to_vk(),
            logic_op: LogicOp::Clear.to_vk(),
            attachment_count: color_attachments.len() as _,
            p_attachments: color_attachments.as_ptr(),
            blend_constants: [0.0; 4],
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

        let color_formats = pipeline_state
            .color_attachments
            .iter()
            .map(|attach| {
                self.resolve_resource::<VkImageView>(&attach.image_view)
                    .format
                    .to_vk()
            })
            .collect::<Vec<_>>();
        let depth_format = pipeline_state.depth_format;
        let stencil_format = ImageFormat::Undefined.to_vk();
        let pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(stencil_format)
            .build();

        let create_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: addr_of!(pipeline_rendering_create_info) as *const c_void,
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
            self.logical_device.create_graphics_pipelines(
                self.vk_pipeline_cache,
                &[create_info],
                get_allocation_callbacks(),
            )
        }
        .expect("Failed to create pipelines");
        info!("Created a new graphics pipeline");
        pipeline[0]
    }

    fn create_compute_pipeline(
        &self,
        pipeline_state: &ComputePipelineState,
        layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let main_name = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: self
                .resolve_resource::<VkShaderModule>(&pipeline_state.shader)
                .inner,
            p_name: main_name.as_ptr(),
            p_specialization_info: std::ptr::null(),
        };
        let create_info = vk::ComputePipelineCreateInfo {
            s_type: StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage,
            layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
        };

        let pipeline = unsafe {
            self.logical_device.create_compute_pipelines(
                self.vk_pipeline_cache,
                &[create_info],
                get_allocation_callbacks(),
            )
        }
        .expect("Failed to create pipelines");
        info!("Created a new compute pipeline");
        pipeline[0]
    }

    pub(crate) fn get_compute_pipeline(
        &self,
        pipeline_state: &ComputePipelineState,
        layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        self.compute_pipeline_cache.get(pipeline_state, || {
            self.create_compute_pipeline(pipeline_state, layout)
        })
    }

    pub(crate) fn create_descriptor_set(
        &self,
        info: &DescriptorSetInfo2,
        layout: vk::DescriptorSetLayout,
    ) -> DescriptorSetAllocation {
        let descriptor_set_bindings = info.descriptor_set_layout();

        let pool_hash = quick_hash(info);

        self.descriptor_pool_cache.use_ref_mut(
            info,
            |pools| {
                let (pool_index, descriptor_pool) = {
                    if let Some((idx, pool)) = pools
                        .iter_mut()
                        .enumerate()
                        .find(|(_, p)| p.allocated_descriptors < p.max_descriptors)
                    {
                        (idx, pool)
                    } else {
                        const MAX_DESCRIPTORS: u32 = 100;
                        let mut samplers = 0;
                        let mut combined_image_samplers = 0;
                        let mut uniform_buffers = 0;
                        let mut storage_buffers = 0;
                        let mut input_attachments = 0;
                        let mut storage_images = 0;

                        for binding in &descriptor_set_bindings.elements {
                            match binding.binding_type {
                                crate::BindingType::Uniform => uniform_buffers += 1,
                                crate::BindingType::Storage => storage_buffers += 1,
                                crate::BindingType::Sampler => samplers += 1,
                                crate::BindingType::CombinedImageSampler => {
                                    combined_image_samplers += 1
                                }
                                crate::BindingType::InputAttachment => input_attachments += 1,
                                crate::BindingType::StorageImage => storage_images += 1,
                            }
                        }

                        let mut pool_sizes = vec![];
                        if samplers > 0 {
                            pool_sizes.push(vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLER,
                                descriptor_count: samplers as _,
                            });
                        }
                        if combined_image_samplers > 0 {
                            pool_sizes.push(vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                descriptor_count: combined_image_samplers as _,
                            });
                        }
                        if uniform_buffers > 0 {
                            pool_sizes.push(vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: uniform_buffers as _,
                            });
                        }
                        if storage_buffers > 0 {
                            pool_sizes.push(vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                descriptor_count: storage_buffers as _,
                            });
                        }
                        if input_attachments > 0 {
                            pool_sizes.push(vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::INPUT_ATTACHMENT,
                                descriptor_count: input_attachments as _,
                            });
                        }
                        if storage_images > 0 {
                            pool_sizes.push(vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_IMAGE,
                                descriptor_count: storage_images as _,
                            });
                        }
                        let pool = unsafe {
                            self.logical_device.create_descriptor_pool(
                                &vk::DescriptorPoolCreateInfo {
                                    s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                                    p_next: std::ptr::null(),
                                    flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                                    max_sets: MAX_DESCRIPTORS,
                                    pool_size_count: pool_sizes.len() as _,
                                    p_pool_sizes: pool_sizes.as_ptr() as *const _,
                                },
                                get_allocation_callbacks(),
                            )
                        }
                        .expect("Failed to create pool");
                        let pool_allocation = DescriptorPoolAllocation::new(pool, MAX_DESCRIPTORS);
                        let idx = pools.len();

                        pools.push(pool_allocation);
                        (idx, &mut pools[idx])
                    }
                };
                let set = unsafe {
                    self.logical_device
                        .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                            s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                            p_next: std::ptr::null(),
                            descriptor_pool: descriptor_pool.pool,
                            descriptor_set_count: 1,
                            p_set_layouts: addr_of!(layout) as *const _,
                        })
                        .expect("Failed to allocate descriptor set")[0]
                };
                descriptor_pool.allocated_descriptors += 1;
                info!("Created a new descriptor set");
                DescriptorSetAllocation {
                    set,
                    pool_hash,
                    pool_index,
                }
            },
            Vec::new,
        )
    }

    pub(crate) fn get_pipeline_layout(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> vk::PipelineLayout {
        self.pipeline_layout_cache.get(descriptor_state, || {
            self.create_pipeline_layout(descriptor_state, &self.descriptor_set_layout_cache)
        })
    }

    fn write_descriptor_set(&self, set: vk::DescriptorSet, info: &DescriptorSetInfo2) {
        let mut buffer_descriptors = vec![];
        let mut image_descriptors = vec![];
        info.bindings.iter().for_each(|b| match &b.ty {
            crate::DescriptorBindingType::StorageBuffer {
                handle,
                offset,
                range,
            } => buffer_descriptors.push((
                b.location,
                DescriptorBufferInfo {
                    buffer: self.resolve_resource::<VkBuffer>(handle).inner,
                    offset: *offset,
                    range: if *range as vk::DeviceSize == crate::WHOLE_SIZE {
                        vk::WHOLE_SIZE
                    } else {
                        *range as _
                    },
                },
                vk::DescriptorType::STORAGE_BUFFER,
            )),
            crate::DescriptorBindingType::UniformBuffer {
                handle,
                offset,
                range,
            } => buffer_descriptors.push((
                b.location,
                DescriptorBufferInfo {
                    buffer: self.resolve_resource::<VkBuffer>(handle).inner,
                    offset: *offset,
                    range: if *range as vk::DeviceSize == crate::WHOLE_SIZE {
                        vk::WHOLE_SIZE
                    } else {
                        *range as _
                    },
                },
                vk::DescriptorType::UNIFORM_BUFFER,
            )),
            crate::DescriptorBindingType::ImageView {
                image_view_handle,
                sampler_handle,
                layout,
            } => image_descriptors.push((
                b.location,
                DescriptorImageInfo {
                    sampler: self.resolve_resource::<VkSampler>(sampler_handle).inner,
                    image_view: self
                        .resolve_resource::<VkImageView>(image_view_handle)
                        .inner,
                    image_layout: layout.to_vk(),
                },
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            )),
            crate::DescriptorBindingType::InputAttachment {
                image_view_handle,
                layout,
            } => image_descriptors.push((
                b.location,
                DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: self
                        .resolve_resource::<VkImageView>(image_view_handle)
                        .inner,
                    image_layout: layout.to_vk(),
                },
                vk::DescriptorType::INPUT_ATTACHMENT,
            )),
            crate::DescriptorBindingType::StorageImage { handle: image } => {
                image_descriptors.push((
                    b.location,
                    DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: self.resolve_resource::<VkImageView>(image).inner,
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    vk::DescriptorType::STORAGE_IMAGE,
                ))
            }
        });

        let mut write_descriptor_sets = vec![];

        for (bind, desc, ty) in &buffer_descriptors {
            write_descriptor_sets.push(WriteDescriptorSet {
                s_type: StructureType::WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: set,
                dst_binding: *bind as _,
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
                dst_set: set,
                dst_binding: *bind as _,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: *ty,
                p_image_info: addr_of!(*desc),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            });
        }
        unsafe {
            self.logical_device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }
    }

    fn create_pipeline_layout(
        &self,
        info: &DescriptorSetState,
        descriptor_set_layout_cache: &LifetimedCache<vk::DescriptorSetLayout>,
    ) -> vk::PipelineLayout {
        info!("Creating a new Pipeline Layout");

        let mut descriptor_set_layouts = vec![];
        for set in &info.sets {
            let layout =
                descriptor_set_layout_cache.get(set, || self.create_descriptor_set_layout(set));
            descriptor_set_layouts.push(layout);
        }
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
            set_layout_count: descriptor_set_layouts.len() as _,
            p_set_layouts: descriptor_set_layouts.as_ptr() as *const _,
            push_constant_range_count: constant_ranges.len() as _,
            p_push_constant_ranges: constant_ranges.as_ptr() as *const _,
        };

        unsafe {
            self.logical_device
                .create_pipeline_layout(&create_info, get_allocation_callbacks())
                .expect("Failed to create pipeline layout")
        }
    }
    pub(crate) fn get_descriptor_sets(
        &self,
        descriptor_set_state: &DescriptorSetState,
    ) -> Vec<vk::DescriptorSet> {
        let mut descriptors = vec![];

        for set_info in &descriptor_set_state.sets {
            let layout = self
                .descriptor_set_layout_cache
                .get(set_info, || self.create_descriptor_set_layout(set_info));
            let descriptor = self.descriptor_set_cache.use_ref(
                &set_info,
                |s| s.set,
                || {
                    let set = self.create_descriptor_set(set_info, layout);
                    self.write_descriptor_set(set.set, set_info);
                    set
                },
            );
            descriptors.push(descriptor)
        }

        descriptors
    }

    fn create_descriptor_set_layout(&self, info: &DescriptorSetInfo2) -> vk::DescriptorSetLayout {
        let descriptor_set_bindings = info.descriptor_set_layout().vk_set_layout_bindings();
        info!("Created a new descriptor set layout");
        unsafe {
            self.logical_device
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

    #[allow(dead_code)]
    pub(crate) fn get_framebuffer(
        &self,
        render_pass_info: &BeginRenderPassInfo,
        render_pass: vk::RenderPass,
    ) -> vk::Framebuffer {
        self.framebuffer_cache.get(render_pass_info, || {
            self.create_framebuffer(render_pass_info, render_pass)
        })
    }

    #[allow(dead_code)]
    fn create_framebuffer(
        &self,
        render_pass_info: &BeginRenderPassInfo,
        render_pass: vk::RenderPass,
    ) -> vk::Framebuffer {
        let mut attachments = render_pass_info
            .color_attachments
            .iter()
            .map(|att| self.resolve_resource::<VkImageView>(&att.image_view).inner)
            .collect::<Vec<_>>();
        if let Some(ref depth) = render_pass_info.depth_attachment {
            attachments.push(
                self.resolve_resource::<VkImageView>(&depth.image_view)
                    .inner,
            );
        }
        let create_info = vk::FramebufferCreateInfo {
            s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass,
            attachment_count: attachments.len() as _,
            p_attachments: attachments.as_ptr(),
            width: render_pass_info.render_area.extent.width,
            height: render_pass_info.render_area.extent.height,
            layers: 1,
        };
        unsafe {
            self.logical_device
                .create_framebuffer(&create_info, get_allocation_callbacks())
                .expect("Failed to create framebuffer")
        }
    }
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
impl VkGpu {
    const MB_64: u64 = 1024 * 1024 * 64;

    pub fn new(configuration: GpuConfiguration) -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let mut instance_extensions = if let Some(window) = configuration.window {
            enumerate_window_extensions(window)?
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
            &device_extensions,
            &instance,
            physical_device,
            &queue_families,
        )?;
        trace!("Created logical device");

        let graphics_queue = Self::get_device_queues(&logical_device, &queue_families)?;
        trace!("Created queues");

        trace!(
            "Created a GPU from a device with name '{}'",
            &description.name
        );

        let gpu_memory_allocator =
            PasstroughAllocator::new(&instance, physical_device.physical_device, &logical_device)?;

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
        let mut command_pools = Vec::with_capacity(crate::constants::MAX_FRAMES_IN_FLIGHT);
        for i in 0..crate::constants::MAX_FRAMES_IN_FLIGHT {
            let graphics_command_pool = VkCommandPool::new(
                logical_device.clone(),
                &queue_families,
                &CommandPoolCreateInfo {
                    queue_type: QueueType::Graphics,
                    flags: CommandPoolCreateFlags::empty(),
                },
            )?;

            let primary_graphics_command_buffer = unsafe {
                logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool: graphics_command_pool.inner,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })
                    .unwrap()
            }[0];

            let async_compute_command_pool = VkCommandPool::new(
                logical_device.clone(),
                &queue_families,
                &CommandPoolCreateInfo {
                    queue_type: QueueType::AsyncCompute,
                    flags: CommandPoolCreateFlags::empty(),
                },
            )?;
            let primary_async_compute_command_buffer = unsafe {
                logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool: async_compute_command_pool.inner,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })
                    .unwrap()
            }[0];

            let transfer_command_pool = VkCommandPool::new(
                logical_device.clone(),
                &queue_families,
                &CommandPoolCreateInfo {
                    queue_type: QueueType::Transfer,
                    flags: CommandPoolCreateFlags::empty(),
                },
            )?;
            let primary_transfer_command_buffer = unsafe {
                logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool: transfer_command_pool.inner,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })
                    .unwrap()
            }[0];

            if let Some(utils) = &debug_utilities {
                let graphics_name = CString::new("Primary Graphics Command Buffer").unwrap();
                let transfer_name = CString::new("Primary Transfer Command Buffer").unwrap();
                let compute_name = CString::new("Primary Async Compute Command Buffer").unwrap();

                unsafe {
                    utils.set_debug_utils_object_name(
                        logical_device.handle(),
                        &DebugUtilsObjectNameInfoEXT {
                            s_type: DebugUtilsObjectNameInfoEXT::STRUCTURE_TYPE,
                            p_next: std::ptr::null(),
                            object_type: ObjectType::COMMAND_BUFFER,
                            object_handle: vk::Handle::as_raw(primary_graphics_command_buffer),
                            p_object_name: graphics_name.as_ptr(),
                        },
                    )?;

                    utils.set_debug_utils_object_name(
                        logical_device.handle(),
                        &DebugUtilsObjectNameInfoEXT {
                            s_type: DebugUtilsObjectNameInfoEXT::STRUCTURE_TYPE,
                            p_next: std::ptr::null(),
                            object_type: ObjectType::COMMAND_BUFFER,
                            object_handle: vk::Handle::as_raw(primary_transfer_command_buffer),
                            p_object_name: transfer_name.as_ptr(),
                        },
                    )?;

                    utils.set_debug_utils_object_name(
                        logical_device.handle(),
                        &DebugUtilsObjectNameInfoEXT {
                            s_type: DebugUtilsObjectNameInfoEXT::STRUCTURE_TYPE,
                            p_next: std::ptr::null(),
                            object_type: ObjectType::COMMAND_BUFFER,
                            object_handle: vk::Handle::as_raw(primary_async_compute_command_buffer),
                            p_object_name: compute_name.as_ptr(),
                        },
                    )?;
                };
            }
            let command_pool = InFlightFrame {
                graphics_command_buffer: PrimaryCommandBufferInfo {
                    command_pool: graphics_command_pool,
                    command_buffer: primary_graphics_command_buffer,
                    was_started: AtomicBool::new(false),
                    work_done_semaphore: make_semaphore(&logical_device),
                    work_done_fence: make_fence(&logical_device, i > 0),
                },
            };
            command_pools.push(command_pool);
        }

        let staging_buffer = RwLock::new(VkStagingBuffer::new(
            &logical_device,
            physical_device.physical_device,
            &instance,
            Self::MB_64,
            queue_families.graphics_family.index,
            graphics_queue,
        )?);
        let state = Arc::new(GpuThreadSharedState {
            entry,
            instance,
            logical_device: logical_device.clone(),
            physical_device,
            graphics_queue,

            description,
            queue_families,
            debug_utilities: debug_utilities.clone(),
            features: supported_features,
            vk_pipeline_cache: pipeline_cache,
            gpu_memory_allocator: Arc::new(RwLock::new(gpu_memory_allocator)),
            messenger,
            compute_pipeline_cache: LifetimedCache::new(60),
            graphics_pipeline_cache: LifetimedCache::new(60 * 60),
            pipeline_layout_cache: LifetimedCache::new(60 * 60),
            descriptor_set_layout_cache: LifetimedCache::new(60 * 60),
            descriptor_pool_cache: LifetimedCache::new(lifetime_cache_constants::NEVER_DEALLOCATE),
            descriptor_set_cache: LifetimedCache::new(60 * 60),
            framebuffer_cache: LifetimedCache::new(lifetime_cache_constants::NEVER_DEALLOCATE),
            dynamic_rendering,
            allocated_resources: Arc::new(RwLock::new(GpuResourceMap::new())),
            destroyed_resources: DestroyedResources::default(),
            command_pools,
            current_command_pools: AtomicUsize::new(0),

            render_graph: RwLock::new(RenderGraph::new(
                logical_device.clone(),
                debug_utilities.clone(),
            )),
            image_layouts: Default::default(),
            buffer_layouts: Default::default(),
        });

        Ok(VkGpu {
            state,
            staging_buffer,
        })
    }

    fn create_instance(
        entry: &Entry,
        configuration: &GpuConfiguration,
        instance_extensions: &[String],
    ) -> VkResult<Instance> {
        const ENGINE_NAME: &str = "PlaygroundEngine";
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
        for (index, queue_family) in all_queue_families.iter().enumerate() {
            if queue_family.queue_count == 0 {
                continue;
            }

            if queue_family
                .queue_flags
                .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            {
                graphics_queue = Some((index as u32, *queue_family));
            }
        }

        match graphics_queue {
            Some(g) => Ok(QueueFamilies {
                graphics_family: QueueFamily {
                    index: g.0,
                    count: g.1.queue_count,
                },
                indices: vec![g.0],
            }),
            _ => Err(GpuError::NoQueueFamilyFound(graphics_queue)),
        }
    }

    fn create_device(
        device_extensions: &[String],
        instance: &Instance,
        selected_device: SelectedPhysicalDevice,
        queue_indices: &QueueFamilies,
    ) -> VkResult<Device> {
        let priority_one: f32 = 1.0;

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
        let queue_create_infos = [make_queue_create_info(queue_indices.graphics_family.index)];

        let device_features = PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,
            depth_clamp: vk::TRUE,

            ..Default::default()
        };

        let mut dynamic_state_features = PhysicalDeviceDynamicRenderingFeaturesKHR {
            p_next: std::ptr::null_mut(),
            s_type: StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
            dynamic_rendering: vk::TRUE,
        };

        let mut synchronization2 = PhysicalDeviceSynchronization2Features {
            s_type: StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            p_next: addr_of_mut!(dynamic_state_features).cast::<c_void>(),
            synchronization2: true.to_vk(),
        };
        let device_features_2 = PhysicalDeviceFeatures2KHR {
            p_next: addr_of_mut!(synchronization2).cast(),
            s_type: StructureType::PHYSICAL_DEVICE_FEATURES_2_KHR,
            features: device_features,
        };

        let create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            p_next: addr_of!(device_features_2).cast(),
            flags: DeviceCreateFlags::empty(),
            queue_create_info_count: queue_create_infos.len() as _,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_extension_count: c_ptr_device_extensions.len() as u32,
            pp_enabled_extension_names: c_ptr_device_extensions.as_ptr(),
            p_enabled_features: std::ptr::null(),

            ..Default::default()
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

    fn get_device_queues(device: &Device, queues: &QueueFamilies) -> Result<Queue, GpuError> {
        let graphics_queue = unsafe { device.get_device_queue(queues.graphics_family.index, 0) };

        Ok(graphics_queue)
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
                    .unwrap_or_default()
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
    pub fn vk_logical_device(&self) -> Device {
        self.state.logical_device.clone()
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

    pub fn wait_device_idle_impl(&self) -> VkResult<()> {
        unsafe { self.vk_logical_device().device_wait_idle() }
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

        let shader_info =
            Self::reflect_spirv(code).expect("TOOO: change return type to anyhow::Result");

        let vk_create_info = vk::ShaderModuleCreateInfo {
            s_type: StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: ShaderModuleCreateFlags::empty(),
            code_size: create_info.code.len() as _,
            p_code,
        };

        let shader = VkShaderModule::create(
            self.vk_logical_device(),
            create_info.label.to_owned(),
            &vk_create_info,
            shader_info,
        )?;

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
                &vk::PipelineCacheCreateInfo {
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

    pub(crate) fn allocated_resources(&self) -> RwLockReadGuard<GpuResourceMap> {
        self.state.allocated_resources()
    }

    fn reflect_spirv(code: &[u32]) -> anyhow::Result<ShaderInfo> {
        let spirv_module = spirv_reflect::ShaderModule::load_u32_data(code);
        let spirv_module = match spirv_module {
            Ok(m) => m,
            Err(e) => anyhow::bail!(e.to_string()),
        };

        let mut info = ShaderInfo {
            entry_point: spirv_module.get_entry_point_name(),
            ..Default::default()
        };

        for input in spirv_module
            .enumerate_input_variables(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            let shader_input = ShaderAttribute {
                name: input.name,
                format: input.format.into(),
                location: input.location,
            };
            info.inputs.push(shader_input);
        }

        for output in spirv_module
            .enumerate_output_variables(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            let shader_output = ShaderAttribute {
                name: output.name,
                format: output.format.into(),
                location: output.location,
            };
            info.outputs.push(shader_output);
        }

        for set in spirv_module
            .enumerate_descriptor_sets(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            let mut set_description = DescriptorSetDescription { bindings: vec![] };

            for binding in &set.bindings {
                let members = parse_members(&binding.block.members);
                info.uniform_variables.extend(members.into_iter());
                let binding = DescriptorBindingInfo {
                    name: binding.name.clone(),
                    binding: binding.binding,
                    ty: match binding.descriptor_type {
                        spirv_reflect::types::ReflectDescriptorType::Undefined => unreachable!(),
                        spirv_reflect::types::ReflectDescriptorType::Sampler => {
                            crate::BindingType::Sampler
                        }
                        spirv_reflect::types::ReflectDescriptorType::CombinedImageSampler => {
                            crate::BindingType::CombinedImageSampler
                        }
                        spirv_reflect::types::ReflectDescriptorType::SampledImage => todo!(),
                        spirv_reflect::types::ReflectDescriptorType::StorageImage => {
                            crate::BindingType::StorageImage
                        }
                        spirv_reflect::types::ReflectDescriptorType::UniformTexelBuffer => todo!(),
                        spirv_reflect::types::ReflectDescriptorType::StorageTexelBuffer => todo!(),
                        spirv_reflect::types::ReflectDescriptorType::UniformBuffer => {
                            crate::BindingType::Uniform
                        }
                        spirv_reflect::types::ReflectDescriptorType::StorageBuffer => {
                            crate::BindingType::Storage
                        }
                        spirv_reflect::types::ReflectDescriptorType::UniformBufferDynamic => {
                            todo!()
                        }
                        spirv_reflect::types::ReflectDescriptorType::StorageBufferDynamic => {
                            todo!()
                        }
                        spirv_reflect::types::ReflectDescriptorType::InputAttachment => {
                            crate::BindingType::InputAttachment
                        }
                        spirv_reflect::types::ReflectDescriptorType::AccelerationStructureNV => {
                            todo!()
                        }
                    },
                };

                set_description.bindings.push(binding);
            }
            info.descriptor_layouts.push(set_description);
        }
        for push_constant in spirv_module
            .enumerate_push_constant_blocks(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            let push_constant_description = PushConstantBlockDescription {
                name: push_constant.name,
                size: push_constant.size,
            };
            info.push_constant_ranges.push(push_constant_description);
        }

        Ok(info)
    }

    pub fn resolve_resource<H: HasAssociatedHandle + Clone + 'static>(
        &self,
        handle: &H::AssociatedHandle,
    ) -> H {
        self.state.resolve_resource(handle)
    }
}

fn enumerate_window_extensions(window: &Window) -> anyhow::Result<&[*const i8; 2]> {
    let display_handle = window.display_handle().unwrap().as_raw();
    let extensions = match display_handle {
        RawDisplayHandle::Windows(_) => {
            const WINDOWS_EXTS: [*const c_char; 2] = [
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(),
            ];
            &WINDOWS_EXTS
        }

        RawDisplayHandle::Wayland(_) => {
            const WAYLAND_EXTS: [*const c_char; 2] = [
                khr::Surface::name().as_ptr(),
                khr::WaylandSurface::name().as_ptr(),
            ];
            &WAYLAND_EXTS
        }

        RawDisplayHandle::Xlib(_) => {
            const XLIB_EXTS: [*const c_char; 2] = [
                khr::Surface::name().as_ptr(),
                khr::XlibSurface::name().as_ptr(),
            ];
            &XLIB_EXTS
        }

        RawDisplayHandle::Xcb(_) => {
            const XCB_EXTS: [*const c_char; 2] = [
                khr::Surface::name().as_ptr(),
                khr::XcbSurface::name().as_ptr(),
            ];
            &XCB_EXTS
        }

        RawDisplayHandle::Android(_) => {
            const ANDROID_EXTS: [*const c_char; 2] = [
                khr::Surface::name().as_ptr(),
                khr::AndroidSurface::name().as_ptr(),
            ];
            &ANDROID_EXTS
        }

        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            const METAL_EXTS: [*const c_char; 2] = [
                khr::Surface::name().as_ptr(),
                ext::MetalSurface::name().as_ptr(),
            ];
            &METAL_EXTS
        }

        _ => return Err(anyhow::Error::from(vk::Result::ERROR_EXTENSION_NOT_PRESENT)),
    };
    Ok(extensions)
}

fn make_semaphore(logical_device: &Device) -> vk::Semaphore {
    unsafe {
        logical_device
            .create_semaphore(
                &vk::SemaphoreCreateInfo {
                    s_type: StructureType::SEMAPHORE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: SemaphoreCreateFlags::empty(),
                },
                get_allocation_callbacks(),
            )
            .unwrap()
    }
}

fn make_fence(logical_device: &Device, signaled: bool) -> vk::Fence {
    unsafe {
        logical_device
            .create_fence(
                &vk::FenceCreateInfo {
                    s_type: StructureType::FENCE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: if signaled {
                        FenceCreateFlags::SIGNALED
                    } else {
                        FenceCreateFlags::empty()
                    },
                },
                get_allocation_callbacks(),
            )
            .unwrap()
    }
}

fn ensure_map_is_empty<H: HasAssociatedHandle + Clone + std::fmt::Debug>(
    map: &AllocatedResourceMap<H>,
) {
    if !map.is_empty() {
        map.for_each(|f| {
            error!(
                "Leaked resource of type {:?}: {f:?}",
                H::AssociatedHandle::handle_type()
            );
        });
    }
}

fn parse_members(
    members: &[spirv_reflect::types::variable::ReflectBlockVariable],
) -> HashMap<String, UniformVariableDescription> {
    let mut map = HashMap::new();
    for member in members {
        let uniform_description = UniformVariableDescription {
            name: member.name.clone(),
            offset: member.offset,
            size: member.size,
            inner_members: parse_members(&member.members),
        };
        map.insert(member.name.clone(), uniform_description);
    }
    map
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
            sharing_mode: SharingMode::EXCLUSIVE,
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
            .write()
            .unwrap()
            .allocate(allocation_requirements)?;
        unsafe {
            self.state
                .logical_device
                .bind_buffer_memory(buffer, allocation.device_memory, 0)
        }?;

        self.set_object_debug_name(create_info.label, buffer)?;

        VkBuffer::create(create_info.label, buffer, memory_domain, allocation)
    }

    fn set_object_debug_name<T: ash::vk::Handle>(
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
            self.staging_buffer
                .write()
                .expect("Failed to lock staging buffer")
                .write_buffer(&self.state.logical_device, buffer.inner, offset, data)
                .expect("Write buffer");
        }
        Ok(())
    }

    pub fn create_image(
        &self,
        create_info: &ImageCreateInfo,
        memory_domain: MemoryDomain,
    ) -> VkResult<(VkImage, ImageFormat)> {
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
                sharing_mode: SharingMode::EXCLUSIVE,
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
            .write()
            .unwrap()
            .allocate(allocation_requirements)?;
        unsafe {
            self.state
                .logical_device
                .bind_image_memory(image, allocation.device_memory, 0)
        }?;
        self.set_object_debug_name(create_info.label, image)?;

        let image = VkImage::create(image, create_info.label, allocation, format)?;

        Ok((image, format))
    }

    pub fn create_sampler(&self, create_info: &SamplerCreateInfo) -> VkResult<VkSampler> {
        VkSampler::create(self.vk_logical_device(), None, create_info)
    }

    pub fn transition_image_layout_impl(
        &self,
        image: ImageHandle,
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
        Ok(())
    }

    pub fn transition_image_layout_in_command_buffer(
        &self,
        image: ImageHandle,
        command_buffer: &mut VkCommandBuffer,
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

    pub fn create_command_buffer(&self, queue_type: QueueType) -> VkResult<VkCommandBuffer> {
        let pool = &self.state.command_pools[self
            .state
            .current_command_pools
            .load(std::sync::atomic::Ordering::Relaxed)];
        let primary_buffer = match queue_type {
            QueueType::Graphics => &pool.graphics_command_buffer,
            QueueType::AsyncCompute => &pool.graphics_command_buffer,
            QueueType::Transfer => &pool.graphics_command_buffer,
        };

        if !primary_buffer
            .was_started
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            primary_buffer
                .was_started
                .store(true, std::sync::atomic::Ordering::Relaxed);
            unsafe {
                self.vk_logical_device()
                    .begin_command_buffer(
                        primary_buffer.command_buffer,
                        &CommandBufferBeginInfo::builder().build(),
                    )
                    .unwrap()
            };
        }
        VkCommandBuffer::new(self.state.clone(), primary_buffer.command_buffer)
    }

    pub fn save_pipeline_cache_impl(&self, path: &str) -> VkResult<()> {
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

    fn update_cycle_for_deleted_resources<T: std::fmt::Debug, F: Fn(&T)>(
        &self,
        deleted_resources: &mut Vec<DestroyedResource<T>>,
        fun: F,
    ) {
        deleted_resources.retain(|el| {
            if el.destroyed_frame_counter == 0 {
                debug!(
                    "Destroyed resource {:?} of type {:?}",
                    el.resource, el.handle_type
                );
                fun(&el.resource);
                false
            } else {
                true
            }
        });
        deleted_resources.iter_mut().for_each(|r| {
            r.destroyed_frame_counter -= 1;
        });
    }

    fn clear_in_flight_frame_state(&self, current_pools: usize) -> anyhow::Result<()> {
        let command_pools = &self.state.command_pools[current_pools];
        let device = &self.state.logical_device;
        unsafe {
            device.wait_for_fences(
                &[command_pools.graphics_command_buffer.work_done_fence],
                true,
                u64::MAX,
            )?;
            device.reset_fences(&[command_pools.graphics_command_buffer.work_done_fence])?;
            device.reset_command_buffer(
                command_pools.graphics_command_buffer.command_buffer,
                CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;

            command_pools
                .graphics_command_buffer
                .was_started
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
    }

    fn destroy_resource<H: HasAssociatedHandle + Clone + std::fmt::Debug + 'static>(
        &self,
        handle: H::AssociatedHandle,
        destroyed_resources: &RwLock<Vec<DestroyedResource<H>>>,
    ) {
        let resources = self
            .state
            .allocated_resources
            .write()
            .expect("Failed to lock resource map");
        let mut map = resources.get_map_mut::<H>();
        let id = handle.id();
        if let Some(resource) = map.remove(handle) {
            destroyed_resources
                .write()
                .expect("Failed to block destroyed resources vec")
                .push(DestroyedResource::new(
                    resource,
                    H::AssociatedHandle::handle_type(),
                ));
        } else {
            warn!("Attempted to destroy resource of type {:?} with id {} which is not valid (freed earlier?): this will cause a panic in the future", H::AssociatedHandle::handle_type(), id);
        }
    }
}

impl Gpu for VkGpu {
    fn update(&self) {
        let device = self.vk_logical_device();
        self.state.descriptor_set_cache.update(|set| unsafe {
            info!("Destroying descriptor set");
            self.state
                .descriptor_pool_cache
                .use_ref_mut_raw(&set.pool_hash, |owner_pool| {
                    let owner_pool = &mut owner_pool[set.pool_index];
                    device
                        .free_descriptor_sets(owner_pool.pool, &[set.set])
                        .expect("Failed to free descriptor set");
                    owner_pool.allocated_descriptors -= 1;
                });
        });
        self.state.descriptor_set_layout_cache.update(|l| unsafe {
            info!("Destroying descriptor layout");
            device.destroy_descriptor_set_layout(l, get_allocation_callbacks());
        });
        self.state.pipeline_layout_cache.update(|p| unsafe {
            info!("Destroying pipeline layout");
            device.destroy_pipeline_layout(p, get_allocation_callbacks());
        });
        self.state.graphics_pipeline_cache.update(|p| unsafe {
            info!("Destroying graphics pipeline");
            device.destroy_pipeline(p, get_allocation_callbacks());
        });
        self.state.compute_pipeline_cache.update(|p| unsafe {
            info!("Destroying compute pipeline");
            device.destroy_pipeline(p, get_allocation_callbacks());
        });

        self.update_cycle_for_deleted_resources(
            &mut self
                .state
                .destroyed_resources
                .destroyed_semaphores
                .write()
                .unwrap(),
            |sm| unsafe {
                self.vk_logical_device()
                    .destroy_semaphore(sm.inner, get_allocation_callbacks())
            },
        );
        self.update_cycle_for_deleted_resources(
            &mut self
                .state
                .destroyed_resources
                .destroyed_fences
                .write()
                .unwrap(),
            |sm| unsafe {
                self.vk_logical_device()
                    .destroy_fence(sm.inner, get_allocation_callbacks())
            },
        );
        self.update_cycle_for_deleted_resources(
            &mut self
                .state
                .destroyed_resources
                .destroyed_shader_modules
                .write()
                .unwrap(),
            |sm| unsafe {
                self.vk_logical_device()
                    .destroy_shader_module(sm.inner, get_allocation_callbacks())
            },
        );
        self.update_cycle_for_deleted_resources(
            &mut self
                .state
                .destroyed_resources
                .destroyed_image_views
                .write()
                .unwrap(),
            |view| unsafe {
                self.vk_logical_device()
                    .destroy_image_view(view.inner, get_allocation_callbacks());
            },
        );
        self.update_cycle_for_deleted_resources(
            &mut self
                .state
                .destroyed_resources
                .destroyed_samplers
                .write()
                .unwrap(),
            |sam| unsafe {
                self.vk_logical_device()
                    .destroy_sampler(sam.inner, get_allocation_callbacks());
            },
        );
        self.update_cycle_for_deleted_resources(
            &mut self
                .state
                .destroyed_resources
                .destroyed_buffers
                .write()
                .unwrap(),
            |buf| unsafe {
                self.vk_logical_device()
                    .destroy_buffer(buf.inner, get_allocation_callbacks());
                self.state
                    .gpu_memory_allocator
                    .write()
                    .unwrap()
                    .deallocate(&buf.allocation);
            },
        );
        self.update_cycle_for_deleted_resources(
            &mut self
                .state
                .destroyed_resources
                .destroyed_images
                .write()
                .unwrap(),
            |img| {
                // Otherwise it's a wrapped image, e.g a swapchain image
                if let Some(allocation) = &img.allocation {
                    unsafe {
                        self.vk_logical_device()
                            .destroy_image(img.inner, get_allocation_callbacks());
                        self.state
                            .gpu_memory_allocator
                            .write()
                            .unwrap()
                            .deallocate(allocation);
                    }
                }
            },
        );
    }

    fn make_semaphore(
        &self,
        create_info: &crate::SemaphoreCreateInfo,
    ) -> anyhow::Result<SemaphoreHandle> {
        let semaphore = VkSemaphore::new(self, &create_info.to_vk(), create_info.label)?;
        let handle = SemaphoreHandle::new();
        self.state
            .allocated_resources
            .write()
            .unwrap()
            .insert(&handle, semaphore);
        Ok(handle)
    }

    fn make_fence(&self, create_info: &crate::FenceCreateInfo) -> anyhow::Result<FenceHandle> {
        let fence = VkFence::new(self, &create_info.to_vk(), create_info.label)?;
        let handle = FenceHandle::new();
        self.state
            .allocated_resources
            .write()
            .unwrap()
            .insert(&handle, fence);
        Ok(handle)
    }

    fn make_shader_module(
        &self,
        info: &ShaderModuleCreateInfo,
    ) -> anyhow::Result<crate::ShaderModuleHandle> {
        let buffer = self.create_shader_module(info)?;
        let handle = ShaderModuleHandle::new();
        self.state
            .allocated_resources
            .write()
            .unwrap()
            .insert(&handle, buffer);

        Ok(handle)
    }

    fn make_buffer(
        &self,
        buffer_info: &BufferCreateInfo,
        memory_domain: MemoryDomain,
    ) -> anyhow::Result<crate::BufferHandle> {
        let buffer = self.create_buffer(buffer_info, memory_domain)?;
        let handle = BufferHandle::new();
        self.state
            .allocated_resources
            .write()
            .unwrap()
            .insert(&handle, buffer);

        Ok(handle)
    }

    fn write_buffer(&self, buffer: &BufferHandle, offset: u64, data: &[u8]) -> anyhow::Result<()> {
        let buffer = self.allocated_resources().resolve(buffer);
        self.write_buffer_data_with_offset(&buffer, offset, data)?;
        Ok(())
    }

    fn read_buffer(
        &self,
        output_buffer: &BufferHandle,
        offset: u64,
        size: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let buffer = self.resolve_resource::<VkBuffer>(output_buffer);
        assert!(size > 0, "Cannot read 0 bytes on a buffer");
        assert!(offset < buffer.allocation.size);
        assert!(size as u64 + offset <= buffer.allocation.size);

        let address = unsafe {
            buffer
                .allocation
                .persistent_ptr
                .expect("Tried to read from a buffer without a persistent ptr!")
                .as_ptr::<u8>()
                .add(offset as _)
        };
        let slice = unsafe { std::slice::from_raw_parts_mut(address, size) };

        Ok(slice.to_vec())
    }

    fn make_image(
        &self,
        info: &ImageCreateInfo,
        memory_domain: MemoryDomain,
        data: Option<&[u8]>,
    ) -> anyhow::Result<ImageHandle> {
        let (image, format) = self.create_image(info, memory_domain)?;

        let handle = ImageHandle::new();

        self.state
            .allocated_resources
            .write()
            .unwrap()
            .insert(&handle, image);

        let write_layered_image = |data: &[u8]| -> anyhow::Result<()> {
            let texel_size = format.texel_size_bytes();
            for i in 0..info.layers {
                let layer_size = (info.width * info.height * texel_size) as usize;
                let offset = i as usize * layer_size;

                self.write_image(
                    &handle,
                    &data[offset..offset + layer_size],
                    Rect2D {
                        offset: Offset2D { x: 0, y: 0 },
                        extent: Extent2D {
                            width: info.width,
                            height: info.height,
                        },
                    },
                    i,
                )?;
            }
            Ok(())
        };

        if let Some(data) = data {
            assert!(!data.is_empty());
            if info.format == ImageFormat::Rgb8 && !self.state.features.supports_rgb_images {
                let mut rgba_data = vec![];
                let rgba_size = info.width * info.height * 4;
                rgba_data.reserve(rgba_size as _);
                for chunk in data.chunks(3) {
                    rgba_data.push(chunk[0]);
                    rgba_data.push(chunk[1]);
                    rgba_data.push(chunk[2]);
                    rgba_data.push(255);
                }

                write_layered_image(&rgba_data)?;
            } else {
                write_layered_image(data)?;
            }
        } else {
            self.transition_image_layout(
                &handle,
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
                    aspect_mask: info.format.aspect_mask(),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            )?;
        }

        Ok(handle)
    }

    fn write_image(
        &self,
        handle: &ImageHandle,
        data: &[u8],
        region: Rect2D,
        layer: u32,
    ) -> anyhow::Result<()> {
        let offset = region.offset;
        let extent = region.extent;
        let vk_image = self.resolve_resource::<VkImage>(handle);

        let device = &self.state.logical_device;

        self.staging_buffer
            .write()
            .expect("Failed to lock staging buffer")
            .write_image(
                device,
                vk_image.inner,
                vk::Offset3D {
                    x: offset.x,
                    y: offset.y,
                    z: 0,
                },
                vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                vk_image.format.texel_size_bytes(),
                vk::ImageSubresourceLayers {
                    aspect_mask: vk_image.format.aspect_mask().to_vk(),
                    mip_level: 0,
                    base_array_layer: layer,
                    layer_count: 1,
                },
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                data,
            )?;
        let new_layout = ImageLayoutInfo {
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            access_flags: vk::AccessFlags2::SHADER_READ,
            pipeline_flags: vk::PipelineStageFlags2::ALL_GRAPHICS,
        };

        self.state
            .image_layouts
            .write()
            .unwrap()
            .entry(vk_image.inner)
            .and_modify(|i| *i = new_layout)
            .or_insert(new_layout);
        Ok(())
    }

    fn make_image_view(
        &self,
        info: &ImageViewCreateInfo,
    ) -> anyhow::Result<crate::ImageViewHandle> {
        let image = self.resolve_resource::<VkImage>(&info.image);
        let view = {
            let create_info: &ImageViewCreateInfo = info;

            let gpu_view_format: ImageFormat = create_info.format;
            let format = if gpu_view_format == image.format {
                create_info.format
            } else {
                warn!(
                    "Creating an image view of an image with a different format: Requested {:?} but image uses {:?}! Using Image format",
                gpu_view_format,
                image.format);
                image.format
            };

            // TODO: implement ToVk for ImageViewCreateInfo
            let vk_create_info = vk::ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ImageViewCreateFlags::empty(),
                image: image.inner,
                view_type: create_info.view_type.to_vk(),
                format: format.to_vk(),
                components: create_info.components.to_vk(),
                subresource_range: create_info.subresource_range.to_vk(),
            };
            VkImageView::create(
                self.vk_logical_device(),
                None,
                &vk_create_info,
                gpu_view_format,
                info.image,
                VkImageViewFlags::empty(),
            )
        }?;
        {
            if let (Some(label), Some(debug)) = (info.label, &self.state.debug_utilities) {
                unsafe {
                    let c_label = CString::new(label).unwrap();
                    debug.set_debug_utils_object_name(
                        self.state.logical_device.handle(),
                        &DebugUtilsObjectNameInfoEXT {
                            s_type: StructureType::DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                            p_next: std::ptr::null(),
                            object_type: vk::ObjectType::IMAGE_VIEW,
                            object_handle: vk::Handle::as_raw(view.inner),
                            p_object_name: c_label.as_ptr(),
                        },
                    )?
                }
            };
        };
        let handle = ImageViewHandle::new();
        self.state
            .allocated_resources
            .write()
            .unwrap()
            .insert(&handle, view);
        Ok(handle)
    }

    fn make_sampler(&self, info: &SamplerCreateInfo) -> anyhow::Result<crate::SamplerHandle> {
        let sampler = self.create_sampler(info)?;
        let handle = SamplerHandle::new();
        self.state
            .allocated_resources
            .write()
            .unwrap()
            .insert(&handle, sampler);

        Ok(handle)
    }

    fn get_shader_info(&self, shader_handle: &ShaderModuleHandle) -> ShaderInfo {
        self.resolve_resource::<VkShaderModule>(shader_handle)
            .shader_info
            .clone()
    }

    fn transition_image_layout(
        &self,
        image: &ImageHandle,
        old_layout: TransitionInfo,
        new_layout: TransitionInfo,
        subresource_range: ImageSubresourceRange,
    ) -> anyhow::Result<()> {
        self.transition_image_layout_impl(*image, old_layout, new_layout, subresource_range)?;
        Ok(())
    }

    fn wait_device_idle(&self) -> anyhow::Result<()> {
        self.wait_device_idle_impl()?;
        Ok(())
    }

    fn save_pipeline_cache(&self, path: &str) -> anyhow::Result<()> {
        self.save_pipeline_cache_impl(path)?;
        Ok(())
    }

    fn start_command_buffer(&self, queue_type: QueueType) -> anyhow::Result<CommandBuffer> {
        let command_buffer = self.create_command_buffer(queue_type)?;
        Ok(CommandBuffer {
            pimpl: Box::new(command_buffer),
        })
    }

    fn create_swapchain(&self, window: &Window) -> anyhow::Result<Swapchain> {
        let pimpl = Box::new(VkSwapchain::new(self, window)?);
        Ok(Swapchain { pimpl })
    }

    fn destroy(&self) {
        self.staging_buffer
            .write()
            .expect("Failed to lock staging buffer")
            .destroy(&self.state.logical_device);
        let resources = self
            .state
            .allocated_resources
            .read()
            .expect("Failed to read allocated resources");
        ensure_map_is_empty(&resources.get_map::<VkImageView>());
        ensure_map_is_empty(&resources.get_map::<VkImage>());
        ensure_map_is_empty(&resources.get_map::<VkBuffer>());
        ensure_map_is_empty(&resources.get_map::<VkShaderModule>());
    }

    fn destroy_semaphore(&self, semaphore: SemaphoreHandle) {
        self.destroy_resource(
            semaphore,
            &self.state.destroyed_resources.destroyed_semaphores,
        );
    }

    fn destroy_fence(&self, fence: FenceHandle) {
        self.destroy_resource(fence, &self.state.destroyed_resources.destroyed_fences);
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        self.destroy_resource(buffer, &self.state.destroyed_resources.destroyed_buffers);
    }

    fn destroy_image(&self, image: ImageHandle) {
        self.destroy_resource(image, &self.state.destroyed_resources.destroyed_images);
    }

    fn destroy_image_view(&self, image_view: ImageViewHandle) {
        self.destroy_resource(
            image_view,
            &self.state.destroyed_resources.destroyed_image_views,
        );
    }

    fn destroy_sampler(&self, sampler: SamplerHandle) {
        self.destroy_resource(sampler, &self.state.destroyed_resources.destroyed_samplers);
    }

    fn destroy_shader_module(&self, shader_module: ShaderModuleHandle) {
        self.destroy_resource(
            shader_module,
            &self.state.destroyed_resources.destroyed_shader_modules,
        );
    }

    fn submit_work(&self) {
        let graphics_buffer = self
            .create_command_buffer(QueueType::Graphics)
            .unwrap()
            .vk_command_buffer();

        let mut staging_buffer = self
            .staging_buffer
            .write()
            .expect("Failed to lock staging buffer");
        staging_buffer
            .flush_operations(&self.state.logical_device)
            .unwrap();

        unsafe {
            let memory_barrier = [vk::MemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
                ..Default::default()
            }];
            self.state.logical_device.cmd_pipeline_barrier2(
                graphics_buffer,
                &DependencyInfo::builder()
                    .memory_barriers(&memory_barrier)
                    .build(),
            );
        }

        let pool = self.state.current_frame();
        self.state
            .render_graph
            .write()
            .expect("Failed to lock render graph")
            .execute(RenderGraphExecutionContext {
                graphics_command_buffer: graphics_buffer,
                graphics_queue: self.state.graphics_queue,
                graphics_queue_index: self.state.queue_families.graphics_family.index,

                image_layouts: &mut self.state.image_layouts.write().unwrap(),
                buffer_layouts: &mut self.state.buffer_layouts.write().unwrap(),
            })
            .unwrap();

        // self.ensure_any_swapchain_image_is_presentable(graphics_buffer);

        unsafe {
            self.state
                .logical_device
                .end_command_buffer(graphics_buffer)
                .unwrap();

            {
                let signal_semaphores = [pool.graphics_command_buffer.work_done_semaphore];
                let command_buffers = [graphics_buffer];
                self.state
                    .logical_device
                    .queue_submit(
                        self.state.graphics_queue,
                        &[SubmitInfo::builder()
                            .signal_semaphores(&signal_semaphores)
                            .command_buffers(&command_buffers)
                            .build()],
                        pool.graphics_command_buffer.work_done_fence,
                    )
                    .unwrap();
            }
        }
    }

    fn end_frame(&self) {
        let next_command_pools = (self
            .state
            .current_command_pools
            .load(std::sync::atomic::Ordering::Relaxed)
            + 1)
            % crate::constants::MAX_FRAMES_IN_FLIGHT;
        self.state
            .current_command_pools
            .store(next_command_pools, std::sync::atomic::Ordering::Relaxed);
        self.clear_in_flight_frame_state(next_command_pools)
            .expect("Failed to clear the command pools");
    }
}
