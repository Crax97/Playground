mod util;

#[cfg(feature = "swapchain")]
mod swapchain;

use crate::hal::vulkan::util::{
    DescriptorPoolInfo, LayoutInfo, ToVk, VulkanBindingSet, VulkanBuffer, VulkanImage,
    VulkanSampler,
};
use crate::hal::{CommandRecorder, CommandRecorderAllocator, Hal, QueueType, ResourceAccessMode};
#[cfg(debug_assertions)]
use crate::util::check;
use crate::util::{hash_type, Handle};
use crate::{
    AttachmentAccessMode, AttachmentStoreOp, BindingSet, BindingSetDescription, BindingSetElement,
    BindingSetElementKind, BindingSetLayout, BindingSetLayoutInfo, Buffer, BufferDescription,
    ComputePipeline, ComputePipelineDescription, DeviceConfiguration, DeviceFeatures,
    DevicePreference, FilterMode, Framebuffer, GraphicsPipeline, GraphicsPipelineDescription,
    GraphicsPipelineLayout, Image, ImageDescription, ImageDimension, ImageFormat, ImageRegion,
    ImageSubresource, MemoryDomain, MgpuError, MgpuResult, OwnedBindingSetLayout,
    OwnedBindingSetLayoutInfo, PresentMode, PushConstantInfo, RenderPassInfo, Sampler,
    SamplerDescription, ShaderAttribute, ShaderModule, ShaderModuleLayout, ShaderStageFlags,
    ShaderVariable, StorageAccessMode, SwapchainCreationInfo, VariableType, VertexAttributeFormat,
};
#[cfg(debug_assertions)]
use std::collections::HashSet;

#[cfg(feature = "swapchain")]
use crate::SwapchainInfo;
use ash::vk::{ComponentMapping, Handle as AshHandle, ImageLayout, QueueFlags};
use ash::{vk, Entry, Instance};
use gpu_allocator::vulkan::{
    AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use log::info;
use raw_window_handle::{DisplayHandle, WindowHandle};
use spirq::ty::{AccessType, CombinedImageSamplerType, Type, VectorType};
use spirq::var::Variable;
use spirq::ReflectConfig;
use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::{self, c_char, CStr, CString};
use std::hash::{Hash, Hasher};
use std::iter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use self::swapchain::{SwapchainError, VulkanSwapchain};
use self::util::{
    DescriptorPoolInfos, DescriptorSetAllocation, FromVk, ResolveVulkan, SpirvShaderModule,
    VulkanComputePipelineInfo, VulkanGraphicsPipelineInfo, VulkanImageView, VulkanResolver,
};

use super::{ComputePipelineLayout, RenderState, ResourceTransition, SubmitInfo};

pub(crate) const FLIP_VIEWPORT: bool = true;

pub struct VulkanHal {
    entry: Entry,
    instance: Instance,
    physical_device: VulkanPhysicalDevice,
    logical_device: VulkanLogicalDevice,
    debug_utilities: Option<VulkanDebugUtilities>,
    configuration: VulkanHalConfiguration,
    resolver: VulkanResolver,
    frames_in_flight: Arc<Mutex<FramesInFlight>>,

    framebuffers: RwLock<HashMap<u64, vk::Framebuffer>>,
    render_passes: RwLock<HashMap<u64, vk::RenderPass>>,
    descriptor_set_layouts: RwLock<HashMap<u64, vk::DescriptorSetLayout>>,
    pipeline_layouts: RwLock<HashMap<u64, vk::PipelineLayout>>,
    command_buffer_states: RwLock<HashMap<vk::CommandBuffer, VulkanCommandBufferState>>,
    descriptor_pool_infos: Mutex<HashMap<vk::DescriptorSetLayout, DescriptorPoolInfos>>,

    memory_allocator: RwLock<Allocator>,
}

pub struct VulkanHalConfiguration {
    frames_in_flight: usize,
}

#[allow(dead_code)]
pub struct VulkanPhysicalDevice {
    handle: vk::PhysicalDevice,
    name: String,
    limits: vk::PhysicalDeviceLimits,
    device_id: u32,
    features: VulkanDeviceFeatures,
}

#[allow(dead_code)]
pub struct VulkanQueueFamily {
    pub index: u32,
    pub requested_flags: QueueFlags,
}

#[allow(dead_code)]
pub struct VulkanQueueFamilies {
    pub families: Vec<VulkanQueueFamily>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct VulkanQueue {
    handle: vk::Queue,
    family_index: u32,
    queue_index: u32,
}

pub struct VulkanLogicalDevice {
    handle: ash::Device,
    graphics_queue: VulkanQueue,
    compute_queue: VulkanQueue,
    transfer_queue: VulkanQueue,
}

#[allow(dead_code)]
pub(crate) struct VulkanDebugUtilities {
    debug_messenger: vk::DebugUtilsMessengerEXT,
    debug_instance: ash::ext::debug_utils::Instance,
    debug_device: ash::ext::debug_utils::Device,
}
pub(crate) struct FrameInFlight {
    graphics_command_pool: vk::CommandPool,
    compute_command_pool: vk::CommandPool,
    transfer_command_pool: vk::CommandPool,

    graphics_work_ended_fence: vk::Fence,
    compute_work_ended_fence: vk::Fence,
    transfer_work_ended_fence: vk::Fence,
    allocated_semaphores: Vec<vk::Semaphore>,
    cached_semaphores: Vec<vk::Semaphore>,

    allocated_semaphores_binary: Vec<vk::Semaphore>,
    cached_semaphores_binary: Vec<vk::Semaphore>,
    work_ended_semaphore: vk::Semaphore,

    atomic_semaphore_counter: AtomicU64,
}

pub(crate) struct FramesInFlight {
    frames: Vec<FrameInFlight>,
    current_frame_in_flight: usize,
}

#[derive(Clone, Copy, Default)]
struct VulkanCommandBufferState {
    current_render_pass: vk::RenderPass,
    current_subpass: u32,
}

pub struct VulkanDeviceFeatures {
    swapchain_support: bool,
}

#[derive(Debug)]
pub enum VulkanHalError {
    NoSuitableDevice(Option<DevicePreference>),
    ApiError(vk::Result),
    LayerNotAvailable(std::borrow::Cow<'static, str>),
    ExtensionNotAvailable(std::borrow::Cow<'static, str>),
    NoSuitableQueueFamily(vk::QueueFlags),

    GpuAllocatorError(gpu_allocator::AllocationError),

    #[cfg(feature = "swapchain")]
    SwapchainError(SwapchainError),
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
            frames_in_flight: self.configuration.frames_in_flight,
        }
    }

    #[cfg(feature = "swapchain")]
    fn create_swapchain_impl(
        &self,
        swapchain_info: &crate::SwapchainCreationInfo,
    ) -> MgpuResult<SwapchainInfo> {
        let swapchain = VulkanSwapchain::create(self, swapchain_info)?;

        let format = swapchain.data.current_format.format.to_mgpu();
        let present_mode = PresentMode::Immediate;

        let handle = self.resolver.add(swapchain);
        let info = SwapchainInfo {
            id: handle.to_u64(),
            format,
            present_mode,
        };
        Ok(info)
    }

    #[cfg(feature = "swapchain")]
    fn swapchain_acquire_next_image(&self, id: u64) -> MgpuResult<crate::SwapchainImage> {
        use crate::SwapchainImage;

        Ok(self
            .resolver
            .apply_mut::<VulkanSwapchain, SwapchainImage>(
                unsafe { Handle::from_u64(id) },
                |swapchain| {
                    let (index, suboptimal) = unsafe {
                        swapchain
                            .swapchain_device
                            .acquire_next_image(
                                swapchain.handle,
                                u64::MAX,
                                vk::Semaphore::null(),
                                swapchain.acquire_fence,
                            )
                            .expect("Failed to get swapchain")
                    };
                    if suboptimal {
                        info!("Acquired swapchain image {index} is suboptimal, recreate the swapchain");
                    }

                    unsafe {
                        self.logical_device.handle.wait_for_fences(
                            &[swapchain.acquire_fence],
                            true,
                            u64::MAX,
                        )?;

                        self.logical_device
                            .handle
                            .reset_fences(&[swapchain.acquire_fence])?;
                    };
                    swapchain.current_image_index = Some(index);

                    Ok(swapchain.data.images[index as usize])
                },
            )
            .expect("TODO Correct error"))
    }

    unsafe fn begin_rendering(&self) -> MgpuResult<RenderState> {
        use ash::vk::Handle;
        let mut current_frame = self.frames_in_flight.lock().unwrap();
        let current_frame = current_frame.current_mut();

        Ok(RenderState {
            graphics_compute_allocator: CommandRecorderAllocator {
                id: current_frame.graphics_command_pool.as_raw(),
                queue_type: QueueType::Graphics,
            },
            async_compute_allocator: CommandRecorderAllocator {
                id: current_frame.compute_command_pool.as_raw(),
                queue_type: QueueType::AsyncCompute,
            },
            async_transfer_allocator: CommandRecorderAllocator {
                id: current_frame.transfer_command_pool.as_raw(),
                queue_type: QueueType::AsyncTransfer,
            },
        })
    }

    unsafe fn submit(&self, end_rendering_info: SubmitInfo) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_submission_info(&end_rendering_info)?;
        let mut current_frame = self.frames_in_flight.lock().unwrap();
        let current_frame = current_frame.current_mut();

        let current_semaphore_value = current_frame
            .atomic_semaphore_counter
            .fetch_add(1, Ordering::Relaxed)
            + 1;

        let mut command_buffer_states = self
            .command_buffer_states
            .write()
            .expect("Failed to lock command buffer states");
        command_buffer_states.clear();

        let mut all_semaphores_to_signal = vec![];
        let mut all_command_buffers = vec![];
        let mut all_semaphores_to_wait = vec![vec![]];
        let mut all_semaphores_values_to_wait = vec![];

        let device = &self.logical_device.handle;

        let mut graphics_queue_submit = vec![];
        let mut async_compute_queue_submit = vec![];
        let mut transfer_queue_submit = vec![];

        let semaphore_values = [current_semaphore_value; 256];

        for (i, group) in end_rendering_info.submission_groups.iter().enumerate() {
            let mut signals = vec![];
            let mut command_buffers = vec![];
            for cb in &group.command_recorders {
                let semaphore_to_signal = current_frame.allocate_semaphore(device)?;
                signals.push([semaphore_to_signal]);
                command_buffers.push([vk::CommandBuffer::from_raw(cb.id)]);
            }
            all_semaphores_to_signal.push(signals);
            all_command_buffers.push(command_buffers);
            let num_semaphores_to_wait = if i > 0 {
                let semaphores_to_wait = all_semaphores_to_signal[i - 1]
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>();
                let num_semaphores_to_wait = semaphores_to_wait.len();
                all_semaphores_to_wait.push(semaphores_to_wait);
                num_semaphores_to_wait
            } else {
                0
            };

            let timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .signal_semaphore_values(&semaphore_values[0..1])
                .wait_semaphore_values(&semaphore_values[0..num_semaphores_to_wait]);

            all_semaphores_values_to_wait.push(timeline_info);
        }

        let all_stages = [vk::PipelineStageFlags::TOP_OF_PIPE; 256];
        for (i, group) in end_rendering_info.submission_groups.iter().enumerate() {
            let semaphores_to_wait = all_semaphores_to_wait[i].as_slice();
            for (c, cb) in group.command_recorders.iter().enumerate() {
                let semaphore_to_signal = &all_semaphores_to_signal[i][c];
                let vk_cb = &all_command_buffers[i][c];

                let submission_queue = match cb.queue_type {
                    QueueType::Graphics => &mut graphics_queue_submit,
                    QueueType::AsyncCompute => &mut async_compute_queue_submit,
                    QueueType::AsyncTransfer => &mut transfer_queue_submit,
                };

                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(vk_cb)
                    .signal_semaphores(semaphore_to_signal.as_slice())
                    .wait_dst_stage_mask(&all_stages)
                    .wait_semaphores(semaphores_to_wait)
                    .push_next(unsafe {
                        (&mut all_semaphores_values_to_wait[i]
                            as *mut vk::TimelineSemaphoreSubmitInfo)
                            .as_mut()
                            .unwrap()
                    });

                submission_queue.push(submit_info);
            }
        }

        let values = &[current_semaphore_value];
        let (wait_semaphores, values) = if let Some(info) = graphics_queue_submit.last() {
            (
                unsafe { std::slice::from_raw_parts(info.p_signal_semaphores, 1) },
                values.as_slice(),
            )
        } else {
            ([].as_slice(), [].as_slice())
        };

        let mut present_timeline_submit_info =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(values);
        let work_done = [current_frame.work_ended_semaphore];
        graphics_queue_submit.push(
            vk::SubmitInfo::default()
                .wait_dst_stage_mask(&all_stages[0..1])
                .wait_semaphores(wait_semaphores)
                .signal_semaphores(&work_done)
                .push_next(&mut present_timeline_submit_info),
        );

        {
            unsafe {
                device.queue_submit(
                    self.logical_device.graphics_queue.handle,
                    &graphics_queue_submit,
                    current_frame.graphics_work_ended_fence,
                )?;
                device.queue_submit(
                    self.logical_device.transfer_queue.handle,
                    &transfer_queue_submit,
                    current_frame.transfer_work_ended_fence,
                )?;
                device.queue_submit(
                    self.logical_device.compute_queue.handle,
                    &async_compute_queue_submit,
                    current_frame.compute_work_ended_fence,
                )?;
            }
        }
        Ok(())
    }
    unsafe fn end_rendering(&self) -> MgpuResult<()> {
        let mut ff = self.frames_in_flight.lock().unwrap();

        let current_frame_idx = ff.current_frame_in_flight;
        let current_frame = ff.current_mut();
        current_frame.cached_semaphores = current_frame.allocated_semaphores.to_vec();
        current_frame.cached_semaphores_binary = current_frame.allocated_semaphores_binary.to_vec();

        ff.current_frame_in_flight = (current_frame_idx + 1) % self.configuration.frames_in_flight;
        self.resolver.update(self)?;
        Ok(())
    }

    fn swapchain_on_resized(
        &self,
        id: u64,
        new_size: crate::Extents2D,
        window_handle: WindowHandle,
        display_handle: DisplayHandle,
    ) -> MgpuResult<()> {
        if new_size.width == 0 || new_size.height == 0 {
            // Don't recreate the swapchain when reduced to icon
            return Ok(());
        }
        self.resolver.apply_mut(
            unsafe { Handle::<VulkanSwapchain>::from_u64(id) },
            |swapchain| {
                swapchain.rebuild(
                    self,
                    &crate::SwapchainCreationInfo {
                        display_handle,
                        window_handle,
                        preferred_format: Some(swapchain.data.current_format.format.to_mgpu()),
                        preferred_present_mode: Some(swapchain.data.current_present_mode.to_mgpu()),
                    },
                )
            },
        )
    }

    fn create_image(
        &self,
        image_description: &crate::ImageDescription,
    ) -> MgpuResult<crate::Image> {
        let flags = image_description.creation_flags.to_vk();
        let tiling = if image_description.memory_domain == MemoryDomain::Gpu {
            vk::ImageTiling::OPTIMAL
        } else {
            vk::ImageTiling::LINEAR
        };
        let image_create_info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_description.dimension.to_vk())
            .format(image_description.format.to_vk())
            .extent(image_description.extents.to_vk())
            .mip_levels(image_description.mips.get())
            .array_layers(image_description.array_layers.get())
            .samples(image_description.samples.to_vk())
            .tiling(tiling)
            .usage(image_description.usage_flags.to_vk())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let device = &self.logical_device.handle;
        let image = unsafe { device.create_image(&image_create_info, get_allocation_callbacks())? };

        self.try_assign_debug_name(
            image,
            image_description
                .label
                .unwrap_or(&format!("Image {:?}", image)),
        )?;

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let location = match image_description.memory_domain {
            MemoryDomain::Cpu => MemoryLocation::CpuToGpu,
            MemoryDomain::Gpu => MemoryLocation::GpuOnly,
        };
        let fallback_name = format!("Memory allocation for image {:?}", image);
        let allocation_create_desc = AllocationCreateDesc {
            name: image_description.label.unwrap_or(&fallback_name),
            requirements,
            location,
            linear: image_description.memory_domain == MemoryDomain::Gpu,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .memory_allocator
            .write()
            .expect("Failed to lock memory allocator")
            .allocate(&allocation_create_desc)
            .map_err(|e| MgpuError::VulkanError(VulkanHalError::GpuAllocatorError(e)))?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let mips = iter::repeat(LayoutInfo::default())
            .take(image_description.mips.get() as usize)
            .collect::<Vec<_>>();
        let mut layouts = vec![];
        layouts.resize(image_description.array_layers.get() as usize, mips);
        let vulkan_image = VulkanImage {
            label: image_description.label.map(ToOwned::to_owned),
            handle: image,
            external: false,
            allocation: Some(allocation),
            subresource_layouts: layouts,
        };

        info!(
            "Created image {:?} {:?} dimension {:?} layers {} mips {} format {:?}",
            image_description.label,
            image_description.extents,
            image_description.dimension,
            image_description.array_layers.get(),
            image_description.mips.get(),
            image_description.format,
        );

        let handle = self.resolver.add(vulkan_image);
        let image = Image {
            id: handle.to_u64(),
            usage_flags: image_description.usage_flags,
            creation_flags: image_description.creation_flags,
            extents: image_description.extents,
            dimension: image_description.dimension,
            num_mips: image_description.mips,
            array_layers: image_description.array_layers,
            samples: image_description.samples,
            format: image_description.format,
        };
        Ok(image)
    }
    fn destroy_image(&self, image: crate::Image) -> MgpuResult<()> {
        self.resolver
            .remove::<VulkanImage>(unsafe { Handle::from_u64(image.id) })
    }

    fn destroy_image_view(&self, image_view: crate::ImageView) -> MgpuResult<()> {
        self.resolver
            .remove::<VulkanImageView>(unsafe { Handle::from_u64(image_view.id) })
    }

    fn image_name(&self, image: Image) -> MgpuResult<Option<String>> {
        self.resolver
            .apply::<VulkanImage, Option<String>>(image, |vk_image| Ok(vk_image.label.clone()))
    }

    fn create_buffer(&self, buffer_description: &BufferDescription) -> MgpuResult<Buffer> {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .usage(buffer_description.usage_flags.to_vk())
            .size(buffer_description.size as _);
        let device = &self.logical_device.handle;
        let buffer =
            unsafe { device.create_buffer(&buffer_create_info, get_allocation_callbacks())? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        if let Some(label) = buffer_description.label {
            self.try_assign_debug_name(buffer, label)?;
        }

        let fallback_name = format!("Memory allocation for buffer {:?}", buffer);
        let name = buffer_description.label.unwrap_or(&fallback_name);
        let allocation_description = AllocationCreateDesc {
            name,
            requirements: memory_requirements,
            location: match buffer_description.memory_domain {
                MemoryDomain::Cpu => MemoryLocation::CpuToGpu,
                MemoryDomain::Gpu => MemoryLocation::GpuOnly,
            },
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let mut allocator = self
            .memory_allocator
            .write()
            .expect("Failed to lock memory allocator");
        let allocation = allocator
            .allocate(&allocation_description)
            .map_err(|e| MgpuError::VulkanError(VulkanHalError::GpuAllocatorError(e)))?;

        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        let vulkan_buffer = VulkanBuffer {
            label: buffer_description.label.map(ToOwned::to_owned),
            handle: buffer,
            allocation,
            current_access_mask: vk::AccessFlags2::empty(),
            current_stage_mask: vk::PipelineStageFlags2::empty(),
        };
        let handle = self.resolver.add(vulkan_buffer);

        info!(
            "Created buffer {:?} of size {}",
            buffer_description.label, buffer_description.size,
        );
        Ok(Buffer {
            id: handle.to_u64(),
            usage_flags: buffer_description.usage_flags,
            size: buffer_description.size,
            memory_domain: buffer_description.memory_domain,
        })
    }

    fn buffer_name(&self, buffer: Buffer) -> MgpuResult<Option<String>> {
        self.resolver
            .apply::<VulkanBuffer, Option<String>>(buffer, |vk_buffer| Ok(vk_buffer.label.clone()))
    }

    fn destroy_buffer(&self, buffer: Buffer) -> MgpuResult<()> {
        self.resolver.remove(buffer)
    }

    unsafe fn write_host_visible_buffer(
        &self,
        buffer: Buffer,
        params: &crate::BufferWriteParams,
    ) -> MgpuResult<()> {
        debug_assert!(params.data.len() >= params.size);
        debug_assert!(buffer.size >= params.size);
        self.resolver.apply(buffer, |buffer| {
            let allocation = &buffer.allocation;
            let ptr = allocation
                .mapped_ptr()
                .expect("Tried to do a host visible write without a peristent ptr");
            let ptr = unsafe { ptr.cast::<u8>().as_ptr().add(params.offset) };
            unsafe { ptr.copy_from(params.data.as_ptr(), params.size) };
            Ok(())
        })
    }

    fn create_shader_module(
        &self,
        shader_module_description: &crate::ShaderModuleDescription,
    ) -> MgpuResult<ShaderModule> {
        let layout = self.reflect_layout(shader_module_description)?;
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default()
            .flags(vk::ShaderModuleCreateFlags::default())
            .code(shader_module_description.source);
        let shader_module = unsafe {
            self.logical_device
                .handle
                .create_shader_module(&shader_module_create_info, get_allocation_callbacks())?
        };
        if let Some(name) = shader_module_description.label {
            self.try_assign_debug_name(shader_module, name)?;
        }

        let vulkan_shader_module = SpirvShaderModule {
            label: shader_module_description.label.map(ToOwned::to_owned),
            handle: shader_module,
            layout,
        };

        info!(
            "Created shader module {:?}",
            shader_module_description.label
        );

        let handle = self.resolver.add(vulkan_shader_module);
        Ok(ShaderModule {
            id: handle.to_u64(),
        })
    }

    fn get_shader_module_layout(
        &self,
        shader_module: ShaderModule,
    ) -> MgpuResult<ShaderModuleLayout> {
        self.resolver
            .apply(shader_module, |module| Ok(module.layout.clone()))
    }

    fn destroy_shader_module(&self, shader_module: ShaderModule) -> MgpuResult<()> {
        self.resolver.remove(shader_module)
    }

    fn create_graphics_pipeline(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<GraphicsPipeline> {
        // The actual creation of the pipeline is deferred until a render pass using the pipeline is executed

        #[cfg(debug_assertions)]
        self.validate_graphics_pipeline_shader_layouts(graphics_pipeline_description);

        let bind_set_layouts = graphics_pipeline_description
            .binding_set_layouts
            .iter()
            .map(Into::into)
            .collect();

        let owned_info = graphics_pipeline_description.to_vk_owned(bind_set_layouts);

        let pipeline_layout = self.get_pipeline_layout(
            graphics_pipeline_description.label,
            &owned_info.binding_sets_infos,
            owned_info.push_constant_range.as_ref(),
        )?;
        let vulkan_graphics_pipeline_info = VulkanGraphicsPipelineInfo {
            label: graphics_pipeline_description.label.map(ToOwned::to_owned),
            layout: owned_info,
            vk_layout: pipeline_layout,
            pipelines: vec![],
        };
        let handle = self.resolver.add(vulkan_graphics_pipeline_info);

        info!(
            "Created GraphicsPipeline {:?}",
            graphics_pipeline_description.label
        );

        Ok(GraphicsPipeline {
            id: handle.to_u64(),
        })
    }

    fn create_compute_pipeline(
        &self,
        compute_pipeline_description: &ComputePipelineDescription,
    ) -> MgpuResult<ComputePipeline> {
        #[cfg(debug_assertions)]
        self.validate_compute_pipeline_shader_layouts(compute_pipeline_description);
        let bind_set_layouts = compute_pipeline_description
            .binding_set_layouts
            .iter()
            .map(Into::into)
            .collect();

        let owned_info = compute_pipeline_description.to_vk_owned(bind_set_layouts);

        let pipeline_layout = self.get_pipeline_layout(
            compute_pipeline_description.label,
            &owned_info.binding_sets_infos,
            owned_info.push_constant_range.as_ref(),
        )?;
        let compute_entrypt = CString::new(compute_pipeline_description.entry_point).unwrap();
        let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout)
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(
                        self.resolver
                            .resolve_vulkan(compute_pipeline_description.shader)
                            .ok_or(MgpuError::InvalidHandle)?,
                    )
                    .name(&compute_entrypt),
            );

        let pipeline = unsafe {
            self.logical_device
                .handle
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_create_info],
                    get_allocation_callbacks(),
                )
                .map_err(|(_, err)| VulkanHalError::ApiError(err))?[0]
        };

        if let Some(name) = compute_pipeline_description.label {
            self.try_assign_debug_name(pipeline, name)?;
        }

        let vk_pipeline = VulkanComputePipelineInfo {
            label: compute_pipeline_description.label.map(ToOwned::to_owned),
            handle: pipeline,
            layout: owned_info,
            vk_layout: pipeline_layout,
        };
        let handle = self.resolver.add(vk_pipeline);
        info!(
            "Created vulkan compute pipeline {:?}",
            compute_pipeline_description.label
        );
        Ok(ComputePipeline {
            id: handle.to_u64(),
        })
    }
    fn get_compute_pipeline_layout(
        &self,
        compute_pipeline: ComputePipeline,
    ) -> MgpuResult<ComputePipelineLayout> {
        self.resolver
            .apply(compute_pipeline, |pipeline| Ok(pipeline.layout.clone()))
    }

    fn destroy_graphics_pipeline(&self, graphics_pipeline: GraphicsPipeline) -> MgpuResult<()> {
        self.resolver.remove(graphics_pipeline)
    }

    unsafe fn request_command_recorder(
        &self,
        allocator: super::CommandRecorderAllocator,
    ) -> MgpuResult<super::CommandRecorder> {
        use ash::vk::Handle;
        let command_pool = vk::CommandPool::from_raw(allocator.id);
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let cb = unsafe {
            self.logical_device
                .handle
                .allocate_command_buffers(&command_buffer_allocate_info)?
        }[0];

        let state = VulkanCommandBufferState::default();
        let mut states = self
            .command_buffer_states
            .write()
            .expect("Failed to lock command recorder states");
        let old_state = states.insert(cb, state);

        assert!(old_state.is_none());

        unsafe {
            self.logical_device
                .handle
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
        }

        Ok(CommandRecorder {
            id: cb.as_raw(),
            queue_type: allocator.queue_type,
        })
    }

    unsafe fn present_image(&self, swapchain_id: u64, image: Image) -> MgpuResult<()> {
        let mut current_frame = self.frames_in_flight.lock().unwrap();
        let current_frame = current_frame.current_mut();
        let swapchain = unsafe { Handle::from_u64(swapchain_id) };
        let mut images = self.resolver.get_mut::<VulkanImage>();
        let mut swapchains = self.resolver.get_mut::<VulkanSwapchain>();
        let swapchain = swapchains.resolve_mut(swapchain).unwrap();
        let vk_image = images.resolve_mut(image).unwrap();
        let image_layout = vk_image.get_subresource_layout(image.whole_subresource());

        let queue = self.logical_device.graphics_queue.handle;
        let current_index = swapchain
            .current_image_index
            .take()
            .expect("Either a Present has already been issued, or acquire has never been called");
        let indices = [current_index];
        let swapchains = [swapchain.handle];
        let mut queue_wait_semaphore = current_frame.work_ended_semaphore;
        let wait_semaphores = [queue_wait_semaphore];

        if image_layout.image_layout != vk::ImageLayout::PRESENT_SRC_KHR {
            let device = &self.logical_device.handle;
            let cb = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(current_frame.graphics_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0];

            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                        .image(vk_image.handle)
                        .src_access_mask(image_layout.access_mask)
                        .src_stage_mask(image_layout.stage_mask)
                        .old_layout(image_layout.image_layout)
                        .dst_access_mask(vk::AccessFlags2::default())
                        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })]),
            );
            device.end_command_buffer(cb)?;

            vk_image.set_subresource_layout(
                image.whole_subresource(),
                LayoutInfo {
                    image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    access_mask: vk::AccessFlags2::empty(),
                    stage_mask: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                },
            );

            let present_transition_done_semaphore =
                current_frame.allocate_binary_semaphore(device)?;
            let signal = [present_transition_done_semaphore];
            device.queue_submit(
                self.logical_device.graphics_queue.handle,
                &[vk::SubmitInfo::default()
                    .command_buffers(&[cb])
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::BOTTOM_OF_PIPE])
                    .signal_semaphores(&signal)],
                vk::Fence::null(),
            )?;
            queue_wait_semaphore = present_transition_done_semaphore;
        }

        let wait_semaphores = [queue_wait_semaphore];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&wait_semaphores)
            .image_indices(&indices);
        let swapchain_device = &swapchain.swapchain_device;

        let suboptimal = unsafe { swapchain_device.queue_present(queue, &present_info)? };
        if suboptimal {
            swapchain.rebuild(
                self,
                &SwapchainCreationInfo {
                    preferred_format: Some(swapchain.data.current_format.format.to_mgpu()),
                    preferred_present_mode: Some(swapchain.data.current_present_mode.to_mgpu()),
                    display_handle: unsafe {
                        DisplayHandle::borrow_raw(swapchain.raw_display_handle)
                    },
                    window_handle: unsafe { WindowHandle::borrow_raw(swapchain.raw_window_handle) },
                },
            )?;
        }
        Ok(())
    }

    unsafe fn finalize_command_recorder(&self, command_buffer: CommandRecorder) -> MgpuResult<()> {
        let mut states = self
            .command_buffer_states
            .write()
            .expect("Failed to lock command recorder states");
        let cb = vk::CommandBuffer::from_raw(command_buffer.id);
        states
            .remove(&cb)
            .expect("Finalizing a command recorder with no state");
        unsafe {
            self.logical_device
                .handle
                .end_command_buffer(vk::CommandBuffer::from_raw(command_buffer.id))?;
        }
        Ok(())
    }
    unsafe fn begin_render_pass(
        &self,
        command_recorder: CommandRecorder,
        render_pass_info: &RenderPassInfo,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        let render_pass = self.resolve_render_pass(render_pass_info)?;
        let framebuffer =
            self.resolve_framebuffer_for_render_pass(render_pass, render_pass_info)?;

        let mut states = self
            .command_buffer_states
            .write()
            .expect("Failed to lock command buffer states");
        let state = states.get_mut(&cb).unwrap();

        let clear_values = render_pass_info
            .framebuffer
            .render_targets
            .iter()
            .map(|rt| match rt.load_op {
                crate::RenderTargetLoadOp::Clear(color) => vk::ClearValue {
                    color: vk::ClearColorValue { float32: color },
                },
                _ => Default::default(),
            })
            .chain(
                render_pass_info
                    .framebuffer
                    .depth_stencil_target
                    .iter()
                    .map(|rt| match rt.load_op {
                        crate::DepthStencilTargetLoadOp::Clear(depth, stencil) => vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                        },
                        _ => Default::default(),
                    }),
            )
            .collect::<Vec<_>>();
        let device = &self.logical_device.handle;
        unsafe {
            device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default()
                    .framebuffer(framebuffer)
                    .render_pass(render_pass)
                    .render_area(render_pass_info.render_area.to_vk())
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );

            let viewport = if FLIP_VIEWPORT {
                vk::Viewport::default()
                    .width(render_pass_info.render_area.extents.width as f32)
                    .height(-(render_pass_info.render_area.extents.height as f32))
                    .x(render_pass_info.render_area.offset.x as f32)
                    .y(render_pass_info.render_area.offset.y as f32
                        + render_pass_info.render_area.extents.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)
            } else {
                vk::Viewport::default()
                    .width(render_pass_info.render_area.extents.width as f32)
                    .height(render_pass_info.render_area.extents.height as f32)
                    .x(render_pass_info.render_area.offset.x as f32)
                    .y(render_pass_info.render_area.offset.y as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)
            };
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[render_pass_info.render_area.to_vk()]);
            state.current_render_pass = render_pass;
        };

        for rt in &render_pass_info.framebuffer.render_targets {
            self.resolver.apply_mut(rt.view.owner, |image| {
                let image_layout = match rt.store_op {
                    crate::AttachmentStoreOp::DontCare => vk::ImageLayout::UNDEFINED,
                    crate::AttachmentStoreOp::Store => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    crate::AttachmentStoreOp::Present => vk::ImageLayout::PRESENT_SRC_KHR,
                };
                let stage_mask = match rt.store_op {
                    AttachmentStoreOp::DontCare => Default::default(),
                    AttachmentStoreOp::Store => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    AttachmentStoreOp::Present => Default::default(),
                };

                let access_mask = match rt.store_op {
                    AttachmentStoreOp::DontCare => Default::default(),
                    AttachmentStoreOp::Store => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    AttachmentStoreOp::Present => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                };
                image.set_subresource_layout(
                    rt.view.subresource,
                    LayoutInfo {
                        image_layout,
                        access_mask,
                        stage_mask,
                    },
                );
                Ok(())
            })?;
        }

        if let Some(rt) = render_pass_info.framebuffer.depth_stencil_target {
            self.resolver.apply_mut(rt.view.owner, |image| {
                let image_layout = match rt.store_op {
                    crate::AttachmentStoreOp::DontCare => vk::ImageLayout::UNDEFINED,
                    crate::AttachmentStoreOp::Store => {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    }
                    crate::AttachmentStoreOp::Present => vk::ImageLayout::PRESENT_SRC_KHR,
                };
                let stage_mask = match rt.store_op {
                    AttachmentStoreOp::DontCare => Default::default(),
                    AttachmentStoreOp::Store => {
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                    }
                    AttachmentStoreOp::Present => Default::default(),
                };

                let access_mask = match rt.store_op {
                    AttachmentStoreOp::DontCare => Default::default(),
                    AttachmentStoreOp::Store => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    AttachmentStoreOp::Present => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                };
                image.set_subresource_layout(
                    rt.view.subresource,
                    LayoutInfo {
                        image_layout,
                        access_mask,
                        stage_mask,
                    },
                );
                Ok(())
            })?;
        }

        Ok(())
    }

    unsafe fn bind_graphics_pipeline(
        &self,
        command_recorder: CommandRecorder,
        graphics_pipeline: GraphicsPipeline,
    ) -> MgpuResult<()> {
        let vk_command_buffer = vk::CommandBuffer::from_raw(command_recorder.id);
        let command_buffer_states = self
            .command_buffer_states
            .read()
            .expect("Failed to lock command buffer states");
        let command_buffer_state = command_buffer_states[&vk_command_buffer];
        let pipeline = self.resolver.apply_mut(graphics_pipeline, |vk_pipeline| {
            if let Some(pipeline) = vk_pipeline
                .pipelines
                .get(command_buffer_state.current_subpass as usize)
                .copied()
            {
                debug_assert!(pipeline != vk::Pipeline::null());
                Ok(pipeline)
            } else {
                let pipeline =
                    self.create_vulkan_graphics_pipeline(vk_pipeline, &command_buffer_state)?;
                if vk_pipeline.pipelines.len() <= command_buffer_state.current_subpass as usize {
                    vk_pipeline.pipelines.resize(
                        command_buffer_state.current_subpass as usize + 1,
                        vk::Pipeline::default(),
                    )
                }
                vk_pipeline.pipelines[command_buffer_state.current_subpass as usize] = pipeline;
                Ok(pipeline)
            }
        })?;

        unsafe {
            self.logical_device.handle.cmd_bind_pipeline(
                vk_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
        }

        Ok(())
    }

    unsafe fn set_vertex_buffers(
        &self,
        command_recorder: CommandRecorder,
        vertex_buffers: &[Buffer],
    ) -> MgpuResult<()> {
        if vertex_buffers.is_empty() {
            return Ok(());
        }
        let map = self.resolver.get::<VulkanBuffer>();
        let mut vulkan_buffers = vec![];
        for &buffer in vertex_buffers {
            let vk_buffer = map.resolve(buffer).ok_or(MgpuError::InvalidHandle)?.handle;
            vulkan_buffers.push(vk_buffer);
        }

        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        let offsets = vertex_buffers.iter().map(|_| 0u64).collect::<Vec<_>>();
        unsafe {
            self.logical_device
                .handle
                .cmd_bind_vertex_buffers(cb, 0, &vulkan_buffers, &offsets);
        }

        Ok(())
    }

    unsafe fn set_index_buffer(
        &self,
        command_recorder: CommandRecorder,
        index_buffer: Buffer,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);

        unsafe {
            self.logical_device.handle.cmd_bind_index_buffer(
                cb,
                self.resolver
                    .resolve_vulkan(index_buffer)
                    .ok_or(MgpuError::InvalidHandle)?,
                0,
                vk::IndexType::UINT32,
            )
        }

        Ok(())
    }

    unsafe fn set_graphics_push_constant(
        &self,
        command_recorder: CommandRecorder,
        graphics_pipeline: GraphicsPipeline,
        data: &[u8],
        visibility: ShaderStageFlags,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        let graphics_pipeline = self
            .resolver
            .apply(graphics_pipeline, |g| Ok(g.vk_layout))?;

        unsafe {
            self.logical_device.handle.cmd_push_constants(
                cb,
                graphics_pipeline,
                visibility.to_vk(),
                0,
                data,
            );
        }
        Ok(())
    }

    unsafe fn set_compute_push_constant(
        &self,
        command_recorder: CommandRecorder,
        compute_pipeline: ComputePipeline,
        data: &[u8],
        visibility: ShaderStageFlags,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        let compute_pipeline = self.resolver.apply(compute_pipeline, |g| Ok(g.vk_layout))?;

        unsafe {
            self.logical_device.handle.cmd_push_constants(
                cb,
                compute_pipeline,
                visibility.to_vk(),
                0,
                data,
            );
        }
        Ok(())
    }
    unsafe fn bind_graphics_binding_sets(
        &self,
        command_recorder: CommandRecorder,
        binding_sets: &[BindingSet],
        graphics_pipeline: GraphicsPipeline,
    ) -> MgpuResult<()> {
        if binding_sets.is_empty() {
            return Ok(());
        }
        let sets = self.resolver.get::<VulkanBindingSet>();
        let vk_sets = binding_sets
            .iter()
            .map(|s| {
                sets.resolve(s)
                    .map(|b| b.handle)
                    .expect("Binding Set handle not valid")
            })
            .collect::<Vec<_>>();
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        let graphics_pipeline = self
            .resolver
            .apply(graphics_pipeline, |g| Ok(g.vk_layout))?;

        unsafe {
            self.logical_device.handle.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
                0,
                &vk_sets,
                &[],
            );
        }

        Ok(())
    }

    unsafe fn draw(
        &self,
        command_recorder: CommandRecorder,
        vertices: usize,
        indices: usize,
        first_vertex: usize,
        first_instance: usize,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        unsafe {
            self.logical_device.handle.cmd_draw(
                cb,
                vertices as _,
                indices as _,
                first_vertex as _,
                first_instance as _,
            );
        }

        Ok(())
    }

    unsafe fn draw_indexed(
        &self,
        command_recorder: CommandRecorder,
        indices: usize,
        instances: usize,
        first_index: usize,
        vertex_offset: i32,
        first_instance: usize,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        unsafe {
            self.logical_device.handle.cmd_draw_indexed(
                cb,
                indices as _,
                instances as _,
                first_index as _,
                vertex_offset as _,
                first_instance as _,
            );
        }
        Ok(())
    }

    unsafe fn advance_to_next_step(&self, command_recorder: CommandRecorder) -> MgpuResult<()> {
        let mut states = self
            .command_buffer_states
            .write()
            .expect("Failed to lock command recorder states");
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        states
            .get_mut(&cb)
            .expect("Command buffer with no state")
            .current_subpass += 1;
        unsafe {
            self.logical_device.handle.cmd_next_subpass(
                vk::CommandBuffer::from_raw(command_recorder.id),
                vk::SubpassContents::INLINE,
            );
        }

        Ok(())
    }

    unsafe fn end_render_pass(&self, command_recorder: CommandRecorder) -> MgpuResult<()> {
        let mut states = self
            .command_buffer_states
            .write()
            .expect("Failed to lock command recorder states");
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        states.get_mut(&cb).unwrap().current_render_pass = vk::RenderPass::null();

        unsafe {
            self.logical_device
                .handle
                .cmd_end_render_pass(vk::CommandBuffer::from_raw(command_recorder.id))
        }

        Ok(())
    }

    unsafe fn bind_compute_pipeline(
        &self,
        command_recorder: CommandRecorder,
        pipeline: ComputePipeline,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        let pipeline = self
            .resolver
            .resolve_vulkan(pipeline)
            .ok_or(MgpuError::InvalidHandle)?;
        unsafe {
            self.logical_device.handle.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        }
        Ok(())
    }

    unsafe fn bind_compute_binding_sets(
        &self,
        command_recorder: CommandRecorder,
        binding_sets: &[BindingSet],
        pipeline: ComputePipeline,
    ) -> MgpuResult<()> {
        if binding_sets.is_empty() {
            return Ok(());
        }
        let sets = self.resolver.get::<VulkanBindingSet>();
        let vk_sets = binding_sets
            .iter()
            .map(|s| {
                sets.resolve(s)
                    .map(|b| b.handle)
                    .expect("Binding Set handle not valid")
            })
            .collect::<Vec<_>>();
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        let compute_pipeline = self.resolver.apply(pipeline, |g| Ok(g.vk_layout))?;

        unsafe {
            self.logical_device.handle.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                compute_pipeline,
                0,
                &vk_sets,
                &[],
            );
        }

        Ok(())
    }

    unsafe fn dispatch(
        &self,
        command_recorder: CommandRecorder,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> MgpuResult<()> {
        let cb = vk::CommandBuffer::from_raw(command_recorder.id);
        unsafe {
            self.logical_device.handle.cmd_dispatch(
                cb,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
        Ok(())
    }
    fn device_wait_idle(&self) -> MgpuResult<()> {
        unsafe { self.logical_device.handle.device_wait_idle() }?;
        Ok(())
    }

    unsafe fn cmd_copy_buffer_to_buffer(
        &self,
        command_buffer: CommandRecorder,
        source: Buffer,
        dest: Buffer,
        source_offset: usize,
        dest_offset: usize,
        size: usize,
    ) -> MgpuResult<()> {
        let vk_command_buffer = vk::CommandBuffer::from_raw(command_buffer.id);

        let region = vk::BufferCopy::default()
            .src_offset(source_offset as _)
            .dst_offset(dest_offset as _)
            .size(size as _);

        unsafe {
            self.logical_device.handle.cmd_copy_buffer(
                vk_command_buffer,
                self.resolver
                    .resolve_vulkan(source)
                    .ok_or(MgpuError::InvalidHandle)?,
                self.resolver
                    .resolve_vulkan(dest)
                    .ok_or(MgpuError::InvalidHandle)?,
                &[region],
            );
        }
        Ok(())
    }

    unsafe fn cmd_copy_buffer_to_image(
        &self,
        command_buffer: CommandRecorder,
        source: Buffer,
        dest: Image,
        source_offset: usize,
        dest_region: ImageRegion,
    ) -> MgpuResult<()> {
        let vk_command_buffer = vk::CommandBuffer::from_raw(command_buffer.id);
        let image_subresource = vk::ImageSubresourceLayers::default()
            .mip_level(dest_region.mip)
            .base_array_layer(dest_region.base_array_layer)
            .layer_count(dest_region.num_layers.get())
            .aspect_mask(dest.format.aspect_mask());

        let copy = vk::BufferImageCopy::default()
            .buffer_offset(source_offset as _)
            .image_offset(dest_region.offset.to_vk())
            .image_extent(dest_region.extents.to_vk())
            .image_subresource(image_subresource);

        let (image, layout) = self.resolver.apply_mut(dest, |image| {
            Ok((
                image.handle,
                image.get_subresource_layout(dest_region.to_image_subresource()),
            ))
        })?;
        let buffer = self
            .resolver
            .resolve_vulkan(source)
            .ok_or(MgpuError::InvalidHandle)?;

        let device = &self.logical_device.handle;

        let valid_transfer_layouts = [
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ];

        let mut copy_layout = layout.image_layout;

        let src_stage_mask = if command_buffer.queue_type == QueueType::AsyncTransfer {
            vk::PipelineStageFlags2::ALL_COMMANDS
        } else {
            layout.stage_mask
        };

        if !valid_transfer_layouts.contains(&copy_layout) {
            // transition the image to transfer_dst
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .image(image)
                .src_access_mask(layout.access_mask)
                .src_stage_mask(src_stage_mask)
                .old_layout(layout.image_layout)
                .dst_access_mask(ResourceAccessMode::TransferDst.access_mask())
                .dst_stage_mask(ResourceAccessMode::TransferDst.pipeline_flags())
                .new_layout(ResourceAccessMode::TransferDst.image_layout())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(dest.format.aspect_mask())
                        .base_array_layer(dest_region.base_array_layer)
                        .base_mip_level(dest_region.mip)
                        .layer_count(dest_region.num_layers.get())
                        .level_count(dest_region.num_mips.get()),
                );
            device.cmd_pipeline_barrier2(
                vk_command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&[image_barrier])
                    .dependency_flags(vk::DependencyFlags::BY_REGION),
            );
            copy_layout = ResourceAccessMode::TransferDst.image_layout();
        }

        unsafe {
            device.cmd_copy_buffer_to_image(vk_command_buffer, buffer, image, copy_layout, &[copy]);
        }

        if !valid_transfer_layouts.contains(&layout.image_layout) {
            // transition the image to old_layout
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .image(image)
                .dst_access_mask(layout.access_mask)
                .dst_stage_mask(layout.stage_mask)
                .new_layout(layout.image_layout)
                .src_access_mask(ResourceAccessMode::TransferDst.access_mask())
                .src_stage_mask(ResourceAccessMode::TransferDst.pipeline_flags())
                .old_layout(ResourceAccessMode::TransferDst.image_layout())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(dest.format.aspect_mask())
                        .base_array_layer(dest_region.base_array_layer)
                        .base_mip_level(dest_region.mip)
                        .layer_count(dest_region.num_layers.get())
                        .level_count(dest_region.num_mips.get()),
                );
            device.cmd_pipeline_barrier2(
                vk_command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&[image_barrier])
                    .dependency_flags(vk::DependencyFlags::BY_REGION),
            );
        }
        Ok(())
    }

    unsafe fn cmd_blit_image(
        &self,
        command_buffer: CommandRecorder,
        source: Image,
        source_region: ImageRegion,
        dest: Image,
        dest_region: ImageRegion,
        filter: FilterMode,
    ) -> MgpuResult<()> {
        let vk_command_buffer = vk::CommandBuffer::from_raw(command_buffer.id);
        let device = &self.logical_device.handle;
        let filter = filter.to_vk();

        let (source_image, mut source_layout) = self.resolver.apply(source, |img| {
            Ok((
                img.handle,
                img.get_subresource_layout(source_region.to_image_subresource()),
            ))
        })?;

        let (dest_image, mut dest_layout) = self.resolver.apply(dest, |img| {
            Ok((
                img.handle,
                img.get_subresource_layout(dest_region.to_image_subresource()),
            ))
        })?;

        debug_assert!(source_layout.image_layout != vk::ImageLayout::UNDEFINED);
        debug_assert!(dest_layout.image_layout != vk::ImageLayout::UNDEFINED);

        if source_layout.image_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
            device.cmd_pipeline_barrier2(
                vk_command_buffer,
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                        .src_access_mask(source_layout.access_mask)
                        .src_stage_mask(source_layout.stage_mask)
                        .old_layout(source_layout.image_layout)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .image(source_image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: source.format.aspect_mask(),
                            base_mip_level: source_region.mip,
                            level_count: source_region.num_mips.get(),
                            base_array_layer: source_region.base_array_layer,
                            layer_count: source_region.num_layers.get(),
                        })]),
            );

            source_layout = LayoutInfo {
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                stage_mask: vk::PipelineStageFlags2::TRANSFER,
            };

            self.resolver.apply_mut(source, |i| {
                i.set_subresource_layout(source_region.to_image_subresource(), source_layout);
                Ok(())
            })?;
        }
        if dest_layout.image_layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            device.cmd_pipeline_barrier2(
                vk_command_buffer,
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                        .src_access_mask(dest_layout.access_mask)
                        .src_stage_mask(dest_layout.stage_mask)
                        .old_layout(dest_layout.image_layout)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(dest_image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: dest.format.aspect_mask(),
                            base_mip_level: dest_region.mip,
                            level_count: dest_region.num_mips.get(),
                            base_array_layer: dest_region.base_array_layer,
                            layer_count: dest_region.num_layers.get(),
                        })]),
            );

            dest_layout = LayoutInfo {
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                stage_mask: vk::PipelineStageFlags2::TRANSFER,
            };

            self.resolver.apply_mut(dest, |i| {
                i.set_subresource_layout(dest_region.to_image_subresource(), dest_layout);
                Ok(())
            })?;
        }

        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: source.format.aspect_mask(),
                mip_level: source_region.mip,
                base_array_layer: source_region.base_array_layer,
                layer_count: source_region.num_layers.get(),
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dest.format.aspect_mask(),
                mip_level: dest_region.mip,
                base_array_layer: dest_region.base_array_layer,
                layer_count: dest_region.num_layers.get(),
            },
            src_offsets: [
                source_region.offset.to_vk(),
                (source_region.offset + source_region.extents).to_vk(),
            ],
            dst_offsets: [
                dest_region.offset.to_vk(),
                (dest_region.offset + dest_region.extents).to_vk(),
            ],
        };

        device.cmd_blit_image(
            vk_command_buffer,
            source_image,
            source_layout.image_layout,
            dest_image,
            dest_layout.image_layout,
            &[blit],
            filter,
        );

        Ok(())
    }
    unsafe fn enqueue_synchronization(
        &self,
        infos: &[super::SynchronizationInfo],
    ) -> MgpuResult<()> {
        let device = &self.logical_device.handle;
        for info in infos {
            let vk_buffer_src = info
                .source_command_recorder
                .map(|cr| vk::CommandBuffer::from_raw(cr.id));
            let vk_buffer_dst = vk::CommandBuffer::from_raw(info.destination_command_recorder.id);
            let source_qfi = match info.source_queue {
                super::QueueType::Graphics => self.logical_device.graphics_queue.family_index,
                super::QueueType::AsyncCompute => self.logical_device.compute_queue.family_index,
                super::QueueType::AsyncTransfer => self.logical_device.transfer_queue.family_index,
            };

            let dest_qfi = match info.destination_queue {
                super::QueueType::Graphics => self.logical_device.graphics_queue.family_index,
                super::QueueType::AsyncCompute => self.logical_device.compute_queue.family_index,
                super::QueueType::AsyncTransfer => self.logical_device.transfer_queue.family_index,
            };

            if source_qfi != dest_qfi {
                let mut buffers = self.resolver.get_mut::<VulkanBuffer>();
                let mut images = self.resolver.get_mut::<VulkanImage>();
                let mut image_memory_barrier_source = vec![];
                let mut buffer_memory_barrier_source = vec![];

                let mut image_memory_barrier_dest = vec![];
                let mut buffer_memory_barrier_dest = vec![];
                for resource in &info.resources {
                    assert!(resource.new_usage != ResourceAccessMode::Undefined);
                    let dst_stage_mask = resource.new_usage.pipeline_flags();
                    let dst_access_mask = resource.new_usage.access_mask();
                    match &resource.resource {
                        super::Resource::Image {
                            image,
                            subresource: region,
                        } => {
                            let new_layout = resource.new_usage.image_layout();
                            let vk_image = images.resolve_mut(*image).unwrap();
                            let current_layout_info = vk_image.get_subresource_layout(*region);

                            let src_stage_mask = if info.source_queue == QueueType::AsyncTransfer {
                                vk::PipelineStageFlags2::ALL_COMMANDS
                            } else {
                                current_layout_info.stage_mask
                            };
                            if vk_buffer_src.is_some() {
                                image_memory_barrier_source.push(
                                    vk::ImageMemoryBarrier2::default()
                                        .image(vk_image.handle)
                                        .subresource_range(
                                            vk::ImageSubresourceRange::default()
                                                .aspect_mask(image.format.aspect_mask())
                                                .base_array_layer(region.base_array_layer)
                                                .base_mip_level(region.mip)
                                                .layer_count(region.num_layers.get())
                                                .level_count(region.num_mips.get()),
                                        )
                                        .old_layout(current_layout_info.image_layout)
                                        .new_layout(new_layout)
                                        .src_access_mask(current_layout_info.access_mask)
                                        .src_stage_mask(src_stage_mask)
                                        .src_queue_family_index(source_qfi)
                                        .dst_queue_family_index(dest_qfi),
                                );

                                image_memory_barrier_dest.push(
                                    vk::ImageMemoryBarrier2::default()
                                        .image(vk_image.handle)
                                        .subresource_range(
                                            vk::ImageSubresourceRange::default()
                                                .aspect_mask(image.format.aspect_mask())
                                                .base_array_layer(region.base_array_layer)
                                                .base_mip_level(region.mip)
                                                .layer_count(region.num_layers.get())
                                                .level_count(region.num_mips.get()),
                                        )
                                        .old_layout(current_layout_info.image_layout)
                                        .new_layout(new_layout)
                                        .dst_access_mask(dst_access_mask)
                                        .dst_stage_mask(dst_stage_mask)
                                        .src_queue_family_index(source_qfi)
                                        .dst_queue_family_index(dest_qfi),
                                );
                            } else {
                                image_memory_barrier_dest.push(
                                    vk::ImageMemoryBarrier2::default()
                                        .image(vk_image.handle)
                                        .subresource_range(
                                            vk::ImageSubresourceRange::default()
                                                .aspect_mask(image.format.aspect_mask())
                                                .base_array_layer(region.base_array_layer)
                                                .base_mip_level(region.mip)
                                                .layer_count(region.num_layers.get())
                                                .level_count(region.num_mips.get()),
                                        )
                                        .old_layout(vk::ImageLayout::UNDEFINED)
                                        .new_layout(new_layout)
                                        .src_access_mask(vk::AccessFlags2::empty())
                                        .dst_access_mask(dst_access_mask)
                                        .src_stage_mask(vk::PipelineStageFlags2::empty())
                                        .dst_stage_mask(dst_stage_mask)
                                        .src_queue_family_index(source_qfi)
                                        .dst_queue_family_index(dest_qfi),
                                );
                            }
                            vk_image.set_subresource_layout(
                                *region,
                                LayoutInfo {
                                    image_layout: new_layout,
                                    access_mask: dst_access_mask,
                                    stage_mask: dst_stage_mask,
                                },
                            );
                        }
                        super::Resource::Buffer {
                            buffer,
                            offset,
                            size,
                        } => {
                            let buffer = buffers.resolve_mut(*buffer).unwrap();

                            if vk_buffer_src.is_some() {
                                buffer_memory_barrier_source.push(
                                    vk::BufferMemoryBarrier2::default()
                                        .buffer(buffer.handle)
                                        .offset(*offset as _)
                                        .size(*size as _)
                                        .src_queue_family_index(source_qfi)
                                        .dst_queue_family_index(dest_qfi)
                                        .src_access_mask(buffer.current_access_mask)
                                        // .dst_access_mask(dst_access_mask)
                                        .src_stage_mask(buffer.current_stage_mask), // .dst_stage_mask(dst_stage_mask),
                                );
                                buffer_memory_barrier_dest.push(
                                    vk::BufferMemoryBarrier2::default()
                                        .buffer(buffer.handle)
                                        .offset(*offset as _)
                                        .size(*size as _)
                                        .src_queue_family_index(source_qfi)
                                        .dst_queue_family_index(dest_qfi)
                                        // .src_access_mask(buffer.current_access_mask)
                                        .dst_access_mask(dst_access_mask)
                                        // .src_stage_mask(buffer.current_stage_mask)
                                        .dst_stage_mask(dst_stage_mask),
                                );
                            } else {
                                buffer_memory_barrier_dest.push(
                                    vk::BufferMemoryBarrier2::default()
                                        .buffer(buffer.handle)
                                        .offset(*offset as _)
                                        .size(*size as _)
                                        .src_queue_family_index(source_qfi)
                                        .dst_queue_family_index(dest_qfi)
                                        .src_access_mask(vk::AccessFlags2::empty())
                                        .dst_access_mask(dst_access_mask)
                                        .src_stage_mask(vk::PipelineStageFlags2::empty())
                                        .dst_stage_mask(dst_stage_mask),
                                );
                            }
                            buffer.current_access_mask = dst_access_mask;
                            buffer.current_stage_mask = dst_stage_mask;
                        }
                    }
                }

                let depedency_info_source = vk::DependencyInfo::default()
                    .buffer_memory_barriers(&buffer_memory_barrier_source)
                    .image_memory_barriers(&image_memory_barrier_source)
                    .dependency_flags(vk::DependencyFlags::BY_REGION);

                let depedency_info_dest = vk::DependencyInfo::default()
                    .buffer_memory_barriers(&buffer_memory_barrier_dest)
                    .image_memory_barriers(&image_memory_barrier_dest)
                    .dependency_flags(vk::DependencyFlags::BY_REGION);

                unsafe {
                    if let Some(cb) = vk_buffer_src {
                        device.cmd_pipeline_barrier2(cb, &depedency_info_source);
                    }
                    device.cmd_pipeline_barrier2(vk_buffer_dst, &depedency_info_dest);
                }
            } else {
                self.transition_resources(info.destination_command_recorder, &info.resources)?;
            }
        }
        Ok(())
    }

    fn transition_resources(
        &self,
        command_recorder: CommandRecorder,
        resources: &[ResourceTransition],
    ) -> MgpuResult<()> {
        if resources.is_empty() {
            return Ok(());
        }
        let mut image_resolver = self.resolver.get_mut::<VulkanImage>();
        let mut buffer_resolver = self.resolver.get_mut::<VulkanBuffer>();
        let command_buffer = vk::CommandBuffer::from_raw(command_recorder.id);
        let mut image_transitions = vec![];
        let mut buffer_transitions = vec![];
        for resource in resources {
            let dst_access_mask = resource.new_usage.access_mask();
            let dst_pipeline_flags = resource.new_usage.pipeline_flags();
            match resource.resource {
                crate::hal::Resource::Image {
                    image,
                    subresource: region,
                } => {
                    let new_image_layout = resource.new_usage.image_layout();
                    let vk_image = image_resolver
                        .resolve_mut(image)
                        .ok_or(MgpuError::InvalidHandle)?;
                    let old_layout = vk_image.get_subresource_layout(region);

                    let subresource = vk::ImageSubresourceRange::default()
                        .aspect_mask(image.format.aspect_mask())
                        .base_array_layer(region.base_array_layer)
                        .base_mip_level(region.mip)
                        .layer_count(region.num_layers.get())
                        .level_count(region.num_mips.get());
                    image_transitions.push(
                        vk::ImageMemoryBarrier2::default()
                            .image(vk_image.handle)
                            .old_layout(old_layout.image_layout)
                            .new_layout(new_image_layout)
                            .src_access_mask(old_layout.access_mask)
                            .dst_access_mask(dst_access_mask)
                            .src_stage_mask(old_layout.stage_mask)
                            .dst_stage_mask(dst_pipeline_flags)
                            .subresource_range(subresource),
                    );

                    vk_image.set_subresource_layout(
                        region,
                        LayoutInfo {
                            image_layout: new_image_layout,
                            access_mask: dst_access_mask,
                            stage_mask: dst_pipeline_flags,
                        },
                    );
                }
                crate::hal::Resource::Buffer {
                    buffer,
                    offset,
                    size,
                } => {
                    let vk_buffer = buffer_resolver
                        .resolve_mut(buffer)
                        .ok_or(MgpuError::InvalidHandle)?;
                    buffer_transitions.push(
                        vk::BufferMemoryBarrier2::default()
                            .buffer(vk_buffer.handle)
                            .dst_access_mask(dst_access_mask)
                            .dst_stage_mask(dst_pipeline_flags)
                            .src_access_mask(vk_buffer.current_access_mask)
                            .src_stage_mask(vk_buffer.current_stage_mask)
                            .offset(offset as _)
                            .size(size as _),
                    );

                    vk_buffer.current_access_mask = dst_access_mask;
                    vk_buffer.current_stage_mask = dst_pipeline_flags;
                }
            }
        }
        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_transitions)
            .image_memory_barriers(&image_transitions)
            .dependency_flags(vk::DependencyFlags::BY_REGION);

        unsafe {
            self.logical_device
                .handle
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };
        Ok(())
    }

    fn get_graphics_pipeline_layout(
        &self,
        graphics_pipeline: GraphicsPipeline,
    ) -> MgpuResult<GraphicsPipelineLayout> {
        self.resolver
            .resolve_clone(graphics_pipeline)
            .ok_or(MgpuError::InvalidHandle)
            .map(|layout| layout.layout)
    }

    fn create_sampler(&self, sampler_description: &SamplerDescription) -> MgpuResult<Sampler> {
        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(sampler_description.mag_filter.to_vk())
            .min_filter(sampler_description.min_filter.to_vk())
            .mipmap_mode(sampler_description.mipmap_mode.to_vk())
            .address_mode_u(sampler_description.address_mode_u.to_vk())
            .address_mode_v(sampler_description.address_mode_v.to_vk())
            .address_mode_w(sampler_description.address_mode_w.to_vk())
            .mip_lod_bias(sampler_description.lod_bias)
            .compare_enable(sampler_description.compare_op.is_some())
            .compare_op(sampler_description.compare_op.unwrap_or_default().to_vk())
            .min_lod(sampler_description.min_lod)
            .max_lod(sampler_description.max_lod)
            .border_color(sampler_description.border_color.to_vk())
            .unnormalized_coordinates(sampler_description.unnormalized_coordinates);

        let sampler = unsafe {
            self.logical_device
                .handle
                .create_sampler(&sampler_create_info, get_allocation_callbacks())?
        };

        if let Some(name) = sampler_description.label {
            self.try_assign_debug_name(sampler, name)?;
        }

        let sampler = VulkanSampler {
            label: sampler_description.label.map(ToOwned::to_owned),
            handle: sampler,
        };

        let handle = self.resolver.add(sampler);

        info!("Created sampler {:?}", sampler_description.label);

        Ok(Sampler {
            id: handle.to_u64(),
        })
    }

    fn destroy_sampler(&self, sampler: Sampler) -> MgpuResult<()> {
        self.resolver.remove(sampler)
    }

    fn create_binding_set(
        &self,
        description: &BindingSetDescription,
        layout: &BindingSetLayout,
    ) -> MgpuResult<BindingSet> {
        let descriptor_set_layout = {
            let mut ds_layouts = self.descriptor_set_layouts.write().unwrap();
            self.get_descriptor_set_layout_for_binding_layout(&layout.into(), &mut ds_layouts)
        }?;
        let allocation = self.allocate_descriptor_set(layout, descriptor_set_layout)?;

        self.update_descriptor_set(allocation.descriptor_set, description)?;

        let binding_set = VulkanBindingSet {
            label: description.label.map(ToOwned::to_owned),
            handle: allocation.descriptor_set,
            allocation,
        };

        if let Some(name) = description.label {
            self.try_assign_debug_name(binding_set.allocation.descriptor_set, name)?;
        }

        info!("Created BindingSet {:?}", description.label);

        let handle = self.resolver.add(binding_set);
        Ok(BindingSet {
            id: handle.to_u64(),
            bindings: description.bindings.to_vec(),
        })
    }

    fn destroy_binding_set(&self, binding_set: BindingSet) -> MgpuResult<()> {
        self.resolver.remove(binding_set)
    }

    fn create_image_view(
        &self,
        image_view_description: &crate::ImageViewDescription,
    ) -> MgpuResult<crate::ImageView> {
        let image = image_view_description.image;

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(
                self.resolver
                    .resolve_vulkan(image)
                    .ok_or(MgpuError::InvalidHandle)?,
            )
            .format(image_view_description.format.to_vk())
            .view_type(image_view_description.dimension.image_view_type())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_view_description.format.aspect_mask())
                    .base_array_layer(image_view_description.image_subresource.base_array_layer)
                    .base_mip_level(image_view_description.image_subresource.mip)
                    .level_count(image_view_description.image_subresource.num_mips.get())
                    .layer_count(image_view_description.image_subresource.num_layers.get()),
            )
            .components(ComponentMapping::default());

        let image_view = unsafe {
            self.logical_device
                .handle
                .create_image_view(&image_view_create_info, get_allocation_callbacks())?
        };

        if let Some(name) = image_view_description.label {
            self.try_assign_debug_name(image_view, name)?;
        }

        info!("Created ImageView {:?}", image_view_description.label);

        Ok(unsafe {
            self.wrap_raw_image_view(
                image,
                image_view,
                image_view_description.image_subresource,
                image_view_description.label,
            )?
        })
    }

    unsafe fn prepare_next_frame(&self) -> MgpuResult<()> {
        let mut current_frame = self.frames_in_flight.lock().unwrap();
        let current_frame = current_frame.current_mut();
        let device = &self.logical_device.handle;
        unsafe {
            device.wait_for_fences(
                &[
                    current_frame.graphics_work_ended_fence,
                    current_frame.transfer_work_ended_fence,
                    current_frame.compute_work_ended_fence,
                ],
                true,
                u64::MAX,
            )?
        };
        unsafe {
            device.reset_fences(&[
                current_frame.graphics_work_ended_fence,
                current_frame.transfer_work_ended_fence,
                current_frame.compute_work_ended_fence,
            ])?
        };

        unsafe {
            device.reset_command_pool(
                current_frame.graphics_command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
            device.reset_command_pool(
                current_frame.compute_command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
            device.reset_command_pool(
                current_frame.transfer_command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
        }
        Ok(())
    }

    fn try_swapchain_set_present_mode(
        &self,
        id: u64,
        present_mode: crate::PresentMode,
    ) -> MgpuResult<crate::PresentMode> {
        self.resolver.apply_mut::<VulkanSwapchain, PresentMode>(
            unsafe { Handle::from_u64(id) },
            |swapchain| {
                let vk_present_mode = present_mode.to_vk();
                if swapchain.data.present_modes.contains(&vk_present_mode) {
                    swapchain.rebuild(
                        self,
                        &SwapchainCreationInfo {
                            preferred_format: Some(swapchain.data.current_format.format.to_mgpu()),
                            preferred_present_mode: Some(present_mode),
                            display_handle: unsafe {
                                DisplayHandle::borrow_raw(swapchain.raw_display_handle)
                            },
                            window_handle: unsafe {
                                WindowHandle::borrow_raw(swapchain.raw_window_handle)
                            },
                        },
                    )?;
                    Ok(present_mode)
                } else {
                    swapchain.rebuild(
                        self,
                        &SwapchainCreationInfo {
                            preferred_format: Some(swapchain.data.current_format.format.to_mgpu()),
                            preferred_present_mode: Some(PresentMode::Immediate),
                            display_handle: unsafe {
                                DisplayHandle::borrow_raw(swapchain.raw_display_handle)
                            },
                            window_handle: unsafe {
                                WindowHandle::borrow_raw(swapchain.raw_window_handle)
                            },
                        },
                    )?;
                    Ok(PresentMode::Immediate)
                }
            },
        )
    }

    fn swapchain_destroy(&self, id: u64) -> MgpuResult<()> {
        self.resolver
            .remove::<VulkanSwapchain>(unsafe { Handle::from_u64(id) })
    }
}

impl VulkanHal {
    const VULKAN_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 272);
    pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry, configuration)?;
        let physical_device = Self::pick_physical_device(&instance, configuration)?;

        let queue_families = Self::pick_queue_families(&instance, physical_device.handle)?;
        let logical_device =
            Self::create_logical_device(&instance, physical_device.handle, &queue_families)?;

        let use_debug_features = configuration
            .features
            .contains(DeviceFeatures::HAL_DEBUG_LAYERS);

        let debug_utilities = if use_debug_features {
            Some(Self::create_debug_utilities(
                &entry,
                &instance,
                &logical_device.handle,
            )?)
        } else {
            None
        };

        let frames_in_flight = Self::create_frames_in_flight(&logical_device, configuration)?;
        let allocator_create_desc = AllocatorCreateDesc {
            instance: instance.clone(),
            device: logical_device.handle.clone(),
            physical_device: physical_device.handle,
            debug_settings: AllocatorDebugSettings {
                log_memory_information: use_debug_features,
                log_leaks_on_shutdown: use_debug_features,
                store_stack_traces: false,
                log_allocations: use_debug_features,
                log_frees: use_debug_features,
                log_stack_traces: false,
            },
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        };

        info!("Vulkan HAL: using device '{}'", physical_device.name);
        info!("\tGraphics queue info {:?}", logical_device.graphics_queue);
        info!("\tCompute queue info {:?}", logical_device.compute_queue);
        info!("\tTransfer queue info {:?}", logical_device.transfer_queue);

        let hal = Self {
            entry,
            instance,
            physical_device,
            debug_utilities,
            logical_device,
            configuration: VulkanHalConfiguration {
                frames_in_flight: configuration.desired_frames_in_flight,
            },
            resolver: VulkanResolver::new(configuration.desired_frames_in_flight),
            frames_in_flight: Arc::new(Mutex::new(frames_in_flight)),
            framebuffers: RwLock::default(),
            render_passes: RwLock::default(),

            memory_allocator: RwLock::new(
                Allocator::new(&allocator_create_desc)
                    .map_err(VulkanHalError::GpuAllocatorError)?,
            ),
            command_buffer_states: Default::default(),
            descriptor_set_layouts: Default::default(),
            pipeline_layouts: Default::default(),
            descriptor_pool_infos: Default::default(),
        };

        Ok(Arc::new(hal))
    }

    fn create_instance(
        entry: &Entry,
        configuration: &DeviceConfiguration,
    ) -> VulkanHalResult<ash::Instance> {
        const LAYER_KHRONOS_VALIDATION: &CStr =
            unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

        static KHRONOS_VALIDATION_IGNORE_FILTER: &CStr =
            unsafe { CStr::from_bytes_with_nul_unchecked(b"message_id_filter\0") };
        // Timeline semaphores can cause false positives with vkQueueSubmit: see
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2441
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/4427
        // All of these aren't really valid, since each command buffer is synced using timeline semaphores,
        // and queue submissions are done out of order.
        // As stated in the issues, those VUID should be validated at execution time, not queue submission time
        static KHRONOS_VALIDATION_IGNORE_VUIDS: &[&[u8]] = &[
            b"VUID-VkImageMemoryBarrier2-oldLayout-01197,UNASSIGNED-VkBufferMemoryBarrier-buffer-00004,UNASSIGNED-VkImageMemoryBarrier-image-00004,UNASSIGNED-CoreValidation-DrawState-InvalidImageLayout\0",
        ];

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
        let mut requested_instance_extensions: Vec<*const c_char> = vec![];

        if configuration
            .features
            .contains(DeviceFeatures::HAL_DEBUG_LAYERS)
        {
            requested_layers.push(LAYER_KHRONOS_VALIDATION.as_ptr());
            requested_instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        if cfg!(feature = "swapchain") {
            if let Some(display_handle) = &configuration.display_handle {
                let extensions = ash_window::enumerate_required_extensions(*display_handle)?;
                requested_instance_extensions.extend(extensions);
            }
        }

        Self::ensure_requested_layers_are_avaliable(entry, &requested_layers)?;
        Self::ensure_requested_instance_extensions_are_available(
            entry,
            &requested_instance_extensions,
        )?;

        let mut message_id_filter = vk::LayerSettingEXT::default()
            .layer_name(LAYER_KHRONOS_VALIDATION)
            .setting_name(KHRONOS_VALIDATION_IGNORE_FILTER)
            .ty(vk::LayerSettingTypeEXT::STRING);
        message_id_filter.p_values = KHRONOS_VALIDATION_IGNORE_VUIDS.as_ptr().cast();
        message_id_filter.value_count = 1;
        let layer_settings = [message_id_filter];

        let mut layer_settings_create_info =
            vk::LayerSettingsCreateInfoEXT::default().settings(&layer_settings);

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&requested_layers)
            .enabled_extension_names(&requested_instance_extensions)
            .push_next(&mut layer_settings_create_info);
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
        device: &ash::Device,
    ) -> VulkanHalResult<VulkanDebugUtilities> {
        let debug_utils_ext = ash::ext::debug_utils::Instance::new(entry, instance);
        let debug_utils_device = ash::ext::debug_utils::Device::new(instance, device);

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
            debug_device: debug_utils_device,
        })
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_families: &VulkanQueueFamilies,
    ) -> VulkanHalResult<VulkanLogicalDevice> {
        const KHR_SWAPCHAIN_EXTENSION: &CStr =
            unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain\0") };

        #[derive(Default)]
        struct QueueInfo {
            count: u32,
            priorities: Vec<f32>,
        }
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
        let device_features = Self::get_physical_device_features(supported_device_features)?;

        let mut required_extensions = vec![];
        if cfg!(feature = "swapchain") {
            required_extensions.push(KHR_SWAPCHAIN_EXTENSION.as_ptr());
        }
        Self::ensure_requested_device_extensions_are_available(
            instance,
            physical_device,
            &required_extensions,
        )?;
        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let mut features_12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(true);

        let mut physical_device_features_2 = vk::PhysicalDeviceFeatures2::default()
            .features(device_features)
            .push_next(&mut features_13)
            .push_next(&mut features_12);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&required_extensions)
            .push_next(&mut physical_device_features_2);

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
            graphics_queue: VulkanQueue {
                handle: graphics_queue,
                family_index: g_f,
                queue_index: g_i,
            },
            compute_queue: VulkanQueue {
                handle: compute_queue,
                family_index: c_f,
                queue_index: c_i,
            },
            transfer_queue: VulkanQueue {
                handle: transfer_queue,
                family_index: t_f,
                queue_index: t_i,
            },
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
        _supported_device_features: vk::PhysicalDeviceFeatures,
    ) -> VulkanHalResult<vk::PhysicalDeviceFeatures> {
        let physical_device_features = vk::PhysicalDeviceFeatures::default();
        Ok(physical_device_features)
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

    unsafe fn wrap_raw_image(
        &self,
        image: vk::Image,
        image_description: &ImageDescription,
        layout: ImageLayout,
        access_flags: vk::AccessFlags2,
        stage_mask: vk::PipelineStageFlags2,
    ) -> VulkanHalResult<crate::Image> {
        if let Some(name) = image_description.label {
            self.try_assign_debug_name(image, name)?;
        }
        let mips = iter::repeat(LayoutInfo {
            image_layout: layout,
            access_mask: access_flags,
            stage_mask,
        })
        .take(image_description.mips.get() as usize)
        .collect::<Vec<_>>();
        let mut layouts = vec![];
        layouts.resize(image_description.array_layers.get() as usize, mips);
        let vulkan_image = VulkanImage {
            label: image_description.label.map(ToOwned::to_owned),
            handle: image,
            external: true,
            allocation: None,
            subresource_layouts: layouts,
        };
        let handle = self.resolver.add(vulkan_image);
        Ok(crate::Image {
            id: handle.to_u64(),
            creation_flags: image_description.creation_flags,
            usage_flags: image_description.usage_flags,
            extents: image_description.extents,
            dimension: image_description.dimension,
            num_mips: image_description.mips,
            array_layers: image_description.array_layers,
            samples: image_description.samples,
            format: image_description.format,
        })
    }

    unsafe fn wrap_raw_image_view(
        &self,
        image: crate::Image,
        view: vk::ImageView,
        subresource: ImageSubresource,
        name: Option<&str>,
    ) -> VulkanHalResult<crate::ImageView> {
        if let Some(name) = name {
            self.try_assign_debug_name(view, name)?;
        }

        let vulkan_image_view = VulkanImageView {
            label: name.map(ToOwned::to_owned),
            handle: view,
            external: true,
        };
        let handle = self.resolver.add(vulkan_image_view);
        Ok(crate::ImageView {
            id: handle.to_u64(),
            owner: image,
            subresource,
        })
    }

    fn try_assign_debug_name<T: ash::vk::Handle + Copy>(
        &self,
        object: T,
        name: &str,
    ) -> VulkanHalResult<()> {
        if let Some(debug_utils) = &self.debug_utilities {
            let string = CString::new(name).expect("Failed to create string name");
            let object_name = string.as_c_str();
            let debug_object_info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(object)
                .object_name(object_name);

            let debug_object_tag = vk::DebugUtilsObjectTagInfoEXT::default()
                .object_handle(object)
                .tag(object_name.to_bytes());
            unsafe {
                debug_utils
                    .debug_device
                    .set_debug_utils_object_name(&debug_object_info)?;
                debug_utils
                    .debug_device
                    .set_debug_utils_object_tag(&debug_object_tag)?;
            };
        }
        Ok(())
    }

    fn create_frames_in_flight(
        logical_device: &VulkanLogicalDevice,
        configuration: &DeviceConfiguration,
    ) -> VulkanHalResult<FramesInFlight> {
        let mut frames = vec![];
        let device = &logical_device.handle;
        for _ in 0..configuration.desired_frames_in_flight {
            let make_fence = || unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    get_allocation_callbacks(),
                )
            };
            let make_semaphore = || unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default(),
                    get_allocation_callbacks(),
                )
            };
            let make_command_pool = |qfi| {
                let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::default())
                    .queue_family_index(qfi);
                unsafe {
                    device
                        .create_command_pool(&command_pool_create_info, get_allocation_callbacks())
                }
            };
            frames.push(FrameInFlight {
                graphics_work_ended_fence: make_fence()?,
                compute_work_ended_fence: make_fence()?,
                transfer_work_ended_fence: make_fence()?,
                work_ended_semaphore: make_semaphore()?,
                graphics_command_pool: make_command_pool(
                    logical_device.graphics_queue.family_index,
                )?,
                compute_command_pool: make_command_pool(logical_device.compute_queue.family_index)?,
                transfer_command_pool: make_command_pool(
                    logical_device.transfer_queue.family_index,
                )?,
                allocated_semaphores: vec![],
                cached_semaphores: vec![],
                allocated_semaphores_binary: vec![],
                cached_semaphores_binary: vec![],
                atomic_semaphore_counter: AtomicU64::from(0),
            });
        }
        Ok(FramesInFlight {
            frames,
            current_frame_in_flight: 0,
        })
    }

    fn resolve_framebuffer_for_render_pass(
        &self,
        render_pass: vk::RenderPass,
        render_pass_info: &RenderPassInfo,
    ) -> MgpuResult<vk::Framebuffer> {
        let framebuffer_hash = hash_type(&render_pass_info.framebuffer);
        let mut framebuffers = self
            .framebuffers
            .write()
            .expect("Failed to lock framebuffers");

        if let Some(framebuffer) = framebuffers.get(&framebuffer_hash) {
            Ok(*framebuffer)
        } else {
            let framebuffer = self.create_famebuffer(render_pass, &render_pass_info.framebuffer)?;
            framebuffers.insert(framebuffer_hash, framebuffer);
            Ok(framebuffer)
        }
    }

    fn resolve_render_pass(
        &self,
        render_pass_info: &RenderPassInfo,
    ) -> VulkanHalResult<vk::RenderPass> {
        let render_pass_hash = hash_render_pass_info(render_pass_info);
        let mut render_passes = self
            .render_passes
            .write()
            .expect("Failed to lock render_passes");

        if let Some(render_pass) = render_passes.get(&render_pass_hash) {
            Ok(*render_pass)
        } else {
            let render_pass = self.create_render_pass(render_pass_info)?;
            render_passes.insert(render_pass_hash, render_pass);
            Ok(render_pass)
        }
    }

    fn create_famebuffer(
        &self,
        render_pass: vk::RenderPass,
        framebuffer: &Framebuffer,
    ) -> MgpuResult<vk::Framebuffer> {
        let image_views = framebuffer
            .render_targets
            .iter()
            .map(|rt| self.resolver.resolve_vulkan(rt.view))
            .chain(
                framebuffer
                    .depth_stencil_target
                    .iter()
                    .map(|rt| self.resolver.resolve_vulkan(rt.view)),
            )
            .map(|s| s.ok_or(MgpuError::InvalidHandle))
            .collect::<MgpuResult<Vec<_>>>()?;

        // TODO: Framebuffer Multi Layers
        let framebuffer_create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachment_count(image_views.len() as _)
            .attachments(&image_views)
            .width(framebuffer.extents.width)
            .height(framebuffer.extents.height)
            .layers(1);
        let framebuffer = unsafe {
            self.logical_device
                .handle
                .create_framebuffer(&framebuffer_create_info, get_allocation_callbacks())?
        };
        Ok(framebuffer)
    }

    fn create_render_pass(
        &self,
        render_pass_info: &RenderPassInfo,
    ) -> VulkanHalResult<vk::RenderPass> {
        let attachments = render_pass_info
            .framebuffer
            .render_targets
            .iter()
            .map(|rt| {
                vk::AttachmentDescription::default()
                    .format(rt.view.owner.format.to_vk())
                    .initial_layout(match rt.load_op {
                        crate::RenderTargetLoadOp::DontCare
                        | crate::RenderTargetLoadOp::Clear(_) => vk::ImageLayout::UNDEFINED,
                        crate::RenderTargetLoadOp::Load => {
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        }
                    })
                    .final_layout(match rt.store_op {
                        crate::AttachmentStoreOp::DontCare => vk::ImageLayout::UNDEFINED,
                        crate::AttachmentStoreOp::Store => {
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        }
                        crate::AttachmentStoreOp::Present => vk::ImageLayout::PRESENT_SRC_KHR,
                    })
                    .load_op(match rt.load_op {
                        crate::RenderTargetLoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                        crate::RenderTargetLoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                        crate::RenderTargetLoadOp::Load => vk::AttachmentLoadOp::LOAD,
                    })
                    .store_op(match rt.store_op {
                        crate::AttachmentStoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
                        crate::AttachmentStoreOp::Store | crate::AttachmentStoreOp::Present => {
                            vk::AttachmentStoreOp::STORE
                        }
                    })
                    .samples(rt.sample_count.to_vk())
                    .flags(vk::AttachmentDescriptionFlags::empty())
            })
            .chain(
                render_pass_info
                    .framebuffer
                    .depth_stencil_target
                    .iter()
                    .map(|rt| {
                        vk::AttachmentDescription::default()
                            .format(rt.view.owner.format.to_vk())
                            .initial_layout(match rt.load_op {
                                crate::DepthStencilTargetLoadOp::DontCare
                                | crate::DepthStencilTargetLoadOp::Clear(..) => {
                                    vk::ImageLayout::UNDEFINED
                                }
                                crate::DepthStencilTargetLoadOp::Load => {
                                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                                }
                            })
                            .final_layout(match rt.store_op {
                                crate::AttachmentStoreOp::DontCare => vk::ImageLayout::UNDEFINED,
                                crate::AttachmentStoreOp::Store => {
                                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                }
                                crate::AttachmentStoreOp::Present => unreachable!(),
                            })
                            .load_op(match rt.load_op {
                                crate::DepthStencilTargetLoadOp::DontCare => {
                                    vk::AttachmentLoadOp::DONT_CARE
                                }
                                crate::DepthStencilTargetLoadOp::Clear(..) => {
                                    vk::AttachmentLoadOp::CLEAR
                                }
                                crate::DepthStencilTargetLoadOp::Load => vk::AttachmentLoadOp::LOAD,
                            })
                            .store_op(match rt.store_op {
                                crate::AttachmentStoreOp::DontCare => {
                                    vk::AttachmentStoreOp::DONT_CARE
                                }
                                crate::AttachmentStoreOp::Store => vk::AttachmentStoreOp::STORE,
                                crate::AttachmentStoreOp::Present => unreachable!(),
                            })
                            .samples(rt.sample_count.to_vk())
                            .flags(vk::AttachmentDescriptionFlags::empty())
                    }),
            )
            .collect::<Vec<_>>();
        let mut depth_stencil_attachment_references = vec![];
        let mut color_attachment_references = vec![];
        let mut subpasses = vec![];
        for sp in &render_pass_info.steps {
            let mut this_attachment_references = vec![];
            this_attachment_references.resize(
                render_pass_info.framebuffer.render_targets.len(),
                vk::AttachmentReference {
                    attachment: vk::ATTACHMENT_UNUSED,
                    layout: vk::ImageLayout::UNDEFINED,
                },
            );
            for (i, ar) in sp.color_attachments.iter().enumerate() {
                let reference = &mut this_attachment_references[i];
                reference.attachment = i as _;
                reference.layout = match ar.access_mode {
                    AttachmentAccessMode::Read => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    AttachmentAccessMode::Write => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                };
            }
            color_attachment_references.push(this_attachment_references);

            if let Some(ar) = &sp.depth_stencil_attachment {
                let depth_subpass_desc = vk::AttachmentReference::default()
                    .attachment(render_pass_info.framebuffer.render_targets.len() as _)
                    .layout(match ar.access_mode {
                        AttachmentAccessMode::Read => {
                            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        }
                        AttachmentAccessMode::Write => {
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                        }
                    });
                depth_stencil_attachment_references.push(depth_subpass_desc);
            } else {
                depth_stencil_attachment_references.push(vk::AttachmentReference {
                    attachment: vk::ATTACHMENT_UNUSED,
                    layout: Default::default(),
                })
            };
        }
        for (spi, _) in render_pass_info.steps.iter().enumerate() {
            let subpass_description = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachment_references[spi])
                .depth_stencil_attachment(&depth_stencil_attachment_references[spi]);

            subpasses.push(subpass_description);
        }
        let mut src_stage_mask = vk::PipelineStageFlags::default();
        let mut src_access_mask = vk::AccessFlags::default();
        let subpass_dependencies = subpasses
            .iter()
            .enumerate()
            .map(|(d_idx, _)| {
                let src_subpass = if d_idx == 0 {
                    vk::SUBPASS_EXTERNAL
                } else {
                    d_idx as u32 - 1
                };
                let step = &render_pass_info.steps[d_idx];
                let mut dst_access_mask = vk::AccessFlags::default();
                let mut dst_stage_mask = vk::PipelineStageFlags::default();
                for color_rt in &step.color_attachments {
                    if color_rt.access_mode == AttachmentAccessMode::Write {
                        dst_stage_mask |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                        dst_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
                    } else if render_pass_info.framebuffer.render_targets[color_rt.index].store_op
                        == AttachmentStoreOp::Present
                    {
                        dst_access_mask = vk::AccessFlags::empty();
                        dst_stage_mask = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                    } else {
                        dst_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                    }
                }

                if let Some(depth_rt) = &step.depth_stencil_attachment {
                    if depth_rt.access_mode == AttachmentAccessMode::Write {
                        dst_stage_mask |= vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
                        dst_access_mask |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    } else {
                        dst_access_mask |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                    }
                }
                let sd = vk::SubpassDependency::default()
                    .src_subpass(src_subpass)
                    .dst_subpass(d_idx as _)
                    .src_stage_mask(src_stage_mask)
                    .dst_stage_mask(dst_stage_mask)
                    .src_access_mask(src_access_mask)
                    .dst_access_mask(dst_access_mask);

                src_access_mask = dst_access_mask;
                src_stage_mask = dst_stage_mask;

                sd
            })
            .collect::<Vec<_>>();
        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies)
            .flags(vk::RenderPassCreateFlags::default());
        let rp = unsafe {
            self.logical_device
                .handle
                .create_render_pass(&render_pass_create_info, get_allocation_callbacks())
        }?;
        Ok(rp)
    }

    fn create_vulkan_graphics_pipeline(
        &self,
        pipeline_info: &VulkanGraphicsPipelineInfo,
        command_buffer_state: &VulkanCommandBufferState,
    ) -> MgpuResult<vk::Pipeline> {
        let pipeline_layout = &pipeline_info.layout;
        let vertex_shader_entrypt = CString::new(pipeline_layout.vertex_stage.entry_point.as_str())
            .expect("Failed to convert String to CString");
        let fragment_shader_entrypt = CString::new(
            pipeline_layout
                .fragment_stage
                .as_ref()
                .map(|f| f.entry_point.clone())
                .unwrap_or_default(),
        )
        .expect("Failed to convert String to CString");
        let mut stages = vec![];
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .flags(vk::PipelineShaderStageCreateFlags::default())
                .module(
                    self.resolver
                        .resolve_vulkan(pipeline_layout.vertex_stage.shader)
                        .ok_or(MgpuError::InvalidHandle)?,
                )
                .name(vertex_shader_entrypt.as_c_str())
                .stage(vk::ShaderStageFlags::VERTEX),
        );
        if let Some(fragment_stage) = &pipeline_layout.fragment_stage {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .flags(vk::PipelineShaderStageCreateFlags::default())
                    .module(
                        self.resolver
                            .resolve_vulkan(fragment_stage.shader)
                            .ok_or(MgpuError::InvalidHandle)?,
                    )
                    .name(fragment_shader_entrypt.as_c_str())
                    .stage(vk::ShaderStageFlags::FRAGMENT),
            );
        }

        let vertex_attribute_descriptions = pipeline_layout
            .vertex_stage
            .vertex_inputs
            .iter()
            .enumerate()
            .map(|(i, attribute)| {
                vk::VertexInputAttributeDescription::default()
                    .binding(i as _)
                    .location(attribute.location as _)
                    .format(attribute.format.to_vk())
                    .offset(attribute.offset as _)
            })
            .collect::<Vec<_>>();
        let vertex_binding_descriptions = pipeline_layout
            .vertex_stage
            .vertex_inputs
            .iter()
            .enumerate()
            .map(|(i, attrib)| {
                vk::VertexInputBindingDescription::default()
                    .binding(i as _)
                    .stride(attrib.stride as _)
                    .input_rate(attrib.frequency.to_vk())
            })
            .collect::<Vec<_>>();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_binding_descriptions);
        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .primitive_restart_enable(pipeline_layout.primitive_restart_enabled)
            .topology(pipeline_layout.primitive_topology.to_vk());
        // The viewports will be set when drawing
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .scissor_count(1)
            .viewport_count(1);

        let pipeline_rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(pipeline_layout.polygon_mode.to_vk())
            .line_width(pipeline_layout.polygon_mode.line_width())
            .cull_mode(pipeline_layout.cull_mode.to_vk())
            .front_face(pipeline_layout.front_face.to_vk());

        let multisample_state = if let Some(_state) = &pipeline_layout.multisample_state {
            todo!()
        } else {
            vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_compare_op(pipeline_layout.depth_stencil_state.depth_compare_op.to_vk())
            .depth_test_enable(pipeline_layout.depth_stencil_state.depth_test_enabled)
            .depth_write_enable(pipeline_layout.depth_stencil_state.depth_write_enabled);
        let color_attachments = pipeline_layout
            .fragment_stage
            .as_ref()
            .map(|fs| {
                fs.render_targets
                    .iter()
                    .map(|attachment| {
                        if let Some(settings) = &attachment.blend {
                            vk::PipelineColorBlendAttachmentState::default()
                                .blend_enable(true)
                                .src_color_blend_factor(settings.src_color_blend_factor.to_vk())
                                .dst_color_blend_factor(settings.dst_color_blend_factor.to_vk())
                                .color_blend_op(settings.color_blend_op.to_vk())
                                .src_alpha_blend_factor(settings.src_alpha_blend_factor.to_vk())
                                .dst_alpha_blend_factor(settings.dst_alpha_blend_factor.to_vk())
                                .alpha_blend_op(settings.alpha_blend_op.to_vk())
                                .color_write_mask(settings.write_mask.to_vk())
                        } else {
                            vk::PipelineColorBlendAttachmentState::default()
                                .color_write_mask(vk::ColorComponentFlags::RGBA)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_attachments)
            .blend_constants([1.0; 4])
            .logic_op(vk::LogicOp::default())
            .logic_op_enable(false);

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
        ];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state_create_info)
            .viewport_state(&viewport_state)
            .rasterization_state(&pipeline_rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_info.vk_layout)
            .render_pass(command_buffer_state.current_render_pass)
            .subpass(command_buffer_state.current_subpass);

        let pipeline = unsafe {
            self.logical_device
                .handle
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_create_info],
                    get_allocation_callbacks(),
                )
                .map_err(|(_, err)| VulkanHalError::ApiError(err))?[0]
        };

        if let Some(label) = pipeline_info.label.as_deref() {
            self.try_assign_debug_name(pipeline, label)?;
        }

        info!(
            "Created vulkan graphics pipeline {:?} subpass {}",
            pipeline_info.label, command_buffer_state.current_subpass
        );

        Ok(pipeline)
    }

    fn get_pipeline_layout(
        &self,
        label: Option<&str>,
        binding_sets_infos: &[OwnedBindingSetLayoutInfo],
        push_constant_ranges: Option<&PushConstantInfo>,
    ) -> MgpuResult<vk::PipelineLayout> {
        let descriptor_layouts = self.get_descriptor_set_layouts(binding_sets_infos)?;

        let hash = {
            let value = &descriptor_layouts;
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            push_constant_ranges.hash(&mut hasher);
            hasher.finish()
        };
        let mut pipeline_layouts = self
            .pipeline_layouts
            .write()
            .expect("Failed to lock pipeline layouts");
        if let Some(layout) = pipeline_layouts.get(&hash) {
            Ok(*layout)
        } else {
            let vk_push_constant_ranges = push_constant_ranges
                .iter()
                .map(|pc| pc.to_vk())
                .collect::<Vec<_>>();
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&descriptor_layouts)
                .push_constant_ranges(&vk_push_constant_ranges);
            let layout = unsafe {
                self.logical_device.handle.create_pipeline_layout(
                    &pipeline_layout_create_info,
                    get_allocation_callbacks(),
                )?
            };

            if let Some(label) = label {
                self.try_assign_debug_name(layout, label)?;
            }

            info!("Created pipeline layout for pipeline {:?}", label);

            pipeline_layouts.insert(hash, layout);
            Ok(layout)
        }
    }
    fn reflect_layout(
        &self,
        shader_module_description: &crate::ShaderModuleDescription<'_>,
    ) -> MgpuResult<ShaderModuleLayout> {
        use spirq::ty::ImageFormat as SpirqImageFormat;
        let spirv_mgpu_err =
            |s: spirq::prelude::Error| MgpuError::Dynamic(format!("SpirV reflection error: {s:?}"));
        let module_entry_points = ReflectConfig::new()
            .spv(shader_module_description.source)
            .ref_all_rscs(true)
            .reflect()
            .map_err(spirv_mgpu_err)?;
        let entry_points: Vec<_> = module_entry_points
            .iter()
            .map(|entry| entry.name.clone())
            .collect();

        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut descriptors = HashMap::<u32, HashMap<u32, _>>::default();
        let mut variables = vec![];
        let mut push_constant: Option<ShaderStageFlags> = None;

        module_entry_points.iter().for_each(|entry| {
            let shader_stage = entry_point_shader_stage(entry);
            for var in &entry.vars {
                match var {
                    Variable::Input { location, ty, .. } => {
                        inputs.push(ShaderAttribute {
                            location: location.loc() as usize,
                            format: ty_to_vertex_attribute_format(ty),
                        });
                    }
                    Variable::Output { location, ty, .. } => {
                        outputs.push(ShaderAttribute {
                            location: location.loc() as usize,
                            format: ty_to_vertex_attribute_format(ty),
                        });
                    }
                    Variable::Descriptor {
                        name,
                        desc_bind,
                        desc_ty,
                        ty,
                        ..
                    } => {
                        descriptors
                            .entry(desc_bind.set())
                            .or_default()
                            .entry(desc_bind.bind())
                            .or_insert((
                                name.clone(),
                                desc_ty.clone(),
                                ty.clone(),
                                ShaderStageFlags::empty(),
                            ))
                            .3 |= shader_stage;
                    }
                    Variable::PushConstant { .. } => {
                        push_constant.replace(push_constant.unwrap_or_default() | shader_stage);
                    }
                    Variable::SpecConstant { .. } => {
                        todo!()
                    }
                }
            }
        });

        let access_mode = |ty: Option<AccessType>| match ty.unwrap_or(AccessType::ReadWrite) {
            AccessType::ReadOnly => StorageAccessMode::Read,
            AccessType::WriteOnly => StorageAccessMode::Write,
            AccessType::ReadWrite => StorageAccessMode::ReadWrite,
        };
        let format = |fmt: SpirqImageFormat| match fmt {
            SpirqImageFormat::Unknown => unreachable!(),
            SpirqImageFormat::Rgba32f => todo!(),
            SpirqImageFormat::Rgba16f => todo!(),
            SpirqImageFormat::R32f => todo!(),
            SpirqImageFormat::Rgba8 => ImageFormat::Rgba8,
            SpirqImageFormat::Rgba8Snorm => todo!(),
            SpirqImageFormat::Rg32f => todo!(),
            SpirqImageFormat::Rg16f => todo!(),
            SpirqImageFormat::R11fG11fB10f => todo!(),
            SpirqImageFormat::R16f => todo!(),
            SpirqImageFormat::Rgba16 => todo!(),
            SpirqImageFormat::Rgb10A2 => todo!(),
            SpirqImageFormat::Rg16 => todo!(),
            SpirqImageFormat::Rg8 => todo!(),
            SpirqImageFormat::R16 => todo!(),
            SpirqImageFormat::R8 => todo!(),
            SpirqImageFormat::Rgba16Snorm => todo!(),
            SpirqImageFormat::Rg16Snorm => todo!(),
            SpirqImageFormat::Rg8Snorm => todo!(),
            SpirqImageFormat::R16Snorm => todo!(),
            SpirqImageFormat::R8Snorm => todo!(),
            SpirqImageFormat::Rgba32i => todo!(),
            SpirqImageFormat::Rgba16i => todo!(),
            SpirqImageFormat::Rgba8i => todo!(),
            SpirqImageFormat::R32i => todo!(),
            SpirqImageFormat::Rg32i => todo!(),
            SpirqImageFormat::Rg16i => todo!(),
            SpirqImageFormat::Rg8i => todo!(),
            SpirqImageFormat::R16i => todo!(),
            SpirqImageFormat::R8i => todo!(),
            SpirqImageFormat::Rgba32ui => todo!(),
            SpirqImageFormat::Rgba16ui => todo!(),
            SpirqImageFormat::Rgba8ui => todo!(),
            SpirqImageFormat::R32ui => todo!(),
            SpirqImageFormat::Rgb10a2ui => todo!(),
            SpirqImageFormat::Rg32ui => todo!(),
            SpirqImageFormat::Rg16ui => todo!(),
            SpirqImageFormat::Rg8ui => todo!(),
            SpirqImageFormat::R16ui => todo!(),
            SpirqImageFormat::R8ui => todo!(),
            SpirqImageFormat::R64ui => todo!(),
            SpirqImageFormat::R64i => todo!(),
        };
        let binding_sets = descriptors
            .into_iter()
            .map(|(set_idx, elements)| {
                let elements = elements
                    .into_iter()
                    .map(|(bind_idx, (name, desc_ty, ty, flags))| {
                        let kind = match desc_ty {
                            spirq::ty::DescriptorType::Sampler() => BindingSetElementKind::Sampler,
                            spirq::ty::DescriptorType::CombinedImageSampler() => {
                                let CombinedImageSamplerType { sampled_image_ty } =
                                    ty.as_combined_image_sampler().unwrap();
                                BindingSetElementKind::CombinedImageSampler {
                                    format: if sampled_image_ty.is_depth.unwrap_or_default() {
                                        ImageFormat::Depth32
                                    } else {
                                        match sampled_image_ty.scalar_ty {
                                            spirq::ty::ScalarType::Void => todo!(),
                                            spirq::ty::ScalarType::Boolean => todo!(),
                                            spirq::ty::ScalarType::Integer { is_signed, bits } => {
                                                match (is_signed, bits) {
                                                    (true, 8) => ImageFormat::Rgba8,
                                                    e => todo!("Handle case {e:?}"),
                                                }
                                            }
                                            spirq::ty::ScalarType::Float { .. } => todo!(),
                                        }
                                    },
                                    dimension: match sampled_image_ty.dim {
                                        spirq::ty::Dim::Dim1D => ImageDimension::D1,
                                        spirq::ty::Dim::Dim2D => ImageDimension::D2,
                                        spirq::ty::Dim::Dim3D => ImageDimension::D3,
                                        spirq::ty::Dim::DimCube => todo!(),
                                        spirq::ty::Dim::DimRect => todo!(),
                                        spirq::ty::Dim::DimBuffer => todo!(),
                                        spirq::ty::Dim::DimSubpassData => todo!(),
                                        spirq::ty::Dim::DimTileImageDataEXT => todo!(),
                                    },
                                }
                            }
                            spirq::ty::DescriptorType::SampledImage() => {
                                variables.push(ShaderVariable {
                                    name,
                                    binding_set: set_idx as _,
                                    binding_index: bind_idx as _,
                                    ty: VariableType::Texture(StorageAccessMode::Read),
                                });

                                BindingSetElementKind::SampledImage
                            }
                            spirq::ty::DescriptorType::StorageImage(access) => {
                                let storage_image = ty.as_storage_image().unwrap();
                                let access_mode = access_mode(Some(access));
                                variables.push(ShaderVariable {
                                    name,
                                    binding_set: set_idx as _,
                                    binding_index: bind_idx as _,
                                    ty: VariableType::Texture(access_mode),
                                });
                                BindingSetElementKind::StorageImage {
                                    format: format(storage_image.fmt),
                                    access_mode,
                                }
                            }
                            spirq::ty::DescriptorType::UniformTexelBuffer() => todo!(),
                            spirq::ty::DescriptorType::StorageTexelBuffer(_) => todo!(),
                            spirq::ty::DescriptorType::UniformBuffer() => {
                                if let Some(struc) = ty.as_struct() {
                                    variables
                                        .extend(get_struct_variables(struc, set_idx, bind_idx));
                                }

                                BindingSetElementKind::Buffer {
                                    ty: crate::BufferType::Uniform,
                                    access_mode: StorageAccessMode::Read,
                                }
                            }
                            spirq::ty::DescriptorType::StorageBuffer(access) => {
                                BindingSetElementKind::Buffer {
                                    access_mode: access_mode(Some(access)),
                                    ty: crate::BufferType::Storage,
                                }
                            }
                            _ => todo!(),
                        };
                        BindingSetElement {
                            binding: bind_idx as _,
                            array_length: 1,
                            ty: kind,
                            shader_stage_flags: flags,
                        }
                    })
                    .collect();

                OwnedBindingSetLayoutInfo {
                    set: set_idx as _,
                    layout: OwnedBindingSetLayout {
                        binding_set_elements: elements,
                    },
                }
            })
            .collect();
        Ok(ShaderModuleLayout {
            entry_points,
            inputs,
            outputs,
            binding_sets,
            variables,
            push_constant,
        })
    }

    fn get_descriptor_set_layouts(
        &self,
        binding_sets_infos: &[OwnedBindingSetLayoutInfo],
    ) -> MgpuResult<Vec<vk::DescriptorSetLayout>> {
        let mut cached_layouts = self
            .descriptor_set_layouts
            .write()
            .expect("Failed to lock ds layouts");
        let mut layouts = vec![];
        layouts.resize(binding_sets_infos.len(), vk::DescriptorSetLayout::default());
        for (idx, set) in binding_sets_infos.iter().enumerate() {
            let set_layout = &set.layout;

            let layout =
                self.get_descriptor_set_layout_for_binding_layout(set_layout, &mut cached_layouts)?;
            layouts[idx] = layout;
        }

        Ok(layouts)
    }

    fn get_descriptor_set_layout_for_binding_layout(
        &self,
        set_layout: &OwnedBindingSetLayout,
        cached_layouts: &mut HashMap<u64, vk::DescriptorSetLayout>,
    ) -> Result<vk::DescriptorSetLayout, MgpuError> {
        let hash = hash_type(set_layout);
        let layout = if let Some(layout) = cached_layouts.get(&hash) {
            *layout
        } else {
            let binding_types = set_layout
                .binding_set_elements
                .iter()
                .map(|ty| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(ty.binding as _)
                        .descriptor_count(ty.array_length as _)
                        .descriptor_type(ty.ty.to_vk())
                        .stage_flags(ty.shader_stage_flags.to_vk())
                })
                .collect::<Vec<_>>();

            let ds_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&binding_types)
                .flags(vk::DescriptorSetLayoutCreateFlags::default());
            let ds_layout = unsafe {
                self.logical_device.handle.create_descriptor_set_layout(
                    &ds_layout_create_info,
                    get_allocation_callbacks(),
                )?
            };

            cached_layouts.insert(hash, ds_layout);

            ds_layout
        };
        Ok(layout)
    }

    #[cfg(debug_assertions)]
    fn validate_shader_layout_against_binding_layouts(
        layout: &ShaderModuleLayout,
        binding_set_layouts: &[BindingSetLayoutInfo],
    ) {
        for shader_set in &layout.binding_sets {
            let entry = binding_set_layouts.iter().find(|s| s.set == shader_set.set);
            check!(
                entry.is_some(),
                "Pipeline expects a binding set at index {}",
                shader_set.set
            );
            let ds_binding = entry.unwrap();
            for shader_binding in &shader_set.layout.binding_set_elements {
                let pipeline_binding = ds_binding
                    .layout
                    .binding_set_elements
                    .iter()
                    .find(|s| s.binding == shader_binding.binding);
                check!(
                    pipeline_binding.is_some(),
                    "Shader expects a binding element at index {} in set {}, but none was provided",
                    shader_binding.binding,
                    shader_set.set
                );
                let pipeline_binding = pipeline_binding.unwrap();

                check!(
                    pipeline_binding
                        .shader_stage_flags
                        .contains(shader_binding.shader_stage_flags),
                    "Binding set {} binding {} expects at least a visibility of {:?}, got {:?}",
                    shader_set.set,
                    shader_binding.binding,
                    shader_binding.shader_stage_flags,
                    pipeline_binding.shader_stage_flags
                );

                check!(
                    pipeline_binding.array_length == shader_binding.array_length,
                    "Differing array lengths for set {} binding {}, got {} expected {}",
                    shader_set.set,
                    shader_binding.binding,
                    pipeline_binding.array_length,
                    shader_binding.array_length
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    fn validate_submission_info(&self, submission_info: &SubmitInfo) -> MgpuResult<()> {
        let mut command_buffer_ids = HashSet::new();
        for group in &submission_info.submission_groups {
            for &recorder in &group.command_recorders {
                check!(
                    !command_buffer_ids.contains(&recorder),
                    &format!(
                        "Tried submitting the same command recorder twice on queue {:?}",
                        recorder.queue_type
                    )
                );

                command_buffer_ids.insert(recorder);
            }
        }
        Ok(())
    }

    fn allocate_descriptor_set(
        &self,
        binding_set_layout: &BindingSetLayout,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> MgpuResult<DescriptorSetAllocation> {
        const MAX_SETS_PER_POOL: usize = 1000;
        let make_ds_pool = || {
            let mut descriptor_counts_map: HashMap<vk::DescriptorType, u32> = HashMap::new();

            for binding in binding_set_layout.binding_set_elements {
                let ty = binding.ty.to_vk();
                *descriptor_counts_map.entry(ty).or_default() += 1;
            }
            let mut descriptor_counts = Vec::with_capacity(descriptor_counts_map.len());

            for (ty, ds) in descriptor_counts_map {
                descriptor_counts.push(vk::DescriptorPoolSize {
                    ty,
                    descriptor_count: ds,
                });
            }

            for count in &mut descriptor_counts {
                count.descriptor_count *= MAX_SETS_PER_POOL as u32;
            }

            let ds_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(MAX_SETS_PER_POOL as _)
                .pool_sizes(&descriptor_counts);
            let pool = unsafe {
                self.logical_device
                    .handle
                    .create_descriptor_pool(&ds_pool_create_info, get_allocation_callbacks())?
            };

            let info = DescriptorPoolInfo {
                pool,
                allocated: 0,
                max: MAX_SETS_PER_POOL,
            };

            info!(
                "Created new descriptor pool for layout {:?}, max sets: {MAX_SETS_PER_POOL}",
                descriptor_set_layout
            );

            MgpuResult::Ok(info)
        };
        let mut infos = self.descriptor_pool_infos.lock().unwrap();
        let infos = infos.entry(descriptor_set_layout).or_default();
        if let Some(free) = infos.freed.pop() {
            infos.pools[free.pool_index].allocated += 1;
            Ok(free)
        } else {
            if let Some(pool) = infos.pools.last().copied() {
                if pool.allocated == pool.max {
                    let new_pool = make_ds_pool()?;
                    infos.pools.push(new_pool);
                }
            } else {
                let pool = make_ds_pool()?;
                infos.pools.push(pool);
            }
            let pool = infos.pools.last_mut().unwrap();
            pool.allocated += 1;

            let set = unsafe {
                let layouts = [descriptor_set_layout];
                let allocate_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool.pool)
                    .set_layouts(&layouts);

                self.logical_device
                    .handle
                    .allocate_descriptor_sets(&allocate_info)?
            }[0];

            Ok(DescriptorSetAllocation {
                descriptor_set: set,
                layout: descriptor_set_layout,
                pool_index: infos.pools.len() - 1,
            })
        }
    }

    fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        description: &BindingSetDescription,
    ) -> MgpuResult<()> {
        let mut image_ops = vec![];
        let mut buffer_ops = vec![];
        // let mut buffer_view_ops = vec![];
        for binding in description.bindings {
            match binding.ty {
                crate::BindingType::Sampler(sampler) => image_ops.push((
                    binding.binding,
                    binding.ty.binding_type().to_vk(),
                    vk::DescriptorImageInfo::default().sampler(
                        self.resolver
                            .resolve_vulkan(sampler)
                            .ok_or(MgpuError::InvalidHandle)?,
                    ),
                )),
                crate::BindingType::UniformBuffer {
                    buffer,
                    offset,
                    range,
                } => buffer_ops.push((
                    binding.binding,
                    binding.ty.binding_type().to_vk(),
                    vk::DescriptorBufferInfo::default()
                        .buffer(
                            self.resolver
                                .resolve_vulkan(buffer)
                                .ok_or(MgpuError::InvalidHandle)?,
                        )
                        .offset(offset as _)
                        .range(range as _),
                )),
                crate::BindingType::SampledImage { view, .. } => image_ops.push((
                    binding.binding,
                    binding.ty.binding_type().to_vk(),
                    vk::DescriptorImageInfo::default()
                        .image_view(
                            self.resolver
                                .resolve_vulkan(view)
                                .ok_or(MgpuError::InvalidHandle)?,
                        )
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                )),
                crate::BindingType::StorageImage { view, access_mode } => image_ops.push((
                    binding.binding,
                    binding.ty.binding_type().to_vk(),
                    vk::DescriptorImageInfo::default()
                        .image_view(
                            self.resolver
                                .resolve_vulkan(view)
                                .ok_or(MgpuError::InvalidHandle)?,
                        )
                        .image_layout(match access_mode {
                            StorageAccessMode::Read => vk::ImageLayout::GENERAL,
                            StorageAccessMode::Write => vk::ImageLayout::GENERAL,
                            StorageAccessMode::ReadWrite => vk::ImageLayout::GENERAL,
                        }),
                )),
            }
        }

        let mut writes = vec![];

        for (idx, ty, image) in &image_ops {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(*idx as _)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(*ty);
            write.p_image_info = std::ptr::addr_of!(*image);
            writes.push(write);
        }

        for (idx, ty, buf) in &buffer_ops {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(*idx as _)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(*ty);
            write.p_buffer_info = std::ptr::addr_of!(*buf);
            writes.push(write);
        }

        unsafe {
            self.logical_device
                .handle
                .update_descriptor_sets(&writes, &[])
        };

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn validate_graphics_pipeline_shader_layouts(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription<'_>,
    ) {
        let vs_layout = self
            .get_shader_module_layout(*graphics_pipeline_description.vertex_stage.shader)
            .expect("Invalid vertex shader handle");
        Self::validate_shader_layout_against_binding_layouts(
            &vs_layout,
            graphics_pipeline_description.binding_set_layouts,
        );
        if let Some(fs) = graphics_pipeline_description.fragment_stage {
            let fs_layout = self
                .get_shader_module_layout(*fs.shader)
                .expect("Invalid fragment shader handle");
            Self::validate_shader_layout_against_binding_layouts(
                &fs_layout,
                graphics_pipeline_description.binding_set_layouts,
            );
        }
    }
    #[cfg(debug_assertions)]
    fn validate_compute_pipeline_shader_layouts(
        &self,
        compute_pipeline_description: &ComputePipelineDescription<'_>,
    ) {
        let cs_layout = self
            .get_shader_module_layout(compute_pipeline_description.shader)
            .expect("Invalid handle");
        Self::validate_shader_layout_against_binding_layouts(
            &cs_layout,
            compute_pipeline_description.binding_set_layouts,
        );
    }
}

fn entry_point_shader_stage(entry: &spirq::prelude::EntryPoint) -> ShaderStageFlags {
    match entry.exec_model {
        spirq::spirv::ExecutionModel::Vertex => ShaderStageFlags::VERTEX,
        spirq::spirv::ExecutionModel::Fragment => ShaderStageFlags::FRAGMENT,
        spirq::spirv::ExecutionModel::GLCompute => ShaderStageFlags::COMPUTE,
        _ => todo!(),
    }
}

fn get_struct_variables(
    struc: &spirq::ty::StructType,
    set_idx: u32,
    bind_idx: u32,
) -> Vec<ShaderVariable> {
    let mut variables = vec![];
    for member in &struc.members {
        variables.push(ShaderVariable {
            name: member.name.clone(),
            binding_set: set_idx as _,
            binding_index: bind_idx as _,
            ty: match &member.ty {
                Type::Scalar(_) | Type::Vector(_) | Type::Matrix(_) | Type::Array(_) => {
                    VariableType::Field {
                        offset: member.offset.unwrap(),
                        format: ty_to_vertex_attribute_format(&member.ty),
                    }
                }
                Type::Struct(struc) => {
                    let variables = get_struct_variables(struc, set_idx, bind_idx);
                    VariableType::Compound(variables)
                }
                _ => todo!(),
            },
        });
    }
    variables
}

fn ty_to_vertex_attribute_format(ty: &Type) -> VertexAttributeFormat {
    match ty {
        Type::Scalar(s) => match s {
            spirq::ty::ScalarType::Void => unreachable!(),
            spirq::ty::ScalarType::Boolean => todo!(),
            spirq::ty::ScalarType::Integer { is_signed, .. } => {
                if *is_signed {
                    VertexAttributeFormat::Int
                } else {
                    VertexAttributeFormat::Uint
                }
            }
            spirq::ty::ScalarType::Float { .. } => VertexAttributeFormat::Float,
        },
        Type::Vector(VectorType { scalar_ty, nscalar }) => match scalar_ty {
            spirq::ty::ScalarType::Void => unreachable!(),
            spirq::ty::ScalarType::Boolean => todo!(),
            spirq::ty::ScalarType::Integer { is_signed, .. } => match (is_signed, nscalar) {
                (true, 2) => VertexAttributeFormat::Int2,
                (true, 3) => VertexAttributeFormat::Int3,
                (true, 4) => VertexAttributeFormat::Int4,
                (false, 2) => VertexAttributeFormat::Uint2,
                (false, 3) => VertexAttributeFormat::Uint3,
                (false, 4) => VertexAttributeFormat::Uint4,
                _ => {
                    todo!()
                }
            },
            spirq::ty::ScalarType::Float { .. } => match nscalar {
                2 => VertexAttributeFormat::Float2,
                3 => VertexAttributeFormat::Float3,
                4 => VertexAttributeFormat::Float4,
                _ => unreachable!(),
            },
        },
        Type::Matrix(m) => match (m.nvector, &m.vector_ty) {
            (
                2,
                VectorType {
                    scalar_ty: spirq::ty::ScalarType::Float { .. },
                    nscalar: 2,
                },
            ) => VertexAttributeFormat::Mat2x2,
            (
                3,
                VectorType {
                    scalar_ty: spirq::ty::ScalarType::Float { .. },
                    nscalar: 3,
                },
            ) => VertexAttributeFormat::Mat3x3,
            (
                4,
                VectorType {
                    scalar_ty: spirq::ty::ScalarType::Float { .. },
                    nscalar: 4,
                },
            ) => VertexAttributeFormat::Mat4x4,
            _ => todo!("Unhandled case of matrix {:?}", (m.nvector, &m.vector_ty)),
        },

        _ => todo!("{:?}", ty),
    }
}

impl Drop for VulkanHal {
    fn drop(&mut self) {
        self.device_wait_idle().unwrap();
        self.resolver.on_destroy(self);

        let pools = self.descriptor_pool_infos.lock().unwrap();
        for (_, pool) in pools.iter() {
            for pool in &pool.pools {
                unsafe {
                    self.logical_device
                        .handle
                        .destroy_descriptor_pool(pool.pool, get_allocation_callbacks())
                };
            }
        }
    }
}

fn hash_render_pass_info(render_pass_info: &RenderPassInfo) -> u64 {
    let mut hasher = DefaultHasher::new();
    render_pass_info.label.hash(&mut hasher);

    for rt in &render_pass_info.framebuffer.render_targets {
        rt.view.owner.format.hash(&mut hasher);
        rt.sample_count.hash(&mut hasher);
        rt.load_op.hash(&mut hasher);
        rt.store_op.hash(&mut hasher);
    }

    if let Some(rt) = &render_pass_info.framebuffer.depth_stencil_target {
        rt.view.owner.format.hash(&mut hasher);
        rt.sample_count.hash(&mut hasher);
        rt.load_op.hash(&mut hasher);
        rt.store_op.hash(&mut hasher);
    }

    hasher.finish()
}
impl FrameInFlight {
    fn allocate_semaphore(&mut self, device: &ash::Device) -> VulkanHalResult<vk::Semaphore> {
        if let Some(sem) = self.cached_semaphores.pop() {
            Ok(sem)
        } else {
            let mut timeline_semaphore_info =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            let semaphore = unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_semaphore_info),
                    get_allocation_callbacks(),
                )?
            };
            self.allocated_semaphores.push(semaphore);
            info!("Allocated new semaphore for current frame in flight");
            Ok(semaphore)
        }
    }
    fn allocate_binary_semaphore(
        &mut self,
        device: &ash::Device,
    ) -> VulkanHalResult<vk::Semaphore> {
        if let Some(sem) = self.cached_semaphores_binary.pop() {
            Ok(sem)
        } else {
            let semaphore = unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default(),
                    get_allocation_callbacks(),
                )?
            };
            self.allocated_semaphores_binary.push(semaphore);
            info!("Allocated new binary semaphore for current frame in flight");
            Ok(semaphore)
        }
    }
}

impl FramesInFlight {
    fn current_mut(&mut self) -> &mut FrameInFlight {
        let current = self.current_frame_in_flight;
        &mut self.frames[current]
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
            VulkanHalError::GpuAllocatorError(error) => {
                format!("Error allocating a resource: {error}")
            }
            #[cfg(feature = "swapchain")]
            VulkanHalError::SwapchainError(err) => format!("Swapchain error: {err:?}"),
        };
        MgpuError::Dynamic(format!("Vulkan Hal error: {}", message))
    }
}

impl From<vk::Result> for VulkanHalError {
    fn from(value: vk::Result) -> Self {
        VulkanHalError::ApiError(value)
    }
}

impl From<vk::Result> for MgpuError {
    fn from(value: vk::Result) -> Self {
        let vk_error = VulkanHalError::ApiError(value);
        MgpuError::VulkanError(vk_error)
    }
}

impl From<ash::LoadingError> for MgpuError {
    fn from(value: ash::LoadingError) -> Self {
        MgpuError::Dynamic(value.to_string())
    }
}

impl From<gpu_allocator::AllocationError> for VulkanHalError {
    fn from(value: gpu_allocator::AllocationError) -> Self {
        Self::GpuAllocatorError(value)
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if crate::util::ERROR_HAPPENED.load(Ordering::Relaxed) {
        return vk::FALSE;
    }
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

    vk::FALSE
}
