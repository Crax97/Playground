mod util;

#[cfg(feature = "swapchain")]
mod swapchain;

use crate::hal::vulkan::util::{ToVk, VulkanBuffer, VulkanImage};
use crate::hal::{CommandRecorder, CommandRecorderAllocator, Hal, QueueType};
use crate::rdg::PassGroup;
use crate::util::{hash_type, Handle};
use crate::{
    AccessMode, AttachmentAccessMode, BindingSetElement, BindingSetElementKind, BindingSetLayout,
    Buffer, BufferDescription, DeviceConfiguration, DeviceFeatures, DevicePreference, Framebuffer,
    GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription, ImageDimension,
    ImageFormat, MemoryDomain, MgpuError, MgpuResult, RenderPassInfo, ShaderAttribute,
    ShaderModule, ShaderModuleLayout, VertexAttributeFormat,
};
use ash::vk::{DebugUtilsMessageSeverityFlagsEXT, Handle as AshHandle, QueueFlags};
use ash::{vk, Entry, Instance};
use gpu_allocator::vulkan::{
    AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use raw_window_handle::{DisplayHandle, WindowHandle};
use spirv_reflect::types::{ReflectDecorationFlags, ReflectInterfaceVariable};
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::{self, c_char, CStr, CString};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

use self::swapchain::{SwapchainError, VulkanSwapchain};
use self::util::{ResolveVulkan, SpirvShaderModule, VulkanImageView, VulkanResolver};

use super::{RenderState, SubmissionGroup, SubmitInfo};

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
    pipelines: RwLock<HashMap<GraphicsPipeline, VulkanPipelineInfo>>,
    command_buffer_states: RwLock<HashMap<vk::CommandBuffer, VulkanCommandBufferState>>,

    memory_allocator: RwLock<Allocator>,
}

pub struct VulkanHalConfiguration {
    frames_in_flight: usize,
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

pub(crate) struct VulkanDebugUtilities {
    debug_messenger: vk::DebugUtilsMessengerEXT,
    debug_instance: ash::ext::debug_utils::Instance,
    debug_device: ash::ext::debug_utils::Device,
}
pub(crate) struct FrameInFlight {
    graphics_command_pool: vk::CommandPool,
    compute_command_pool: vk::CommandPool,
    transfer_command_pool: vk::CommandPool,

    command_buffers: RwLock<Vec<CommandBuffers>>,

    graphics_work_ended_fence: vk::Fence,
    compute_work_ended_fence: vk::Fence,
    transfer_work_ended_fence: vk::Fence,
    allocated_semaphores: Vec<vk::Semaphore>,
    cached_semaphores: Vec<vk::Semaphore>,
    work_ended_semaphores: Vec<vk::Semaphore>,
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

#[derive(Default)]
pub struct VulkanPipelineInfo {
    pipelines: Vec<vk::Pipeline>,
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

// Any of these might be null
#[derive(Default, Clone, Copy)]
struct CommandBuffers {
    graphics: vk::CommandBuffer,
    graphics_semaphore: vk::Semaphore,

    compute: vk::CommandBuffer,
    compute_semaphore: vk::Semaphore,

    transfer: vk::CommandBuffer,
    transfer_semaphore: vk::Semaphore,
}

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
    ) -> MgpuResult<u64> {
        let swapchain = VulkanSwapchain::create(self, swapchain_info)?;

        let handle = self.resolver.add(swapchain);
        Ok(handle.to_u64())
    }

    #[cfg(feature = "swapchain")]
    fn swapchain_acquire_next_image(&self, id: u64) -> MgpuResult<crate::SwapchainImage> {
        use crate::SwapchainImage;

        Ok(self
            .resolver
            .apply_mut::<VulkanSwapchain, SwapchainImage>(
                unsafe { Handle::from_u64(id) },
                |swapchain| {
                    let (index, _) = unsafe {
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
        current_frame.cached_semaphores = std::mem::take(&mut current_frame.allocated_semaphores);
        current_frame.work_ended_semaphores.clear();
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

    unsafe fn submit(&self, mut end_rendering_info: SubmitInfo) -> MgpuResult<()> {
        let mut current_frame = self.frames_in_flight.lock().unwrap();
        let current_frame = current_frame.current_mut();
        let device = &self.logical_device.handle;

        let mut command_buffer_states = self
            .command_buffer_states
            .write()
            .expect("Failed to lock command buffer states");
        command_buffer_states.clear();

        let mut all_semaphores_to_signal = vec![];
        let mut all_command_buffers = vec![];
        let mut all_semaphores_to_wait = vec![vec![]];

        if end_rendering_info.submission_groups.is_empty() {
            let semaphore = current_frame.allocate_semaphore(device)?;
            all_semaphores_to_signal.push(vec![[semaphore]]);
        }

        let device = &self.logical_device.handle;

        let mut graphics_queue_submit = vec![];
        let mut async_compute_queue_submit = vec![];
        let mut transfer_queue_submit = vec![];

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
            if i > 0 {
                all_semaphores_to_wait.push(
                    all_semaphores_to_signal[i - 1]
                        .iter()
                        .flatten()
                        .cloned()
                        .collect::<Vec<_>>(),
                );
            }
        }

        let all_stages = [vk::PipelineStageFlags::ALL_COMMANDS; 256];
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
                    .wait_semaphores(semaphores_to_wait);
                submission_queue.push(submit_info);
            }
        }

        // Signal at least one semaphore on the graphics queue for the swapchain
        if end_rendering_info.submission_groups.is_empty() {
            let submit_info = vk::SubmitInfo::default()
                .signal_semaphores(all_semaphores_to_signal[0][0].as_slice());

            graphics_queue_submit.push(submit_info);
        }

        {
            unsafe {
                device.queue_submit(
                    self.logical_device.transfer_queue.handle,
                    &transfer_queue_submit,
                    current_frame.transfer_work_ended_fence,
                )?;

                device.queue_submit(
                    self.logical_device.graphics_queue.handle,
                    &graphics_queue_submit,
                    current_frame.graphics_work_ended_fence,
                )?;

                device.queue_submit(
                    self.logical_device.compute_queue.handle,
                    &async_compute_queue_submit,
                    current_frame.compute_work_ended_fence,
                )?;
            }
        }

        current_frame.work_ended_semaphores = all_semaphores_to_signal
            .last()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect();
        Ok(())
    }
    unsafe fn end_rendering(&self) -> MgpuResult<()> {
        let mut ff = self.frames_in_flight.lock().unwrap();

        ff.current_frame_in_flight =
            (ff.current_frame_in_flight + 1) % self.configuration.frames_in_flight;
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
                        preferred_format: None,
                    },
                )
            },
        )
    }

    fn create_image(
        &self,
        image_description: &crate::ImageDescription,
    ) -> MgpuResult<crate::Image> {
        let mut flags = vk::ImageCreateFlags::default();
        if image_description.extents.depth > 1 {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        let tiling = if image_description.memory_domain == MemoryDomain::DeviceLocal {
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
            MemoryDomain::HostVisible | MemoryDomain::HostCoherent => MemoryLocation::CpuToGpu,
            MemoryDomain::DeviceLocal => MemoryLocation::GpuOnly,
        };
        let fallback_name = format!("Memory allocation for image {:?}", image);
        let allocation_create_desc = AllocationCreateDesc {
            name: image_description.label.unwrap_or(&fallback_name),
            requirements,
            location,
            linear: image_description.memory_domain == MemoryDomain::DeviceLocal,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .memory_allocator
            .write()
            .expect("Failed to lock memory allocator")
            .allocate(&allocation_create_desc)
            .map_err(|e| MgpuError::VulkanError(VulkanHalError::GpuAllocatorError(e)))?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let vulkan_image = VulkanImage {
            label: image_description.label.map(ToOwned::to_owned),
            handle: image,
            external: false,
            allocation: Some(allocation),
        };

        let handle = self.resolver.add(vulkan_image);
        let image = Image {
            id: handle.to_u64(),
            usage_flags: image_description.usage_flags,
            extents: image_description.extents,
            dimension: image_description.dimension,
            mips: image_description.mips,
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

        let fallback_name = format!("Memory allocation for buffer {:?}", buffer);
        let name = buffer_description.label.unwrap_or(&fallback_name);
        let allocation_description = AllocationCreateDesc {
            name,
            requirements: memory_requirements,
            location: match buffer_description.memory_domain {
                MemoryDomain::HostVisible | MemoryDomain::HostCoherent => MemoryLocation::CpuToGpu,
                MemoryDomain::DeviceLocal => MemoryLocation::GpuOnly,
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
        let layout = self.reflect_layout(&shader_module_description)?;
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
        let owned_info = graphics_pipeline_description.to_vk_owned();
        let handle = self.resolver.add(owned_info);

        Ok(GraphicsPipeline {
            id: handle.to_u64(),
        })
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

    unsafe fn present_image(&self, swapchain_id: u64, _image: Image) -> MgpuResult<()> {
        let current_frame = self.frames_in_flight.lock().unwrap();
        let current_frame = current_frame.current();
        let swapchain = unsafe { Handle::from_u64(swapchain_id) };
        self.resolver
            .apply_mut::<VulkanSwapchain, ()>(swapchain, |swapchain| {
                let queue = self.logical_device.graphics_queue.handle;
                let current_index = swapchain.current_image_index.take().expect(
                    "Either a Present has already been issued, or acquire has never been called",
                );
                let indices = [current_index];
                let swapchains = [swapchain.handle];
                let present_info = vk::PresentInfoKHR::default()
                    .swapchains(&swapchains)
                    .wait_semaphores(&current_frame.work_ended_semaphores)
                    .image_indices(&indices);
                let swapchain_device = &swapchain.swapchain_device;

                let suboptimal = unsafe { swapchain_device.queue_present(queue, &present_info)? };
                Ok(())
            })
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

            let viewport = vk::Viewport::default()
                .width(render_pass_info.render_area.extents.width as f32)
                .height(render_pass_info.render_area.extents.height as f32)
                .x(render_pass_info.render_area.offset.x as f32)
                .y(render_pass_info.render_area.offset.y as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[render_pass_info.render_area.to_vk()]);
            state.current_render_pass = render_pass;
        };

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
        let pipeline = {
            let pipelines = self.pipelines.read().expect("Failed to lock pipelines");
            pipelines.get(&graphics_pipeline).and_then(|info| {
                info.pipelines
                    .get(command_buffer_state.current_subpass as usize)
                    .copied()
            })
        };

        let pipeline = if let Some(pipeline) = pipeline {
            pipeline
        } else {
            let pipeline_info = self
                .resolver
                .resolve_clone(graphics_pipeline)
                .ok_or(MgpuError::InvalidHandle)?;
            let mut pipelines = self.pipelines.write().expect("Failed to lock pipelines");
            let pipeline =
                self.create_vulkan_graphics_pipeline(&pipeline_info, &command_buffer_state)?;
            pipelines
                .entry(graphics_pipeline)
                .or_default()
                .pipelines
                .push(pipeline);
            pipeline
        };

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

    unsafe fn set_binding_sets(
        &self,
        command_recorder: CommandRecorder,
        binding_sets: &[crate::BindingSet],
    ) -> MgpuResult<()> {
        for set in binding_sets {
            todo!()
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

    unsafe fn enqueue_synchronization(
        &self,
        infos: &[super::SynchronizationInfo],
    ) -> MgpuResult<()> {
        let mut buffers = self.resolver.get_mut::<VulkanBuffer>();

        let device = &self.logical_device.handle;
        for info in infos {
            let vk_buffer_src = vk::CommandBuffer::from_raw(info.source_command_recorder.id);
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
                // let mut image_memory_barrier_source = vec![];
                let mut buffer_memory_barrier_source = vec![];

                // let mut image_memory_barrier_dest = vec![];
                let mut buffer_memory_barrier_dest = vec![];

                for resource in &info.resources {
                    let dst_stage_mask = match resource.access_mode {
                        super::ResourceAccessMode::AttachmentRead(ty) => match ty {
                            super::AttachmentType::Color => {
                                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                            }
                            super::AttachmentType::DepthStencil => {
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                            }
                        },
                        super::ResourceAccessMode::AttachmentWrite(ty) => match ty {
                            super::AttachmentType::Color => {
                                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                            }
                            super::AttachmentType::DepthStencil => {
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                            }
                        },
                        super::ResourceAccessMode::ShaderRead => todo!(),
                        super::ResourceAccessMode::ShaderWrite => todo!(),
                        super::ResourceAccessMode::TransferSrc => vk::PipelineStageFlags2::TRANSFER,
                        super::ResourceAccessMode::TransferDst => vk::PipelineStageFlags2::TRANSFER,
                        super::ResourceAccessMode::VertexInput => {
                            vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                        }
                    };
                    let dst_access_mask = match resource.access_mode {
                        super::ResourceAccessMode::AttachmentRead(ty) => match ty {
                            super::AttachmentType::Color => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                            super::AttachmentType::DepthStencil => {
                                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            }
                        },
                        super::ResourceAccessMode::AttachmentWrite(ty) => match ty {
                            super::AttachmentType::Color => {
                                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                            }
                            super::AttachmentType::DepthStencil => {
                                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                            }
                        },
                        super::ResourceAccessMode::ShaderRead => todo!(),
                        super::ResourceAccessMode::ShaderWrite => todo!(),
                        super::ResourceAccessMode::TransferSrc => vk::AccessFlags2::MEMORY_READ,
                        super::ResourceAccessMode::TransferDst => vk::AccessFlags2::MEMORY_WRITE,
                        super::ResourceAccessMode::VertexInput => {
                            vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                        }
                    };
                    match &resource.resource {
                        super::Resource::Image { image } => todo!(),
                        super::Resource::ImageView { view } => todo!(),
                        super::Resource::Buffer {
                            buffer,
                            offset,
                            size,
                        } => {
                            let buffer = buffers.resolve_mut(*buffer).unwrap();
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
                            buffer.current_access_mask = dst_access_mask;
                            buffer.current_stage_mask = dst_stage_mask;
                        }
                    }
                }

                let depedency_info_source = vk::DependencyInfo::default()
                    .buffer_memory_barriers(&buffer_memory_barrier_source)
                    .dependency_flags(vk::DependencyFlags::BY_REGION);

                let depedency_info_dest = vk::DependencyInfo::default()
                    .buffer_memory_barriers(&buffer_memory_barrier_dest)
                    .dependency_flags(vk::DependencyFlags::BY_REGION);

                unsafe {
                    device.cmd_pipeline_barrier2(vk_buffer_src, &depedency_info_source);
                    device.cmd_pipeline_barrier2(vk_buffer_dst, &depedency_info_dest);
                }
            } else {
                panic!("Pipeline barrier");
            }
        }
        Ok(())
    }
}

impl VulkanHal {
    const VULKAN_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
    pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry, configuration)?;
        let physical_device = Self::pick_physical_device(&instance, configuration)?;

        let queue_families = Self::pick_queue_families(&instance, physical_device.handle)?;
        let logical_device =
            Self::create_device(&instance, physical_device.handle, &queue_families)?;

        let use_debug_features = configuration
            .features
            .contains(DeviceFeatures::DEBUG_FEATURES);

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
            pipelines: RwLock::default(),

            memory_allocator: RwLock::new(
                Allocator::new(&allocator_create_desc)
                    .map_err(VulkanHalError::GpuAllocatorError)?,
            ),
            command_buffer_states: Default::default(),
            descriptor_set_layouts: Default::default(),
            pipeline_layouts: Default::default(),
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
        let mut requested_instance_extensions: Vec<*const c_char> = vec![];

        if configuration
            .features
            .contains(DeviceFeatures::DEBUG_FEATURES)
        {
            requested_layers.push(LAYER_KHRONOS_VALIDATION.as_ptr());
            requested_instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

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
            .enabled_layer_names(&requested_layers)
            .enabled_extension_names(&requested_instance_extensions);
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

    fn create_device(
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
        let mut physical_device_buffer_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&required_extensions)
            .push_next(&mut features_13)
            .push_next(&mut physical_device_buffer_address);

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
        supported_device_features: vk::PhysicalDeviceFeatures,
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
        description: &ImageDescription,
    ) -> VulkanHalResult<crate::Image> {
        if let Some(name) = description.label {
            self.try_assign_debug_name(image, name)?;
        }
        let vulkan_image = VulkanImage {
            label: description.label.map(ToOwned::to_owned),
            handle: image,
            external: true,
            allocation: None,
        };
        let handle = self.resolver.add(vulkan_image);
        Ok(crate::Image {
            id: handle.to_u64(),
            usage_flags: description.usage_flags,
            extents: description.extents,
            dimension: description.dimension,
            mips: description.mips,
            array_layers: description.array_layers,
            samples: description.samples,
            format: description.format,
        })
    }

    unsafe fn wrap_raw_image_view(
        &self,
        image: crate::Image,
        view: vk::ImageView,
        name: Option<&str>,
    ) -> VulkanHalResult<crate::ImageView> {
        if let Some(name) = name {
            self.try_assign_debug_name(view, name)?;
        }
        let vulkan_image = self
            .resolver
            .resolve_vulkan(image)
            .expect("Failed to resolve resource");
        let vulkan_image = VulkanImageView {
            label: name.map(ToOwned::to_owned),
            handle: view,
            owner: vulkan_image,
            external: true,
        };
        let handle = self.resolver.add(vulkan_image);
        Ok(crate::ImageView {
            id: handle.to_u64(),
            owner: image,
        })
    }

    fn try_assign_debug_name<T: ash::vk::Handle>(
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
            unsafe {
                debug_utils
                    .debug_device
                    .set_debug_utils_object_name(&debug_object_info)?
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
                work_ended_semaphores: vec![],
                graphics_command_pool: make_command_pool(
                    logical_device.graphics_queue.family_index,
                )?,
                compute_command_pool: make_command_pool(logical_device.compute_queue.family_index)?,
                transfer_command_pool: make_command_pool(
                    logical_device.transfer_queue.family_index,
                )?,
                command_buffers: Default::default(),
                allocated_semaphores: vec![],
                cached_semaphores: vec![],
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
                                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
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
            .map(|(d_idx, sd)| {
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
        pipeline_info: &util::VulkanGraphicsPipelineDescription,
        command_buffer_state: &VulkanCommandBufferState,
    ) -> MgpuResult<vk::Pipeline> {
        let pipeline_layout = self.get_pipeline_layout(pipeline_info)?;

        let vertex_shader_entrypt = CString::new(pipeline_info.vertex_stage.entry_point.as_str())
            .expect("Failed to convert String to CString");
        let fragment_shader_entrypt = CString::new(
            pipeline_info
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
                        .resolve_vulkan(pipeline_info.vertex_stage.shader)
                        .ok_or(MgpuError::InvalidHandle)?,
                )
                .name(vertex_shader_entrypt.as_c_str())
                .stage(vk::ShaderStageFlags::VERTEX),
        );
        if let Some(fragment_stage) = &pipeline_info.fragment_stage {
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

        let vertex_attribute_descriptions = pipeline_info
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
        let vertex_binding_descriptions = pipeline_info
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
            .primitive_restart_enable(pipeline_info.primitive_restart_enabled)
            .topology(pipeline_info.primitive_topology.to_vk());
        // The viewports will be set when drawing
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .scissor_count(1)
            .viewport_count(1);

        let pipeline_rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(pipeline_info.polygon_mode.to_vk())
            .line_width(pipeline_info.polygon_mode.line_width())
            .cull_mode(pipeline_info.cull_mode.to_vk())
            .front_face(pipeline_info.front_face.to_vk());

        let multisample_state = if let Some(state) = &pipeline_info.multisample_state {
            todo!()
        } else {
            vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_compare_op(pipeline_info.depth_stencil_state.depth_compare_op.to_vk())
            .depth_test_enable(pipeline_info.depth_stencil_state.depth_test_enabled)
            .depth_write_enable(pipeline_info.depth_stencil_state.depth_write_enabled);
        let color_attachments = pipeline_info
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
            .layout(pipeline_layout)
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

        Ok(pipeline)
    }

    fn get_pipeline_layout(
        &self,
        pipeline_info: &util::VulkanGraphicsPipelineDescription,
    ) -> MgpuResult<vk::PipelineLayout> {
        let descriptor_layouts = self.get_descriptor_set_layouts(&pipeline_info)?;

        let hash = hash_type(&descriptor_layouts);
        let mut pipeline_layouts = self
            .pipeline_layouts
            .write()
            .expect("Failed to lock pipeline layouts");
        if let Some(layout) = pipeline_layouts.get(&hash) {
            Ok(*layout)
        } else {
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_layouts);
            let layout = unsafe {
                self.logical_device.handle.create_pipeline_layout(
                    &pipeline_layout_create_info,
                    get_allocation_callbacks(),
                )?
            };

            pipeline_layouts.insert(hash, layout);
            Ok(layout)
        }
    }

    fn reflect_layout(
        &self,
        shader_module_description: &&crate::ShaderModuleDescription<'_>,
    ) -> MgpuResult<ShaderModuleLayout> {
        use spirv_reflect::*;
        let spirv_mgpu_err = |s: &str| MgpuError::Dynamic(format!("SpirV reflection error: {s}"));
        let module = ShaderModule::load_u32_data(shader_module_description.source)
            .map_err(spirv_mgpu_err)?;
        let entry_points = module
            .enumerate_entry_points()
            .map_err(spirv_mgpu_err)?
            .into_iter()
            .map(|entry| entry.name)
            .collect();

        let inputs = module
            .enumerate_input_variables(None)
            .map_err(spirv_mgpu_err)?;

        let outputs = module
            .enumerate_output_variables(None)
            .map_err(spirv_mgpu_err)?;

        let descriptor_layouts = module
            .enumerate_descriptor_sets(None)
            .map_err(spirv_mgpu_err)?;
        let filter_out_semantics = |input: &ReflectInterfaceVariable| {
            !input
                .decoration_flags
                .contains(ReflectDecorationFlags::BUILT_IN)
        };
        let spirv_to_attribute_input = |input: ReflectInterfaceVariable| {
            let format = match input.format {
                types::ReflectFormat::Undefined => unreachable!("{input:?}"),
                types::ReflectFormat::R32_UINT => VertexAttributeFormat::Uint,
                types::ReflectFormat::R32_SINT => VertexAttributeFormat::Int,
                types::ReflectFormat::R32_SFLOAT => VertexAttributeFormat::Float,
                types::ReflectFormat::R32G32_UINT => VertexAttributeFormat::Uint2,
                types::ReflectFormat::R32G32_SINT => VertexAttributeFormat::Int2,
                types::ReflectFormat::R32G32_SFLOAT => VertexAttributeFormat::Float2,
                types::ReflectFormat::R32G32B32_UINT => VertexAttributeFormat::Uint3,
                types::ReflectFormat::R32G32B32_SINT => VertexAttributeFormat::Int3,
                types::ReflectFormat::R32G32B32_SFLOAT => VertexAttributeFormat::Float3,
                types::ReflectFormat::R32G32B32A32_UINT => VertexAttributeFormat::Uint4,
                types::ReflectFormat::R32G32B32A32_SINT => VertexAttributeFormat::Int4,
                types::ReflectFormat::R32G32B32A32_SFLOAT => VertexAttributeFormat::Float4,
            };
            ShaderAttribute {
                name: input.name,
                location: input.location as _,
                format,
            }
        };

        let inputs = inputs
            .into_iter()
            .filter(filter_out_semantics)
            .map(spirv_to_attribute_input)
            .collect::<Vec<_>>();
        let outputs = outputs
            .into_iter()
            .filter(filter_out_semantics)
            .map(spirv_to_attribute_input)
            .collect::<Vec<_>>();
        let binding_sets = descriptor_layouts
            .into_iter()
            .map(|set| {
                let elements = set
                    .bindings
                    .into_iter()
                    .map(|element| {
                        println!("element {element:#?}");
                        let image_format = match element.image.image_format {
                            types::ReflectImageFormat::Undefined => ImageFormat::Unknown,
                            types::ReflectImageFormat::RGBA8 => ImageFormat::Rgba8,
                            _ => todo!("Unknown image format {:?}", element.image.image_format),
                        };

                        let dimension = match element.image.dim {
                            types::ReflectDimension::Undefined => None,
                            types::ReflectDimension::Type1d => Some(ImageDimension::D1),
                            types::ReflectDimension::Type2d => Some(ImageDimension::D2),
                            types::ReflectDimension::Type3d => Some(ImageDimension::D3),
                            _ => todo!("Unknown dimension {:?}", element.image.dim),
                        };
                        let access_mode = match element.resource_type {
                            types::ReflectResourceType::Undefined => unreachable!(),
                            types::ReflectResourceType::Sampler => AccessMode::Read,
                            types::ReflectResourceType::CombinedImageSampler => AccessMode::Read,
                            types::ReflectResourceType::ConstantBufferView => todo!(),
                            types::ReflectResourceType::ShaderResourceView => AccessMode::Read,
                            types::ReflectResourceType::UnorderedAccessView => todo!(),
                        };

                        let kind = match element.descriptor_type {
                            types::ReflectDescriptorType::Undefined => unreachable!(),
                            types::ReflectDescriptorType::Sampler => {
                                BindingSetElementKind::Sampler {
                                    access_mode: todo!("{element:?}"),
                                }
                            }
                            types::ReflectDescriptorType::CombinedImageSampler => {
                                BindingSetElementKind::SampledImage {
                                    format: image_format,
                                    dimension: dimension.unwrap(),
                                }
                            }
                            types::ReflectDescriptorType::SampledImage => {
                                BindingSetElementKind::SampledImage {
                                    format: image_format,
                                    dimension: dimension.unwrap(),
                                }
                            }
                            types::ReflectDescriptorType::StorageImage => {
                                BindingSetElementKind::StorageImage {
                                    format: image_format,
                                    access_mode: todo!("{element:?}"),
                                    dimension: dimension.unwrap(),
                                }
                            }
                            types::ReflectDescriptorType::UniformTexelBuffer => todo!(),
                            types::ReflectDescriptorType::StorageTexelBuffer => todo!(),
                            types::ReflectDescriptorType::UniformBuffer => {
                                BindingSetElementKind::Buffer {
                                    ty: crate::BufferType::Uniform,
                                    access_mode: crate::AccessMode::Read,
                                }
                            }
                            types::ReflectDescriptorType::StorageBuffer => {
                                BindingSetElementKind::Buffer {
                                    ty: crate::BufferType::Storage,
                                    access_mode: todo!("{element:?}"),
                                }
                            }
                            types::ReflectDescriptorType::UniformBufferDynamic => todo!(),
                            types::ReflectDescriptorType::StorageBufferDynamic => todo!(),
                            types::ReflectDescriptorType::InputAttachment => todo!(),
                            types::ReflectDescriptorType::AccelerationStructureNV => todo!(),
                        };
                        BindingSetElement {
                            name: element.name,
                            binding: element.binding as _,
                            ty: kind,
                        }
                    })
                    .collect();
                BindingSetLayout {
                    set: set.set as _,
                    bindings: elements,
                }
            })
            .collect();
        Ok(ShaderModuleLayout {
            entry_points,
            inputs,
            outputs,
            binding_sets,
        })
    }

    fn get_descriptor_set_layouts(
        &self,
        pipeline_info: &&util::VulkanGraphicsPipelineDescription,
    ) -> MgpuResult<Vec<vk::DescriptorSetLayout>> {
        #[derive(Default, Hash, Clone, Copy)]
        struct DescriptorBindingType {
            ty: vk::DescriptorType,
            count: u32,
            shader_stage_flags: vk::ShaderStageFlags,
        }
        #[derive(Default, Hash)]
        struct DescriptorTypes {
            bindings: Vec<DescriptorBindingType>,
        }
        let mut descriptor_set_layouts: HashMap<usize, DescriptorTypes> = HashMap::default();
        let mut count_descriptor_types =
            |layout: &ShaderModuleLayout, flags: vk::ShaderStageFlags| {
                for set in &layout.binding_sets {
                    let entry = descriptor_set_layouts.entry(set.set).or_default();
                    for binding in &set.bindings {
                        if entry.bindings.len() < binding.binding {
                            entry
                                .bindings
                                .resize(binding.binding + 1, Default::default())
                        }
                        let ds_binding = &mut entry.bindings[binding.binding];
                        ds_binding.shader_stage_flags |= flags;
                        ds_binding.count = 1;
                        let ty = match binding.ty {
                            BindingSetElementKind::Buffer { ty, .. } => match ty {
                                crate::BufferType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                                crate::BufferType::Storage => vk::DescriptorType::STORAGE_BUFFER,
                            },
                            BindingSetElementKind::Sampler { .. } => vk::DescriptorType::SAMPLER,
                            BindingSetElementKind::SampledImage { .. } => {
                                vk::DescriptorType::SAMPLED_IMAGE
                            }
                            BindingSetElementKind::StorageImage { .. } => {
                                vk::DescriptorType::STORAGE_IMAGE
                            }
                        };
                        ds_binding.ty = ty;
                    }
                }
            };
        let vs_layout = self.get_shader_module_layout(pipeline_info.vertex_stage.shader)?;
        count_descriptor_types(&vs_layout, vk::ShaderStageFlags::VERTEX);

        if let Some(fs) = &pipeline_info.fragment_stage {
            let fs_layout = self.get_shader_module_layout(fs.shader)?;
            count_descriptor_types(&fs_layout, vk::ShaderStageFlags::FRAGMENT);
        }

        let mut cached_layouts = self
            .descriptor_set_layouts
            .write()
            .expect("Failed to lock ds layouts");
        let mut layouts = vec![];
        layouts.resize(
            descriptor_set_layouts.len(),
            vk::DescriptorSetLayout::default(),
        );
        for (idx, types) in descriptor_set_layouts {
            let hash = hash_type(&types);

            if let Some(layout) = cached_layouts.get(&hash) {
                layouts[idx] = *layout;
            } else {
                let binding_types = types
                    .bindings
                    .iter()
                    .map(|ty| {
                        vk::DescriptorSetLayoutBinding::default()
                            .descriptor_count(ty.count)
                            .descriptor_type(ty.ty)
                            .stage_flags(ty.shader_stage_flags)
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

                layouts[idx] = ds_layout;
            }
        }

        Ok(layouts)
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
    fn get_last_command_buffers(
        &self,
        device: &ash::Device,
        current_frame: &FrameInFlight,
        group: &PassGroup,
    ) -> VulkanHalResult<CommandBuffers> {
        let mut command_buffer_list = self
            .command_buffers
            .write()
            .expect("Failed to lock command buffers");
        if command_buffer_list.is_empty() {
            let mut command_buffers = CommandBuffers::default();

            let allocate_command_buffer = |command_pool| {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY);

                let command_buffers = unsafe { device.allocate_command_buffers(&info)? };
                VulkanHalResult::Ok(command_buffers[0])
            };
            let make_semaphore = || {
                let info =
                    vk::SemaphoreCreateInfo::default().flags(vk::SemaphoreCreateFlags::empty());

                unsafe { device.create_semaphore(&info, get_allocation_callbacks()) }
            };
            if !group.graphics_nodes.is_empty() {
                command_buffers.graphics =
                    allocate_command_buffer(current_frame.graphics_command_pool)?;
                command_buffers.graphics_semaphore = make_semaphore()?;
            }
            if !group.compute_nodes.is_empty() {
                command_buffers.compute =
                    allocate_command_buffer(current_frame.compute_command_pool)?;
                command_buffers.compute_semaphore = make_semaphore()?;
            }

            command_buffer_list.push(command_buffers);
            Ok(command_buffers)
        } else {
            let last = command_buffer_list.last().unwrap();
            Ok(*last)
        }
    }

    fn read_command_buffers(&self) -> RwLockReadGuard<'_, Vec<CommandBuffers>> {
        self.command_buffers
            .read()
            .expect("Failed to lock command buffers")
    }

    fn allocate_semaphore(&mut self, device: &ash::Device) -> VulkanHalResult<vk::Semaphore> {
        if let Some(sem) = self.cached_semaphores.pop() {
            Ok(sem)
        } else {
            let semaphore = unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default(),
                    get_allocation_callbacks(),
                )?
            };
            self.allocated_semaphores.push(semaphore);
            Ok(semaphore)
        }
    }
}

impl FramesInFlight {
    fn current(&self) -> &FrameInFlight {
        let current = self.current_frame_in_flight;
        &self.frames[current]
    }
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

    if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        panic!("Invalid vulkan state: check error log above.");
    }

    vk::FALSE
}
