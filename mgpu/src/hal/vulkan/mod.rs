mod util;

#[cfg(feature = "swapchain")]
mod swapchain;

use crate::hal::vulkan::util::{ToVk, VulkanBuffer, VulkanImage};
use crate::hal::{CommandRecorder, CommandRecorderAllocator, Hal};
use crate::rdg::PassGroup;
use crate::util::{hash_type, Handle};
use crate::{
    AttachmentAccessMode, Buffer, BufferDescription, DeviceConfiguration, DeviceFeatures,
    DevicePreference, Framebuffer, GraphicsPipeline, GraphicsPipelineDescription, Image,
    ImageDescription, MemoryDomain, MgpuError, MgpuResult, RenderPassInfo, ShaderModule,
};
use ash::vk::{DebugUtilsMessageSeverityFlagsEXT, Handle as AshHandle, QueueFlags};
use ash::{vk, Entry, Instance};
use gpu_allocator::vulkan::{
    AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use raw_window_handle::{DisplayHandle, WindowHandle};
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::{self, c_char, CStr, CString};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard};

use self::swapchain::{SwapchainError, VulkanSwapchain};
use self::util::{ResolveVulkan, VulkanImageView, VulkanResolver, VulkanShaderModule};

use super::{RenderState, SubmitInfo};

pub struct VulkanHal {
    entry: Entry,
    instance: Instance,
    physical_device: VulkanPhysicalDevice,
    logical_device: VulkanLogicalDevice,
    debug_utilities: Option<VulkanDebugUtilities>,
    configuration: VulkanHalConfiguration,
    resolver: VulkanResolver,
    frames_in_flight: Arc<FramesInFlight>,

    framebuffers: RwLock<HashMap<u64, vk::Framebuffer>>,
    render_passes: RwLock<HashMap<u64, vk::RenderPass>>,

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

    work_ended_fence: vk::Fence,
    work_ended_semaphore: vk::Semaphore,
}

pub(crate) struct FramesInFlight {
    frames: Vec<FrameInFlight>,
    current_frame_in_flight: AtomicUsize,
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
        let current_frame = self.frames_in_flight.current();
        let device = &self.logical_device.handle;
        unsafe { device.wait_for_fences(&[current_frame.work_ended_fence], true, u64::MAX)? };
        unsafe { device.reset_fences(&[current_frame.work_ended_fence])? };

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
            },
            async_compute_allocator: CommandRecorderAllocator {
                id: current_frame.compute_command_pool.as_raw(),
            },
        })
    }

    unsafe fn submit(&self, end_rendering_info: SubmitInfo) -> MgpuResult<()> {
        let current_frame = self.frames_in_flight.current();
        let device = &self.logical_device.handle;

        if !end_rendering_info.graphics_command_recorders.is_empty() {
            let command_buffers = end_rendering_info
                .graphics_command_recorders
                .iter()
                .map(|cb| vk::CommandBuffer::from_raw(cb.id))
                .collect::<Vec<_>>();
            // let pipeline_stage_flags = [vk::PipelineStageFlags::BOTTOM_OF_PIPE];
            let signal_semaphores = [current_frame.work_ended_semaphore];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                // .wait_dst_stage_mask(&pipeline_stage_flags)
                .signal_semaphores(&signal_semaphores);
            unsafe {
                device.queue_submit(
                    self.logical_device.graphics_queue.handle,
                    &[submit_info],
                    current_frame.work_ended_fence,
                )?;
            }
        }

        if !end_rendering_info
            .async_compute_command_recorders
            .is_empty()
        {
            todo!()
        }

        Ok(())
    }
    unsafe fn end_rendering(&self) -> MgpuResult<()> {
        let current_frame_index = self
            .frames_in_flight
            .current_frame_in_flight
            .load(Ordering::Relaxed);
        self.frames_in_flight.current_frame_in_flight.store(
            (current_frame_index + 1) % self.configuration.frames_in_flight,
            Ordering::Relaxed,
        );
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
        let image = self
            .resolver
            .remove::<VulkanImage>(unsafe { Handle::from_u64(image.id) })
            .ok_or(MgpuError::InvalidHandle)?;

        if image.external {
            return Ok(());
        }
        if let Some(allocation) = image.allocation {
            let mut allocator = self
                .memory_allocator
                .write()
                .expect("Failed to lock memory allocator");
            allocator
                .free(allocation)
                .map_err(|e| MgpuError::VulkanError(VulkanHalError::GpuAllocatorError(e)))?;
        }
        unsafe {
            self.logical_device
                .handle
                .destroy_image(image.handle, get_allocation_callbacks());
        }
        Ok(())
    }

    fn destroy_image_view(&self, image_view: crate::ImageView) -> MgpuResult<()> {
        let view = self
            .resolver
            .remove::<VulkanImageView>(unsafe { Handle::from_u64(image_view.id) })
            .ok_or(MgpuError::InvalidHandle)?;

        if view.external {
            return Ok(());
        }
        unsafe {
            self.logical_device
                .handle
                .destroy_image_view(view.handle, get_allocation_callbacks());
        }
        Ok(())
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

        let vulkan_buffer = VulkanBuffer {
            label: buffer_description.label.map(ToOwned::to_owned),
            handle: buffer,
            allocation,
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
        let buffer = self
            .resolver
            .remove(buffer)
            .ok_or(MgpuError::InvalidHandle)?;

        let mut allocator = self
            .memory_allocator
            .write()
            .expect("Failed to lock memory allocator");
        allocator
            .free(buffer.allocation)
            .map_err(|e| MgpuError::VulkanError(VulkanHalError::GpuAllocatorError(e)))?;

        unsafe {
            self.logical_device
                .handle
                .destroy_buffer(buffer.handle, get_allocation_callbacks());
        }
        Ok(())
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

        let vulkan_shader_module = VulkanShaderModule {
            label: shader_module_description.label.map(ToOwned::to_owned),
            handle: shader_module,
        };

        let handle = self.resolver.add(vulkan_shader_module);
        Ok(ShaderModule {
            id: handle.to_u64(),
        })
    }

    fn destroy_shader_module(&self, shader_module: ShaderModule) -> MgpuResult<()> {
        let shader_module = self
            .resolver
            .remove(shader_module)
            .ok_or(MgpuError::InvalidHandle)?;

        unsafe {
            self.logical_device
                .handle
                .destroy_shader_module(shader_module.handle, get_allocation_callbacks());
        }
        Ok(())
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
        todo!()
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

        unsafe {
            self.logical_device
                .handle
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
        }

        Ok(CommandRecorder { id: cb.as_raw() })
    }

    unsafe fn present_image(&self, swapchain_id: u64, _image: Image) -> MgpuResult<()> {
        let current_frame = self.frames_in_flight.current();
        let swapchain = unsafe { Handle::from_u64(swapchain_id) };
        self.resolver
            .apply_mut::<VulkanSwapchain, ()>(swapchain, |swapchain| {
                let queue = self.logical_device.graphics_queue.handle;
                let current_index = swapchain.current_image_index.take().expect(
                    "Either a Present has already been issued, or acquire has never been called",
                );
                let indices = [current_index];
                let swapchains = [swapchain.handle];
                let wait_semaphores = [current_frame.work_ended_semaphore];
                let present_info = vk::PresentInfoKHR::default()
                    .swapchains(&swapchains)
                    .wait_semaphores(&wait_semaphores)
                    .image_indices(&indices);
                let swapchain_device = &swapchain.swapchain_device;

                let suboptimal = unsafe { swapchain_device.queue_present(queue, &present_info)? };
                Ok(())
            })
    }

    unsafe fn finalize_command_recorder(&self, command_buffer: CommandRecorder) -> MgpuResult<()> {
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
        let render_pass = self.resolve_render_pass(render_pass_info)?;
        let framebuffer =
            self.resolve_framebuffer_for_render_pass(render_pass, render_pass_info)?;

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
        unsafe {
            self.logical_device.handle.cmd_begin_render_pass(
                vk::CommandBuffer::from_raw(command_recorder.id),
                &vk::RenderPassBeginInfo::default()
                    .framebuffer(framebuffer)
                    .render_pass(render_pass)
                    .render_area(render_pass_info.render_area.to_vk())
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            )
        };

        Ok(())
    }

    unsafe fn bind_graphics_pipeline(
        &self,
        command_recorder: CommandRecorder,
        pipeline: GraphicsPipeline,
    ) -> MgpuResult<()> {
        todo!()
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
        todo!()
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
        unsafe {
            self.logical_device.handle.cmd_next_subpass(
                vk::CommandBuffer::from_raw(command_recorder.id),
                vk::SubpassContents::INLINE,
            );
        }

        Ok(())
    }

    unsafe fn end_render_pass(&self, command_recorder: CommandRecorder) -> MgpuResult<()> {
        unsafe {
            self.logical_device
                .handle
                .cmd_end_render_pass(vk::CommandBuffer::from_raw(command_recorder.id))
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
            resolver: Default::default(),
            frames_in_flight: Arc::new(frames_in_flight),
            framebuffers: RwLock::default(),
            render_passes: RwLock::default(),

            memory_allocator: RwLock::new(
                Allocator::new(&allocator_create_desc)
                    .map_err(VulkanHalError::GpuAllocatorError)?,
            ),
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
            let fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    get_allocation_callbacks(),
                )?
            };
            let semaphore = unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default(),
                    get_allocation_callbacks(),
                )?
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
                work_ended_fence: fence,
                work_ended_semaphore: semaphore,
                graphics_command_pool: make_command_pool(
                    logical_device.graphics_queue.family_index,
                )?,
                compute_command_pool: make_command_pool(logical_device.compute_queue.family_index)?,
                transfer_command_pool: make_command_pool(
                    logical_device.transfer_queue.family_index,
                )?,
                command_buffers: Default::default(),
            });
        }
        Ok(FramesInFlight {
            frames,
            current_frame_in_flight: AtomicUsize::new(0),
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

    let render_pass_hash = hasher.finish();
    render_pass_hash
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
}

impl FramesInFlight {
    fn current(&self) -> &FrameInFlight {
        let current = self.current_frame_in_flight.load(Ordering::Relaxed);
        &self.frames[current]
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
