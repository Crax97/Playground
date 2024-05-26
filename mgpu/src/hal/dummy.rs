use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::{Extents2D, Extents3D, ImageCreationFlags, ImageFormat, ImageUsageFlags};

use super::Hal;

#[derive(Default)]
pub struct DummyHal {
    command_buffers: AtomicU64,
    buffers: AtomicU64,
    images: AtomicU64,
    image_views: AtomicU64,
    shader_modules: AtomicU64,
    pipelines: AtomicU64,
    samplers: AtomicU64,
    binding_sets: AtomicU64,
}

impl Hal for DummyHal {
    fn device_wait_idle(&self) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn device_wait_queue(&self, _queue: super::QueueType) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn prepare_next_frame(&self) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn begin_rendering(&self) -> crate::MgpuResult<super::RenderState> {
        Ok(super::RenderState {
            graphics_compute_allocator: super::CommandRecorderAllocator {
                id: 0,
                queue_type: super::QueueType::Graphics,
            },
            async_compute_allocator: super::CommandRecorderAllocator {
                id: 1,
                queue_type: super::QueueType::AsyncCompute,
            },
            async_transfer_allocator: super::CommandRecorderAllocator {
                id: 2,
                queue_type: super::QueueType::AsyncTransfer,
            },
        })
    }

    unsafe fn request_oneshot_command_recorder(
        &self,
        queue_type: super::QueueType,
    ) -> crate::MgpuResult<super::CommandRecorder> {
        Ok(super::CommandRecorder {
            id: self.command_buffers.fetch_add(1, Relaxed),
            queue_type,
        })
    }

    unsafe fn submit_command_recorder_immediate(
        &self,
        _command_recorder: super::CommandRecorder,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn request_command_recorder(
        &self,
        allocator: super::CommandRecorderAllocator,
    ) -> crate::MgpuResult<super::CommandRecorder> {
        Ok(super::CommandRecorder {
            id: self.command_buffers.fetch_add(1, Relaxed),
            queue_type: allocator.queue_type,
        })
    }

    unsafe fn finalize_command_recorder(
        &self,
        _command_buffer: super::CommandRecorder,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn set_graphics_push_constant(
        &self,
        _command_recorder: super::CommandRecorder,
        _graphics_pipeline: crate::GraphicsPipeline,
        _data: &[u8],
        _visibility: crate::ShaderStageFlags,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn set_compute_push_constant(
        &self,
        _command_recorder: super::CommandRecorder,
        _compute_pipeline: crate::ComputePipeline,
        _data: &[u8],
        _visibility: crate::ShaderStageFlags,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn begin_render_pass(
        &self,
        _command_recorder: super::CommandRecorder,
        _render_pass_info: &crate::RenderPassInfo,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn bind_graphics_pipeline(
        &self,
        _command_recorder: super::CommandRecorder,
        _pipeline: crate::GraphicsPipeline,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn set_vertex_buffers(
        &self,
        _command_recorder: super::CommandRecorder,
        _vertex_buffers: &[crate::Buffer],
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn set_index_buffer(
        &self,
        _command_recorder: super::CommandRecorder,
        _index_buffer: crate::Buffer,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn set_scissor_rect(
        &self,
        _command_recorder: super::CommandRecorder,
        _scissor_rect: crate::Rect2D,
    ) {
    }

    unsafe fn bind_graphics_binding_sets(
        &self,
        _command_recorder: super::CommandRecorder,
        _binding_sets: &[crate::BindingSet],
        _graphics_pipeline: crate::GraphicsPipeline,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn draw(
        &self,
        _command_recorder: super::CommandRecorder,
        _vertices: usize,
        _indices: usize,
        _first_vertex: usize,
        _first_index: usize,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn draw_indexed(
        &self,
        _command_recorder: super::CommandRecorder,
        _indices: usize,
        _instances: usize,
        _first_index: usize,
        _vertex_offset: i32,
        _first_instance: usize,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn advance_to_next_step(
        &self,
        _command_recorder: super::CommandRecorder,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn end_render_pass(
        &self,
        _command_recorder: super::CommandRecorder,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn present_image(
        &self,
        _swapchain_id: u64,
        _image: crate::Image,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn submit(&self, _end_rendering_info: super::SubmitInfo) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn end_rendering(&self) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn bind_compute_pipeline(
        &self,
        _command_recorder: super::CommandRecorder,
        _pipeline: crate::ComputePipeline,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn bind_compute_binding_sets(
        &self,
        _command_recorder: super::CommandRecorder,
        _binding_sets: &[crate::BindingSet],
        _pipeline: crate::ComputePipeline,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn dispatch(
        &self,
        _command_recorder: super::CommandRecorder,
        _group_count_x: u32,
        _group_count_y: u32,
        _group_count_z: u32,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn cmd_copy_buffer_to_buffer(
        &self,
        _command_buffer: super::CommandRecorder,
        _source: crate::Buffer,
        _dest: crate::Buffer,
        _source_offset: usize,
        _dest_offset: usize,
        _size: usize,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn cmd_copy_buffer_to_image(
        &self,
        _command_buffer: super::CommandRecorder,
        _source: crate::Buffer,
        _dest: crate::Image,
        _source_offset: usize,
        _dest_region: crate::ImageRegion,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn cmd_blit_image(
        &self,
        _command_buffer: super::CommandRecorder,
        _source: crate::Image,
        _source_region: crate::ImageRegion,
        _dest: crate::Image,
        _dest_region: crate::ImageRegion,
        _filter: crate::FilterMode,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn enqueue_synchronization(
        &self,
        _infos: &[super::SynchronizationInfo],
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn transition_resources(
        &self,
        _command_recorder: super::CommandRecorder,
        _resources: &[super::ResourceTransition],
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn create_swapchain_impl(
        &self,
        swapchain_info: &crate::SwapchainCreationInfo,
    ) -> crate::MgpuResult<crate::SwapchainInfo> {
        Ok(crate::SwapchainInfo {
            id: 0,
            format: swapchain_info
                .preferred_format
                .unwrap_or(crate::ImageFormat::Rgba8),
            present_mode: swapchain_info
                .preferred_present_mode
                .unwrap_or(crate::PresentMode::Fifo),
        })
    }

    fn swapchain_destroy(&self, _id: u64) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn swapchain_acquire_next_image(&self, _id: u64) -> crate::MgpuResult<crate::SwapchainImage> {
        let image = crate::Image {
            id: 0,
            usage_flags: ImageUsageFlags::COLOR_ATTACHMENT,
            creation_flags: ImageCreationFlags::default(),
            extents: Extents3D {
                width: 512,
                height: 512,
                depth: 1,
            },
            dimension: crate::ImageDimension::D2,
            num_mips: 1.try_into().unwrap(),
            array_layers: 1.try_into().unwrap(),
            samples: crate::SampleCount::One,
            format: ImageFormat::Rgba8,
        };
        Ok(crate::SwapchainImage {
            image,
            extents: Extents2D {
                width: 512,
                height: 512,
            },
            view: crate::ImageView {
                owner: image,
                subresource: image.whole_subresource(),
                id: 0,
            },
        })
    }

    fn swapchain_on_resized(
        &self,
        _id: u64,
        _new_size: crate::Extents2D,
        _window_handle: raw_window_handle::WindowHandle,
        _display_handle: raw_window_handle::DisplayHandle,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn try_swapchain_set_present_mode(
        &self,
        _id: u64,
        present_mode: crate::PresentMode,
    ) -> crate::MgpuResult<crate::PresentMode> {
        Ok(present_mode)
    }

    fn device_info(&self) -> crate::DeviceInfo {
        crate::DeviceInfo {
            name: "DummyHAL SuperGPU3000".into(),
            api_description: "DummyHAL".into(),
            swapchain_support: true,
            frames_in_flight: 3,
        }
    }

    fn create_image(
        &self,
        image_description: &crate::ImageDescription,
    ) -> crate::MgpuResult<crate::Image> {
        Ok(crate::Image {
            id: self.images.fetch_add(1, Relaxed),
            usage_flags: image_description.usage_flags,
            creation_flags: image_description.creation_flags,
            extents: image_description.extents,
            dimension: image_description.dimension,
            num_mips: image_description.mips,
            array_layers: image_description.array_layers,
            samples: image_description.samples,
            format: image_description.format,
        })
    }

    fn image_name(&self, _image: crate::Image) -> crate::MgpuResult<Option<String>> {
        Ok(None)
    }

    fn destroy_image(&self, _image: crate::Image) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn create_image_view(
        &self,
        image_view_description: &crate::ImageViewDescription,
    ) -> crate::MgpuResult<crate::ImageView> {
        Ok(crate::ImageView {
            owner: image_view_description.image,
            subresource: image_view_description.image_subresource,
            id: self.image_views.fetch_add(1, Relaxed),
        })
    }

    fn create_buffer(
        &self,
        buffer_description: &crate::BufferDescription,
    ) -> crate::MgpuResult<crate::Buffer> {
        Ok(crate::Buffer {
            id: self.buffers.fetch_add(1, Relaxed),
            usage_flags: buffer_description.usage_flags,
            size: buffer_description.size,
            memory_domain: buffer_description.memory_domain,
        })
    }

    fn buffer_name(&self, _buffer: crate::Buffer) -> crate::MgpuResult<Option<String>> {
        Ok(None)
    }

    fn destroy_buffer(&self, _buffer: crate::Buffer) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn create_graphics_pipeline(
        &self,
        _graphics_pipeline_description: &crate::GraphicsPipelineDescription,
    ) -> crate::MgpuResult<crate::GraphicsPipeline> {
        Ok(crate::GraphicsPipeline {
            id: self.pipelines.fetch_add(1, Relaxed),
        })
    }

    fn create_compute_pipeline(
        &self,
        _compute_pipeline_description: &crate::ComputePipelineDescription,
    ) -> crate::MgpuResult<crate::ComputePipeline> {
        Ok(crate::ComputePipeline {
            id: self.pipelines.fetch_add(1, Relaxed),
        })
    }

    fn get_graphics_pipeline_layout(
        &self,
        _graphics_pipeline: crate::GraphicsPipeline,
    ) -> crate::MgpuResult<crate::GraphicsPipelineLayout> {
        todo!()
    }

    fn get_compute_pipeline_layout(
        &self,
        _compute_pipeline: crate::ComputePipeline,
    ) -> crate::MgpuResult<super::ComputePipelineLayout> {
        todo!()
    }

    fn destroy_graphics_pipeline(
        &self,
        _graphics_pipeline: crate::GraphicsPipeline,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn destroy_compute_pipeline(&self, _pipeline: crate::ComputePipeline) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn create_shader_module(
        &self,
        _shader_module_description: &crate::ShaderModuleDescription,
    ) -> crate::MgpuResult<crate::ShaderModule> {
        Ok(crate::ShaderModule {
            id: self.shader_modules.fetch_add(1, Relaxed),
        })
    }

    fn get_shader_module_layout(
        &self,
        _shader_module: crate::ShaderModule,
    ) -> crate::MgpuResult<crate::ShaderModuleLayout> {
        todo!()
    }

    fn destroy_shader_module(&self, _shader_module: crate::ShaderModule) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn create_sampler(
        &self,
        _sampler_description: &crate::SamplerDescription,
    ) -> crate::MgpuResult<crate::Sampler> {
        Ok(crate::Sampler {
            id: self.samplers.fetch_add(1, Relaxed),
        })
    }

    fn destroy_sampler(&self, _sampler: crate::Sampler) -> crate::MgpuResult<()> {
        todo!()
    }

    fn create_binding_set(
        &self,
        description: &crate::BindingSetDescription,
        _layout: &crate::BindingSetLayout,
    ) -> crate::MgpuResult<crate::BindingSet> {
        Ok(crate::BindingSet {
            id: self.binding_sets.fetch_add(1, Relaxed),
            bindings: description.bindings.to_vec(),
        })
    }

    fn destroy_binding_set(&self, _binding_set: crate::BindingSet) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn destroy_image_view(&self, _image_view: crate::ImageView) -> crate::MgpuResult<()> {
        Ok(())
    }

    unsafe fn write_host_visible_buffer(
        &self,
        _buffer: crate::Buffer,
        _params: &crate::BufferWriteParams,
    ) -> crate::MgpuResult<()> {
        Ok(())
    }

    fn begin_debug_region(
        &self,
        _command_recorder: super::CommandRecorder,
        _region_name: &str,
        _color: [f32; 3],
    ) {
    }

    fn end_debug_region(&self, _command_recorder: super::CommandRecorder) {}

    unsafe fn cmd_clear_image(
        &self,
        _command_recorder: super::CommandRecorder,
        _target: crate::ImageView,
        _color: [f32; 4],
    ) {
    }
}
