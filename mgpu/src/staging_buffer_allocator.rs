use crate::{hal::Hal, Buffer, BufferDescription, BufferUsageFlags, MemoryDomain, MgpuResult};

pub(crate) struct StagingBufferAllocator {
    staging_buffer: Buffer,
    allocated_regions: Vec<StagingBufferAllocatedRegion>,
    current_frame: usize,
    head: usize,
    tail: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StagingBufferAllocation {
    pub buffer: Buffer,
    pub offset: usize,
}

#[derive(Default, Clone, Copy)]
struct StagingBufferAllocatedRegion {
    pub tip: usize,
}

impl StagingBufferAllocator {
    pub fn new(
        hal: &dyn Hal,
        staging_area_size: usize,
        frames_in_flight: usize,
    ) -> MgpuResult<Self> {
        let staging_buffer = hal.create_buffer(&BufferDescription {
            label: Some("Staging buffer"),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::TRANSFER_SRC,
            size: staging_area_size,
            memory_domain: MemoryDomain::HostVisible,
        })?;

        let allocated_regions = std::iter::repeat(StagingBufferAllocatedRegion::default())
            .take(frames_in_flight)
            .collect::<Vec<_>>();

        Ok(Self {
            staging_buffer,
            allocated_regions,
            current_frame: 0,
            head: 0,
            tail: 0,
        })
    }

    // Returns a suitable, host-visible, buffer big enough to write 'size' bytes to it
    pub(crate) fn allocate_staging_buffer_region(
        &mut self,
        size: usize,
    ) -> MgpuResult<StagingBufferAllocation> {
        self.allocate_inner(size, false)
    }

    fn allocate_inner(
        &mut self,
        size: usize,
        recursed: bool,
    ) -> Result<StagingBufferAllocation, crate::MgpuError> {
        if self.head < self.tail {
            if self.head + size > self.tail {
                panic!("Staging buffer ran out of space");
            } else {
                let allocation = StagingBufferAllocation {
                    buffer: self.staging_buffer,
                    offset: self.head,
                };
                self.head += size;
                Ok(allocation)
            }
        } else if self.head + size > self.staging_buffer.size {
            self.head = 0;
            if recursed {
                panic!("Bug! Recursed twice in allocate_inner")
            }
            return self.allocate_inner(size, true);
        } else {
            let allocation = StagingBufferAllocation {
                buffer: self.staging_buffer,
                offset: self.head,
            };
            self.head += size;
            return Ok(allocation);
        }
    }

    pub fn end_frame(&mut self) {
        self.allocated_regions[self.current_frame] =
            StagingBufferAllocatedRegion { tip: self.head };
        self.current_frame = (self.current_frame + 1) % self.allocated_regions.len();
        self.tail = self.allocated_regions[self.current_frame].tip;
        self.allocated_regions[self.current_frame] = Default::default();
    }
}

#[cfg(test)]
mod tests {
    use crate::{hal::Hal, SwapchainInfo};

    use super::StagingBufferAllocator;

    struct DummyHal;

    impl Hal for DummyHal {
        fn device_wait_idle(&self) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn begin_rendering(&self) -> crate::MgpuResult<crate::hal::RenderState> {
            todo!()
        }

        unsafe fn request_command_recorder(
            &self,
            allocator: crate::hal::CommandRecorderAllocator,
        ) -> crate::MgpuResult<crate::hal::CommandRecorder> {
            todo!()
        }

        unsafe fn finalize_command_recorder(
            &self,
            command_buffer: crate::hal::CommandRecorder,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn begin_render_pass(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            render_pass_info: &crate::RenderPassInfo,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn bind_graphics_pipeline(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            pipeline: crate::GraphicsPipeline,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn set_vertex_buffers(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            vertex_buffers: &[crate::Buffer],
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn set_index_buffer(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            index_buffer: crate::Buffer,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn set_binding_sets(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            binding_sets: &[crate::BindingSet],
            graphics_pipeline: crate::GraphicsPipeline,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn draw(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            vertices: usize,
            indices: usize,
            first_vertex: usize,
            first_index: usize,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn draw_indexed(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            indices: usize,
            instances: usize,
            first_index: usize,
            vertex_offset: i32,
            first_instance: usize,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn advance_to_next_step(
            &self,
            command_recorder: crate::hal::CommandRecorder,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn end_render_pass(
            &self,
            command_recorder: crate::hal::CommandRecorder,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn present_image(
            &self,
            swapchain_id: u64,
            image: crate::Image,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn submit(
            &self,
            end_rendering_info: crate::hal::SubmitInfo,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn end_rendering(&self) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn cmd_copy_buffer_to_buffer(
            &self,
            command_buffer: crate::hal::CommandRecorder,
            source: crate::Buffer,
            dest: crate::Buffer,
            source_offset: usize,
            dest_offset: usize,
            size: usize,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn cmd_copy_buffer_to_image(
            &self,
            command_buffer: crate::hal::CommandRecorder,
            source: crate::Buffer,
            dest: crate::Image,
            source_offset: usize,
            dest_region: crate::ImageRegion,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn cmd_blit_image(
            &self,
            command_buffer: crate::hal::CommandRecorder,
            source: crate::Image,
            source_region: crate::ImageRegion,
            dest: crate::Image,
            dest_region: crate::ImageRegion,
            filter: crate::FilterMode,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn enqueue_synchronization(
            &self,
            infos: &[crate::hal::SynchronizationInfo],
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        fn transition_resources(
            &self,
            command_recorder: crate::hal::CommandRecorder,
            resources: &[crate::hal::ResourceTransition],
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        fn create_swapchain_impl(
            &self,
            swapchain_info: &crate::SwapchainCreationInfo,
        ) -> crate::MgpuResult<SwapchainInfo> {
            todo!()
        }

        fn swapchain_acquire_next_image(
            &self,
            id: u64,
        ) -> crate::MgpuResult<crate::SwapchainImage> {
            todo!()
        }

        fn swapchain_on_resized(
            &self,
            id: u64,
            new_size: crate::Extents2D,
            window_handle: raw_window_handle::WindowHandle,
            display_handle: raw_window_handle::DisplayHandle,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        fn device_info(&self) -> crate::DeviceInfo {
            todo!()
        }

        fn create_image(
            &self,
            image_description: &crate::ImageDescription,
        ) -> crate::MgpuResult<crate::Image> {
            todo!()
        }

        fn image_name(&self, image: crate::Image) -> crate::MgpuResult<Option<String>> {
            todo!()
        }

        fn destroy_image(&self, image: crate::Image) -> crate::MgpuResult<()> {
            todo!()
        }

        fn create_image_view(
            &self,
            image_view_description: &crate::ImageViewDescription,
        ) -> crate::MgpuResult<crate::ImageView> {
            todo!()
        }

        fn create_buffer(
            &self,
            buffer_description: &crate::BufferDescription,
        ) -> crate::MgpuResult<crate::Buffer> {
            Ok(crate::Buffer {
                id: 0,
                usage_flags: buffer_description.usage_flags,
                size: buffer_description.size,
                memory_domain: buffer_description.memory_domain,
            })
        }

        fn buffer_name(&self, buffer: crate::Buffer) -> crate::MgpuResult<Option<String>> {
            todo!()
        }

        fn destroy_buffer(&self, buffer: crate::Buffer) -> crate::MgpuResult<()> {
            todo!()
        }

        fn create_graphics_pipeline(
            &self,
            graphics_pipeline_description: &crate::GraphicsPipelineDescription,
        ) -> crate::MgpuResult<crate::GraphicsPipeline> {
            todo!()
        }

        fn get_graphics_pipeline_layout(
            &self,
            graphics_pipeline: crate::GraphicsPipeline,
        ) -> crate::MgpuResult<crate::GraphicsPipelineLayout> {
            todo!()
        }

        fn destroy_graphics_pipeline(
            &self,
            graphics_pipeline: crate::GraphicsPipeline,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        fn create_shader_module(
            &self,
            shader_module_description: &crate::ShaderModuleDescription,
        ) -> crate::MgpuResult<crate::ShaderModule> {
            todo!()
        }

        fn get_shader_module_layout(
            &self,
            shader_module: crate::ShaderModule,
        ) -> crate::MgpuResult<crate::ShaderModuleLayout> {
            todo!()
        }

        fn destroy_shader_module(
            &self,
            shader_module: crate::ShaderModule,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        fn create_sampler(
            &self,
            sampler_description: &crate::SamplerDescription,
        ) -> crate::MgpuResult<crate::Sampler> {
            todo!()
        }

        fn destroy_sampler(&self, sampler: crate::Sampler) -> crate::MgpuResult<()> {
            todo!()
        }

        fn create_binding_set(
            &self,
            description: &crate::BindingSetDescription,
            layout: &crate::BindingSetLayout,
        ) -> crate::MgpuResult<crate::BindingSet> {
            todo!()
        }

        fn destroy_binding_set(&self, binding_set: crate::BindingSet) -> crate::MgpuResult<()> {
            todo!()
        }

        fn destroy_image_view(&self, image_view: crate::ImageView) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn write_host_visible_buffer(
            &self,
            buffer: crate::Buffer,
            params: &crate::BufferWriteParams,
        ) -> crate::MgpuResult<()> {
            todo!()
        }

        unsafe fn prepare_next_frame(&self) -> crate::MgpuResult<()> {
            todo!()
        }

        fn try_swapchain_set_present_mode(
            &self,
            id: u64,
            present_mode: crate::PresentMode,
        ) -> crate::MgpuResult<crate::PresentMode> {
            todo!()
        }
    }

    #[test]
    fn test_staging_buffer_looping() {
        let hal = DummyHal;

        let mut staging_buffer = StagingBufferAllocator::new(&hal, 32, 4).unwrap();
        for _ in 0..10 {
            let alloc = staging_buffer.allocate_staging_buffer_region(5).unwrap();
            println!("{alloc:?}");
            staging_buffer.end_frame();
        }
    }
}
