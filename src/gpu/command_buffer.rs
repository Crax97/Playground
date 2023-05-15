use ash::{
    prelude::VkResult,
    vk::{
        self, ClearValue, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsageFlags, Offset2D, PipelineBindPoint, PipelineStageFlags, Rect2D,
        RenderPassBeginInfo, StructureType, SubmitInfo, SubpassContents, Viewport,
    },
};
use log::trace;

use super::{material::RenderPass, Gpu, Material};

#[derive(Default)]
pub enum QueueType {
    #[default]
    Graphics,
    AsyncCompute,
    Transfer,
}

#[derive(Default)]
pub struct CommandBufferSubmitInfo {
    pub wait_semaphores: Vec<vk::Semaphore>,
    pub wait_stages: Vec<PipelineStageFlags>,
    pub signal_semaphores: Vec<vk::Semaphore>,
    pub fence: Option<vk::Fence>,
    pub target_queue: QueueType,
}

pub struct CommandBuffer<'g> {
    gpu: &'g Gpu,
    inner_command_buffer: vk::CommandBuffer,
    has_recorded_anything: bool,
    info: CommandBufferSubmitInfo,
}

pub struct RenderPassCommand<'c, 'g>
where
    'g: 'c,
{
    command_buffer: &'c mut CommandBuffer<'g>,
    viewport_area: Option<Viewport>,
    scissor_area: Option<Rect2D>,
    has_draw_command: bool,
    render_area: Rect2D,
}
impl<'g> CommandBuffer<'g> {
    pub fn new(gpu: &'g Gpu, info: CommandBufferSubmitInfo) -> VkResult<Self> {
        let device = gpu.vk_logical_device();
        let inner_command_buffer = unsafe {
            device.allocate_command_buffers(&CommandBufferAllocateInfo {
                s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                command_pool: match info.target_queue {
                    QueueType::Graphics => gpu.thread_local_state.graphics_command_pool,
                    QueueType::AsyncCompute => gpu.thread_local_state.compute_command_pool,
                    QueueType::Transfer => gpu.thread_local_state.transfer_command_pool,
                },
                level: CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
            })
        }?[0];

        unsafe {
            device.begin_command_buffer(
                inner_command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )
        }?;

        Ok(Self {
            gpu,
            inner_command_buffer,
            has_recorded_anything: false,
            info,
        })
    }
    pub fn begin_render_pass<'p>(
        &'p mut self,
        info: &BeginRenderPassInfo<'p>,
    ) -> RenderPassCommand<'p, 'g> {
        RenderPassCommand::<'p, 'g>::new(self, &info)
    }

    pub fn inner(&self) -> vk::CommandBuffer {
        self.inner_command_buffer.clone()
    }
}

#[derive(Clone, Copy)]
pub struct BeginRenderPassInfo<'a> {
    pub framebuffer: vk::Framebuffer,
    pub render_pass: &'a RenderPass,
    pub clear_color_values: &'a [ClearValue],
    pub render_area: Rect2D,
}

impl<'c, 'g> RenderPassCommand<'c, 'g> {
    fn new(command_buffer: &'c mut CommandBuffer<'g>, info: &BeginRenderPassInfo<'c>) -> Self {
        let device = command_buffer.gpu.vk_logical_device();
        let create_info = RenderPassBeginInfo {
            s_type: StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: std::ptr::null(),
            render_pass: info.render_pass.inner,
            framebuffer: info.framebuffer,
            render_area: info.render_area,
            clear_value_count: info.clear_color_values.len() as _,
            p_clear_values: info.clear_color_values.as_ptr(),
        };
        unsafe {
            device.cmd_begin_render_pass(
                command_buffer.inner_command_buffer,
                &create_info,
                SubpassContents::INLINE,
            );
            let viewport = Viewport {
                x: 0 as f32,
                y: 0 as f32,
                width: info.render_area.extent.width as f32,
                height: info.render_area.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            let scissor = Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: info.render_area.extent,
            };
            device.cmd_set_viewport(command_buffer.inner(), 0, &[viewport]);
            device.cmd_set_scissor(command_buffer.inner(), 0, &[scissor]);
        };

        Self {
            command_buffer,
            has_draw_command: false,
            viewport_area: None,
            scissor_area: None,
            render_area: info.render_area,
        }
    }

    pub fn bind_material(&mut self, material: &Material) {
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_bind_pipeline(
                self.command_buffer.inner_command_buffer,
                PipelineBindPoint::GRAPHICS,
                material.pipeline,
            )
        }
    }

    pub(crate) fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.has_draw_command = true;
        self.command_buffer.has_recorded_anything = true;
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_draw_indexed(
                self.command_buffer.inner(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
}

impl<'c, 'g> Drop for RenderPassCommand<'c, 'g> {
    fn drop(&mut self) {
        let device = &self.command_buffer.gpu.vk_logical_device();
        unsafe { device.cmd_end_render_pass(self.command_buffer.inner_command_buffer) };
    }
}

impl<'g> Drop for CommandBuffer<'g> {
    fn drop(&mut self) {
        if !self.has_recorded_anything {
            return;
        }

        trace!("Submitting command buffer");
        let device = self.gpu.vk_logical_device();
        unsafe {
            device
                .end_command_buffer(self.inner())
                .expect("Failed to end inner command buffer");
            let target_queue = match self.info.target_queue {
                QueueType::Graphics => self.gpu.state.graphics_queue.clone(),
                QueueType::AsyncCompute => self.gpu.state.async_compute_queue.clone(),
                QueueType::Transfer => self.gpu.state.transfer_queue.clone(),
            };

            device
                .queue_submit(
                    target_queue,
                    &[SubmitInfo {
                        s_type: StructureType::SUBMIT_INFO,
                        p_next: std::ptr::null(),
                        wait_semaphore_count: self.info.wait_semaphores.len() as _,
                        p_wait_semaphores: self.info.wait_semaphores.as_ptr(),
                        p_wait_dst_stage_mask: self.info.wait_stages.as_ptr(),
                        command_buffer_count: 1,
                        p_command_buffers: [self.inner_command_buffer].as_ptr(),
                        signal_semaphore_count: self.info.signal_semaphores.len() as _,
                        p_signal_semaphores: self.info.signal_semaphores.as_ptr(),
                    }],
                    if let Some(fence) = self.info.fence {
                        fence
                    } else {
                        vk::Fence::null()
                    },
                )
                .unwrap();
        }
    }
}
