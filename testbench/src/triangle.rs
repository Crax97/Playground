mod app;
mod utils;

use app::{bootstrap, App};

use engine::Backbuffer;
use gpu::{
    AccessFlags, BufferCreateInfo, BufferHandle, BufferUsageFlags, ColorAttachment, Gpu,
    ImageAspectFlags, ImageMemoryBarrier, MemoryDomain, PipelineStageFlags, PresentMode,
    ShaderModuleHandle, VkCommandBuffer,
};
use imgui::Ui;
use nalgebra::*;
use winit::event_loop::EventLoop;

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}

pub struct TriangleApp {
    triangle_buffer: BufferHandle,
    index_buffer: BufferHandle,

    fragment_module: ShaderModuleHandle,
    vertex_module: ShaderModuleHandle,
}

impl App for TriangleApp {
    fn window_name(&self, _app_state: &engine::AppState) -> String {
        "planes".to_owned()
    }

    fn create(app_state: &engine::AppState, _: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let vertex_module =
            utils::read_file_to_shader_module(&app_state.gpu, "./shaders/vertex_simple.spirv")?;
        let fragment_module =
            utils::read_file_to_shader_module(&app_state.gpu, "./shaders/fragment.spirv")?;

        let vertices = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0];

        let indices: [u32; 3] = [0, 1, 2];

        let triangle_buffer = app_state.gpu.make_buffer(
            &BufferCreateInfo {
                label: Some("Triangle buffer"),
                size: std::mem::size_of_val(&vertices) as _,
                usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            },
            MemoryDomain::DeviceLocal,
        )?;
        app_state
            .gpu
            .write_buffer(triangle_buffer, 0, bytemuck::cast_slice(&vertices))?;

        let index_buffer = app_state.gpu.make_buffer(
            &BufferCreateInfo {
                label: Some("Triangle index buffer"),
                size: std::mem::size_of_val(&indices) as _,
                usage: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            },
            MemoryDomain::DeviceLocal,
        )?;
        app_state
            .gpu
            .write_buffer(triangle_buffer, 0, bytemuck::cast_slice(&indices))?;

        engine::app_state_mut()
            .swapchain_mut()
            .select_present_mode(PresentMode::Mailbox)?;

        Ok(Self {
            vertex_module,
            fragment_module,
            triangle_buffer,
            index_buffer,
        })
    }

    fn draw(&mut self, backbuffer: &Backbuffer) -> anyhow::Result<VkCommandBuffer> {
        let gpu = &engine::app_state().gpu;
        let mut command_buffer = gpu.create_command_buffer(gpu::QueueType::Graphics)?;

        command_buffer.pipeline_barrier(&gpu::PipelineBarrierInfo {
            src_stage_mask: PipelineStageFlags::TOP_OF_PIPE,
            dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            memory_barriers: &[],
            buffer_memory_barriers: &[],
            image_memory_barriers: &[ImageMemoryBarrier {
                src_access_mask: AccessFlags::empty(),
                dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                old_layout: gpu::ImageLayout::Undefined,
                new_layout: gpu::ImageLayout::ColorAttachment,
                src_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
                image: &backbuffer.image,
                subresource_range: gpu::ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            }],
        });
        {
            let color_attachments = vec![ColorAttachment {
                image_view: backbuffer.image_view,
                load_op: gpu::ColorLoadOp::Clear([0.3, 0.0, 0.3, 1.0]),
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: gpu::ImageLayout::ColorAttachment,
            }];

            let mut pass = command_buffer.begin_render_pass(&gpu::BeginRenderPassInfo {
                color_attachments: &color_attachments,
                depth_attachment: None,
                stencil_attachment: None,
                render_area: gpu::Rect2D {
                    offset: gpu::Offset2D::default(),
                    extent: backbuffer.size,
                },
            });

            pass.set_vertex_shader(self.vertex_module);
            pass.set_fragment_shader(self.fragment_module);
            // pass.set_vertex_buffers(&self, &[&VertexBindingInfo {
            //      handle: self.triangle_buffer,
            //      location: 0,
            //      offset: 0,
            //      stride: std::mem::size_of::<[f32; 3]>(),
            //      input_rate: InputRate::PerVertex,
            // });
            // pass.set_cull_mode(CullMode::None);
            // pass.draw_indexed(&self.index_buffer, IndexType::Uint32, 0, 3, 0);
        }
        Ok(command_buffer)
    }

    fn input(
        &mut self,
        _app_state: &engine::AppState,
        _event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, _app_state: &mut engine::AppState, _ui: &mut Ui) -> anyhow::Result<()> {
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<TriangleApp>()
}
