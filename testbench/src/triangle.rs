mod app;
mod utils;

use app::{bootstrap, App};

use engine::Backbuffer;
use engine_macros::glsl;
use gpu::{
    AccessFlags, BufferCreateInfo, BufferHandle, BufferUsageFlags, ColorAttachment, CullMode, Gpu,
    ImageAspectFlags, ImageFormat, ImageMemoryBarrier, IndexType, InputRate, MemoryDomain,
    PipelineStageFlags, PresentMode, ShaderModuleHandle, ShaderStage, VertexBindingInfo,
    VkCommandBuffer,
};
use imgui::Ui;
use nalgebra::*;
use winit::event_loop::EventLoop;

const VERTEX_SHADER: &[u32] = glsl!(
    source = "
#version 460

layout(location = 0) in vec3 in_position;
layout(push_constant) uniform constants {
    mat4 render_matrix;
};

void main() {
    gl_Position = vec4(in_position, 1.0);
}
    ",
    kind = vertex,
    entry_point = "main"
);

const FRAGMENT_SHADER: &[u32] = glsl!(
    source = "
#version 460

layout(location = 0) out vec4 color;

void main() {
    color = vec4(0.33, 0.45, 0.41, 1.0);
}
    ",
    kind = fragment,
    entry_point = "main"
);

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
        let vertex_module = app_state
            .gpu
            .make_shader_module(&gpu::ShaderModuleCreateInfo {
                code: bytemuck::cast_slice(VERTEX_SHADER),
            })?;
        let fragment_module = app_state
            .gpu
            .make_shader_module(&gpu::ShaderModuleCreateInfo {
                code: bytemuck::cast_slice(FRAGMENT_SHADER),
            })?;

        let vertices = [0.5f32, -0.5, 0.0, 0.0, 0.5, 0.0, -0.5, -0.5, 0.0];

        let indices = [0u32, 1, 2];

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
            .write_buffer(index_buffer, 0, bytemuck::cast_slice(&indices))?;

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

            let model = Matrix4::<f32>::identity().data.0;

            pass.set_vertex_shader(self.vertex_module);
            pass.set_fragment_shader(self.fragment_module);
            pass.set_vertex_buffers(&[VertexBindingInfo {
                handle: self.triangle_buffer,
                location: 0,
                offset: 0,
                stride: std::mem::size_of::<[f32; 3]>() as _,
                format: ImageFormat::RgbFloat32,
                input_rate: InputRate::PerVertex,
            }]);
            pass.set_index_buffer(self.index_buffer, IndexType::Uint32, 0);
            pass.set_cull_mode(CullMode::None);
            pass.push_constants(0, bytemuck::cast_slice(&[model]), ShaderStage::VERTEX);
            pass.draw_indexed_handle(3, 1, 0, 0, 0)?;
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
