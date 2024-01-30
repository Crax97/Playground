mod utils;

use std::borrow::Cow;
use std::io::BufReader;

use engine::app::{app_state::*, bootstrap, App};
use engine::{Backbuffer, Time};
use engine_macros::glsl;
use gpu::{
    AttachmentReference, Binding, Binding2, BufferCreateInfo, BufferHandle, BufferUsageFlags,
    ColorAttachment, CommandBuffer, CullMode, FramebufferColorAttachment, ImageAspectFlags,
    ImageCreateInfo, ImageFormat, ImageHandle, ImageLayout, ImageUsageFlags, ImageViewCreateInfo,
    ImageViewHandle, IndexType, InputRate, MemoryDomain, PresentMode, SamplerCreateInfo,
    SamplerHandle, ShaderModuleHandle, ShaderStage, SubpassDescription, VertexBindingInfo,
};
use nalgebra::*;
use winit::event_loop::EventLoop;

const VERTEX_SHADER: &[u32] = glsl!(
    source = "
#version 460

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(push_constant) uniform constants {
    mat4 render_matrix;
};

layout(location = 0) out vec2 uv;

void main() {
    gl_Position = render_matrix * vec4(in_position, 1.0);
    uv = in_uv;
}
    ",
    kind = vertex,
    entry_point = "main"
);

const FRAGMENT_SHADER: &[u32] = glsl!(
    source = "
#version 460

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) out vec4 color;

layout(location = 0) in vec2 uv;

void main() {
    color = texture(tex, uv);
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
    time: Time,
    triangle_buffer: BufferHandle,
    uv_buffer: BufferHandle,
    index_buffer: BufferHandle,

    david_image: ImageHandle,
    david_image_view: ImageViewHandle,

    david_sampler: SamplerHandle,

    fragment_module: ShaderModuleHandle,
    vertex_module: ShaderModuleHandle,

    y_rotation: f32,
}

impl App for TriangleApp {
    fn window_name(&self, _app_state: &AppState) -> Cow<str> {
        Cow::Borrowed("triangle")
    }

    fn create(app_state: &mut AppState, _: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        app_state
            .swapchain_mut()
            .select_present_mode(PresentMode::Mailbox)?;

        let vertex_module = app_state
            .gpu
            .make_shader_module(&gpu::ShaderModuleCreateInfo {
                label: Some("Triangle vertex shader"),
                code: bytemuck::cast_slice(VERTEX_SHADER),
            })?;
        let fragment_module = app_state
            .gpu
            .make_shader_module(&gpu::ShaderModuleCreateInfo {
                label: Some("Triangle fragment shader"),
                code: bytemuck::cast_slice(FRAGMENT_SHADER),
            })?;

        let vertices = [0.5f32, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0];

        let uvs = [0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let indices = [0u32, 1, 2];

        let david_image = image::load(
            BufReader::new(std::fs::File::open("images/texture.jpg")?),
            image::ImageFormat::Jpeg,
        )?;
        let david_image_data = david_image.into_rgba8();

        let david_image = app_state.gpu.make_image(
            &ImageCreateInfo {
                label: Some("David image"),
                width: david_image_data.width(),
                height: david_image_data.height(),
                depth: 1,
                mips: 1,
                layers: 1,
                samples: gpu::SampleCount::Sample1,
                format: ImageFormat::Rgba8,
                usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
            },
            MemoryDomain::DeviceLocal,
            None,
        )?;

        let david_image_view = app_state.gpu.make_image_view(&ImageViewCreateInfo {
            label: Some("David image view"),
            image: david_image,
            view_type: gpu::ImageViewType::Type2D,
            format: ImageFormat::Rgba8,
            components: gpu::ComponentMapping::default(),
            subresource_range: gpu::ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        })?;

        let david_sampler = app_state.gpu.make_sampler(&SamplerCreateInfo {
            mag_filter: gpu::Filter::Linear,
            min_filter: gpu::Filter::Linear,
            address_u: gpu::SamplerAddressMode::ClampToEdge,
            address_v: gpu::SamplerAddressMode::ClampToEdge,
            address_w: gpu::SamplerAddressMode::ClampToEdge,
            mip_lod_bias: 0.0,
            compare_function: None,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: [1.0; 4],
        })?;

        app_state.gpu.write_image(
            &david_image,
            &david_image_data,
            gpu::Rect2D {
                offset: gpu::Offset2D::default(),
                extent: gpu::Extent2D {
                    width: david_image_data.width() as _,
                    height: david_image_data.height() as _,
                },
            },
            0,
        )?;

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
            .write_buffer(&triangle_buffer, 0, bytemuck::cast_slice(&vertices))?;

        let uv_buffer = app_state.gpu.make_buffer(
            &BufferCreateInfo {
                label: Some("Uv buffer"),
                size: std::mem::size_of_val(&uvs) as _,
                usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            },
            MemoryDomain::DeviceLocal,
        )?;
        app_state
            .gpu
            .write_buffer(&uv_buffer, 0, bytemuck::cast_slice(&uvs))?;

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
            .write_buffer(&index_buffer, 0, bytemuck::cast_slice(&indices))?;

        // app_state
        //     .swapchain_mut()
        //     .select_present_mode(PresentMode::Mailbox)?;

        Ok(Self {
            vertex_module,
            uv_buffer,
            fragment_module,
            triangle_buffer,
            index_buffer,

            david_image,
            david_image_view,
            david_sampler,

            y_rotation: 0.0,
            time: Time::new(),
        })
    }

    fn draw<'a>(
        &'a mut self,
        app_state: &'a mut AppState,
        backbuffer: &Backbuffer,
    ) -> anyhow::Result<CommandBuffer> {
        let gpu = &app_state.gpu;
        let mut command_buffer = gpu.start_command_buffer(gpu::QueueType::Graphics)?;
        {
            let mut pass = command_buffer.start_render_pass_2(&gpu::BeginRenderPassInfo2 {
                label: Some("Triangle rendering"),
                color_attachments: &[ColorAttachment {
                    image_view: backbuffer.image_view,
                    load_op: gpu::ColorLoadOp::Clear([0.3, 0.0, 0.3, 1.0]),
                    store_op: gpu::AttachmentStoreOp::Store,
                }],
                depth_attachment: None,
                stencil_attachment: None,
                render_area: backbuffer.whole_area(),
            });

            let projection = Matrix4::<f32>::new_perspective(
                backbuffer.size.width as f32 / backbuffer.size.height as f32,
                90.0f32.to_radians(),
                0.001,
                1000.0,
            );
            let view = Matrix4::look_at_rh(
                &point![0.0, 0.0, 0.0],
                &point![0.0, 0.0, 1.0],
                &vector![0.0, 1.0, 0.0],
            );
            let model = Matrix4::<f32>::new_translation(&vector![0.0, -0.4, 1.0])
                * Matrix4::new_rotation(vector![0.0, self.y_rotation, 0.0]);
            let mvp = projection * view * model;

            pass.set_vertex_shader(self.vertex_module);
            pass.set_fragment_shader(self.fragment_module);
            pass.set_vertex_buffers(&[
                VertexBindingInfo {
                    handle: self.triangle_buffer,
                    location: 0,
                    offset: 0,
                    stride: std::mem::size_of::<[f32; 3]>() as _,
                    format: ImageFormat::RgbFloat32,
                    input_rate: InputRate::PerVertex,
                },
                VertexBindingInfo {
                    handle: self.uv_buffer,
                    location: 1,
                    offset: 0,
                    stride: std::mem::size_of::<[f32; 2]>() as _,
                    format: ImageFormat::RgFloat32,
                    input_rate: InputRate::PerVertex,
                },
            ]);
            pass.set_index_buffer(self.index_buffer, IndexType::Uint32, 0);
            pass.bind_resources_2(
                0,
                &[Binding2::image_view(
                    self.david_image_view,
                    self.david_sampler,
                )],
            );
            pass.set_cull_mode(CullMode::None);
            pass.push_constants(
                0,
                0,
                bytemuck::cast_slice(mvp.as_slice()),
                ShaderStage::VERTEX,
            );
            pass.draw_indexed(3, 1, 0, 0, 0)?;
        }
        Ok(command_buffer)
    }

    fn input(
        &mut self,
        _app_state: &AppState,
        _event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn begin_frame(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        self.time.begin_frame();
        Ok(())
    }
    fn update(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        self.y_rotation += self.time.delta_frame() * 6.0;
        Ok(())
    }

    fn end_frame(&mut self, _app_state: &AppState) {
        self.time.end_frame();
    }

    fn on_shutdown(&mut self, app_state: &mut AppState) {
        let gpu = app_state.gpu();
        gpu.destroy_buffer(self.triangle_buffer);
        gpu.destroy_buffer(self.uv_buffer);
        gpu.destroy_buffer(self.index_buffer);
        gpu.destroy_image_view(self.david_image_view);
        gpu.destroy_image(self.david_image);
        gpu.destroy_sampler(self.david_sampler);
        gpu.destroy_shader_module(self.fragment_module);
        gpu.destroy_shader_module(self.vertex_module);
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<TriangleApp>()
}
