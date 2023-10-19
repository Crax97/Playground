mod app;
mod utils;

use app::{bootstrap, App};

use engine::{Backbuffer, Camera};
use gpu::{PresentMode, VkCommandBuffer, BufferHandle, Gpu, BufferCreateInfo, BufferUsageFlags, MemoryDomain, ShaderModuleHandle};
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

        let vertices = [
            -0.5, -0.5, 0.0,
            0.5, -0.5, 0.0,
            0.5, 0.5, 0.0,
        ];

        let indices: [u32; 3] = [0, 1, 2];

        let triangle_buffer = app_state.gpu.make_buffer(&BufferCreateInfo {
            label: Some("Triangle buffer"),
            size: std::mem::size_of_val(&vertices) as _,
            usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
        }, MemoryDomain::DeviceLocal)?;
        app_state.gpu.write_buffer(triangle_buffer, 0, bytemuck::cast_slice(&vertices))?;

        let index_buffer = app_state.gpu.make_buffer(&BufferCreateInfo {
            label: Some("Triangle index buffer"),
            size: std::mem::size_of_val(&indices) as _,
            usage: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
        }, MemoryDomain::DeviceLocal)?;
        app_state.gpu.write_buffer(triangle_buffer, 0, bytemuck::cast_slice(&indices))?;

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
        todo!()
    }

    fn input(
        &mut self,
        app_state: &engine::AppState,
        event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, app_state: &mut engine::AppState, ui: &mut Ui) -> anyhow::Result<()> {
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<TriangleApp>()
}
