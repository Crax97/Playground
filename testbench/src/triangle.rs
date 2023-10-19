mod app;
mod utils;

use app::{bootstrap, App};

use engine::{Backbuffer, Camera};
use gpu::{PresentMode, VkCommandBuffer, BufferHandle};
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
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/vertex_simple.spirv")?;
        let fragment_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/fragment.spirv")?;

        let vertices = vec![
            vector![-0.5, -0.5, 0.0],
            vector![0.5, -0.5, 0.0],
            vector![0.5, 0.5, 0.0],
        ];

        engine::app_state_mut()
            .swapchain_mut()
            .select_present_mode(PresentMode::Mailbox)?;

        Ok(Self {
            triangle_buffer: todo!(),
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
