mod app;
mod fps_camera;
mod gltf_loader;
mod input;
mod utils;

use app::{bootstrap, App};
use ash::vk::PresentModeKHR;

use fps_camera::FpsCamera;
use gpu::CommandBuffer;
use imgui::Ui;
use input::InputState;
use winit::dpi::{PhysicalPosition, Position};

use crate::gltf_loader::{GltfLoadOptions, GltfLoader};
use engine::{
    AppState, Backbuffer, DeferredRenderingPipeline, Light, LightType, RenderingPipeline, Scene,
};
use nalgebra::*;
use resource_map::ResourceMap;
use winit::event::MouseButton;
use winit::event_loop::EventLoop;

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}

pub struct GLTFViewer {
    resource_map: ResourceMap,
    camera: FpsCamera,
    scene_renderer: DeferredRenderingPipeline,
    gltf_loader: GltfLoader,
    input: InputState,
}

impl App for GLTFViewer {
    fn window_name(&self, app_state: &AppState) -> String {
        format!("GLTF Viewer - FPS {}", 1.0 / app_state.time().delta_frame())
    }

    fn create(app_state: &AppState, _event_loop: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut resource_map = ResourceMap::new();

        let screen_quad_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/screen_quad.spirv")?;
        let gbuffer_combine_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/gbuffer_combine.spirv")?;
        let texture_copy_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/texture_copy.spirv")?;
        let tonemap_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/tonemap.spirv")?;

        let mut scene_renderer = DeferredRenderingPipeline::new(
            &app_state.gpu,
            screen_quad_module,
            gbuffer_combine_module,
            texture_copy_module,
            tonemap_module,
        )?;

        let mut gltf_loader = GltfLoader::load(
            "gltf_models/Sponza/glTF/Sponza.gltf",
            &app_state.gpu,
            &mut scene_renderer,
            &mut resource_map,
            GltfLoadOptions {},
        )?;

        add_scene_lights(gltf_loader.scene_mut());

        engine::app_state_mut()
            .gpu
            .swapchain_mut()
            .select_present_mode(PresentModeKHR::IMMEDIATE)?;

        Ok(Self {
            resource_map,
            scene_renderer,
            gltf_loader,
            input: InputState::new(),
            camera: FpsCamera::default(),
        })
    }

    fn on_event(
        &mut self,
        event: &winit::event::Event<()>,
        _app_state: &AppState,
    ) -> anyhow::Result<()> {
        self.input.update(&event);
        Ok(())
    }

    fn input(
        &mut self,
        _app_state: &AppState,
        _event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, app_state: &mut AppState, ui: &mut Ui) -> anyhow::Result<()> {
        let mut settings = self.scene_renderer.fxaa_settings();

        ui.text("Hiii");

        ui.slider("FXAA iterations", 0, 12, &mut settings.iterations);
        ui.slider("FXAA subpix", 0.0, 1.0, &mut settings.fxaa_quality_subpix);
        ui.slider(
            "FXAA Edge Threshold",
            0.0,
            1.0,
            &mut settings.fxaa_quality_edge_threshold,
        );
        ui.slider(
            "FXAA Edge Threshold min",
            0.0,
            1.0,
            &mut settings.fxaa_quality_edge_threshold_min,
        );
        self.scene_renderer.set_fxaa_settings_mut(settings);

        if ui.io().want_capture_keyboard || ui.io().want_capture_mouse {
            return Ok(());
        }

        if self.input.is_mouse_button_just_pressed(MouseButton::Right) {
            app_state
                .gpu
                .swapchain()
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)?;
            app_state.gpu.swapchain().window.set_cursor_visible(false);
        }
        if self.input.is_mouse_button_just_released(MouseButton::Right) {
            app_state
                .gpu
                .swapchain()
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::None)?;
            app_state.gpu.swapchain().window.set_cursor_visible(true);
        }

        if self
            .input
            .is_mouse_button_pressed(winit::event::MouseButton::Right)
        {
            self.camera
                .update(&self.input, app_state.time.delta_frame());
            let window_size = app_state.gpu.swapchain().window.inner_size();
            app_state
                .gpu
                .swapchain()
                .window
                .set_cursor_position(Position::Physical(PhysicalPosition {
                    x: window_size.width as i32 / 2,
                    y: window_size.height as i32 / 2,
                }))?;
        }
        self.input.end_frame();
        Ok(())
    }

    fn draw(&mut self, backbuffer: &Backbuffer) -> anyhow::Result<CommandBuffer> {
        let command_buffer = self.scene_renderer.render(
            &self.camera.camera(),
            self.gltf_loader.scene(),
            backbuffer,
            &self.resource_map,
        )?;
        Ok(command_buffer)
    }
}

fn add_scene_lights(scene: &mut Scene) {
    scene.add_light(Light {
        ty: LightType::Point,
        position: vector![0.0, 10.0, 0.0],
        radius: 50.0,
        color: vector![1.0, 0.0, 0.0],
        intensity: 1.0,
        enabled: true,
    });
    scene.add_light(Light {
        ty: LightType::Directional {
            direction: vector![-0.45, -0.45, 0.0],
        },
        position: vector![100.0, 100.0, 0.0],
        radius: 10.0,
        color: vector![1.0, 1.0, 1.0],
        intensity: 1.0,
        enabled: true,
    });
    scene.add_light(Light {
        ty: LightType::Spotlight {
            direction: vector![0.45, -0.45, 0.0],
            inner_cone_degrees: 15.0,
            outer_cone_degrees: 35.0,
        },
        position: vector![100.0, 100.0, 0.0],
        radius: 10.0,
        color: vector![1.0, 1.0, 1.0],
        intensity: 1.0,
        enabled: true,
    });
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
