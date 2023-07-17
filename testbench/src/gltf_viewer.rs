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
    AppState, Backbuffer, DeferredRenderingPipeline, Light, LightHandle, LightType,
    RenderingPipeline, Scene, ShadowSetup,
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
    camera_light: LightHandle,
}
impl GLTFViewer {
    fn lights_ui(&mut self, ui: &mut Ui) {
        ui.separator();
        let group = ui.begin_group();
        self.gltf_loader
            .scene()
            .all_lights()
            .iter()
            .enumerate()
            .for_each(|(i, l)| {
                let light_string = match l.ty {
                    LightType::Point => "Point",
                    LightType::Directional { .. } => "Directional",
                    LightType::Spotlight { .. } => "Spotlight",
                    LightType::Rect { .. } => "Rect",
                };

                let handle = LightHandle(i);

                if ui
                    .selectable_config(&format!("{light_string} light nr. #{i}"))
                    .selected(self.camera_light == handle)
                    .build()
                {
                    self.camera_light = handle;
                }
            });

        let l = self.gltf_loader.scene_mut().edit_light(&self.camera_light);
        ui.indent();
        ui.checkbox("Enabled", &mut l.enabled);
        ui.input_float3("Position", &mut l.position.coords.data.0[0])
            .build();
        ui.input_float3("Color", &mut l.color.data.0[0]).build();
        ui.slider("Intensity", 0.0, 1000.0, &mut l.intensity);
        ui.slider("Radius", 0.0, 1000.0, &mut l.radius);
        match &mut l.ty {
            LightType::Point => {}
            LightType::Directional { direction, size } => {
                ui.input_float3("Direction", &mut direction.data.0[0])
                    .build();
                ui.input_float2("Shadow size", &mut size.data.0[0]).build();
            }
            LightType::Spotlight {
                direction,
                inner_cone_degrees,
                outer_cone_degrees,
            } => {
                ui.input_float3("Direction", &mut direction.data.0[0])
                    .build();
                ui.slider("Outer cone", *inner_cone_degrees, 90.0, outer_cone_degrees);
                ui.slider("Inner cone", 0.0, *outer_cone_degrees, inner_cone_degrees);
            }
            LightType::Rect {
                direction,
                width,
                height,
            } => {
                ui.input_float3("Direction", &mut direction.data.0[0])
                    .build();
                ui.slider("Width", 0.0, 100000.0, width);
                ui.slider("Height", 0.0, 100000.0, height);
            }
        }
        ui.unindent();
        ui.separator();

        group.end();
    }
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

        let camera_light = add_scene_lights(gltf_loader.scene_mut());

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
            camera_light,
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

        self.lights_ui(ui);

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

        if self.input.is_mouse_button_pressed(MouseButton::Left) {
            self.gltf_loader
                .scene_mut()
                .edit_light(&self.camera_light)
                .position = self.camera.location;
            self.gltf_loader
                .scene_mut()
                .edit_light(&self.camera_light)
                .set_direction(self.camera.forward());
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

fn add_scene_lights(scene: &mut Scene) -> LightHandle {
    // scene.add_light(Light {
    //     ty: LightType::Point,
    //     position: point![0.0, 10.0, 0.0],
    //     radius: 50.0,
    //     color: vector![1.0, 0.0, 0.0],
    //     intensity: 1.0,
    //     enabled: true,
    // });
    scene.add_light(Light {
        ty: LightType::Directional {
            direction: vector![-0.45, -0.45, 0.0],
            size: vector![50.0, 50.0],
        },
        position: point![100.0, 100.0, 0.0],
        radius: 100.0,
        color: vector![1.0, 1.0, 1.0],
        intensity: 100.0,
        enabled: true,
        shadow_setup: Some(ShadowSetup {
            width: 2048,
            height: 2048,
        }),
    });
    scene.add_light(Light {
        ty: LightType::Spotlight {
            direction: vector![0.45, -0.45, 0.0],
            inner_cone_degrees: 15.0,
            outer_cone_degrees: 35.0,
        },
        position: point![100.0, 100.0, 0.0],
        radius: 100.0,
        color: vector![1.0, 1.0, 1.0],
        intensity: 10.0,
        enabled: true,
        shadow_setup: Some(ShadowSetup {
            width: 512,
            height: 512,
        }),
    })
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
