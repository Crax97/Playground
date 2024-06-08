mod gltf_to_scene_loader;

use clap::Parser;
use engine::editor::SceneEditor;
use engine::egui_mgpu::EguiMgpuIntegration;
use engine::glam::{vec3, EulerRot, Mat4, Vec4Swizzles};
use engine::glam::{Quat, Vec3};
use engine::math::constants;
use engine::scene_renderer::SceneOutput;
use engine::winit::event::{DeviceEvent, MouseButton, WindowEvent};
use engine::{app, include_spirv};
use engine::{app::AppContext, scene_renderer::PointOfView};
use engine::{
    app::{bootstrap, App, AppDescription},
    asset_map::AssetMap,
    sampler_allocator::SamplerAllocator,
    scene::Scene,
    scene_renderer::{ProjectionMode, SceneRenderer, SceneRenderingParams},
    shader_cache::ShaderCache,
};
use mgpu::{Extents2D, ShaderModuleDescription};

const MOVEMENT_SPEED: f64 = 50.0;
const ROTATION_DEGREES: f64 = 3200.0;
const VIEW_CUBEMAP_VERT: &[u8] = include_spirv!("../spirv/base_pass/view_cubemap.vert.spv");
const VIEW_CUBEMAP_FRAG: &[u8] = include_spirv!("../spirv/base_pass/view_cubemap.frag.spv");

const PBR_VERTEX: &[u8] = include_spirv!("../spirv/base_pass/pbr_vertex.vert.spv");
const PBR_FRAGMENT: &[u8] = include_spirv!("../spirv/base_pass/pbr_fragment.frag.spv");

#[derive(Clone, Copy)]
enum CameraMode {
    Orbit,
    Free,
}

pub struct GltfViewerApplication {
    asset_map: AssetMap,
    scene: Scene,
    scene_renderer: SceneRenderer,
    pov: PointOfView,
    output: SceneOutput,

    cam_roll: f32,
    cam_pitch: f32,
    orbit_distance: f32,
    camera_mode: CameraMode,
    is_mouse_captured: bool,
    scene_editor: SceneEditor,
    egui_integration: EguiMgpuIntegration,
}

impl App for GltfViewerApplication {
    fn create(context: &engine::app::AppContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let view_cubemap_frag = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("View cubemap"),
                source: bytemuck::cast_slice(VIEW_CUBEMAP_FRAG),
            })?;

        let view_cubemap_vert = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("View cubemap vertex"),
                source: bytemuck::cast_slice(VIEW_CUBEMAP_VERT),
            })?;

        let pbr_vertex = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("PBR Vertex Shader"),
                source: bytemuck::cast_slice(PBR_VERTEX),
            })
            .unwrap();
        let pbr_fragment = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("PBR Fragment Shader"),
                source: bytemuck::cast_slice(PBR_FRAGMENT),
            })
            .unwrap();

        let mut shader_cache = ShaderCache::new();
        shader_cache.add_shader("view_cubemap_vert", view_cubemap_vert);
        shader_cache.add_shader("view_cubemap_frag", view_cubemap_frag);
        shader_cache.add_shader("pbr_vertex", pbr_vertex);
        shader_cache.add_shader("pbr_fragment", pbr_fragment);
        let sampler_allocator = SamplerAllocator::default();

        let mut asset_map =
            app::asset_map_with_defaults(&context.device, &sampler_allocator, shader_cache)?;

        asset_map.discover_assets("assets");

        let mut pov = PointOfView::new_perspective(0.01, 1000.0, 75.0, 1920.0 / 1080.0);
        pov.transform.location = vec3(0.0, 0.0, -2.0);

        let scene = Scene::new();
        let scene_renderer = SceneRenderer::new(&context.device, &asset_map)?;
        Ok(Self {
            asset_map,
            scene,
            scene_renderer,

            cam_pitch: -15.0,
            cam_roll: 0.0,
            orbit_distance: 0.5,

            pov,
            output: SceneOutput::FinalImage,
            camera_mode: CameraMode::Orbit,
            is_mouse_captured: false,
            egui_integration: EguiMgpuIntegration::new(&context.device)?,
            scene_editor: SceneEditor::new(),
        })
    }

    fn handle_window_event(
        &mut self,
        event: &WindowEvent,
        context: &AppContext,
    ) -> anyhow::Result<()> {
        if let WindowEvent::Focused(false) = event {
            self.set_cursor_captured(false, context);
        }
        self.egui_integration
            .on_window_event(context.window(), event);
        Ok(())
    }

    fn handle_device_event(
        &mut self,
        _event: &DeviceEvent,
        _context: &AppContext,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        self.egui_integration.begin_frame(context.window());
        let egui_context = self.egui_integration.context();

        self.scene_editor.show(
            &context.device,
            &egui_context,
            &mut self.scene,
            &mut self.asset_map,
        );

        if context.input.is_key_just_pressed(engine::input::Key::F1) {
            self.output = SceneOutput::BaseColor;
        }
        if context.input.is_key_just_pressed(engine::input::Key::F2) {
            self.output = SceneOutput::Normal;
        }
        if context.input.is_key_just_pressed(engine::input::Key::F3) {
            self.output = SceneOutput::EmissiveAO;
        }
        if context.input.is_key_just_pressed(engine::input::Key::F3) {
            self.output = SceneOutput::WorldPosition;
        }
        if context.input.is_key_just_pressed(engine::input::Key::F9) {
            self.output = SceneOutput::FinalImage;
        }

        if context.input.is_key_just_pressed(engine::input::Key::O) {
            self.camera_mode = CameraMode::Orbit;
        }
        if context.input.is_key_just_pressed(engine::input::Key::P) {
            self.camera_mode = CameraMode::Free;
        }

        if context
            .input
            .is_key_just_pressed(engine::input::Key::Semicolon)
        {
            self.set_cursor_captured(!self.is_mouse_captured, context);
        }

        match self.camera_mode {
            CameraMode::Orbit => self.update_orbit_camera(context),
            CameraMode::Free => self.update_fps_camera(context),
        }

        self.scene_editor
            .update_pov(&self.pov, self.egui_integration.pixels_per_point());

        Ok(())
    }

    fn render(
        &mut self,
        context: &engine::app::AppContext,
        render_context: engine::app::RenderContext,
    ) -> anyhow::Result<()> {
        let output = self.egui_integration.end_frame();

        self.egui_integration
            .handle_platform_output(context.window(), output.platform_output);

        self.pov.projection_mode = ProjectionMode::Perspective {
            fov_y_radians: 75.0f32.to_radians(),
            aspect_ratio: render_context.swapchain_image.extents.width as f32
                / render_context.swapchain_image.extents.height as f32,
        };
        self.scene_renderer.render(SceneRenderingParams {
            device: &context.device,
            scene: &self.scene,
            pov: &self.pov,
            asset_map: &mut self.asset_map,
            output_image: render_context.swapchain_image.view,
            output: self.output,
        })?;
        self.egui_integration.paint_frame(
            &context.device,
            render_context.swapchain_image.view,
            output.textures_delta,
            output.shapes,
        )?;
        Ok(())
    }

    fn resized(
        &mut self,
        _context: &engine::app::AppContext,
        _new_extents: mgpu::Extents2D,
    ) -> mgpu::MgpuResult<()> {
        Ok(())
    }

    fn shutdown(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        self.scene.dispose(&mut self.asset_map);
        self.scene_renderer.release_resources(&context.device)?;
        self.egui_integration.destroy(&context.device)?;
        Ok(())
    }

    fn on_window_created(&mut self, context: &AppContext) -> anyhow::Result<()> {
        let window = context.window();
        window.set_cursor_visible(false);
        Ok(())
    }
}

impl GltfViewerApplication {
    fn update_fps_camera(&mut self, context: &AppContext) {
        let mut camera_input = Vec3::default();

        if self.is_mouse_captured {
            if context.input.is_key_pressed(engine::input::Key::A) {
                camera_input.x = 1.0;
            } else if context.input.is_key_pressed(engine::input::Key::D) {
                camera_input.x = -1.0;
            }

            if context.input.is_key_pressed(engine::input::Key::W) {
                camera_input.z = 1.0;
            } else if context.input.is_key_pressed(engine::input::Key::S) {
                camera_input.z = -1.0;
            }
            if context.input.is_key_pressed(engine::input::Key::Q) {
                camera_input.y = 1.0;
            } else if context.input.is_key_pressed(engine::input::Key::E) {
                camera_input.y = -1.0;
            }

            camera_input *= (MOVEMENT_SPEED * context.time.delta_seconds()) as f32;
        }
        let mouse_delta = context.input.mouse_delta();

        self.cam_roll += mouse_delta.x * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
        self.cam_pitch -= mouse_delta.y * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
        self.cam_pitch = self.cam_pitch.clamp(-89.0, 89.0);

        let new_location_offset = camera_input.x * self.pov.transform.left()
            + camera_input.y * self.pov.transform.up()
            + camera_input.z * self.pov.transform.forward();
        self.pov.transform.location += new_location_offset;
        self.pov.transform.rotation =
            Quat::from_euler(EulerRot::XYZ, self.cam_pitch.to_radians(), 0.0, 0.0)
                * Quat::from_euler(EulerRot::XYZ, 0.0, self.cam_roll.to_radians(), 0.0);

        self.orbit_distance = self.pov.transform.location.distance(Vec3::ZERO);
    }

    fn update_orbit_camera(&mut self, context: &AppContext) {
        if self.is_mouse_captured {
            let mouse_delta = context.input.mouse_delta();
            if context.input.is_mouse_button_pressed(MouseButton::Left) {
                self.cam_roll +=
                    mouse_delta.x * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
                self.cam_pitch +=
                    mouse_delta.y * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
                self.cam_pitch = self.cam_pitch.clamp(-89.0, 89.0);
            }
            if context.input.is_mouse_button_pressed(MouseButton::Right) {
                self.orbit_distance +=
                    -mouse_delta.y * (MOVEMENT_SPEED * context.time.delta_seconds()) as f32;
                self.orbit_distance = self.orbit_distance.max(0.1);
            }
        }

        let rotation = Quat::from_euler(
            EulerRot::XYZ,
            self.cam_pitch.to_radians(),
            self.cam_roll.to_radians(),
            0.0,
        );
        let forward = Mat4::from_quat(rotation).row(2).xyz();
        let location = -forward * self.orbit_distance;

        let rotation = Quat::from_mat4(&Mat4::look_at_rh(-location, Vec3::ZERO, constants::UP));
        self.pov.transform.rotation = rotation;
        self.pov.transform.location = location;
    }

    fn set_cursor_captured(&mut self, captured: bool, context: &AppContext) {
        self.is_mouse_captured = captured;
        context.window().set_cursor_visible(!captured);
        context
            .window()
            .set_cursor_grab(if captured {
                engine::winit::window::CursorGrabMode::Confined
            } else {
                engine::winit::window::CursorGrabMode::None
            })
            .unwrap();
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GltfViewerApplication>(AppDescription {
        window_size: Extents2D {
            width: 1920,
            height: 1080,
        },
        initial_title: Some("GLTF Viewer"),
        app_identifier: "GLTFViewerApp",
    })
}
