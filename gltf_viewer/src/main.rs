mod gltf_to_scene_loader;

use clap::Parser;
use engine::glam::{vec3, EulerRot, Mat4, Vec4Swizzles};
use engine::glam::{Quat, Vec3};
use engine::math::constants;
use engine::scene_renderer::SceneOutput;
use engine::winit::dpi::PhysicalPosition;
use engine::winit::event::{DeviceEvent, MouseButton, WindowEvent};
use engine::{app::AppContext, scene_renderer::PointOfView};
use engine::{
    app::{bootstrap, App, AppDescription},
    asset_map::AssetMap,
    sampler_allocator::SamplerAllocator,
    scene::Scene,
    scene_renderer::{ProjectionMode, SceneRenderer, SceneRenderingParams},
    shader_cache::ShaderCache,
};
use mgpu::Extents2D;

const MOVEMENT_SPEED: f64 = 50.0;
const ROTATION_DEGREES: f64 = 320.0;

#[derive(Parser, Debug)]
#[command(version, about)]
struct GltfViewerArgs {
    #[arg(long)]
    file: String,
}

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
}

impl App for GltfViewerApplication {
    fn create(context: &engine::app::AppContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let args = GltfViewerArgs::parse();

        let mut asset_map = AssetMap::new();
        let mut shader_cache = ShaderCache::new();
        let sampler_allocator = SamplerAllocator::default();
        let scene_renderer = SceneRenderer::new(&context.device)?;
        let mut pov = PointOfView::new_perspective(0.01, 1000.0, 75.0, 1920.0 / 1080.0);
        pov.transform.location = vec3(0.0, 0.0, -2.0);

        let scene = gltf_to_scene_loader::load(
            &context.device,
            &args.file,
            &sampler_allocator,
            &mut shader_cache,
            &mut asset_map,
        )?;

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
            is_mouse_captured: true,
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

        if !self.is_mouse_captured {
            return Ok(());
        }

        match self.camera_mode {
            CameraMode::Orbit => self.update_orbit_camera(context),
            CameraMode::Free => self.update_fps_camera(context),
        }

        Ok(())
    }

    fn render(
        &mut self,
        context: &engine::app::AppContext,
        render_context: engine::app::RenderContext,
    ) -> anyhow::Result<()> {
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
        Ok(())
    }

    fn resized(
        &mut self,
        _context: &engine::app::AppContext,
        _new_extents: mgpu::Extents2D,
    ) -> mgpu::MgpuResult<()> {
        Ok(())
    }

    fn shutdown(&mut self, _context: &engine::app::AppContext) -> anyhow::Result<()> {
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

        let mouse_delta = context.input.normalized_mouse_position();

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

        let cursor_position = context.window().inner_size();
        context
            .window()
            .set_cursor_position(PhysicalPosition::new(
                cursor_position.width / 2,
                cursor_position.height / 2,
            ))
            .unwrap();
    }

    fn update_orbit_camera(&mut self, context: &AppContext) {
        let mouse_delta = context.input.normalized_last_mouse_position();
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

        let cursor_position = context.window().inner_size();
        context
            .window()
            .set_cursor_position(PhysicalPosition::new(
                cursor_position.width / 2,
                cursor_position.height / 2,
            ))
            .unwrap();
    }

    fn set_cursor_captured(&mut self, captured: bool, context: &AppContext) {
        self.is_mouse_captured = captured;
        context.window().set_cursor_visible(!captured);
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GltfViewerApplication>(AppDescription {
        window_size: Extents2D {
            width: 1920,
            height: 1080,
        },
        initial_title: Some("Cube Scene"),
        app_identifier: "CubeSceneApp",
    })
}
