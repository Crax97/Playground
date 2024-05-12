mod gltf_to_scene_loader;

use clap::Parser;
use engine::glam::{vec3, EulerRot};
use engine::glam::{Quat, Vec3};
use engine::winit::dpi::PhysicalPosition;
use engine::winit::event::{DeviceEvent, WindowEvent};
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

#[derive(Parser, Debug)]
#[command(version, about)]
struct GltfViewerArgs {
    #[arg(long)]
    file: String,
}

pub struct GltfViewerApplication {
    asset_map: AssetMap,
    scene: Scene,
    scene_renderer: SceneRenderer,
    pov: PointOfView,
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
        pov.transform.location = vec3(0.0, 10.0, -5.0);

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

            pov,
        })
    }

    fn handle_window_event(&mut self, _event: &WindowEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn handle_device_event(&mut self, _event: &DeviceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        update_fps_camera(context, &mut self.pov);

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

pub fn update_fps_camera(context: &AppContext, pov: &mut PointOfView) {
    const MOVEMENT_SPEED: f64 = 100.0;
    const ROTATION_DEGREES: f64 = 90.0;

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
    let pov_transform = pov.transform;

    let (cam_pitch, cam_roll, _) = pov_transform.rotation.to_euler(EulerRot::XYZ);
    let mut cam_roll = cam_roll.to_degrees();
    let mut cam_pitch = cam_pitch.to_degrees();

    cam_roll -= mouse_delta.x * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
    cam_pitch += mouse_delta.y * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
    cam_pitch = cam_pitch.clamp(-89.0, 89.0);

    let new_location_offset = camera_input.x * pov.transform.left()
        + camera_input.y * pov.transform.up()
        + camera_input.z * pov.transform.forward();
    pov.transform.location += new_location_offset;
    pov.transform.rotation = Quat::from_euler(
        EulerRot::XYZ,
        cam_pitch.to_radians(),
        cam_roll.to_radians(),
        0.0,
    );

    let cursor_position = context.window().inner_size();
    context
        .window()
        .set_cursor_position(PhysicalPosition::new(
            cursor_position.width / 2,
            cursor_position.height / 2,
        ))
        .unwrap();
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
