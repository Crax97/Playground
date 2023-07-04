mod app;
mod gltf_loader;
mod utils;

use app::{bootstrap, App};
use ash::vk::PresentModeKHR;

use crate::gltf_loader::{GltfLoadOptions, GltfLoader};
use engine::{
    AppState, Camera, DeferredRenderingPipeline, Light, LightType, RenderingPipeline, Scene,
};
use nalgebra::*;
use resource_map::ResourceMap;
use winit::event::ElementState;

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}
const SPEED: f32 = 0.01;
const ROTATION_SPEED: f32 = 3.0;
const MIN_DELTA: f32 = 1.0;

pub struct GLTFViewer {
    resource_map: ResourceMap,
    camera: Camera,
    forward_movement: f32,
    rotation_movement: f32,
    rot_x: f32,
    rot_y: f32,
    dist: f32,
    movement: Vector3<f32>,
    scene_renderer: DeferredRenderingPipeline,
    gltf_loader: GltfLoader,
}

impl App for GLTFViewer {
    fn window_name(&self, app_state: &AppState) -> String {
        format!(
            "GLTF Viewer - FPS {}",
            1.0 / app_state.time().delta_frame().max(0.0000001)
        )
    }

    fn create(app_state: &AppState) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut resource_map = ResourceMap::new();

        let camera = Camera {
            location: point![2.0, 2.0, 2.0],
            forward: vector![0.0, -1.0, -1.0].normalize(),
            near: 0.01,
            ..Default::default()
        };

        let forward_movement = 0.0;
        let rotation_movement = 0.0;

        let rot_x = 0.0;
        let rot_z = 0.0;
        let dist = 1.0;

        let movement: Vector3<f32> = vector![0.0, 0.0, 0.0];

        let screen_quad_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/screen_quad.spirv")?;
        let gbuffer_combine_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/gbuffer_combine.spirv")?;
        let texture_copy_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/texture_copy.spirv")?;

        let mut scene_renderer = DeferredRenderingPipeline::new(
            &app_state.gpu,
            screen_quad_module,
            gbuffer_combine_module,
            texture_copy_module,
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
            camera,
            forward_movement,
            rotation_movement,
            rot_x,
            rot_y: rot_z,
            dist,
            movement,
            scene_renderer,
            gltf_loader,
        })
    }

    fn input(
        &mut self,
        _app_state: &AppState,
        event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        match event {
            winit::event::DeviceEvent::Button { button, state } => {
                let mul = if state == ElementState::Pressed {
                    1.0
                } else {
                    0.0
                };
                if button == 1 {
                    self.rotation_movement = mul;
                } else if button == 3 {
                    self.forward_movement = mul;
                }
            }

            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.movement.x = (delta.0.abs() as f32 - MIN_DELTA).max(0.0)
                    * delta.0.signum() as f32
                    * ROTATION_SPEED;
                self.movement.y = (delta.1.abs() as f32 - MIN_DELTA).max(0.0)
                    * delta.1.signum() as f32
                    * ROTATION_SPEED;
            }
            _ => {}
        };
        Ok(())
    }

    fn update(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        if self.rotation_movement > 0.0 {
            self.rot_y += self.movement.x;
            self.rot_x += -self.movement.y;
            self.rot_x = self.rot_x.clamp(-89.0, 89.0);
        } else {
            self.dist += self.movement.y * self.forward_movement * SPEED;
        }

        let rotation = Rotation::from_euler_angles(0.0, self.rot_y.to_radians(), 0.0);
        let rotation = rotation * Rotation::from_euler_angles(0.0, 0.0, self.rot_x.to_radians());
        let new_forward = rotation.to_homogeneous();
        let new_forward = new_forward.column(0);

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        let new_position = direction * self.dist;
        let new_position = point![new_position.x, new_position.y, new_position.z];
        self.camera.location = new_position;

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        self.camera.forward = -direction;
        Ok(())
    }

    fn draw(&mut self, app_state: &mut AppState) -> anyhow::Result<()> {
        self.scene_renderer
            .render(
                &self.camera,
                self.gltf_loader.scene(),
                app_state.gpu.swapchain_mut(),
                &self.resource_map,
            )
            .unwrap();

        Ok(())
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
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
