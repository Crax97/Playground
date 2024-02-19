mod utils;

use std::borrow::Cow;

use engine::app::egui_support::EguiSupport;
use engine::app::{app_state::*, bootstrap, App};

use engine::loaders::FileSystemTextureLoader;
use engine::math::shape::BoundingShape;
use engine::{
    Backbuffer, Camera, CvarManager, DeferredRenderingPipeline, MaterialDescription,
    MaterialDomain, MaterialInstance, MaterialInstanceDescription, Mesh, MeshCreateInfo,
    MeshPrimitiveCreateInfo, RenderScene, RenderingPipeline, ResourceMap, ScenePrimitive, Texture,
    TextureInput, Time,
};
use gpu::{CommandBuffer, Offset2D, PresentMode, Rect2D, ShaderStage};
use nalgebra::*;
use winit::{event::ElementState, event_loop::EventLoop};
#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}
const SPEED: f32 = 0.1;
const ROTATION_SPEED: f32 = 3.0;
const MIN_DELTA: f32 = 1.0;
pub struct PlanesApp {
    camera: Camera,
    forward_movement: f32,
    rotation_movement: f32,
    rot_x: f32,
    rot_z: f32,
    dist: f32,
    movement: Vector3<f32>,
    scene_renderer: DeferredRenderingPipeline,
    scene: RenderScene,
    resource_map: ResourceMap,
    cvar_manager: CvarManager,
    egui_integration: EguiSupport,

    time: Time,
}

impl App for PlanesApp {
    fn window_name(&self, _app_state: &AppState) -> Cow<str> {
        Cow::Borrowed("planes")
    }

    fn create(app_state: &mut AppState, _: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut resource_map = ResourceMap::new(app_state.gpu.clone());
        resource_map.install_resource_loader(FileSystemTextureLoader::new(app_state.gpu.clone()));

        let cvar_manager = CvarManager::new();
        let camera = Camera {
            location: point![2.0, 2.0, 2.0],
            forward: vector![0.0, -1.0, -1.0].normalize(),
            ..Default::default()
        };

        let forward_movement = 0.0;
        let rotation_movement = 0.0;

        let rot_x = 45.0;
        let rot_z = 55.0;
        let dist = 5.0;

        let movement: Vector3<f32> = vector![0.0, 0.0, 0.0];
        let vertex_module =
            utils::read_file_to_vk_module(app_state.gpu(), "./shaders/vertex_deferred.spirv")?;
        let fragment_module =
            utils::read_file_to_vk_module(app_state.gpu(), "./shaders/fragment_deferred.spirv")?;

        let cube = utils::load_cube_to_resource_map(app_state.gpu(), &mut resource_map)?;

        let mesh_data = MeshCreateInfo {
            label: Some("Quad mesh"),
            primitives: &[MeshPrimitiveCreateInfo {
                indices: vec![0, 1, 2, 2, 3, 0],
                positions: vec![
                    vector![-0.5, -0.5, 0.0],
                    vector![0.5, -0.5, 0.0],
                    vector![0.5, 0.5, 0.0],
                    vector![-0.5, 0.5, 0.0],
                ],
                colors: vec![
                    vector![1.0, 0.0, 0.0],
                    vector![0.0, 1.0, 0.0],
                    vector![0.0, 0.0, 1.0],
                    vector![1.0, 1.0, 1.0],
                ],
                normals: vec![
                    vector![0.0, 1.0, 0.0],
                    vector![0.0, 1.0, 0.0],
                    vector![0.0, 1.0, 0.0],
                    vector![0.0, 1.0, 0.0],
                ],
                tangents: vec![
                    vector![0.0, 0.0, 1.0],
                    vector![0.0, 0.0, 1.0],
                    vector![0.0, 0.0, 1.0],
                    vector![0.0, 0.0, 1.0],
                ],
                uvs: vec![
                    vector![1.0, 0.0],
                    vector![0.0, 0.0],
                    vector![0.0, 1.0],
                    vector![1.0, 1.0],
                ],
            }],
        };

        let egui_integration = EguiSupport::new(&app_state.window, app_state.gpu())?;
        let mesh = Mesh::new(app_state.gpu(), &mesh_data)?;
        let mesh = resource_map.add(mesh);

        let texture = resource_map.load::<Texture>("images/texture.jpg")?;
        let mut scene_renderer = DeferredRenderingPipeline::new(
            app_state.gpu(),
            &mut resource_map,
            cube.clone(),
            DeferredRenderingPipeline::make_3d_combine_shader(app_state.gpu.as_ref())?,
        )?;

        let master = scene_renderer.create_material(
            app_state.gpu(),
            MaterialDescription {
                name: "Simple",
                domain: MaterialDomain::Surface,
                fragment_module,
                vertex_module,
                texture_inputs: &[TextureInput {
                    name: "texSampler".to_owned(),
                    format: gpu::ImageFormat::Rgba8,
                    shader_stage: ShaderStage::FRAGMENT,
                }],
                material_parameters: Default::default(),
                parameter_shader_visibility: ShaderStage::FRAGMENT,
                cull_mode: gpu::CullMode::Back,
            },
        )?;

        let texture_inputs = vec![texture];
        let material = resource_map.add(master);
        let mat_instance = MaterialInstance::create_instance(
            material,
            &MaterialInstanceDescription {
                name: "simple inst",
                textures: texture_inputs,
                parameter_buffers: vec![],
            },
        )?;

        app_state
            .swapchain_mut()
            .select_present_mode(PresentMode::Mailbox)?;

        let mut scene = RenderScene::new();

        let bounds = BoundingShape::BoundingBox {
            min: point![-1.0, -1.0, 0.0],
            max: point![1.0, 1.0, 1.0],
        };
        scene.add(ScenePrimitive {
            mesh: mesh.clone(),
            materials: vec![mat_instance.clone()],
            transform: Matrix4::identity(),
            bounds,
        });
        scene.add(ScenePrimitive {
            mesh: mesh.clone(),
            materials: vec![mat_instance.clone()],
            transform: Matrix4::new_translation(&vector![0.0, 0.0, 1.0]),
            bounds,
        });
        scene.add(ScenePrimitive {
            mesh,
            materials: vec![mat_instance.clone()],
            transform: Matrix4::new_translation(&vector![0.0, 0.0, -1.0]),
            bounds,
        });

        scene.add(ScenePrimitive {
            mesh: cube.clone(),
            materials: vec![mat_instance.clone()],
            transform: Matrix4::new_scaling(1.0)
                * Matrix4::new_translation(&vector![1.5, 0.0, 0.0]),
            bounds,
        });

        scene.add(ScenePrimitive {
            mesh: cube,
            materials: vec![mat_instance],
            transform: Matrix4::new_translation(&vector![-1.5, 0.0, 0.0])
                * Matrix4::new_scaling(-1.0),

            bounds,
        });

        scene_renderer.ambient_color = vector![1.0, 1.0, 1.0];
        scene_renderer.ambient_intensity = 1.0;

        Ok(Self {
            egui_integration,
            camera,
            forward_movement,
            rotation_movement,
            rot_x,
            rot_z,
            dist,
            movement,
            scene_renderer,
            scene,
            resource_map,
            cvar_manager,
            time: Time::default(),
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
                if button == 3 {
                    self.rotation_movement = mul;
                } else if button == 1 {
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

    fn begin_frame(&mut self, app_state: &mut AppState) -> anyhow::Result<()> {
        self.time.begin_frame();
        self.egui_integration
            .begin_frame(&app_state.window, &self.time);
        Ok(())
    }

    fn update(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        self.resource_map.update();

        if self.rotation_movement > 0.0 {
            self.rot_z += self.movement.y;
            self.rot_z = self.rot_z.clamp(-180.0, 180.0);
        } else {
            self.dist += self.movement.y * self.forward_movement * SPEED;
        }

        let new_forward = Rotation::<f32, 3>::from_axis_angle(
            &Unit::new_normalize(vector![0.0, 0.0, 1.0]),
            self.rot_x.to_radians(),
        ) * Rotation::<f32, 3>::from_axis_angle(
            &Unit::new_normalize(vector![0.0, 1.0, 0.0]),
            -self.rot_z.to_radians(),
        );
        let new_forward = new_forward.to_homogeneous();
        let new_forward = new_forward.column(2);

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        let new_position = direction * self.dist;
        let new_position = point![new_position.x, new_position.y, new_position.z];
        self.camera.location = new_position;

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        self.camera.forward = -direction;

        Ok(())
    }

    fn draw<'a>(
        &'a mut self,
        app_state: &'a mut AppState,
        backbuffer: &Backbuffer,
    ) -> anyhow::Result<CommandBuffer> {
        let mut cb = app_state
            .gpu
            .start_command_buffer(gpu::QueueType::Graphics)?;
        let final_render = self.scene_renderer.render(
            app_state.gpu(),
            &mut cb,
            &self.camera,
            &self.scene,
            &self.resource_map,
            &self.cvar_manager,
        )?;

        let output = self.egui_integration.end_frame(&app_state.window);
        self.scene_renderer.draw_textured_quad(
            &mut cb,
            &backbuffer.image_view,
            &final_render,
            Rect2D {
                offset: Offset2D::default(),
                extent: backbuffer.size,
            },
            true,
            None,
        )?;

        self.egui_integration.paint_frame(
            app_state.gpu(),
            &mut cb,
            backbuffer,
            output.textures_delta,
            output.shapes,
        )?;
        self.egui_integration
            .handle_platform_output(&app_state.window, output.platform_output);
        Ok(cb)
    }
    fn on_shutdown(&mut self, app_state: &mut AppState) {
        let gpu = app_state.gpu();
        self.scene_renderer.destroy(gpu);
        self.resource_map.update();
        self.egui_integration.destroy(gpu);
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<PlanesApp>()
}
