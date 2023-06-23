mod app;
mod utils;

use std::{io::BufReader, rc::Rc};

use app::{bootstrap, App};
use ash::vk::PresentModeKHR;

use engine::{
    AppState, Camera, DeferredRenderingPipeline, MaterialDescription, MaterialDomain, Mesh,
    MeshCreateInfo, RenderingPipeline, Scene, ScenePrimitive, Texture,
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
const SPEED: f32 = 0.1;
const ROTATION_SPEED: f32 = 3.0;
const MIN_DELTA: f32 = 1.0;
pub struct GLTFViewer {
    resource_map: Rc<ResourceMap>,
    camera: Camera,
    forward_movement: f32,
    rotation_movement: f32,
    rot_x: f32,
    rot_z: f32,
    dist: f32,
    movement: Vector3<f32>,
    scene_renderer: DeferredRenderingPipeline,
    scene: Scene,
}

impl GLTFViewer {
    fn read_gltf(
        app_state: &AppState,
        scene_renderer: &mut dyn RenderingPipeline,
        resource_map: Rc<ResourceMap>,
        path: &str,
    ) -> anyhow::Result<Scene> {
        let cpu_image = image::load(
            BufReader::new(std::fs::File::open("images/texture.jpg")?),
            image::ImageFormat::Jpeg,
        )?;
        let cpu_image = cpu_image.into_rgba8();

        let vertex_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/vertex_deferred.spirv")?;
        let fragment_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/fragment_deferred.spirv")?;

        let texture = Texture::new_with_data(
            &app_state.gpu,
            cpu_image.width(),
            cpu_image.height(),
            &cpu_image,
            Some("Quad texture david"),
        )?;
        let texture = resource_map.add(texture);

        let material = scene_renderer.get_context().create_material(
            &app_state.gpu,
            &resource_map,
            MaterialDescription {
                domain: MaterialDomain::Surface,
                uniform_buffers: vec![],
                input_textures: vec![texture.clone()],
                fragment_module: &fragment_module,
                vertex_module: &vertex_module,
            },
        )?;
        let material = resource_map.add(material);

        let (document, buffers, images) = gltf::import(path)?;
        let mut engine_scene = Scene::new();

        let mut meshes = vec![];

        for mesh in document.meshes() {
            let mut indices = vec![];
            let mut positions = vec![];
            let mut colors = vec![];
            let mut normals = vec![];
            let mut tangents = vec![];
            let mut uvs = vec![];

            for prim in mesh.primitives() {
                let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
                if let Some(iter) = reader.read_indices() {
                    for idx in iter.into_u32() {
                        indices.push(idx);
                    }
                }
                if let Some(iter) = reader.read_positions() {
                    for vert in iter {
                        positions.push(vector![vert[0], vert[1], vert[2]]);
                    }
                }
                if let Some(iter) = reader.read_normals() {
                    for vec in iter {
                        normals.push(vector![vec[0], vec[1], vec[2]]);
                    }
                }
                if let Some(iter) = reader.read_tangents() {
                    for vec in iter {
                        tangents.push(vector![vec[0], vec[1], vec[2]]);
                    }
                }
                if let Some(iter) = reader.read_tex_coords(0) {
                    for vec in iter.into_f32() {
                        uvs.push(vector![vec[0], vec[1]]);
                    }
                }
            }

            let label = format!("Mesh #{}", mesh.index());

            let create_info = MeshCreateInfo {
                label: Some(mesh.name().unwrap_or(&label)),
                positions: &positions,
                indices: &indices,
                colors: &colors,
                normals: &normals,
                tangents: &tangents,
                uvs: &uvs,
            };
            let gpu_mesh = Mesh::new(&app_state.gpu, &create_info)?;
            meshes.push(resource_map.add(gpu_mesh));
        }

        for scene in document.scenes() {
            for node in scene.nodes() {
                let node_transform = node.transform();
                let (pos, rot, scale) = node_transform.decomposed();
                let transform = Matrix4::new_translation(&Vector3::from_row_slice(&pos))
                    * Matrix4::new_nonuniform_scaling(&Vector3::from_row_slice(&scale));

                if let Some(mesh) = node.mesh() {
                    engine_scene.add(ScenePrimitive {
                        mesh: meshes[mesh.index()].clone(),
                        material: material.clone(),
                        transform,
                    });
                }
            }
        }

        Ok(engine_scene)
    }
}

impl App for GLTFViewer {
    fn window_name() -> &'static str {
        "GLTF Viewer"
    }

    fn create(app_state: &engine::AppState) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let resource_map = Rc::new(ResourceMap::new());

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

        let screen_quad_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/screen_quad.spirv")?;
        let gbuffer_combine_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/gbuffer_combine.spirv")?;
        let texture_copy_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/texture_copy.spirv")?;

        let mut scene_renderer = DeferredRenderingPipeline::new(
            &app_state.gpu,
            resource_map.clone(),
            screen_quad_module,
            gbuffer_combine_module,
            texture_copy_module,
        )?;

        let scene = Self::read_gltf(
            app_state,
            &mut scene_renderer,
            resource_map.clone(),
            "gltf_models/cube/Cube.gltf",
        )?;
        engine::app_state_mut()
            .swapchain
            .select_present_mode(PresentModeKHR::MAILBOX)?;

        Ok(Self {
            resource_map,
            camera,
            forward_movement,
            rotation_movement,
            rot_x,
            rot_z,
            dist,
            movement,
            scene_renderer,
            scene,
        })
    }

    fn input(
        &mut self,
        app_state: &engine::AppState,
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
                    * delta.1.signum() as f32 as f32
                    * ROTATION_SPEED;
            }
            _ => {}
        };
        Ok(())
    }

    fn draw(&mut self, app_state: &mut engine::AppState) -> anyhow::Result<()> {
        self.scene_renderer
            .render(&self.camera, &self.scene, &mut app_state.swapchain)
            .unwrap();

        Ok(())
    }

    fn update(&mut self, _app_state: &mut engine::AppState) -> anyhow::Result<()> {
        if self.rotation_movement > 0.0 {
            self.rot_z += self.movement.y;
            self.rot_z = self.rot_z.clamp(-89.0, 89.0);
            self.rot_x += self.movement.x;
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
        let new_forward = new_forward.column(0);

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        let new_position = direction * self.dist;
        let new_position = point![new_position.x, new_position.y, new_position.z];
        self.camera.location = new_position;

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        self.camera.forward = -direction;
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
