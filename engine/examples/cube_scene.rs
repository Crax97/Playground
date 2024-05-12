use engine::{
    app::{bootstrap, App, AppContext, AppDescription},
    asset_map::{AssetHandle, AssetMap},
    assets::{
        loaders::FsTextureLoader,
        material::{
            Material, MaterialDescription, MaterialDomain, MaterialParameters, MaterialProperties,
        },
        mesh::{Mesh, MeshDescription},
        texture::Texture,
    },
    input::InputState,
    math::Transform,
    sampler_allocator::SamplerAllocator,
    scene::{serializable_scene::SerializableScene, Scene, SceneMesh, SceneNode, SceneNodeId},
    scene_renderer::{PointOfView, ProjectionMode, SceneRenderer, SceneRenderingParams},
    shader_cache::ShaderCache,
};
use glam::{vec2, vec3, Quat, Vec3};
use gltf::json::Asset;
use mgpu::{Extents2D, ShaderModuleDescription};
use winit::dpi::{LogicalPosition, PhysicalPosition, PhysicalSize, Position};

macro_rules! include_bytes_align_as {
    ($align_ty:ty, $path:literal) => {{
        #[repr(C)]
        pub struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }

        const ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        &ALIGNED.bytes
    }};
}

const VERTEX_SHADER: &[u8] = include_bytes_align_as!(u32, "spirv/simple_vertex.vert.spv");
const FRAGMENT_SHADER: &[u8] = include_bytes_align_as!(u32, "spirv/simple_fragment.frag.spv");
pub struct CubesSceneApplication {
    asset_map: AssetMap,
    scene: Scene,
    first_node_handle: SceneNodeId,
    scene_renderer: SceneRenderer,
    cam_pitch: f32,
    cam_roll: f32,
    pov: PointOfView,
}

impl App for CubesSceneApplication {
    fn create(context: &engine::app::AppContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let vertex_shader_module = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("Simple Vertex Shader"),
                source: bytemuck::cast_slice(VERTEX_SHADER),
            })
            .unwrap();
        let fragment_shader_module = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("Simple Fragment Shader"),
                source: bytemuck::cast_slice(FRAGMENT_SHADER),
            })
            .unwrap();
        let mut asset_map = AssetMap::new();
        let mut shader_cache = ShaderCache::new();

        shader_cache.add_shader("simple_vertex_shader", vertex_shader_module);
        shader_cache.add_shader("simple_fragment_shader", fragment_shader_module);

        let sampler_allocator = SamplerAllocator::default();
        let texture_loader =
            FsTextureLoader::new(context.device.clone(), sampler_allocator.clone());
        asset_map.add_loader(texture_loader);
        let david_texture = AssetHandle::<Texture>::new("assets/images/david.jpg");
        asset_map.load::<Texture>(&david_texture)?;
        let mut scene = Scene::default();
        let cube_mesh = Self::create_cube_mesh(&context.device)?;
        let cube_handle = asset_map.add(cube_mesh, "meshes.cube");
        let material = Material::new(
            &context.device,
            &MaterialDescription {
                label: Some("simple material"),
                vertex_shader: "simple_vertex_shader".into(),
                fragment_shader: "simple_fragment_shader".into(),
                parameters: MaterialParameters::default().texture_parameter("tex", david_texture),
                properties: MaterialProperties {
                    domain: MaterialDomain::Surface,
                },
            },
            &mut asset_map,
            &mut shader_cache,
        );
        let simple_material = material.unwrap();
        let material_handle = asset_map.add(simple_material, "simple_material");
        let first_node_handle = scene.add_node(
            SceneNode::default()
                .label("First Cube")
                .primitive(engine::scene::ScenePrimitive::Mesh(SceneMesh {
                    handle: cube_handle,
                    material: material_handle,
                }))
                .transform(Transform {
                    location: vec3(0.0, 0.0, 10.0),
                    ..Default::default()
                }),
        );

        let scene_renderer = SceneRenderer::new(&context.device)?;
        let mut pov = PointOfView::new_perspective(0.01, 1000.0, 75.0, 1920.0 / 1080.0);
        pov.transform.location = vec3(0.0, 10.0, -5.0);

        Ok(Self {
            asset_map,
            scene,
            first_node_handle,
            scene_renderer,
            cam_pitch: 0.0,
            cam_roll: 0.0,
            pov,
        })
    }

    fn handle_window_event(&mut self, _event: &winit::event::WindowEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn handle_device_event(&mut self, _event: &winit::event::DeviceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        self.update_fps_camera(context);

        const ROTATION_PER_FRAME_DEGS: f64 = 30.0;
        let mut node_transform = self
            .scene
            .get_node_world_transform(self.first_node_handle)
            .unwrap();
        node_transform.add_rotation_euler(
            0.0,
            (ROTATION_PER_FRAME_DEGS * context.time.delta_seconds()) as f32,
            0.0,
        );
        self.scene
            .set_node_world_transform(self.first_node_handle, node_transform);

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

impl CubesSceneApplication {
    fn create_cube_mesh(device: &mgpu::Device) -> anyhow::Result<Mesh> {
        let mesh_description = MeshDescription {
            label: Some("Cube mesh"),
            indices: &[
                0, 1, 2, 3, 1, 0, //Bottom
                6, 5, 4, 4, 5, 7, // Front
                10, 9, 8, 8, 9, 11, // Left
                12, 13, 14, 15, 13, 12, // Right
                16, 17, 18, 19, 17, 16, // Up
                22, 21, 20, 20, 21, 23, // Down
            ],
            positions: &[
                // Back
                vec3(-1.0, -1.0, 1.0),
                vec3(1.0, 1.0, 1.0),
                vec3(-1.0, 1.0, 1.0),
                vec3(1.0, -1.0, 1.0),
                // Front
                vec3(-1.0, -1.0, -1.0),
                vec3(1.0, 1.0, -1.0),
                vec3(-1.0, 1.0, -1.0),
                vec3(1.0, -1.0, -1.0),
                // Left
                vec3(1.0, -1.0, -1.0),
                vec3(1.0, 1.0, 1.0),
                vec3(1.0, 1.0, -1.0),
                vec3(1.0, -1.0, 1.0),
                // Right
                vec3(-1.0, -1.0, -1.0),
                vec3(-1.0, 1.0, 1.0),
                vec3(-1.0, 1.0, -1.0),
                vec3(-1.0, -1.0, 1.0),
                // Up
                vec3(-1.0, 1.0, -1.0),
                vec3(1.0, 1.0, 1.0),
                vec3(1.0, 1.0, -1.0),
                vec3(-1.0, 1.0, 1.0),
                // Down
                vec3(-1.0, -1.0, -1.0),
                vec3(1.0, -1.0, 1.0),
                vec3(1.0, -1.0, -1.0),
                vec3(-1.0, -1.0, 1.0),
            ],
            colors: &[vec3(1.0, 0.0, 0.0)],
            normals: &[
                // Back
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                // Front
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                // Left
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                // Right
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                // Up
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                // Down
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
            ],
            tangents: &[
                // Back
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                // Front
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                // Left
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                // Right
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                // Up
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                // Down
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
            ],
            uvs: &[
                vec2(0.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
            ],
        };
        let cube_mesh = Mesh::new(device, &mesh_description)?;
        Ok(cube_mesh)
    }

    fn update_fps_camera(&mut self, context: &AppContext) {
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
        self.cam_roll -= mouse_delta.x * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
        self.cam_pitch += mouse_delta.y * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
        self.cam_pitch = self.cam_pitch.clamp(-89.0, 89.0);

        let new_location_offset = camera_input.x * self.pov.transform.left()
            + camera_input.y * self.pov.transform.up()
            + camera_input.z * self.pov.transform.forward();
        self.pov.transform.location += new_location_offset;
        self.pov.transform.rotation = Quat::from_euler(
            glam::EulerRot::XYZ,
            self.cam_pitch.to_radians(),
            self.cam_roll.to_radians(),
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
}

fn main() -> anyhow::Result<()> {
    bootstrap::<CubesSceneApplication>(AppDescription {
        window_size: Extents2D {
            width: 1920,
            height: 1080,
        },
        initial_title: Some("Cube Scene"),
        app_identifier: "CubeSceneApp",
    })
}
