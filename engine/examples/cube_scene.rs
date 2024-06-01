mod utils;

use engine::{
    app::{self, bootstrap, App, AppContext, AppDescription},
    asset_map::{AssetHandle, AssetMap},
    assets::{
        material::{
            Material, MaterialDescription, MaterialDomain, MaterialParameters, MaterialProperties,
            MaterialType,
        },
        texture::Texture,
    },
    constants::CUBE_MESH_HANDLE,
    include_spirv,
    math::Transform,
    sampler_allocator::SamplerAllocator,
    scene::{Scene, SceneMesh, SceneNode, SceneNodeId},
    scene_renderer::{
        PointOfView, ProjectionMode, SceneOutput, SceneRenderer, SceneRenderingParams,
    },
    shader_cache::ShaderCache,
};
use glam::vec3;
use mgpu::{Extents2D, ShaderModuleDescription};

const VERTEX_SHADER: &[u8] = include_spirv!("spirv/simple_vertex.vert.spv");
const FRAGMENT_SHADER: &[u8] = include_spirv!("spirv/simple_fragment.frag.spv");
pub struct CubesSceneApplication {
    asset_map: AssetMap,
    scene: Scene,
    first_node_handle: SceneNodeId,
    scene_renderer: SceneRenderer,
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

        let mut shader_cache = ShaderCache::new();
        shader_cache.add_shader("simple_vertex_shader", vertex_shader_module);
        shader_cache.add_shader("simple_fragment_shader", fragment_shader_module);

        let sampler_allocator = SamplerAllocator::default();
        let mut asset_map =
            app::asset_map_with_defaults(&context.device, &sampler_allocator, shader_cache)?;

        asset_map.discover_assets("assets");
        asset_map.preload("assets/images/david");

        let david_texture = AssetHandle::<Texture>::new("assets/images/david");
        let mut scene = Scene::default();
        let material = Material::new(
            &MaterialDescription {
                label: Some("simple material"),
                vertex_shader: "simple_vertex_shader".into(),
                fragment_shader: "simple_fragment_shader".into(),
                parameters: MaterialParameters::default().texture_parameter("tex", david_texture),
                properties: MaterialProperties {
                    domain: MaterialDomain::Surface,
                    ty: MaterialType::Lit,
                    double_sided: false,
                },
            },
            &mut asset_map,
        );
        let simple_material = material.unwrap();
        let material_handle = asset_map.add(simple_material, "simple_material");
        let first_node_handle = scene.add_node(
            SceneNode::default()
                .label("First Cube")
                .primitive(engine::scene::ScenePrimitive::Mesh(SceneMesh {
                    handle: CUBE_MESH_HANDLE.clone(),
                    material: material_handle.clone(),
                }))
                .transform(Transform {
                    location: vec3(0.0, 0.0, 10.0),
                    ..Default::default()
                }),
        );
        let second_cube = scene.add_node(
            SceneNode::default()
                .label("Second Cube")
                .primitive(engine::scene::ScenePrimitive::Mesh(SceneMesh {
                    handle: CUBE_MESH_HANDLE.clone(),
                    material: material_handle.clone(),
                }))
                .transform(Transform {
                    location: vec3(10.0, 0.0, 10.0),
                    ..Default::default()
                }),
        );
        let third_cube = scene.add_node(
            SceneNode::default()
                .label("Third Cube")
                .primitive(engine::scene::ScenePrimitive::Mesh(SceneMesh {
                    handle: CUBE_MESH_HANDLE.clone(),
                    material: material_handle.clone(),
                }))
                .transform(Transform {
                    location: vec3(-10.0, 0.0, 10.0),
                    ..Default::default()
                }),
        );
        scene.add_child(first_node_handle, second_cube);
        scene.add_child(first_node_handle, third_cube);

        let scene_renderer = SceneRenderer::new(&context.device, &asset_map)?;
        let mut pov = PointOfView::new_perspective(0.01, 1000.0, 75.0, 1920.0 / 1080.0);
        pov.transform.location = vec3(0.0, 10.0, -5.0);

        Ok(Self {
            asset_map,
            scene,
            first_node_handle,
            scene_renderer,

            pov,
        })
    }

    fn handle_window_event(
        &mut self,
        _event: &winit::event::WindowEvent,
        _context: &AppContext,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn handle_device_event(
        &mut self,
        _event: &winit::event::DeviceEvent,
        _context: &AppContext,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        utils::update_fps_camera(context, &mut self.pov);

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
            output: SceneOutput::BaseColor,
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
        context
            .window()
            .set_cursor_grab(engine::winit::window::CursorGrabMode::Confined)
            .unwrap();
        Ok(())
    }
}

impl CubesSceneApplication {}

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
