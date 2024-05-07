use engine::{
    app::{bootstrap, App},
    asset_map::AssetMap,
    assets::{
        loaders::FsTextureLoader,
        material::{
            Material, MaterialDescription, MaterialDomain, MaterialParameters, MaterialProperties,
        },
        mesh::{Mesh, MeshDescription},
        texture::Texture,
    },
    compile_glsl,
    input::InputState,
    math::Transform,
    sampler_allocator::SamplerAllocator,
    scene::{Scene, SceneMesh, SceneNode, SceneNodeId},
    scene_renderer::{PointOfView, ProjectionMode, SceneRenderer, SceneRenderingParams},
    shader_cache::ShaderCache,
};
use glam::{vec2, vec3};
use mgpu::ShaderModuleDescription;
use shaderc::ShaderKind;
const VERTEX_SHADER: &str = "
#version 460
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 norm;
layout(location = 2) in vec3 tang;
layout(location = 3) in vec3 color;
layout(location = 4) in vec2 uv;

layout(location = 0) out vec2 fs_uv;

layout(push_constant, std140) uniform ObjectData {
    mat4 model;
};

layout(set = 0, binding = 0, std140) uniform GlobalFrameData {
    mat4 projection;
    mat4 view;
    float frame_time;
};

void main() {
    mat4 mvp = projection * view * model;
    vec4 vs_pos = mvp * vec4(pos, 1.0);
    gl_Position = vs_pos;
    fs_uv = uv;
}
";
const FRAGMENT_SHADER: &str = "
#version 460
layout(set = 1, binding = 1) uniform texture2D tex;
layout(set = 1, binding = 2) uniform sampler tex_sampler;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 color;

void main() {
    color = texture(sampler2D(tex, tex_sampler), uv);
}
";
pub struct CubesSceneApplication {
    asset_map: AssetMap,
    scene: Scene,
    first_node_handle: SceneNodeId,
    input: InputState,
    scene_renderer: SceneRenderer,
    pov: PointOfView,
}

impl App for CubesSceneApplication {
    fn app_name() -> &'static str {
        "Cube Scene"
    }

    fn create(context: &engine::app::AppContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let vertex_shader_source = compile_glsl(VERTEX_SHADER, ShaderKind::Vertex)?;
        let fragment_shader_source = compile_glsl(FRAGMENT_SHADER, ShaderKind::Fragment)?;
        let vertex_shader_module = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("Simple Vertex Shader"),
                source: &vertex_shader_source,
            })
            .unwrap();
        let fragment_shader_module = context
            .device
            .create_shader_module(&ShaderModuleDescription {
                label: Some("Simple Fragment Shader"),
                source: &fragment_shader_source,
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
        let david_texture = asset_map.load::<Texture>("assets/images/david.jpg")?;
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
            input: InputState::default(),
            scene_renderer,
            pov,
        })
    }

    fn handle_window_event(&mut self, event: &winit::event::WindowEvent) -> anyhow::Result<()> {
        self.input.update(event);
        Ok(())
    }

    fn handle_device_event(&mut self, _event: &winit::event::DeviceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        self.input.end_frame();
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
        context: &engine::app::AppContext,
        new_extents: mgpu::Extents2D,
    ) -> mgpu::MgpuResult<()> {
        Ok(())
    }

    fn shutdown(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
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
}

fn main() -> anyhow::Result<()> {
    bootstrap::<CubesSceneApplication>()
}
