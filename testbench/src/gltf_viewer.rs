mod app;
mod utils;

use std::f32::consts::PI;
use std::rc::Rc;

use app::{bootstrap, App};
use ash::vk::{
    BufferUsageFlags, ComponentMapping, Filter, ImageAspectFlags, ImageSubresourceRange,
    ImageUsageFlags, ImageViewType, PresentModeKHR, SamplerAddressMode, SamplerCreateInfo,
};

use engine::{
    AppState, Camera, DeferredRenderingPipeline, ImageResource, MaterialDescription,
    MaterialDomain, Mesh, MeshCreateInfo, RenderingPipeline, SamplerResource, Scene,
    ScenePrimitive, Texture, TextureImageView,
};
use gpu::{BufferCreateInfo, ImageCreateInfo, ImageViewCreateInfo, MemoryDomain, ToVk};
use nalgebra::*;
use resource_map::{ResourceHandle, ResourceMap};
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

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct PbrProperties {
    pub base_color: Vector4<f32>,         // vec4
    pub metallic_roughness: Vector4<f32>, // vec4
    pub emissive_color: Vector4<f32>,     // vec3
}

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
    pub white_texture: ResourceHandle<Texture>,
    pub black_texture: ResourceHandle<Texture>,
}

impl GLTFViewer {
    fn read_gltf(
        app_state: &AppState,
        scene_renderer: &mut dyn RenderingPipeline,
        resource_map: Rc<ResourceMap>,
        white: &ResourceHandle<Texture>,
        black: &ResourceHandle<Texture>,
        path: &str,
    ) -> anyhow::Result<Scene> {
        let vertex_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/vertex_deferred.spirv")?;
        let fragment_module = utils::read_file_to_vk_module(
            &app_state.gpu,
            "./shaders/metallic_roughness_pbr.spirv",
        )?;

        let mut allocated_images = vec![];
        let mut allocated_image_views = vec![];
        let mut allocated_samplers = vec![];
        let mut allocated_textures = vec![];
        let mut allocated_materials = vec![];

        let (document, buffers, mut images) = gltf::import(path)?;

        for (index, gltf_image) in images.iter_mut().enumerate() {
            let vk_format = match gltf_image.format {
                gltf::image::Format::R8G8B8A8 => gpu::ImageFormat::Rgba8.to_vk(),
                gltf::image::Format::R8G8B8 => gpu::ImageFormat::Rgb8.to_vk(),
                gltf::image::Format::R32G32B32A32FLOAT => gpu::ImageFormat::RgbaFloat.to_vk(),
                f => panic!("Unsupported format! {:?}", f),
            };
            let label = format!("glTF Image #{}", index);
            let image_create_info = ImageCreateInfo {
                label: Some(&label),
                width: gltf_image.width,
                height: gltf_image.height,
                format: vk_format,
                usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
            };
            let gpu_image = app_state.gpu.create_image(
                &image_create_info,
                MemoryDomain::DeviceLocal,
                Some(&gltf_image.pixels),
            )?;

            let gpu_image_view = app_state.gpu.create_image_view(&ImageViewCreateInfo {
                image: &gpu_image,
                view_type: ImageViewType::TYPE_2D,
                format: vk_format,
                components: ComponentMapping::default(),
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            })?;
            let img_index = resource_map.add(ImageResource(gpu_image));
            allocated_images.push(img_index.clone());
            let view_index = resource_map.add(TextureImageView {
                image: img_index,
                view: gpu_image_view,
            });
            allocated_image_views.push(view_index);
        }

        for sampler in document.samplers() {
            let builder = SamplerCreateInfo::builder()
                .address_mode_u(match &sampler.wrap_s() {
                    gltf::texture::WrappingMode::ClampToEdge => SamplerAddressMode::CLAMP_TO_EDGE,
                    gltf::texture::WrappingMode::MirroredRepeat => {
                        SamplerAddressMode::MIRRORED_REPEAT
                    }
                    gltf::texture::WrappingMode::Repeat => SamplerAddressMode::REPEAT,
                })
                .address_mode_v(match &sampler.wrap_t() {
                    gltf::texture::WrappingMode::ClampToEdge => SamplerAddressMode::CLAMP_TO_EDGE,
                    gltf::texture::WrappingMode::MirroredRepeat => {
                        SamplerAddressMode::MIRRORED_REPEAT
                    }
                    gltf::texture::WrappingMode::Repeat => SamplerAddressMode::REPEAT,
                })
                .mag_filter(
                    match sampler
                        .mag_filter()
                        .unwrap_or(gltf::texture::MagFilter::Nearest)
                    {
                        gltf::texture::MagFilter::Nearest => Filter::NEAREST,
                        gltf::texture::MagFilter::Linear => Filter::LINEAR,
                    },
                )
                .min_filter(
                    match sampler
                        .min_filter()
                        .unwrap_or(gltf::texture::MinFilter::Nearest)
                    {
                        gltf::texture::MinFilter::Nearest => Filter::NEAREST,
                        gltf::texture::MinFilter::Linear => Filter::LINEAR,
                        x => {
                            log::warn!("glTF: unsupported filter! {:?}", x);
                            Filter::LINEAR
                        }
                    },
                );
            let sam = app_state.gpu.create_sampler(&builder.build())?;
            allocated_samplers.push(resource_map.add(SamplerResource(sam)))
        }
        
        if allocated_samplers.is_empty() {
            // add default sampler
            let builder = SamplerCreateInfo::builder()
                .address_mode_u(SamplerAddressMode::REPEAT)
                .address_mode_v(SamplerAddressMode::REPEAT)
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR);
            let sam = app_state.gpu.create_sampler(&builder.build())?;
            allocated_samplers.push(resource_map.add(SamplerResource(sam)))
        }

        for texture in document.textures() {
            allocated_textures.push(resource_map.add(Texture {
                sampler: allocated_samplers[texture.sampler().index().unwrap_or(0)].clone(),
                image_view: allocated_image_views[texture.source().index()].clone(),
            }))
        }

        for gltf_material in document.materials() {
            let base_texture =
                if let Some(base) = gltf_material.pbr_metallic_roughness().base_color_texture() {
                    allocated_textures[base.texture().index()].clone()
                } else {
                    white.clone()
                };
            let normal_texture = if let Some(base) = gltf_material.normal_texture() {
                allocated_textures[base.texture().index()].clone()
            } else {
                white.clone()
            };
            let occlusion_texture = if let Some(base) = gltf_material.occlusion_texture() {
                allocated_textures[base.texture().index()].clone()
            } else {
                white.clone()
            };
            let emissive_texture = if let Some(base) = gltf_material.emissive_texture() {
                allocated_textures[base.texture().index()].clone()
            } else {
                black.clone()
            };
            let metallic_roughness = if let Some(base) = gltf_material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
            {
                allocated_textures[base.texture().index()].clone()
            } else {
                white.clone()
            };

            let params = {
                let buf = app_state.gpu.create_buffer(
                    &BufferCreateInfo {
                        label: Some("Pbr Properties"),
                        size: std::mem::size_of::<PbrProperties>(),
                        usage: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;

                let gltf_props = gltf_material.pbr_metallic_roughness();
                let mut props = PbrProperties::default();
                let emissive = gltf_material.emissive_factor();
                props.base_color = Vector4::from_row_slice(&gltf_props.base_color_factor());
                props.metallic_roughness.x = gltf_props.metallic_factor();
                props.metallic_roughness.y = gltf_props.roughness_factor();
                props.emissive_color = vector![emissive[0], emissive[1], emissive[2], 1.0];

                app_state.gpu.write_buffer_data(&buf, &[props])?;
                buf
            };

            let material = scene_renderer.get_context().create_material(
                &app_state.gpu,
                &resource_map,
                MaterialDescription {
                    domain: MaterialDomain::Surface,
                    uniform_buffers: vec![params],
                    input_textures: vec![
                        base_texture,
                        normal_texture,
                        occlusion_texture,
                        emissive_texture,
                        metallic_roughness,
                    ],
                    fragment_module: &fragment_module,
                    vertex_module: &vertex_module,
                },
            )?;
            let material = resource_map.add(material);
            allocated_materials.push(material);
        }

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
                if let Some(iter) = reader.read_colors(0) {
                    for vert in iter.into_rgb_f32() {
                        colors.push(vector![vert[0], vert[1], vert[2]]);
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
                let rotation = UnitQuaternion::from_quaternion(Quaternion::new(rot[0],rot[1],rot[2],rot[3]));
                let rot_matrix  = rotation.to_homogeneous();
                
                let transform = Matrix4::new_translation(&Vector3::from_row_slice(&pos))
                    * Matrix4::new_nonuniform_scaling(&Vector3::from_row_slice(&scale))
                    * rot_matrix;
                
                if let Some(mesh) = node.mesh() {
                    let prim = mesh.primitives().next().unwrap();
                    let mat = prim.material().index().unwrap_or(0);
                    engine_scene.add(ScenePrimitive {
                        mesh: meshes[mesh.index()].clone(),
                        material: allocated_materials[mat].clone(),
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
            resource_map.clone(),
            screen_quad_module,
            gbuffer_combine_module,
            texture_copy_module,
        )?;

        let white_texture = Texture::new_with_data(
            &app_state.gpu,
            &resource_map,
            1,
            1,
            &[255, 255, 255, 255],
            Some("White texture"),
        )?;
        let white_texture = resource_map.add(white_texture);
        let black_texture = Texture::new_with_data(
            &app_state.gpu,
            &resource_map,
            1,
            1,
            &[0, 0, 0, 255],
            Some("Black texture"),
        )?;
        let black_texture = resource_map.add(black_texture);

        let scene = Self::read_gltf(
            app_state,
            &mut scene_renderer,
            resource_map.clone(),
            &white_texture,
            &black_texture,
            "gltf_models/bottle/glTF/WaterBottle.gltf",
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
            white_texture,
            black_texture
        })
    }

    fn input(
        &mut self,
        _app_state: &engine::AppState,
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
            self.rot_z += -self.movement.y;
            self.rot_z = self.rot_z.clamp(-89.0, 89.0);
            self.rot_x -= self.movement.x;
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
