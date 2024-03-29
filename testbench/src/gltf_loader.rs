use crate::utils;
use engine::math::shape::BoundingShape;
use engine::resource_map::{ResourceHandle, ResourceMap};
use engine::{
    LightType, MasterMaterial, MaterialDescription, MaterialDomain, MaterialInstance,
    MaterialInstanceDescription, MaterialParameterOffsetSize, Mesh, MeshCreateInfo,
    MeshPrimitiveCreateInfo, RenderScene, RenderingPipeline, ScenePrimitive, Texture, TextureInput,
    TextureSamplerSettings,
};
use gltf::image::Data;
use gltf::Document;
use gpu::{
    ComponentMapping, Filter, Gpu, ImageAspectFlags, ImageCreateInfo, ImageHandle,
    ImageSubresourceRange, ImageUsageFlags, ImageViewCreateInfo, ImageViewHandle, ImageViewType,
    MemoryDomain, SamplerAddressMode, ShaderStage,
};
use nalgebra::{point, vector, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3, Vector4};
use std::collections::HashMap;
use std::mem::size_of;
use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct PbrProperties {
    pub base_color: Vector4<f32>,         // vec4
    pub metallic_roughness: Vector4<f32>, // vec4
    pub emissive_color: Vector4<f32>,     // vec3
}

pub struct GltfLoader {
    engine_scene: RenderScene,
}

pub struct GltfLoadOptions {
    pub use_bvh: bool,
}

impl Default for GltfLoadOptions {
    fn default() -> Self {
        Self { use_bvh: true }
    }
}
struct LoadedTextures {
    white: ResourceHandle<Texture>,
    black: ResourceHandle<Texture>,
    all_textures: Vec<ResourceHandle<Texture>>,
}

impl GltfLoader {
    pub fn load<P: AsRef<Path>, R: RenderingPipeline>(
        path: P,
        gpu: &dyn Gpu,
        scene_renderer: &mut R,
        resource_map: &mut ResourceMap,
        options: GltfLoadOptions,
    ) -> anyhow::Result<Self> {
        let (document, buffers, mut images) = gltf::import(path)?;

        let pbr_master = Self::create_master_pbr_material(gpu, scene_renderer, resource_map)?;
        let (images, image_views) = Self::load_images(gpu, &mut images)?;
        let samplers = Self::load_samplers(&document)?;
        let textures =
            Self::load_textures(gpu, resource_map, images, image_views, samplers, &document)?;
        let allocated_materials = Self::load_materials(gpu, pbr_master, textures, &document)?;
        let meshes = Self::load_meshes(gpu, resource_map, &document, &buffers)?;

        let mut engine_scene = Self::build_engine_scene(document, allocated_materials, meshes);
        engine_scene.use_bvh = options.use_bvh;

        Ok(Self { engine_scene })
    }

    fn build_engine_scene(
        document: Document,
        allocated_materials: Vec<MaterialInstance>,
        meshes: Vec<(ResourceHandle<Mesh>, Point3<f32>, Point3<f32>)>,
    ) -> RenderScene {
        let mut engine_scene = RenderScene::new();
        for scene in document.scenes() {
            for node in scene.nodes() {
                handle_node(node, &mut engine_scene, &allocated_materials, &meshes);
            }
        }
        engine_scene
    }

    fn load_meshes(
        gpu: &dyn Gpu,
        resource_map: &mut ResourceMap,
        document: &Document,
        buffers: &[gltf::buffer::Data],
    ) -> anyhow::Result<Vec<(ResourceHandle<Mesh>, Point3<f32>, Point3<f32>)>> {
        let mut meshes = vec![];
        for mesh in document.meshes() {
            let mut min = point![0.0f32, 0.0, 0.0];
            let mut max = point![0.0f32, 0.0, 0.0];
            let mut primitive_create_infos = vec![];

            for prim in mesh.primitives() {
                let mut indices = vec![];
                let mut positions = vec![];
                let mut colors = vec![];
                let mut normals = vec![];
                let mut tangents = vec![];
                let mut uvs = vec![];
                let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
                if let Some(iter) = reader.read_indices() {
                    for idx in iter.into_u32() {
                        indices.push(idx);
                    }
                }
                if let Some(iter) = reader.read_positions() {
                    for vert in iter {
                        positions.push(vector![vert[0], vert[1], vert[2]]);
                        min.x = min.x.min(vert[0]);
                        min.y = min.y.min(vert[1]);
                        min.z = min.z.min(vert[2]);
                        max.x = max.x.max(vert[0]);
                        max.y = max.y.max(vert[1]);
                        max.z = max.z.max(vert[2]);
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
                primitive_create_infos.push(MeshPrimitiveCreateInfo {
                    positions,
                    indices,
                    colors,
                    normals,
                    tangents,
                    uvs,
                });
            }

            let label = format!("Mesh #{}", mesh.index());
            let create_info = MeshCreateInfo {
                label: Some(mesh.name().unwrap_or(&label)),
                primitives: &primitive_create_infos,
            };
            let gpu_mesh = Mesh::new(gpu, &create_info)?;
            meshes.push((resource_map.add(gpu_mesh), min, max));
        }
        Ok(meshes)
    }

    fn create_master_pbr_material<R: RenderingPipeline>(
        gpu: &dyn Gpu,
        scene_renderer: &mut R,
        resource_map: &mut ResourceMap,
    ) -> anyhow::Result<ResourceHandle<MasterMaterial>> {
        let vertex_module = utils::read_file_to_vk_module(gpu, "./shaders/vertex_deferred.spirv")?;
        let fragment_module =
            utils::read_file_to_vk_module(gpu, "./shaders/metallic_roughness_pbr.spirv")?;

        let mut params = HashMap::new();
        params.insert(
            "base_color".to_owned(),
            MaterialParameterOffsetSize {
                offset: 0,
                size: size_of::<Vector4<f32>>(),
            },
        );
        params.insert(
            "metallic_roughness".to_owned(),
            MaterialParameterOffsetSize {
                offset: size_of::<Vector4<f32>>(),
                size: size_of::<Vector4<f32>>(),
            },
        );
        params.insert(
            "emissive_color".to_owned(),
            MaterialParameterOffsetSize {
                offset: size_of::<Vector4<f32>>() * 2,
                size: size_of::<Vector4<f32>>(),
            },
        );
        let pbr_master = scene_renderer.create_material(
            gpu,
            MaterialDescription {
                name: "PbrMaterial",
                domain: MaterialDomain::Surface,
                fragment_module,
                vertex_module,
                texture_inputs: &[
                    TextureInput {
                        name: "base_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                        shader_stage: ShaderStage::FRAGMENT,
                    },
                    TextureInput {
                        name: "normal_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                        shader_stage: ShaderStage::FRAGMENT,
                    },
                    TextureInput {
                        name: "occlusion_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                        shader_stage: ShaderStage::FRAGMENT,
                    },
                    TextureInput {
                        name: "emissive_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                        shader_stage: ShaderStage::FRAGMENT,
                    },
                    TextureInput {
                        name: "metallic_roughness".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                        shader_stage: ShaderStage::FRAGMENT,
                    },
                ],
                material_parameters: params,
                parameter_shader_visibility: ShaderStage::FRAGMENT,
                cull_mode: gpu::CullMode::Back,
            },
        )?;

        Ok(resource_map.add(pbr_master))
    }

    fn load_images(
        gpu: &dyn Gpu,
        images: &mut [Data],
    ) -> anyhow::Result<(Vec<ImageHandle>, Vec<ImageViewHandle>)> {
        let mut allocated_images = vec![];
        let mut allocated_image_views = vec![];
        for (index, gltf_image) in images.iter_mut().enumerate() {
            let format = match gltf_image.format {
                gltf::image::Format::R8G8B8A8 => gpu::ImageFormat::Rgba8,
                gltf::image::Format::R8G8B8 => gpu::ImageFormat::Rgb8,
                gltf::image::Format::R32G32B32A32FLOAT => gpu::ImageFormat::RgbaFloat32,
                f => panic!("Unsupported format! {:?}", f),
            };
            let label = format!("glTF Image #{}", index);
            let image_create_info = ImageCreateInfo {
                label: Some(&label),
                width: gltf_image.width,
                height: gltf_image.height,
                depth: 1,
                format,
                usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                mips: 1,
                layers: 1,
                samples: gpu::SampleCount::Sample1,
            };
            let gpu_image = gpu.make_image(
                &image_create_info,
                MemoryDomain::DeviceLocal,
                Some(&gltf_image.pixels),
            )?;

            let gpu_image_view = gpu.make_image_view(&ImageViewCreateInfo {
                label: Some(&(label.to_string() + " - View")),
                image: gpu_image,
                view_type: ImageViewType::Type2D,
                format,
                components: ComponentMapping::default(),
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            })?;
            allocated_images.push(gpu_image);
            allocated_image_views.push(gpu_image_view);
        }
        Ok((allocated_images, allocated_image_views))
    }

    fn load_textures(
        gpu: &dyn Gpu,
        resource_map: &mut ResourceMap,
        allocated_images: Vec<ImageHandle>,
        allocated_image_views: Vec<ImageViewHandle>,
        allocated_samplers: Vec<TextureSamplerSettings>,
        document: &Document,
    ) -> anyhow::Result<LoadedTextures> {
        let mut all_textures = vec![];
        for texture in document.textures() {
            all_textures.push(resource_map.add(Texture {
                image: allocated_images[texture.source().index()],
                view: allocated_image_views[texture.source().index()],
                sampler_settings: allocated_samplers[texture.sampler().index().unwrap_or(0)],
            }))
        }
        let white = Texture::new_with_data(
            gpu,
            1,
            1,
            &[255, 255, 255, 255],
            Some("White texture"),
            gpu::ImageFormat::Rgba8,
            ImageViewType::Type2D,
        )?;
        let white = resource_map.add(white);
        let black = Texture::new_with_data(
            gpu,
            1,
            1,
            &[0, 0, 0, 255],
            Some("Black texture"),
            gpu::ImageFormat::Rgba8,
            ImageViewType::Type2D,
        )?;
        let black = resource_map.add(black);

        Ok(LoadedTextures {
            white,
            black,
            all_textures,
        })
    }

    fn load_samplers(document: &Document) -> anyhow::Result<Vec<TextureSamplerSettings>> {
        let mut allocated_samplers = vec![];
        for sampler in document.samplers() {
            let sam_desc = TextureSamplerSettings {
                address_u: match &sampler.wrap_s() {
                    gltf::texture::WrappingMode::ClampToEdge => SamplerAddressMode::ClampToEdge,
                    gltf::texture::WrappingMode::MirroredRepeat => {
                        SamplerAddressMode::MirroredRepeat
                    }
                    gltf::texture::WrappingMode::Repeat => SamplerAddressMode::Repeat,
                },
                address_v: match &sampler.wrap_t() {
                    gltf::texture::WrappingMode::ClampToEdge => SamplerAddressMode::ClampToEdge,
                    gltf::texture::WrappingMode::MirroredRepeat => {
                        SamplerAddressMode::MirroredRepeat
                    }
                    gltf::texture::WrappingMode::Repeat => SamplerAddressMode::Repeat,
                },
                mag_filter: match sampler
                    .mag_filter()
                    .unwrap_or(gltf::texture::MagFilter::Nearest)
                {
                    gltf::texture::MagFilter::Nearest => Filter::Nearest,
                    gltf::texture::MagFilter::Linear => Filter::Linear,
                },
                min_filter: match sampler
                    .min_filter()
                    .unwrap_or(gltf::texture::MinFilter::Nearest)
                {
                    gltf::texture::MinFilter::Nearest => Filter::Nearest,
                    gltf::texture::MinFilter::Linear => Filter::Linear,
                    x => {
                        log::warn!("glTF: unsupported filter! {:?}", x);
                        Filter::Linear
                    }
                },
                ..Default::default()
            };
            allocated_samplers.push(sam_desc)
        }

        if allocated_samplers.is_empty() {
            // add default sampler
            allocated_samplers.push(TextureSamplerSettings::default())
        }

        Ok(allocated_samplers)
    }

    fn load_materials(
        gpu: &dyn Gpu,
        pbr_master: ResourceHandle<MasterMaterial>,
        textures: LoadedTextures,
        document: &Document,
    ) -> anyhow::Result<Vec<MaterialInstance>> {
        let LoadedTextures {
            white,
            black,
            all_textures,
        } = textures;
        let mut allocated_materials = vec![];
        for gltf_material in document.materials() {
            let base_texture =
                if let Some(base) = gltf_material.pbr_metallic_roughness().base_color_texture() {
                    all_textures[base.texture().index()].clone()
                } else {
                    white.clone()
                };
            let normal_texture = if let Some(base) = gltf_material.normal_texture() {
                all_textures[base.texture().index()].clone()
            } else {
                white.clone()
            };
            let occlusion_texture = if let Some(base) = gltf_material.occlusion_texture() {
                all_textures[base.texture().index()].clone()
            } else {
                white.clone()
            };
            let emissive_texture = if let Some(base) = gltf_material.emissive_texture() {
                all_textures[base.texture().index()].clone()
            } else {
                black.clone()
            };
            let metallic_roughness = if let Some(base) = gltf_material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
            {
                all_textures[base.texture().index()].clone()
            } else {
                white.clone()
            };

            let textures = vec![
                base_texture.clone(),
                normal_texture.clone(),
                occlusion_texture.clone(),
                emissive_texture.clone(),
                metallic_roughness.clone(),
            ];

            let material_instance = MaterialInstance::create_instance(
                pbr_master.clone(),
                &MaterialInstanceDescription {
                    name: &format!(
                        "PbrMaterial Instance #{}",
                        gltf_material.index().unwrap_or(0),
                    ),
                    textures,
                    parameter_buffers: vec![MaterialInstance::create_material_parameter_buffer(
                        "PBR Data",
                        gpu,
                        std::mem::size_of::<PbrProperties>(),
                    )?],
                },
            )?;
            let metallic = gltf_material.pbr_metallic_roughness().metallic_factor();
            let roughness = gltf_material.pbr_metallic_roughness().roughness_factor();
            let emissive = gltf_material.emissive_factor();
            material_instance.write_parameters(
                gpu,
                PbrProperties {
                    base_color: Vector4::from_column_slice(
                        &gltf_material.pbr_metallic_roughness().base_color_factor(),
                    ),
                    metallic_roughness: vector![metallic, roughness, 0.0, 1.0],
                    emissive_color: vector![emissive[0], emissive[1], emissive[2], 1.0],
                },
                0,
            )?;
            allocated_materials.push(material_instance);
        }

        Ok(allocated_materials)
    }

    pub fn scene(&self) -> &engine::RenderScene {
        &self.engine_scene
    }

    pub fn scene_mut(&mut self) -> &mut engine::RenderScene {
        &mut self.engine_scene
    }
}

fn handle_node(
    node: gltf::Node<'_>,
    engine_scene: &mut RenderScene,
    allocated_materials: &Vec<MaterialInstance>,
    meshes: &Vec<(
        ResourceHandle<Mesh>,
        nalgebra::OPoint<f32, nalgebra::Const<3>>,
        nalgebra::OPoint<f32, nalgebra::Const<3>>,
    )>,
) {
    let node_transform = node.transform();
    let (pos, rot, scale) = node_transform.decomposed();
    let pos = Vector3::from_row_slice(&pos);
    let rot = Quaternion::new(rot[3], rot[0], rot[1], rot[2]);
    let rot = UnitQuaternion::from_quaternion(rot);

    let rot_matrix = rot.to_homogeneous();
    let forward = rot_matrix.column(2);

    // gltf specifies light directions to be in -z
    let forward = -vector![forward[0], forward[1], forward[2]];

    if let Some(light) = node.light() {
        use gltf::khr_lights_punctual::Kind as GltfLightKind;
        let light_type = match light.kind() {
            GltfLightKind::Directional => LightType::Directional {
                direction: forward,
                size: vector![20.0, 20.0],
            },
            GltfLightKind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => LightType::Spotlight {
                direction: forward,
                inner_cone_degrees: inner_cone_angle.to_degrees(),
                outer_cone_degrees: outer_cone_angle.to_degrees(),
            },
            GltfLightKind::Point => LightType::Point,
        };
        engine_scene.add_light(engine::Light {
            ty: light_type,
            position: Point3::from(pos),
            radius: 500.0,
            color: Vector3::from_row_slice(&light.color()),
            intensity: light.intensity() / 100.0,
            enabled: true,
            shadow_configuration: Some(engine::ShadowConfiguration {
                shadow_map_width: 512,
                shadow_map_height: 512,

                ..Default::default()
            }),
        });
    } else if let Some(mesh) = node.mesh() {
        let transform = Matrix4::new_translation(&pos)
            * Matrix4::new_nonuniform_scaling(&Vector3::from_row_slice(&scale))
            * rot_matrix;

        let mut materials = vec![];
        for prim in mesh.primitives() {
            let material_index = prim.material().index().unwrap_or(0);
            let material = allocated_materials[material_index].clone();
            materials.push(material);
        }

        let (mesh, min, max) = meshes[mesh.index()].clone();
        // TODO: consider the bounding volumes of all mesh primitives when constructing the scene primitive
        let bounds = BoundingShape::BoundingBox { min, max }.transformed(transform);
        engine_scene.add(ScenePrimitive {
            mesh,
            materials,
            transform,
            bounds,
        });
    }

    for node in node.children() {
        handle_node(node, engine_scene, allocated_materials, meshes);
    }
}
