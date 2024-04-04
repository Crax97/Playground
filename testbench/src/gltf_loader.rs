use crate::utils;
use engine::asset_map::{AssetHandle, AssetMap};
use engine::components::Transform;
use engine::material::{Material, MaterialBuilder, Shader};
use engine::math::shape::BoundingShape;
use engine::{
    GameScene, LightType, MaterialDomain, Mesh, MeshCreateInfo, MeshPrimitiveCreateInfo,
    RenderingPipeline, SceneMesh, Texture, TextureSamplerSettings,
};
use gltf::image::Data;
use gltf::Document;
use gpu::{
    ComponentMapping, Filter, Gpu, ImageAspectFlags, ImageCreateInfo, ImageHandle,
    ImageSubresourceRange, ImageUsageFlags, ImageViewCreateInfo, ImageViewHandle, ImageViewType,
    MemoryDomain, SamplerAddressMode,
};
use nalgebra::{point, vector, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3, Vector4};

use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct PbrProperties {
    pub base_color: Vector4<f32>,         // vec4
    pub metallic_roughness: Vector4<f32>, // vec4
    pub emissive_color: Vector4<f32>,     // vec3
}

pub struct GltfLoader {
    engine_scene: GameScene,
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
    white: AssetHandle<Texture>,
    black: AssetHandle<Texture>,
    all_textures: Vec<AssetHandle<Texture>>,
}

impl GltfLoader {
    pub fn load<P: AsRef<Path>, R: RenderingPipeline>(
        path: P,
        gpu: &dyn Gpu,
        _scene_renderer: &mut R,
        resource_map: &mut AssetMap,
        options: GltfLoadOptions,
    ) -> anyhow::Result<GameScene> {
        let (document, buffers, mut images) = gltf::import(path)?;

        let vertex_module = utils::read_file_to_vk_module(gpu, "./shaders/vertex_deferred.spirv")?;
        let fragment_module =
            utils::read_file_to_vk_module(gpu, "./shaders/metallic_roughness_pbr.spirv")?;
        let vertex_module = resource_map.add(
            Shader {
                name: "PBR Vertex".into(),
                handle: vertex_module,
            },
            Some("PBR Vertex"),
        );

        let fragment_module = resource_map.add(
            Shader {
                name: "PBR Fragment".into(),
                handle: fragment_module,
            },
            Some("PBR Fragment"),
        );
        let (images, image_views) = Self::load_images(gpu, &mut images)?;
        let samplers = Self::load_samplers(&document)?;
        let textures =
            Self::load_textures(gpu, resource_map, images, image_views, samplers, &document)?;
        let allocated_materials = Self::load_materials(
            gpu,
            vertex_module,
            fragment_module,
            resource_map,
            textures,
            &document,
        )?;
        let meshes = Self::load_meshes(gpu, resource_map, &document, &buffers)?;

        let mut engine_scene = Self::build_engine_scene(document, allocated_materials, meshes);
        engine_scene.use_bvh = options.use_bvh;

        Ok(engine_scene)
    }

    fn build_engine_scene(
        document: Document,
        allocated_materials: Vec<AssetHandle<Material>>,
        meshes: Vec<(AssetHandle<Mesh>, Point3<f32>, Point3<f32>)>,
    ) -> GameScene {
        let mut engine_scene = GameScene::new();
        for scene in document.scenes() {
            for node in scene.nodes() {
                handle_node(node, &mut engine_scene, &allocated_materials, &meshes);
            }
        }
        engine_scene
    }

    fn load_meshes(
        gpu: &dyn Gpu,
        resource_map: &mut AssetMap,
        document: &Document,
        buffers: &[gltf::buffer::Data],
    ) -> anyhow::Result<Vec<(AssetHandle<Mesh>, Point3<f32>, Point3<f32>)>> {
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
            meshes.push((resource_map.add(gpu_mesh, Some(&label)), min, max));
        }
        Ok(meshes)
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
        resource_map: &mut AssetMap,
        allocated_images: Vec<ImageHandle>,
        allocated_image_views: Vec<ImageViewHandle>,
        allocated_samplers: Vec<TextureSamplerSettings>,
        document: &Document,
    ) -> anyhow::Result<LoadedTextures> {
        let mut all_textures = vec![];
        for texture in document.textures() {
            all_textures.push(resource_map.add(
                Texture {
                    image: allocated_images[texture.source().index()],
                    view: allocated_image_views[texture.source().index()],
                    sampler_settings: allocated_samplers[texture.sampler().index().unwrap_or(0)],
                },
                texture.name(),
            ))
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
        let white = resource_map.add(white, Some("white texture"));
        let black = Texture::new_with_data(
            gpu,
            1,
            1,
            &[0, 0, 0, 255],
            Some("Black texture"),
            gpu::ImageFormat::Rgba8,
            ImageViewType::Type2D,
        )?;
        let black = resource_map.add(black, Some("white texture"));

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
        _gpu: &dyn Gpu,
        pbr_vertex: AssetHandle<Shader>,
        pbr_fragment: AssetHandle<Shader>,
        asset_map: &mut AssetMap,
        textures: LoadedTextures,
        document: &Document,
    ) -> anyhow::Result<Vec<AssetHandle<Material>>> {
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

            let _textures = vec![
                base_texture.clone(),
                normal_texture.clone(),
                occlusion_texture.clone(),
                emissive_texture.clone(),
                metallic_roughness.clone(),
            ];

            let name = gltf_material.name().unwrap_or("GLTF Material");
            let material_instance = MaterialBuilder::new(
                pbr_vertex.clone(),
                pbr_fragment.clone(),
                MaterialDomain::Surface,
            )
            .name(name);
            let metallic = gltf_material.pbr_metallic_roughness().metallic_factor();
            let roughness = gltf_material.pbr_metallic_roughness().roughness_factor();
            let emissive = gltf_material.emissive_factor();
            let material_instance = material_instance
                .parameter(
                    "baseColorSampler",
                    engine::material::MaterialParameter::Texture(base_texture),
                )
                .parameter(
                    "normalSampler",
                    engine::material::MaterialParameter::Texture(normal_texture),
                )
                .parameter(
                    "occlusionSampler",
                    engine::material::MaterialParameter::Texture(occlusion_texture),
                )
                .parameter(
                    "emissiveSampler",
                    engine::material::MaterialParameter::Texture(emissive_texture),
                )
                .parameter(
                    "metallicRoughnessSampler",
                    engine::material::MaterialParameter::Texture(metallic_roughness.clone()),
                )
                .parameter(
                    "pbrProperties.baseColor",
                    engine::material::MaterialParameter::Color([1.0, 1.0, 1.0, 1.0]),
                )
                .parameter(
                    "pbrProperties.metallicRoughness",
                    engine::material::MaterialParameter::Color([metallic, roughness, 0.0, 0.0]),
                )
                .parameter(
                    "pbrProperties.emissiveFactor",
                    engine::material::MaterialParameter::Color([
                        emissive[0],
                        emissive[1],
                        emissive[2],
                        0.0,
                    ]),
                );
            let material = asset_map.add(material_instance.build(), Some(name));
            allocated_materials.push(material);
        }

        Ok(allocated_materials)
    }
}

fn handle_node(
    node: gltf::Node<'_>,
    engine_scene: &mut GameScene,
    allocated_materials: &Vec<AssetHandle<Material>>,
    meshes: &Vec<(
        AssetHandle<Mesh>,
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
    let _forward = -vector![forward[0], forward[1], forward[2]];

    if let Some(light) = node.light() {
        use gltf::khr_lights_punctual::Kind as GltfLightKind;
        let light_type = match light.kind() {
            GltfLightKind::Directional => LightType::Directional {
                size: vector![20.0, 20.0],
            },
            GltfLightKind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => LightType::Spotlight {
                inner_cone_degrees: inner_cone_angle.to_degrees(),
                outer_cone_degrees: outer_cone_angle.to_degrees(),
            },
            GltfLightKind::Point => LightType::Point,
        };
        engine_scene.add_light(
            engine::SceneLightInfo {
                ty: light_type,
                radius: 500.0,
                color: Vector3::from_row_slice(&light.color()),
                intensity: light.intensity() / 100.0,
                enabled: true,
                shadow_configuration: Some(engine::ShadowConfiguration {
                    shadow_map_width: 512,
                    shadow_map_height: 512,

                    ..Default::default()
                }),
            },
            Transform {
                position: Point3::from(pos),
                rotation: rot,
                ..Default::default()
            },
            light.name().map(|s| s.to_string()),
        );
    } else if let Some(mesh) = node.mesh() {
        let transform = Transform {
            position: Point3::from(pos),
            scale: Vector3::from(scale),
            rotation: UnitQuaternion::from_matrix(&rot_matrix.fixed_resize(0.0)),
        };
        let transform_mat = Matrix4::new_translation(&pos)
            * Matrix4::new_nonuniform_scaling(&Vector3::from_row_slice(&scale))
            * rot_matrix;

        let mut materials = vec![];
        for prim in mesh.primitives() {
            let material_index = prim.material().index().unwrap_or(0);
            let material = allocated_materials[material_index].clone();
            materials.push(material);
        }

        let (mesh_handle, min, max) = meshes[mesh.index()].clone();
        // TODO: consider the bounding volumes of all mesh primitives when constructing the scene primitive
        let bounds = BoundingShape::BoundingBox { min, max }.transformed(transform_mat);
        engine_scene.add_mesh(
            SceneMesh {
                mesh: mesh_handle,
                materials,
                bounds,
            },
            transform,
            mesh.name().map(|s| s.to_string()),
        );
    }

    for node in node.children() {
        handle_node(node, engine_scene, allocated_materials, meshes);
    }
}
