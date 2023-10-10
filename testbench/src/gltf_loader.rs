use crate::utils;
use engine::{
    ImageResource, LightType, MasterMaterial, MaterialDescription, MaterialDomain,
    MaterialInstance, MaterialInstanceDescription, MaterialParameterOffsetSize, Mesh,
    MeshCreateInfo, MeshPrimitiveCreateInfo, RenderingPipeline, SamplerResource, Scene,
    ScenePrimitive, Texture, TextureImageView, TextureInput,
};
use gltf::image::Data;
use gltf::Document;
use gpu::{
    ComponentMapping, Filter, ImageAspectFlags, ImageCreateInfo, ImageSubresourceRange,
    ImageUsageFlags, ImageViewCreateInfo, ImageViewType, MemoryDomain, SamplerAddressMode,
    SamplerCreateInfo, VkGpu,
};
use nalgebra::{vector, Matrix4, Point3, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4};
use resource_map::{ResourceHandle, ResourceMap};
use std::collections::HashMap;
use std::mem::size_of;
use std::num::NonZeroU32;
use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct PbrProperties {
    pub base_color: Vector4<f32>,         // vec4
    pub metallic_roughness: Vector4<f32>, // vec4
    pub emissive_color: Vector4<f32>,     // vec3
}

pub struct GltfLoader {
    engine_scene: Scene,
}

pub struct GltfLoadOptions {}

struct LoadedTextures {
    white: ResourceHandle<Texture>,
    black: ResourceHandle<Texture>,
    all_textures: Vec<ResourceHandle<Texture>>,
}

impl GltfLoader {
    pub fn load<P: AsRef<Path>, R: RenderingPipeline>(
        path: P,
        gpu: &VkGpu,
        scene_renderer: &mut R,
        resource_map: &mut ResourceMap,
        _options: GltfLoadOptions,
    ) -> anyhow::Result<Self> {
        let (document, buffers, mut images) = gltf::import(path)?;

        let pbr_master = Self::create_master_pbr_material(gpu, scene_renderer, resource_map)?;
        let image_views = Self::load_images(gpu, resource_map, &mut images)?;
        let samplers = Self::load_samplers(gpu, resource_map, &document)?;
        let textures = Self::load_textures(gpu, resource_map, image_views, samplers, &document)?;
        let allocated_materials =
            Self::load_materials(gpu, resource_map, pbr_master, textures, &document)?;
        let meshes = Self::load_meshes(gpu, resource_map, &document, &buffers)?;

        let engine_scene = Self::build_engine_scene(document, allocated_materials, meshes);

        Ok(Self { engine_scene })
    }

    fn build_engine_scene(
        document: Document,
        allocated_materials: Vec<ResourceHandle<MaterialInstance>>,
        meshes: Vec<ResourceHandle<Mesh>>,
    ) -> Scene {
        let mut engine_scene = Scene::new();
        for scene in document.scenes() {
            for node in scene.nodes() {
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
                        shadow_setup: Some(engine::ShadowSetup {
                            importance: NonZeroU32::new(match light.kind() {
                                GltfLightKind::Directional => 5,
                                GltfLightKind::Point => 3,
                                GltfLightKind::Spot { .. } => 4,
                            })
                            .unwrap(),
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
                    engine_scene.add(ScenePrimitive {
                        mesh: meshes[mesh.index()].clone(),
                        materials,
                        transform,
                    });
                }
            }
        }
        engine_scene
    }

    fn load_meshes(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        document: &Document,
        buffers: &[gltf::buffer::Data],
    ) -> anyhow::Result<Vec<ResourceHandle<Mesh>>> {
        let mut meshes = vec![];
        for mesh in document.meshes() {
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
            meshes.push(resource_map.add(gpu_mesh));
        }
        Ok(meshes)
    }

    fn create_master_pbr_material<R: RenderingPipeline>(
        gpu: &VkGpu,
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
                fragment_module: &fragment_module,
                vertex_module: &vertex_module,
                texture_inputs: &[
                    TextureInput {
                        name: "base_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                    },
                    TextureInput {
                        name: "normal_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                    },
                    TextureInput {
                        name: "occlusion_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                    },
                    TextureInput {
                        name: "emissive_texture".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                    },
                    TextureInput {
                        name: "metallic_roughness".to_owned(),
                        format: gpu::ImageFormat::Rgba8,
                    },
                ],
                material_parameters: params,
            },
        )?;

        Ok(resource_map.add(pbr_master))
    }

    fn load_images(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        images: &mut [Data],
    ) -> anyhow::Result<Vec<ResourceHandle<TextureImageView>>> {
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
            let gpu_image = gpu.create_image(
                &image_create_info,
                MemoryDomain::DeviceLocal,
                Some(&gltf_image.pixels),
            )?;

            let gpu_image_view = gpu.create_image_view(&ImageViewCreateInfo {
                image: &gpu_image,
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
            let img_index = resource_map.add(ImageResource(gpu_image));
            allocated_images.push(img_index.clone());
            let view_index = resource_map.add(TextureImageView {
                image: img_index,
                view: gpu_image_view,
            });
            allocated_image_views.push(view_index);
        }
        Ok(allocated_image_views)
    }

    fn load_textures(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        allocated_image_views: Vec<ResourceHandle<TextureImageView>>,
        allocated_samplers: Vec<ResourceHandle<SamplerResource>>,
        document: &Document,
    ) -> anyhow::Result<LoadedTextures> {
        let mut all_textures = vec![];
        for texture in document.textures() {
            all_textures.push(resource_map.add(Texture {
                sampler: allocated_samplers[texture.sampler().index().unwrap_or(0)].clone(),
                image_view: allocated_image_views[texture.source().index()].clone(),
            }))
        }
        let white = Texture::new_with_data(
            gpu,
            resource_map,
            1,
            1,
            &[255, 255, 255, 255],
            Some("White texture"),
        )?;
        let white = resource_map.add(white);
        let black = Texture::new_with_data(
            gpu,
            resource_map,
            1,
            1,
            &[0, 0, 0, 255],
            Some("Black texture"),
        )?;
        let black = resource_map.add(black);

        Ok(LoadedTextures {
            white,
            black,
            all_textures,
        })
    }

    fn load_samplers(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        document: &Document,
    ) -> anyhow::Result<Vec<ResourceHandle<SamplerResource>>> {
        let mut allocated_samplers = vec![];
        for sampler in document.samplers() {
            let sam_desc = SamplerCreateInfo {
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
            let sam = gpu.create_sampler(&sam_desc)?;
            allocated_samplers.push(resource_map.add(SamplerResource(sam)))
        }

        if allocated_samplers.is_empty() {
            // add default sampler
            let sam = gpu.create_sampler(&SamplerCreateInfo {
                address_u: SamplerAddressMode::Repeat,
                address_v: SamplerAddressMode::Repeat,
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                ..Default::default()
            })?;
            allocated_samplers.push(resource_map.add(SamplerResource(sam)))
        }

        Ok(allocated_samplers)
    }

    fn load_materials(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        pbr_master: ResourceHandle<MasterMaterial>,
        textures: LoadedTextures,
        document: &Document,
    ) -> anyhow::Result<Vec<ResourceHandle<MaterialInstance>>> {
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

            let mut texture_inputs = HashMap::new();
            texture_inputs.insert("base_texture".to_owned(), base_texture.clone());
            texture_inputs.insert("normal_texture".to_owned(), normal_texture.clone());
            texture_inputs.insert("occlusion_texture".to_owned(), occlusion_texture.clone());
            texture_inputs.insert("emissive_texture".to_owned(), emissive_texture.clone());
            texture_inputs.insert("metallic_roughness".to_owned(), metallic_roughness.clone());

            let material_instance = MaterialInstance::create_instance(
                gpu,
                pbr_master.clone(),
                resource_map,
                &MaterialInstanceDescription {
                    name: &format!(
                        "PbrMaterial Instance #{}",
                        gltf_material.index().unwrap_or(0)
                    ),
                    texture_inputs,
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
            )?;
            let material_instance = resource_map.add(material_instance);
            allocated_materials.push(material_instance);
        }

        Ok(allocated_materials)
    }

    pub fn scene(&self) -> &engine::Scene {
        &self.engine_scene
    }

    pub fn scene_mut(&mut self) -> &mut engine::Scene {
        &mut self.engine_scene
    }
}
