use std::path::Path;

use anyhow::Ok;
use engine::{
    asset_map::{AssetHandle, AssetMap},
    assets::{
        material::{
            Material, MaterialDescription, MaterialDomain, MaterialParameters, MaterialProperties,
        },
        mesh::{Mesh, MeshDescription},
        texture::{Texture, TextureDescription, TextureSamplerConfiguration, TextureUsageFlags},
    },
    glam::{vec2, vec3, Mat4},
    include_spirv,
    math::Transform,
    sampler_allocator::SamplerAllocator,
    scene::{Scene, SceneMesh, SceneNode},
    shader_cache::ShaderCache,
};

use gltf::image::Data;
use mgpu::{Device, Extents2D, FilterMode, ShaderModuleDescription};

const VERTEX_SHADER: &[u8] = include_spirv!("../spirv/pbr_vertex.vert.spv");
const FRAGMENT_SHADER: &[u8] = include_spirv!("../spirv/pbr_fragment.frag.spv");

pub fn load(
    device: &Device,
    path: impl AsRef<Path>,
    sampler_allocator: &SamplerAllocator,
    shader_cache: &mut ShaderCache,
    asset_map: &mut AssetMap,
) -> anyhow::Result<Scene> {
    let (document, buffer_data, image_data) = gltf::import(path.as_ref())?;

    let textures = create_textures(&document, asset_map, sampler_allocator, image_data, device)?;
    let materials = create_materials(&document, asset_map, textures, shader_cache, device)?;
    let meshes = create_meshes(&document, buffer_data, asset_map, device)?;

    assemble_scene(document, materials, meshes)
}

fn assemble_scene(
    document: gltf::Document,
    materials: Vec<AssetHandle<Material>>,
    meshes: Vec<Vec<AssetHandle<Mesh>>>,
) -> Result<Scene, anyhow::Error> {
    let mut scene = Scene::default();
    for node in document.nodes() {
        if let Some(mesh) = node.mesh() {
            let transform = node.transform();
            let transform = Transform::from_matrix(Mat4::from_cols_array_2d(&transform.matrix()));
            let primitives = &meshes[mesh.index()];
            for (prim_idx, prim) in mesh.primitives().enumerate() {
                let mesh_info = SceneMesh {
                    handle: primitives[prim_idx].clone(),
                    material: materials[prim.material().index().unwrap()].clone(),
                };

                let scene_node = SceneNode::default()
                    .primitive(engine::scene::ScenePrimitive::Mesh(mesh_info))
                    .transform(transform);

                let _ = scene.add_node(scene_node);
            }
        }
    }

    Ok(scene)
}

fn create_materials(
    document: &gltf::Document,
    asset_map: &mut AssetMap,
    textures: Vec<AssetHandle<Texture>>,
    shader_cache: &mut ShaderCache,
    device: &Device,
) -> anyhow::Result<Vec<AssetHandle<Material>>> {
    let mut materials = vec![];
    let vertex_shader_module = device
        .create_shader_module(&ShaderModuleDescription {
            label: Some("PBR Vertex Shader"),
            source: bytemuck::cast_slice(VERTEX_SHADER),
        })
        .unwrap();
    let fragment_shader_module = device
        .create_shader_module(&ShaderModuleDescription {
            label: Some("PBR Fragment Shader"),
            source: bytemuck::cast_slice(FRAGMENT_SHADER),
        })
        .unwrap();
    shader_cache.add_shader("pbr_vertex", vertex_shader_module);
    shader_cache.add_shader("pbr_fragment", fragment_shader_module);

    for (idx, material) in document.materials().enumerate() {
        let pbr_info = material.pbr_metallic_roughness();
        let base_color_texture = pbr_info.base_color_texture().unwrap();
        let base_color_texture = textures[base_color_texture.texture().index()].clone();
        let identifier = format!("gltf.materials.{}", idx);
        let material = Material::new(
            device,
            &MaterialDescription {
                label: material.name(),
                vertex_shader: "pbr_vertex".into(),
                fragment_shader: "pbr_fragment".into(),
                parameters: MaterialParameters::default()
                    .texture_parameter("tex", base_color_texture),
                properties: MaterialProperties {
                    domain: MaterialDomain::Surface,
                },
            },
            asset_map,
            shader_cache,
        )?;
        let material = asset_map.add(material, &identifier);
        materials.push(material);
    }

    Ok(materials)
}

fn create_meshes(
    document: &gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    asset_map: &mut AssetMap,
    device: &Device,
) -> anyhow::Result<Vec<Vec<AssetHandle<Mesh>>>> {
    let mut meshes = vec![];
    for mesh in document.meshes() {
        let mut primitives = vec![];
        for (prim_idx, prim) in mesh.primitives().enumerate() {
            let identifier = format!("gltf.mesh-{}.primitive-{}", mesh.index(), prim_idx);
            let mut indices = vec![];
            let mut positions = vec![];
            let mut normals = vec![];
            let mut tangents = vec![];
            let mut uvs = vec![];
            let mut colors = vec![];

            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
            for idx in reader.read_indices().unwrap().into_u32() {
                indices.push(idx);
            }

            for pos in reader.read_positions().unwrap() {
                positions.push(vec3(pos[0], pos[1], pos[2]));
            }

            if let Some(gltf_normals) = reader.read_normals() {
                for normal in gltf_normals {
                    normals.push(vec3(normal[0], normal[1], normal[2]))
                }
            } else {
                normals.push(Default::default())
            }

            if let Some(gltf_tangents) = reader.read_tangents() {
                for tangent in gltf_tangents {
                    tangents.push(vec3(tangent[0], tangent[1], tangent[2]))
                }
            } else {
                tangents.push(Default::default())
            }

            if let Some(gltf_colors) = reader.read_colors(0) {
                for color in gltf_colors.into_rgb_f32() {
                    colors.push(vec3(color[0], color[1], color[2]))
                }
            } else {
                colors.push(Default::default())
            }

            if let Some(gltf_uvs) = reader.read_tex_coords(0) {
                for uv in gltf_uvs.into_f32() {
                    uvs.push(vec2(uv[0], uv[1]))
                }
            } else {
                uvs.push(Default::default())
            }

            let mesh = Mesh::new(
                device,
                &MeshDescription {
                    label: Some(&identifier),
                    indices: &indices,
                    positions: &positions,
                    normals: &normals,
                    tangents: &tangents,
                    uvs: &uvs,
                    colors: &colors,
                },
            )?;

            let mesh = asset_map.add(mesh, &identifier);
            primitives.push(mesh);
        }
        meshes.push(primitives)
    }
    Ok(meshes)
}

fn create_textures(
    document: &gltf::Document,
    asset_map: &mut AssetMap,
    sampler_allocator: &SamplerAllocator,
    image_data: Vec<Data>,
    device: &Device,
) -> anyhow::Result<Vec<AssetHandle<Texture>>> {
    let mut textures = vec![];
    for texture in document.textures() {
        let texture_image = texture.source();
        let image_data = &image_data[texture_image.index()];
        let sampler_info = texture.sampler();
        let sampler_configuration = get_sampler_configuration(sampler_info);
        let identifier = format!("gltf.texture.{}", texture.index());

        let mut data_ausiliary = vec![];
        let mut data_ref = image_data.pixels.as_slice();

        let mut format = image_data.format;
        if format == gltf::image::Format::R8G8B8 {
            // unsupported, add fourth channel
            for pixel in image_data.pixels.as_slice().chunks(3) {
                data_ausiliary.push(pixel[0]);
                data_ausiliary.push(pixel[1]);
                data_ausiliary.push(pixel[2]);
                data_ausiliary.push(255);
            }

            data_ref = &data_ausiliary;
            format = gltf::image::Format::R8G8B8A8;
        }

        let texture = Texture::new(
            device,
            &TextureDescription {
                label: texture.name(),
                data: &[data_ref],
                ty: engine::assets::texture::TextureType::D2(Extents2D {
                    width: image_data.width,
                    height: image_data.height,
                }),
                format: match format {
                    gltf::image::Format::R8 => mgpu::ImageFormat::R8,
                    gltf::image::Format::R8G8 => mgpu::ImageFormat::Rg8,
                    gltf::image::Format::R8G8B8A8 => mgpu::ImageFormat::Rgba8,
                    gltf::image::Format::R32G32B32A32FLOAT => mgpu::ImageFormat::Rgba32f,
                    _ => todo!("Add support for 16 bit images"),
                },
                usage_flags: TextureUsageFlags::default(),
                num_mips: Texture::compute_num_mips(image_data.width, image_data.height)
                    .try_into()
                    .unwrap(),
                auto_generate_mips: true,
                sampler_configuration,
            },
            sampler_allocator,
        )?;

        let handle = asset_map.add(texture, &identifier);
        textures.push(handle);
    }
    Ok(textures)
}

fn get_sampler_configuration(sampler_info: gltf::texture::Sampler) -> TextureSamplerConfiguration {
    let minmag_filter = match sampler_info.min_filter() {
        Some(filter) => match filter {
            gltf::texture::MinFilter::Nearest => FilterMode::Nearest,
            gltf::texture::MinFilter::Linear => FilterMode::Linear,
            _ => Default::default(),
        },
        None => Default::default(),
    };

    TextureSamplerConfiguration {
        minmag_filter,
        mipmap_mode: Default::default(),
    }
}
