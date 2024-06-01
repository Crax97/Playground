use std::io::{BufReader, Read};

use anyhow::Context;
use glam::{vec2, vec3, Vec3};
use mgpu::Extents2D;
use russimp::scene::PostProcess;
use serde::{Deserialize, Serialize};
use texture::{TextureDescription, TextureSamplerConfiguration, TextureUsageFlags};

use crate::{
    asset_map::{Asset, AssetMap},
    constants::{BLACK_TEXTURE_HANDLE, WHITE_TEXTURE_HANDLE},
    immutable_string::ImmutableString,
    shader_parameter_writer::ScalarMaterialParameter,
};

use self::{
    material::{Material, MaterialParameters, MaterialProperties, TextureMaterialParameter},
    mesh::Mesh,
    texture::Texture,
};

pub mod material;
pub mod mesh;
pub mod shader;
pub mod texture;

#[derive(Serialize, Deserialize, Default)]
pub struct MeshMetadata {
    pub source: String,
}

impl Asset for Mesh {
    type Metadata = MeshMetadata;
    fn asset_type_name() -> &'static str {
        "Mesh"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device.destroy_buffer(self.index_buffer).unwrap();
        device.destroy_buffer(self.position_component).unwrap();
        device.destroy_buffer(self.normal_component).unwrap();
        device.destroy_buffer(self.tangent_component).unwrap();
        device.destroy_buffer(self.uv_component).unwrap();
        device.destroy_buffer(self.color_component).unwrap();
    }

    fn import(base_id: &str, metadata: Self::Metadata, ctx: &mut AssetMap) -> anyhow::Result<()>
    where
        Self: Sized,
    {
        use russimp::scene::Scene as RussimpScene;
        let scene = RussimpScene::from_file(
            &metadata.source,
            vec![
                PostProcess::CalculateTangentSpace,
                PostProcess::Triangulate,
                PostProcess::GenerateNormals,
            ],
        )?;
        let mesh = &scene.meshes[0];

        let indices = mesh
            .faces
            .iter()
            .flat_map(|face| face.0.iter().copied())
            .collect::<Vec<_>>();

        let positions = mesh
            .vertices
            .iter()
            .map(|v| vec3(v.x, v.y, v.z))
            .collect::<Vec<_>>();

        let normals = mesh
            .normals
            .iter()
            .map(|v| vec3(v.x, v.y, v.z))
            .collect::<Vec<_>>();

        let tangents = mesh
            .tangents
            .iter()
            .map(|v| vec3(v.x, v.y, v.z))
            .collect::<Vec<_>>();

        let uvs = mesh.texture_coords[0]
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| vec2(v.x, v.y))
            .collect::<Vec<_>>();

        let colors = mesh.colors[0]
            .as_ref()
            .map(|c| c.iter().map(|v| vec3(v.r, v.g, v.b)).collect::<Vec<_>>())
            .unwrap_or(vec![Vec3::ZERO]);

        debug_assert!(
            !indices.is_empty(),
            "Meshes without index buffers aren't supported yet"
        );

        let desc = mesh::MeshDescription {
            label: Some(base_id),
            indices: &indices,
            positions: &positions,
            normals: &normals,
            tangents: &tangents,
            uvs: &uvs,
            colors: &colors,
        };

        let engine_mesh = Mesh::new(ctx.device(), &desc)?;

        ctx.add(engine_mesh, base_id);
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct TextureMetadata {
    pub source_path: String,
    pub sampler_configuration: TextureSamplerConfiguration,
}

impl Asset for Texture {
    type Metadata = TextureMetadata;
    fn asset_type_name() -> &'static str {
        "Texture"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device.destroy_image_view(self.view).unwrap();
        device.destroy_image(self.image).unwrap();
    }

    fn import(base_id: &str, metadata: Self::Metadata, ctx: &mut AssetMap) -> anyhow::Result<()>
    where
        Self: Sized,
    {
        let mut reader = BufReader::new(
            std::fs::File::open(&metadata.source_path)
                .with_context(|| metadata.source_path.clone())?,
        );
        let mut content = vec![];
        reader.read_to_end(&mut content)?;
        let image = image::load_from_memory(&content)?;
        let image_rgba_bytes = image.to_rgba8();

        let mips = Texture::compute_num_mips(image.width(), image.height());

        let texture = Texture::new(
            ctx.device(),
            &TextureDescription {
                label: Some(&metadata.source_path),
                data: &[&image_rgba_bytes],
                ty: crate::assets::texture::TextureType::D2(Extents2D {
                    width: image.width(),
                    height: image.height(),
                }),
                format: mgpu::ImageFormat::Rgba8,
                usage_flags: TextureUsageFlags::default(),
                num_mips: mips.try_into().unwrap(),
                auto_generate_mips: true,
                sampler_configuration: TextureSamplerConfiguration::default(),
            },
            ctx.sampler_allocator(),
        )?;

        ctx.add(texture, base_id);
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct MaterialMetadata {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub parameters: MaterialParameters,
    pub properties: MaterialProperties,
}

impl Default for MaterialMetadata {
    fn default() -> Self {
        Self {
            vertex_shader: Default::default(),
            fragment_shader: Default::default(),
            parameters: MaterialParameters {
                scalars: vec![
                    ScalarMaterialParameter {
                        name: "Foo".into(),
                        value: crate::shader_parameter_writer::ScalarParameterType::Vec2(
                            [1.0, 0.5].into(),
                        ),
                    },
                    ScalarMaterialParameter {
                        name: "Bar".into(),
                        value: crate::shader_parameter_writer::ScalarParameterType::Vec3(
                            [1.0, 0.5, 3.0].into(),
                        ),
                    },
                ],
                textures: vec![
                    TextureMaterialParameter {
                        name: "Tex1".into(),
                        texture: WHITE_TEXTURE_HANDLE.clone(),
                    },
                    TextureMaterialParameter {
                        name: "Tex2".into(),
                        texture: BLACK_TEXTURE_HANDLE.clone(),
                    },
                ],
            },
            properties: Default::default(),
        }
    }
}

impl Asset for Material {
    type Metadata = MaterialMetadata;
    fn asset_type_name() -> &'static str {
        "Material"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device
            .destroy_binding_set(self.binding_set.clone())
            .unwrap();
    }

    fn import(base_id: &str, metadata: Self::Metadata, map: &mut AssetMap) -> anyhow::Result<()>
    where
        Self: Sized,
    {
        let material = Material::new(
            &material::MaterialDescription {
                label: Some(base_id),
                vertex_shader: metadata.vertex_shader.into(),
                fragment_shader: metadata.fragment_shader.into(),
                parameters: metadata.parameters,
                properties: metadata.properties,
            },
            map,
        )?;

        map.add(material, base_id);
        Ok(())
    }
}
