use std::collections::{HashMap, HashSet};

use crate::{shader_parameter_writer::ScalarParameterWriter, utils::shader_parameter_writer::*};

use mgpu::{
    Binding, BindingSet, BindingSetDescription, BindingSetElement, BindingSetElementKind,
    BindingSetLayout, BindingSetLayoutInfo, BindingType, BufferDescription, BufferUsageFlags,
    CompareOp, CullMode, DepthStencilState, DepthStencilTargetInfo, FragmentStageInfo, FrontFace,
    GraphicsPipeline, GraphicsPipelineDescription, ImageFormat, OwnedBindingSetLayout, PolygonMode,
    PrimitiveTopology, PushConstantInfo, RenderTargetInfo, ShaderStageFlags, StorageAccessMode,
    VertexAttributeFormat, VertexInputDescription, VertexInputFrequency, VertexStageInfo,
};
use serde::{Deserialize, Serialize};

use crate::{
    asset_map::{AssetHandle, AssetMap},
    immutable_string::ImmutableString,
    scene_renderer::{self, SceneRenderer},
};

use super::texture::Texture;

pub struct Material {
    pub properties: MaterialProperties,

    pub texture_parameter_infos: Vec<TextureParameterInfo>,
    pub parameters: MaterialParameters,

    pub(crate) pipeline: GraphicsPipeline,
    // All the bindings in binding set 1 are user-settable parameters
    pub(crate) binding_set: BindingSet,
    pub(crate) user_binding_set_layout: OwnedBindingSetLayout,
    // The stuff in set(1) binding (0) should be a struct
    pub(crate) scalar_parameter_writer: ScalarParameterWriter,
    user_buffer: Option<mgpu::Buffer>,
}

pub struct MaterialDescription<'a> {
    pub label: Option<&'a str>,
    pub vertex_shader: ImmutableString,
    pub fragment_shader: ImmutableString,
    pub parameters: MaterialParameters,
    pub properties: MaterialProperties,
}
#[derive(Debug)]
pub struct TextureParameterInfo {
    pub name: String,
    pub binding: usize,
    pub is_storage: bool,
    pub access_mode: StorageAccessMode,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TextureMaterialParameter {
    pub name: String,
    pub texture: AssetHandle<Texture>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub enum MaterialType {
    #[default]
    Lit,
    Unlit,
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub enum MaterialDomain {
    #[default]
    Surface,
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct MaterialProperties {
    pub domain: MaterialDomain,
    pub ty: MaterialType,
    pub double_sided: bool,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct MaterialParameters {
    pub scalars: Vec<ScalarMaterialParameter>,
    pub textures: Vec<TextureMaterialParameter>,
}

impl Material {
    pub fn new(
        description: &MaterialDescription,
        asset_map: &mut AssetMap,
    ) -> anyhow::Result<Material> {
        let mut user_bindings = HashMap::<usize, _>::default();
        let vertex_shader = asset_map
            .shader_cache()
            .get_shader_module(&description.vertex_shader);
        let fragment_shader = asset_map
            .shader_cache()
            .get_shader_module(&description.fragment_shader);
        let vertex_shader_layout = asset_map.device().get_shader_module_layout(vertex_shader)?;
        let fragment_shader_layout = asset_map
            .device()
            .get_shader_module_layout(fragment_shader)?;

        if let Some(user_binding) = vertex_shader_layout
            .binding_sets
            .iter()
            .find(|bs| bs.set == 1)
        {
            for binding in user_binding.layout.binding_set_elements.iter() {
                user_bindings
                    .entry(binding.binding)
                    .or_insert(BindingSetElement {
                        binding: binding.binding,
                        array_length: binding.array_length,
                        ty: binding.ty,
                        shader_stage_flags: binding.shader_stage_flags,
                    })
                    .shader_stage_flags |= binding.shader_stage_flags;
            }
        }

        if let Some(user_binding) = fragment_shader_layout
            .binding_sets
            .iter()
            .find(|bs| bs.set == 1)
        {
            for binding in user_binding.layout.binding_set_elements.iter() {
                user_bindings
                    .entry(binding.binding)
                    .or_insert(BindingSetElement {
                        binding: binding.binding,
                        array_length: binding.array_length,
                        ty: binding.ty,
                        shader_stage_flags: binding.shader_stage_flags,
                    })
                    .shader_stage_flags |= binding.shader_stage_flags;
            }
        }

        let mut user_bindings = user_bindings.into_values().collect::<Vec<_>>();
        user_bindings.sort_unstable_by_key(|b| b.binding);

        let user_binding_layout = BindingSetLayout {
            binding_set_elements: &user_bindings,
        };

        let pipeline =
            asset_map
                .device()
                .create_graphics_pipeline(&GraphicsPipelineDescription {
                    label: Some("Graphics pipeline for material"),
                    vertex_stage: &VertexStageInfo {
                        shader: &vertex_shader,
                        entry_point: "main",
                        vertex_inputs: mesh_vertex_inputs(),
                    },
                    fragment_stage: Some(&FragmentStageInfo {
                        shader: &fragment_shader,
                        entry_point: "main",
                        render_targets: &[
                            RenderTargetInfo { blend: None },
                            RenderTargetInfo { blend: None },
                            RenderTargetInfo { blend: None },
                            RenderTargetInfo { blend: None },
                            RenderTargetInfo { blend: None },
                        ],
                        depth_stencil_target: Some(&DepthStencilTargetInfo {
                            format: ImageFormat::Depth32,
                        }),
                    }),
                    primitive_restart_enabled: false,
                    primitive_topology: PrimitiveTopology::TriangleList,
                    polygon_mode: PolygonMode::Filled,
                    cull_mode: if description.properties.double_sided {
                        CullMode::None
                    } else {
                        CullMode::Back
                    },
                    front_face: FrontFace::CounterClockWise,
                    multisample_state: None,
                    depth_stencil_state: DepthStencilState {
                        depth_test_enabled: true,
                        depth_write_enabled: true,
                        depth_compare_op: CompareOp::Less,
                    },
                    binding_set_layouts: &[
                        BindingSetLayoutInfo {
                            set: 0,
                            layout: SceneRenderer::per_object_scene_binding_set_layout(),
                        },
                        BindingSetLayoutInfo {
                            set: 1,
                            layout: &user_binding_layout,
                        },
                    ],
                    push_constant_info: Some(PushConstantInfo {
                        size: std::mem::size_of::<scene_renderer::GPUPerObjectDrawData>(),
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    }),
                })?;

        let mut user_textures = vec![];

        let mut texture_iterator = user_binding_layout
            .binding_set_elements
            .iter()
            .filter(|binding| binding.binding != 0);
        while let Some(bs) = texture_iterator.next() {
            match &bs.ty {
                BindingSetElementKind::Sampler => {
                    continue;
                }
                BindingSetElementKind::CombinedImageSampler { .. } => {
                    todo!("Combined image samplers are unsupported for now in materials")
                }
                BindingSetElementKind::SampledImage => {
                    let name = vertex_shader_layout
                        .variables
                        .iter()
                        .chain(fragment_shader_layout.variables.iter())
                        .filter(|var| var.binding_set == 1)
                        .find(|var| var.binding_index == bs.binding)
                        .unwrap()
                        .name
                        .clone()
                        .expect("Texture parameter without name");
                    user_textures.push(TextureParameterInfo {
                        name,
                        binding: bs.binding,
                        is_storage: false,
                        access_mode: StorageAccessMode::Read,
                    });
                }
                BindingSetElementKind::StorageImage { access_mode, .. } => {
                    let name = vertex_shader_layout
                        .variables
                        .iter()
                        .chain(fragment_shader_layout.variables.iter())
                        .filter(|var| var.binding_set == 1)
                        .find(|var| var.binding_index == bs.binding)
                        .unwrap()
                        .name
                        .clone()
                        .expect("Texture parameter without name");
                    user_textures.push(TextureParameterInfo {
                        name,
                        binding: bs.binding,
                        is_storage: true,
                        access_mode: *access_mode,
                    });
                }

                BindingSetElementKind::Unknown => panic!("Unknown binding set element in bs 1"),
                BindingSetElementKind::Buffer { .. } => {
                    panic!("Bindings > 0 in set 1 cannot be buffers")
                }
            }

            let next = texture_iterator.next();

            assert!(
                next.is_some_and(|bs| matches!(bs.ty, BindingSetElementKind::Sampler)),
                "The user textures must come in pair: for each texture there must be a sampler, it was {:?}",
                bs.ty
            );
        }

        let mut scalar_parameter_writer = ScalarParameterWriter::new(
            asset_map.device(),
            &[&vertex_shader_layout, &fragment_shader_layout],
            1,
            0,
        )?;
        let user_buffer = if !scalar_parameter_writer.scalar_infos.is_empty() {
            Some(asset_map.device().create_buffer(&BufferDescription {
                label: Some("Material user buffer"),
                usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM_BUFFER,
                size: scalar_parameter_writer.binary_blob.len(),
                memory_domain: mgpu::MemoryDomain::Gpu,
            })?)
        } else {
            None
        };

        let binding_set = create_binding_set(
            &user_textures,
            &description.parameters.textures,
            asset_map,
            user_buffer,
            &user_binding_layout,
        )?;

        for parameter in &description.parameters.scalars {
            scalar_parameter_writer.write(&parameter.name, parameter.value);
        }
        if let Some(buffer) = &user_buffer {
            scalar_parameter_writer.update_buffer(asset_map.device(), *buffer)?;
        }

        Ok(Material {
            parameters: description.parameters.clone(),
            properties: description.properties,
            pipeline,
            binding_set,
            texture_parameter_infos: user_textures,
            scalar_parameter_writer,
            user_buffer,
            user_binding_set_layout: OwnedBindingSetLayout {
                binding_set_elements: user_binding_layout.binding_set_elements.to_vec(),
            },
        })
    }

    pub fn recreate_binding_set_layout(&mut self, asset_map: &mut AssetMap) -> anyhow::Result<()> {
        let bs = create_binding_set(
            &self.texture_parameter_infos,
            &self.parameters.textures,
            asset_map,
            self.user_buffer,
            &BindingSetLayout {
                binding_set_elements: &self.user_binding_set_layout.binding_set_elements,
            },
        )?;
        let old = std::mem::replace(&mut self.binding_set, bs);
        asset_map.device().destroy_binding_set(old)?;
        Ok(())
    }

    pub fn get_used_textures(&self) -> Vec<AssetHandle<Texture>> {
        let mut result = HashSet::new();
        for param in &self.parameters.textures {
            result.insert(param.texture.clone());
        }

        result.into_iter().collect()
    }
}

fn create_binding_set(
    user_textures: &[TextureParameterInfo],
    textures: &[TextureMaterialParameter],
    asset_map: &mut AssetMap,
    user_buffer: Option<mgpu::Buffer>,
    user_binding_layout: &BindingSetLayout,
) -> Result<BindingSet, anyhow::Error> {
    let bindings = user_textures
        .iter()
        .filter_map(|tex| {
            let tex_value = textures.iter().find(|param| tex.name == param.name);
            tex_value.map(|param| {
                let texture = asset_map.load(&param.texture);
                [
                    Binding {
                        binding: tex.binding,
                        ty: if tex.is_storage {
                            BindingType::StorageImage {
                                view: texture.view,
                                access_mode: tex.access_mode,
                            }
                        } else {
                            BindingType::SampledImage { view: texture.view }
                        },
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: tex.binding + 1,
                        ty: BindingType::Sampler(texture.sampler),
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                ]
                .into_iter()
            })
        })
        .flatten()
        .chain(user_buffer.iter().map(|&buf| Binding {
            binding: 0,
            ty: BindingType::UniformBuffer {
                buffer: buf,
                offset: 0,
                range: buf.size(),
            },
            visibility: ShaderStageFlags::ALL_GRAPHICS,
        }))
        .collect::<Vec<_>>();
    let binding_set = asset_map.device().create_binding_set(
        &BindingSetDescription {
            label: Some("Binding set for material"),
            bindings: &bindings,
        },
        user_binding_layout,
    )?;
    Ok(binding_set)
}

pub(crate) fn mesh_vertex_inputs() -> &'static [VertexInputDescription] {
    const VERTEX_INPUTS: &[VertexInputDescription] = &[
        VertexInputDescription {
            binding: 0,
            location: 0,
            stride: std::mem::size_of::<[f32; 3]>(),
            offset: 0,
            format: VertexAttributeFormat::Float3,
            frequency: VertexInputFrequency::PerVertex,
        },
        VertexInputDescription {
            binding: 1,
            location: 1,
            stride: std::mem::size_of::<[f32; 3]>(),
            offset: 0,
            format: VertexAttributeFormat::Float3,
            frequency: VertexInputFrequency::PerVertex,
        },
        VertexInputDescription {
            binding: 2,
            location: 2,
            stride: std::mem::size_of::<[f32; 3]>(),
            offset: 0,
            format: VertexAttributeFormat::Float3,
            frequency: VertexInputFrequency::PerVertex,
        },
        VertexInputDescription {
            binding: 3,
            location: 3,
            stride: std::mem::size_of::<[f32; 3]>(),
            offset: 0,
            format: VertexAttributeFormat::Float3,
            frequency: VertexInputFrequency::PerVertex,
        },
        VertexInputDescription {
            binding: 4,
            location: 4,
            stride: std::mem::size_of::<[f32; 2]>(),
            offset: 0,
            format: VertexAttributeFormat::Float2,
            frequency: VertexInputFrequency::PerVertex,
        },
    ];
    VERTEX_INPUTS
}

impl MaterialParameters {
    pub fn scalar_parameter(
        mut self,
        name: impl Into<String>,
        parameter: impl Into<ScalarParameterType>,
    ) -> Self {
        self.scalars.push(ScalarMaterialParameter {
            name: name.into(),
            value: parameter.into(),
        });
        self
    }

    pub fn texture_parameter(
        mut self,
        name: impl Into<String>,
        texture: AssetHandle<Texture>,
    ) -> Self {
        self.textures.push(TextureMaterialParameter {
            name: name.into(),
            texture,
        });
        self
    }
}
