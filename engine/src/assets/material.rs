use std::collections::{HashMap, HashSet};

use glam::{Vec2, Vec3, Vec4};
use mgpu::{
    Binding, BindingSet, BindingSetDescription, BindingSetElement, BindingSetElementKind,
    BindingSetLayout, BindingSetLayoutInfo, BindingType, Buffer, BufferDescription,
    BufferUsageFlags, CompareOp, CullMode, DepthStencilState, DepthStencilTargetInfo, Device,
    FragmentStageInfo, FrontFace, GraphicsPipeline, GraphicsPipelineDescription, ImageFormat,
    PolygonMode, PrimitiveTopology, PushConstantInfo, RenderTargetInfo, ShaderModule,
    ShaderStageFlags, StorageAccessMode, VariableType, VertexAttributeFormat,
    VertexInputDescription, VertexInputFrequency, VertexStageInfo,
};
use serde::{Deserialize, Serialize};

use crate::{
    asset_map::{AssetHandle, AssetMap},
    immutable_string::ImmutableString,
    scene_renderer::{self, SceneRenderer},
    shader_cache::ShaderCache,
};

use super::texture::Texture;

pub struct Material {
    pub properties: MaterialProperties,

    pub scalar_parameters_infos: Vec<ScalarParameterInfo>,
    pub texture_parameter_infos: Vec<TextureParameterInfo>,
    pub parameters: MaterialParameters,

    pub(crate) pipeline: GraphicsPipeline,
    // All the bindings in binding set 1 are user-settable parameters
    pub(crate) binding_set: BindingSet,
    // The stuff in binding (0,0) should be a struct
    pub(crate) scalar_parameter_buffer: Option<Buffer>,
}

pub struct MaterialDescription<'a> {
    pub label: Option<&'a str>,
    pub vertex_shader: ImmutableString,
    pub fragment_shader: ImmutableString,
    pub parameters: MaterialParameters,
    pub properties: MaterialProperties,
}

pub struct ScalarParameterInfo {
    pub name: String,
    pub offset: usize,
    pub ty: VertexAttributeFormat,
}

pub struct TextureParameterInfo {
    pub name: String,
    pub binding: usize,
    pub is_storage: bool,
    pub access_mode: StorageAccessMode,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum ScalarParameterType {
    Scalar(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ScalarMaterialParameter {
    pub name: String,
    pub texture: ScalarParameterType,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TextureMaterialParameter {
    pub name: String,
    pub texture: AssetHandle<Texture>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum MaterialDomain {
    Surface,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MaterialProperties {
    pub domain: MaterialDomain,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct MaterialParameters {
    scalars: Vec<ScalarMaterialParameter>,
    textures: Vec<TextureMaterialParameter>,
}

impl Material {
    pub fn new(
        device: &Device,
        description: &MaterialDescription,
        asset_map: &mut AssetMap,
        shader_cache: &mut ShaderCache,
    ) -> anyhow::Result<Material> {
        let mut user_bindings = HashMap::<usize, _>::default();
        let vertex_shader = shader_cache.get_shader_module(&description.vertex_shader);
        let fragment_shader = shader_cache.get_shader_module(&description.fragment_shader);
        let vertex_shader_layout = device.get_shader_module_layout(vertex_shader)?;
        let fragment_shader_layout = device.get_shader_module_layout(fragment_shader)?;

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
            binding_set_elements: user_bindings,
        };

        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDescription {
            label: Some("Graphics pipeline for material"),
            vertex_stage: &VertexStageInfo {
                shader: &vertex_shader,
                entry_point: "main",
                vertex_inputs: &[
                    VertexInputDescription {
                        location: 0,
                        stride: std::mem::size_of::<[f32; 3]>(),
                        offset: 0,
                        format: VertexAttributeFormat::Float3,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                    VertexInputDescription {
                        location: 1,
                        stride: std::mem::size_of::<[f32; 3]>(),
                        offset: 0,
                        format: VertexAttributeFormat::Float3,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                    VertexInputDescription {
                        location: 2,
                        stride: std::mem::size_of::<[f32; 3]>(),
                        offset: 0,
                        format: VertexAttributeFormat::Float3,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                    VertexInputDescription {
                        location: 3,
                        stride: std::mem::size_of::<[f32; 3]>(),
                        offset: 0,
                        format: VertexAttributeFormat::Float3,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                    VertexInputDescription {
                        location: 4,
                        stride: std::mem::size_of::<[f32; 2]>(),
                        offset: 0,
                        format: VertexAttributeFormat::Float2,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                ],
            },
            fragment_stage: Some(&FragmentStageInfo {
                shader: &fragment_shader,
                entry_point: "main",
                render_targets: &[
                    RenderTargetInfo {
                        format: ImageFormat::Rgba32f,
                        blend: None,
                    },
                    RenderTargetInfo {
                        format: ImageFormat::Rgba32f,
                        blend: None,
                    },
                    RenderTargetInfo {
                        format: ImageFormat::Rgba32f,
                        blend: None,
                    },
                    RenderTargetInfo {
                        format: ImageFormat::Rgba8,
                        blend: None,
                    },
                ],
                depth_stencil_target: Some(&DepthStencilTargetInfo {
                    format: ImageFormat::Depth32,
                }),
            }),
            primitive_restart_enabled: false,
            primitive_topology: PrimitiveTopology::TriangleList,
            polygon_mode: PolygonMode::Filled,
            cull_mode: CullMode::Back,
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
                    layout: SceneRenderer::scene_data_binding_set_layout(),
                },
                BindingSetLayoutInfo {
                    set: 1,
                    layout: user_binding_layout.clone(),
                },
            ],
            push_constant_info: Some(PushConstantInfo {
                size: std::mem::size_of::<scene_renderer::GPUPerObjectDrawData>(),
                visibility: ShaderStageFlags::ALL_GRAPHICS,
            }),
        })?;

        let mut user_scalars = vec![];
        let mut user_textures = vec![];
        for variable in vertex_shader_layout
            .variables
            .iter()
            .filter(|var| var.binding_index == 0 && var.binding_set == 1)
        {
            match variable.ty {
                VariableType::Field { format, offset } => {
                    user_scalars.push(ScalarParameterInfo {
                        name: variable
                            .name
                            .clone()
                            .expect("A shader variable doesn't have a name"),
                        ty: format,
                        offset,
                    });
                }
                _ => panic!(
                    "A user variable (binding set 1, binding 0) can only be composed of fileds"
                ),
            };
        }
        for variable in fragment_shader_layout
            .variables
            .iter()
            .filter(|var| var.binding_index == 0 && var.binding_set == 0)
        {
            match variable.ty {
                VariableType::Field { format, offset } => {
                    user_scalars.push(ScalarParameterInfo {
                        name: variable
                            .name
                            .clone()
                            .expect("A shader variable doesn't have a name"),
                        ty: format,
                        offset,
                    });
                }
                _ => panic!(
                    "A user variable (binding set 1, binding 0) can only be composed of fileds"
                ),
            };
        }
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

        let user_buffer = if !user_scalars.is_empty() {
            let mut size = 0;
            for scalar in &user_scalars {
                let field_size = match &scalar.ty {
                    VertexAttributeFormat::Int => 4,
                    VertexAttributeFormat::Int2 => 8,
                    VertexAttributeFormat::Int3 => 12,
                    VertexAttributeFormat::Int4 => 16,
                    VertexAttributeFormat::Uint => 4,
                    VertexAttributeFormat::Uint2 => 8,
                    VertexAttributeFormat::Uint3 => 12,
                    VertexAttributeFormat::Uint4 => 16,
                    VertexAttributeFormat::Float => 4,
                    VertexAttributeFormat::Float2 => 8,
                    VertexAttributeFormat::Float3 => 12,
                    VertexAttributeFormat::Float4 => 16,
                    VertexAttributeFormat::Mat2x2 => 8 * 8,
                    VertexAttributeFormat::Mat3x3 => 12 * 12,
                    VertexAttributeFormat::Mat4x4 => 16 * 16,
                };

                size += field_size;
            }
            let buffer = device.create_buffer(&BufferDescription {
                label: Some("Material user buffer"),
                usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM_BUFFER,
                size,
                memory_domain: mgpu::MemoryDomain::Gpu,
            })?;
            Some(buffer)
        } else {
            None
        };

        let bindings = user_textures
            .iter()
            .filter_map(|tex| {
                let tex_value = description
                    .parameters
                    .textures
                    .iter()
                    .find(|param| tex.name == param.name);
                tex_value.map(|param| {
                    let texture = asset_map.get(&param.texture).unwrap();
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
        let binding_set = device.create_binding_set(
            &BindingSetDescription {
                label: Some("Binding set for material"),
                bindings: &bindings,
            },
            &user_binding_layout,
        )?;

        Ok(Material {
            parameters: description.parameters.clone(),
            properties: description.properties,
            pipeline,
            binding_set,
            scalar_parameter_buffer: user_buffer,
            scalar_parameters_infos: user_scalars,
            texture_parameter_infos: user_textures,
        })
    }

    pub fn get_used_textures(&self) -> Vec<AssetHandle<Texture>> {
        let mut result = HashSet::new();
        for param in &self.parameters.textures {
            result.insert(param.texture.clone());
        }

        result.into_iter().collect()
    }
}

impl MaterialParameters {
    pub fn scalar_parameter(
        mut self,
        name: impl Into<String>,
        parameter: ScalarParameterType,
    ) -> Self {
        self.scalars.push(ScalarMaterialParameter {
            name: name.into(),
            texture: parameter,
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
