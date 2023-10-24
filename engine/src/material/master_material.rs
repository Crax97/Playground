use std::{collections::HashMap, hash::Hash, mem::size_of, num::NonZeroU32};

use gpu::{
    BindingElement, BindingType, CompareOp, CullMode, DepthStencilState, FragmentStageInfo,
    FrontFace, GlobalBinding, GraphicsPipelineDescription, ImageFormat, LogicOp, PolygonMode,
    PushConstantRange, ShaderModuleHandle, ShaderStage, StencilOpState, VertexAttributeDescription,
    VertexBindingDescription, VertexStageInfo, VkGpu, VkGraphicsPipeline,
};
use nalgebra::{Vector2, Vector3};
use resource_map::Resource;

use crate::{MaterialDomain, MaterialParameterOffsetSize, PipelineTarget, TextureInput};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScalarType {
    Float,
    Uint,
    Int,
}

#[derive(Clone, Debug)]
pub struct MaterialParameter {
    pub name: String,
    pub ty: ScalarType,
    pub element_count: NonZeroU32,
}

pub struct MasterMaterialDescription<'a> {
    pub name: &'a str,
    pub domain: MaterialDomain,
    pub global_inputs: &'a [BindingType],
    pub texture_inputs: &'a [TextureInput],
    pub material_parameters: HashMap<String, MaterialParameterOffsetSize>,
    pub parameters_visibility: ShaderStage,
    pub vertex_info: &'a VertexStageInfo<'a>,
    pub fragment_info: &'a FragmentStageInfo<'a>,
    pub primitive_restart: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub logic_op: Option<LogicOp>,
    pub push_constant_ranges: &'a [PushConstantRange],
}

// Different pipeline targets may have different shader permutations
// E.g a depth target should have one vertex shader, but no fragment shaders
#[derive(Eq, PartialEq)]
pub struct ShaderPermutation {
    pub vertex_shader: ShaderModuleHandle,
    pub fragment_shader: Option<ShaderModuleHandle>,
}

#[derive(Eq, PartialEq)]
pub struct MasterMaterial {
    pub(crate) name: String,
    pub(crate) shader_permutations: HashMap<PipelineTarget, ShaderPermutation>,
    pub(crate) texture_inputs: Vec<TextureInput>,
    pub(crate) material_parameters: HashMap<String, MaterialParameterOffsetSize>,
    pub(crate) parameter_block_size: usize,
    pub(crate) parameter_shader_stages: ShaderStage,
}

impl Hash for MasterMaterial {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Resource for MasterMaterial {
    fn get_description(&self) -> &str {
        &self.name
    }
}

impl MasterMaterial {
    pub fn new(gpu: &VkGpu, description: &MasterMaterialDescription) -> anyhow::Result<Self> {
        let shader_permutations = Self::create_shader_permutations(gpu, description)?;
        let parameter_block_size = size_of::<f32>() * 4 * description.material_parameters.len();
        Ok(MasterMaterial {
            name: description.name.to_owned(),
            shader_permutations,
            texture_inputs: description.texture_inputs.to_vec(),
            material_parameters: description.material_parameters.clone(),
            parameter_block_size,
            parameter_shader_stages: description.parameters_visibility,
        })
    }

    fn create_shader_permutations(
        gpu: &VkGpu,
        description: &MasterMaterialDescription<'_>,
    ) -> anyhow::Result<HashMap<PipelineTarget, ShaderPermutation>> {
        let global_elements: Vec<_> = description
            .global_inputs
            .iter()
            .enumerate()
            .map(|(i, d)| BindingElement {
                binding_type: *d,
                index: i as _,
                stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
            })
            .collect();
        let mut user_elements: Vec<_> = description
            .texture_inputs
            .iter()
            .enumerate()
            .map(|(i, _)| BindingElement {
                binding_type: BindingType::CombinedImageSampler,
                index: i as _,
                stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
            })
            .collect();
        if !description.material_parameters.is_empty() {
            user_elements.push(BindingElement {
                binding_type: BindingType::Uniform,
                index: user_elements.len() as _,
                stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
            })
        }

        match description.domain {
            MaterialDomain::Surface => {
                Self::create_surface_permutations(gpu, description, global_elements, user_elements)
            }
            MaterialDomain::PostProcess => Self::create_post_process_permutations(
                gpu,
                description,
                global_elements,
                user_elements,
            ),
        }
    }

    fn get_inputs_for_material_domain(
        domain: &MaterialDomain,
    ) -> &'static [gpu::VertexBindingDescription<'static>] {
        static SURFACE_INPUTS: &[VertexBindingDescription] = &[
            VertexBindingDescription {
                binding: 0,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 0,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 1,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 1,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 2,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 2,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 3,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 3,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 4,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector2<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 4,
                    format: ImageFormat::RgFloat32,
                    offset: 0,
                }],
            },
        ];
        static PP_INPUTS: &[VertexBindingDescription] = &[
            // No inputs: the vertex shaders outputs vertices directly
        ];
        match domain {
            MaterialDomain::Surface => SURFACE_INPUTS,
            MaterialDomain::PostProcess => PP_INPUTS,
        }
    }

    pub(crate) fn get_permutation(&self, target: PipelineTarget) -> Option<&ShaderPermutation> {
        self.shader_permutations.get(&target)
    }

    fn create_surface_permutations(
        gpu: &VkGpu,
        description: &MasterMaterialDescription,
        global_elements: Vec<BindingElement>,
        user_elements: Vec<BindingElement>,
    ) -> anyhow::Result<HashMap<PipelineTarget, ShaderPermutation>> {
        let mut pipelines = HashMap::new();

        pipelines.insert(
            PipelineTarget::DepthOnly,
            ShaderPermutation {
                vertex_shader: description.vertex_info.module.clone(),
                fragment_shader: None,
            },
        );

        pipelines.insert(
            PipelineTarget::ColorAndDepth,
            ShaderPermutation {
                vertex_shader: description.vertex_info.module.clone(),
                fragment_shader: Some(description.fragment_info.module.clone()),
            },
        );
        Ok(pipelines)
    }
    fn create_post_process_permutations(
        gpu: &VkGpu,
        description: &MasterMaterialDescription,
        global_elements: Vec<BindingElement>,
        user_elements: Vec<BindingElement>,
    ) -> anyhow::Result<HashMap<PipelineTarget, ShaderPermutation>> {
        let mut pipelines = HashMap::new();
        pipelines.insert(
            PipelineTarget::PostProcess,
            ShaderPermutation {
                vertex_shader: description.vertex_info.module.clone(),
                fragment_shader: Some(description.fragment_info.module.clone()),
            },
        );

        Ok(pipelines)
    }
}
