use std::{collections::HashMap, hash::Hash, mem::size_of, num::NonZeroU32};

use crate::resource_map::Resource;
use gpu::{
    BindingElement, BindingType, CullMode, FragmentStageInfo, FrontFace, LogicOp, PolygonMode,
    ShaderModuleHandle, ShaderStage, VertexStageInfo,
};

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
    pub fn new(description: &MasterMaterialDescription) -> anyhow::Result<Self> {
        let shader_permutations = Self::create_shader_permutations(description)?;
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
        description: &MasterMaterialDescription<'_>,
    ) -> anyhow::Result<HashMap<PipelineTarget, ShaderPermutation>> {
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
            MaterialDomain::Surface => Self::create_surface_permutations(description),
            MaterialDomain::PostProcess => Self::create_post_process_permutations(description),
        }
    }

    pub(crate) fn get_permutation(&self, target: PipelineTarget) -> Option<&ShaderPermutation> {
        self.shader_permutations.get(&target)
    }

    fn create_surface_permutations(
        description: &MasterMaterialDescription,
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
        description: &MasterMaterialDescription,
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
