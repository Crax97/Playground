mod master_material;
mod material_instance;
pub mod material_v2;

use std::collections::HashMap;

use gpu::{CullMode, ImageFormat, ShaderModuleHandle, ShaderStage};
pub use material_instance::*;

pub use master_material::*;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaterialDomain {
    Surface,
    PostProcess,
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum PipelineTarget {
    ColorAndDepth,
    DepthOnly,
    PostProcess,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct MaterialParameterOffsetSize {
    pub offset: usize,
    pub size: usize,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct TextureInput {
    pub name: String,
    pub format: ImageFormat,
    pub shader_stage: ShaderStage,
}

pub struct MaterialDescription<'a> {
    pub name: &'a str,
    pub domain: MaterialDomain,
    pub texture_inputs: &'a [TextureInput],
    pub material_parameters: HashMap<String, MaterialParameterOffsetSize>,
    pub parameter_shader_visibility: ShaderStage,
    pub fragment_module: ShaderModuleHandle,
    pub vertex_module: ShaderModuleHandle,
    pub cull_mode: CullMode,
}
