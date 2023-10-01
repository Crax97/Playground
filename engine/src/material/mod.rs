mod master_material;
mod material_instance;

use std::collections::HashMap;

use gpu::{ImageFormat, VkShaderModule};
pub use material_instance::*;

pub use master_material::*;
pub use material_instance::*;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
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
}

pub struct MaterialDescription<'a> {
    pub name: &'a str,
    pub domain: MaterialDomain,
    pub texture_inputs: &'a [TextureInput],
    pub material_parameters: HashMap<String, MaterialParameterOffsetSize>,
    pub fragment_module: &'a VkShaderModule,
    pub vertex_module: &'a VkShaderModule,
}
