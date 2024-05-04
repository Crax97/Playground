use std::path::Path;

use mgpu::{Device, ShaderModule};

use crate::assets::shader::Shader;

pub struct ShaderCache {}

impl ShaderCache {
    pub fn new(device: &Device, shader_path: &Path) -> anyhow::Result<Self> {
        todo!()
    }

    pub fn get_shader_module(
        &mut self,
        device: &Device,
        shader: &Shader,
    ) -> anyhow::Result<ShaderModule> {
        todo!()
    }
}
