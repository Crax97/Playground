use egui::ahash::HashMap;
use gpu::ShaderModuleHandle;
use serde::{Deserialize, Serialize};

use crate::{immutable_string::ImmutableString, Asset, AssetHandle, MaterialDomain, Texture};

pub struct Shader {
    name: ImmutableString,
    handle: ShaderModuleHandle,
}

#[derive(Serialize, Deserialize)]
pub enum MaterialParameter {
    Color([f32; 4]),
    Float(f32),
    Texture(AssetHandle<Texture>),
}

#[derive(Serialize, Deserialize)]
pub struct Material2 {
    pub name: ImmutableString,
    pub vertex_shader: AssetHandle<Shader>,
    pub fragment_shader: AssetHandle<Shader>,
    pub domain: MaterialDomain,
    pub parameters: HashMap<String, MaterialParameter>,
}

impl Asset for Shader {
    fn get_description(&self) -> &str {
        &self.name
    }

    fn destroyed(&mut self, gpu: &dyn gpu::Gpu) {
        gpu.destroy_shader_module(self.handle)
    }
}

impl Asset for Material2 {
    fn get_description(&self) -> &str {
        &self.name
    }

    fn destroyed(&mut self, _gpu: &dyn gpu::Gpu) {}
}
