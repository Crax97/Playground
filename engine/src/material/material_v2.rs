use std::{collections::HashMap, time::SystemTime};

use gpu::{Gpu, ShaderModuleHandle};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{immutable_string::ImmutableString, Asset, AssetHandle, MaterialDomain, Texture};

pub const MATERIAL_PARAMETER_SLOT: u32 = 1;

pub struct Shader {
    pub name: ImmutableString,
    pub handle: ShaderModuleHandle,
}

#[derive(Serialize, Deserialize)]
pub enum MaterialParameter {
    Color([f32; 4]),
    Float(f32),
    Texture(AssetHandle<Texture>),
}

#[derive(Serialize, Deserialize)]
pub struct Material2 {
    pub(crate) uuid: Uuid,
    pub name: ImmutableString,
    pub vertex_shader: AssetHandle<Shader>,
    pub fragment_shader: AssetHandle<Shader>,
    pub domain: MaterialDomain,
    pub(crate) parameters: HashMap<String, MaterialParameter>,

    #[serde(skip)]
    pub(crate) last_tick_change: u128,
}

pub struct MaterialBuilder {
    name: Option<ImmutableString>,
    vertex_shader: AssetHandle<Shader>,
    fragment_shader: AssetHandle<Shader>,
    domain: MaterialDomain,
    parameters: HashMap<String, MaterialParameter>,
}

impl MaterialBuilder {
    pub fn new(
        vertex_shader: AssetHandle<Shader>,
        fragment_shader: AssetHandle<Shader>,
        domain: MaterialDomain,
    ) -> Self {
        Self {
            vertex_shader,
            fragment_shader,
            domain,

            parameters: HashMap::new(),
            name: None,
        }
    }

    pub fn name(mut self, name: impl Into<ImmutableString>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn parameter(mut self, name: impl Into<String>, parameter: MaterialParameter) -> Self {
        self.parameters.insert(name.into(), parameter);
        self
    }
}

impl Material2 {
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
    pub fn set_parameter(&mut self, name: impl Into<String>, parameter: MaterialParameter) {
        self.parameters.insert(name.into(), parameter);
        self.last_tick_change = std::time::SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Failed to compute time from epoch")
            .as_millis();
    }
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
