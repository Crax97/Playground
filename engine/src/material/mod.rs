use std::collections::HashMap;

use gpu::ShaderModuleHandle;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{immutable_string::ImmutableString, Asset, AssetHandle, Texture, Tick};

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
pub struct Material {
    pub(crate) uuid: Uuid,
    pub name: ImmutableString,
    pub vertex_shader: AssetHandle<Shader>,
    pub fragment_shader: AssetHandle<Shader>,
    pub domain: MaterialDomain,
    pub(crate) parameters: HashMap<String, MaterialParameter>,

    #[serde(skip)]
    pub(crate) last_tick_change: Tick,
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

    pub fn build(self) -> Material {
        let uuid = Uuid::new_v4();
        let name = self
            .name
            .unwrap_or_else(|| ImmutableString::from(format!("Material {uuid}")));
        Material {
            uuid,
            name,
            vertex_shader: self.vertex_shader,
            fragment_shader: self.fragment_shader,
            domain: self.domain,
            parameters: self.parameters,
            last_tick_change: Tick::now(),
        }
    }
}

impl Material {
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
    pub fn set_parameter(&mut self, name: impl Into<String>, parameter: MaterialParameter) {
        self.parameters.insert(name.into(), parameter);
        self.last_tick_change = Tick::now();
    }
}

impl Asset for Shader {
    fn get_description(&self) -> &str {
        &self.name
    }

    fn destroyed(&self, gpu: &dyn gpu::Gpu) {
        gpu.destroy_shader_module(self.handle)
    }
}

impl Asset for Material {
    fn get_description(&self) -> &str {
        &self.name
    }

    fn destroyed(&self, _gpu: &dyn gpu::Gpu) {}
}
