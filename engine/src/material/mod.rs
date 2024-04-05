use std::collections::HashMap;

use gpu::{CullMode, FrontFace, PolygonMode, PrimitiveTopology, ShaderModuleHandle};

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
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub primitive_topology: PrimitiveTopology,
    pub polygon_mode: PolygonMode,

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

    front_face: FrontFace,
    cull_mode: CullMode,
    primitive_topology: PrimitiveTopology,
    polygon_mode: PolygonMode,
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

            front_face: Default::default(),
            cull_mode: Default::default(),
            primitive_topology: Default::default(),
            polygon_mode: Default::default(),

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

    pub fn front_face(mut self, front_face: FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    pub fn cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    pub fn polygon_mode(mut self, polygon_mode: PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }

    pub fn primitive_topology(mut self, primitive_topology: PrimitiveTopology) -> Self {
        self.primitive_topology = primitive_topology;
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
            front_face: self.front_face,
            cull_mode: self.cull_mode,
            primitive_topology: self.primitive_topology,
            polygon_mode: self.polygon_mode,
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
