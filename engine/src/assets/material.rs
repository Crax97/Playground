use std::collections::HashMap;

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::asset_map::AssetHandle;

use super::{shader::Shader, texture::Texture};

pub struct Material {
    pub vertex_shader: Shader,
    pub fragment_shader: Shader,
    pub parameters: MaterialParameters,
    pub properties: MaterialProperties,
}

pub struct MaterialDescription {
    pub vertex_shader: Shader,
    pub fragment_shader: Shader,
    pub parameters: MaterialParameters,
    pub properties: MaterialProperties,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum MaterialParameter {
    Texture(AssetHandle<Texture>),
    Scalar(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
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
    parameters: HashMap<String, MaterialParameter>,
}

impl MaterialParameters {
    pub fn parameter(mut self, name: impl Into<String>, parameter: MaterialParameter) -> Self {
        self.parameters.insert(name.into(), parameter);
        self
    }
}
