use crate::asset_map::Asset;

use self::{material::Material, mesh::Mesh, texture::Texture};

pub mod loaders;

pub mod material;
pub mod mesh;
pub mod shader;
pub mod texture;

impl Asset for Mesh {}
impl Asset for Texture {}
impl Asset for Material {}
