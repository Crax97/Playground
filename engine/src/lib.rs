pub const SMALL_NUMBER: f32 = 0.005;
pub mod app;
pub mod asset_map;
pub mod assets;
pub mod core;
pub mod input;
pub mod math;
pub mod scene;
pub mod scene_renderer;

#[macro_use]
pub mod utils;

pub use utils::*;

pub use glam;
pub use winit;

pub mod constants {
    use crate::{asset_map::AssetHandle, assets::mesh::Mesh, immutable_string::ImmutableString};

    pub const CUBE_MESH_HANDLE: AssetHandle<Mesh> =
        AssetHandle::new_const(ImmutableString::new("mesh.cube"));
}

/*
 * Conventions used
 * Coordinates: +Y is up, +Z is forward, -X is right (same as glTF)
 * */
