pub const SMALL_NUMBER: f32 = 0.005;
pub mod app;
pub mod asset_map;
pub mod assets;
pub mod input;
pub mod math;
pub mod scene;
pub mod scene_renderer;
pub mod types;

mod utils;

pub use utils::*;

/*
 * Conventions used
 * Coordinates: +Y is up, +Z is forward, -X is right (same as glTF)
 * */
