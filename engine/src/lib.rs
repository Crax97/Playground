pub const SMALL_NUMBER: f32 = 0.005;
pub mod app;
pub mod components;
pub mod deferred_renderer;
pub mod editor;
pub mod input;
pub mod loaders;
pub mod math;
pub mod physics;
pub mod post_process_pass;
pub mod resource_map;

mod bevy_ecs_app;
mod camera;
mod cvar_manager;
mod material;
mod mesh;
mod render_scene;
mod scene;
mod texture;
mod time;
mod utils;

pub use bevy_ecs;
pub use bevy_ecs_app::*;
pub use bevy_reflect;
pub use camera::*;
pub use cvar_manager::*;
pub use deferred_renderer::*;
pub use material::*;
pub use mesh::*;
pub use resource_map::*;
pub use scene::*;
pub use texture::*;
pub use time::*;
pub use utils::*;

pub use egui;

/*
 * Conventions used
 * Coordinates: +Y is up, +Z is forward, -X is right (same as glTF)
 * */
