pub const SMALL_NUMBER: f32 = 0.005;
pub mod app;
pub mod components;
pub mod editor;
pub mod input;
pub mod loaders;
pub mod math;
pub mod physics;
pub mod post_process_pass;
pub mod resource_map;

mod bevy_ecs_app;
mod cvar_manager;
mod material;
mod mesh;
mod render_scene;
mod texture;
mod time;
mod utils;

pub use bevy_ecs;
pub use bevy_ecs_app::*;
pub use bevy_reflect;
pub use cvar_manager::*;
pub use material::*;
pub use mesh::*;
pub use resource_map::*;
pub use texture::*;
pub use time::*;
pub use utils::*;

pub use egui;

pub use render_scene::{camera::*, deferred_renderer::*, render_scene::*};

/*
 * Conventions used
 * Coordinates: +Y is up, +Z is forward, -X is right (same as glTF)
 * */
