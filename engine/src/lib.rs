pub const SMALL_NUMBER: f32 = 0.005;
pub mod app;
pub mod asset_map;
pub mod components;
pub mod editor;
pub mod fps_camera;
pub mod input;
pub mod kecs_app;
pub mod loaders;
pub mod math;
pub mod physics;
pub mod post_process_pass;

mod bevy_ecs_app;
mod cvar_manager;
mod material;
mod mesh;
mod render_scene;
mod texture;
mod time;
mod utils;

pub use asset_map::*;
pub use bevy_ecs;
pub use bevy_ecs_app::*;
pub use bevy_reflect;
pub use cvar_manager::*;
pub use kecs;
pub use material::*;
pub use mesh::*;
pub use texture::*;
pub use time::*;
pub use utils::*;

pub use rhai::*;

pub use egui;

pub use render_scene::{camera::*, deferred_renderer::*, render_structs::*, scene::*};

/*
 * Conventions used
 * Coordinates: +Y is up, +Z is forward, -X is right (same as glTF)
 * */
