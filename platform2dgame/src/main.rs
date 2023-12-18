mod bitmap_level;
mod character;
mod entities;

use std::default;

use bitmap_level::BitmapLevelLoader;
use character::{player_input_system, player_movement_system, KinematicCharacter};
use engine::app::app_state::app_state;
use engine::bevy_ecs::entity::Entity;
use engine::bevy_ecs::query::With;
use engine::bevy_ecs::schedule::IntoSystemConfigs;
use engine::bevy_ecs::system::{Query, Res, ResMut, Resource};
use engine::components::{EngineWindow, Transform2D};
use engine::editor::{EditorPlugin, EditorPluginBuilder};
use engine::physics::rapier2d::pipeline::QueryFilter;
use engine::physics::{Collider2DHandle, PhysicsContext2D};
use engine::{bevy_ecs, Camera, DeferredRenderingPipeline};
use engine::{
    bevy_ecs::{component::Component, system::Commands},
    BevyEcsApp,
};
use entities::{load_level_system, Star};
use nalgebra::{point, vector, Isometry2};

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Enemy;

#[derive(Resource, Default)]
pub struct GameState {
    pub points: u32,
}

fn main() -> anyhow::Result<()> {
    let mut app = BevyEcsApp::new()?;

    app.renderer()
        .set_combine_shader(DeferredRenderingPipeline::make_2d_combine_shader(
            app_state().gpu.as_ref(),
        )?);

    app.setup_2d();
    app.resource_map()
        .install_resource_loader(BitmapLevelLoader);

    app.register_type::<KinematicCharacter>();
    app.renderer().set_early_z_enabled(false);

    app.startup_schedule().add_systems(load_level_system);

    app.update_schedule()
        .add_systems((
            player_input_system,
            player_movement_system.after(player_input_system),
        ))
        .add_systems(camera_system.after(player_movement_system));
    app.post_update_schedule().add_systems(collision_checking);

    let plugin = EditorPluginBuilder::new().build(&mut app);
    app.add_plugin::<EditorPlugin>(plugin);

    app.world().insert_resource(GameState::default());

    app.run()
}

fn camera_system(
    mut commands: Commands,
    window: Res<EngineWindow>,
    player_query: Query<(&Transform2D, With<Player>)>,
) {
    const WORLD_SIZE: f32 = 100.0;
    let player_transf = player_query.single().0;
    let size = window.inner_size().cast::<f32>();
    let aspect = size.width / size.height;
    commands.insert_resource({
        let mut camera = Camera::new_orthographic(WORLD_SIZE * aspect, WORLD_SIZE, 0.0001, 1000.0);
        camera.location = point![player_transf.position.x, player_transf.position.y, 0.0];
        camera
    });
}

fn collision_checking(
    player_query: Query<(&Collider2DHandle, &Transform2D, With<Player>)>,
    star_query: Query<(Entity, &Collider2DHandle, With<Star>)>,
    mut phys_context: ResMut<PhysicsContext2D>,
    mut state: ResMut<GameState>,
    mut commands: Commands,
) {
    let (player_collider, transform, _) = player_query.single();
    let shape = phys_context
        .get_collider(player_collider)
        .unwrap()
        .shared_shape()
        .clone();

    let stars = star_query
        .iter()
        .map(|(e, c, _)| (e, c))
        .collect::<Vec<_>>();

    if let Some((collider, _)) = phys_context.cast_shape(
        Isometry2::translation(transform.position.x, transform.position.y),
        vector![0.0, 1.0],
        shape.0.as_ref(),
        1.0,
        true,
        QueryFilter::new()
            .exclude_solids()
            .exclude_collider(player_collider.as_ref().clone()),
    ) {
        if let Some((entity, star_collider)) = stars.iter().find(|(e, c)| **c == collider) {
            commands.entity(*entity).despawn();
            phys_context.remove_collider(**star_collider);
            state.points += 1;
            println!("Points now: {}", state.points);
        }
    }
}
