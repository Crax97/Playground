mod bitmap_level;
mod entities;

use bitmap_level::{BitmapLevel, BitmapLevelLoader, EntityType};
use engine::app::app_state::app_state;
use engine::bevy_ecs::query::With;
use engine::bevy_ecs::schedule::IntoSystemConfigs;
use engine::bevy_ecs::system::{Query, Res, ResMut};
use engine::components::{
    DebugName, EngineWindow, SpriteComponent, SpriteComponentDescription, TestComponent,
    Transform2D,
};
use engine::editor::{EditorPlugin, EditorPluginBuilder};
use engine::input::InputState;
use engine::physics::rapier2d::dynamics::RigidBodyBuilder;
use engine::physics::rapier2d::geometry::{ColliderBuilder, SharedShape};
use engine::physics::{PhysicsContext2D, RigidBody2DHandle};
use engine::{
    bevy_ecs, Camera, CommonResources, DeferredRenderingPipeline, ResourceMap, Texture, Time,
};
use engine::{
    bevy_ecs::{component::Component, system::Commands},
    BevyEcsApp,
};
use entities::load_level_system;
use nalgebra::point;

#[derive(Component)]
pub struct Player;

fn main() -> anyhow::Result<()> {
    let mut app = BevyEcsApp::new()?;

    app.renderer()
        .set_combine_shader(DeferredRenderingPipeline::make_2d_combine_shader(
            &app_state().gpu,
        )?);

    app.setup_2d();
    app.resource_map()
        .install_resource_loader(BitmapLevelLoader);

    app.renderer().set_early_z_enabled(false);

    app.startup_schedule().add_systems(load_level_system);

    app.update_schedule()
        .add_systems((camera_system.after(move_player), move_player));

    let plugin = EditorPluginBuilder::new().build(&mut app);
    app.add_plugin::<EditorPlugin>(plugin);

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

fn move_player(
    mut query: Query<(&mut Transform2D, &RigidBody2DHandle, With<Player>)>,
    mut physics: ResMut<PhysicsContext2D>,
    input_state: Res<InputState>,
    time: Res<Time>,
) {
    const PLAYER_SPEED: f32 = 5.0;
    let (mut player_transform, rigidbody, _) = query.single_mut();
    let body = physics.get_rigidbody_mut(rigidbody).unwrap();

    if input_state.is_key_pressed(engine::input::Key::Left) {
        body.add_force([PLAYER_SPEED * time.delta_frame(), 0.0].into(), true);
    }
    if input_state.is_key_pressed(engine::input::Key::Right) {
        body.add_force([-PLAYER_SPEED * time.delta_frame(), 0.0].into(), true);
    }
    if input_state.is_key_just_pressed(engine::input::Key::Space) {
        physics
            .get_rigidbody_mut(rigidbody)
            .unwrap()
            .apply_impulse([0.0, 5.0].into(), true);
    }
}
