mod bitmap_level;
mod character;
mod entities;

use bitmap_level::BitmapLevelLoader;
use character::{player_input_system, player_movement_system, PlayerCharacter};
use engine::app::app_state::app_state;
use engine::bevy_ecs::query::With;
use engine::bevy_ecs::schedule::IntoSystemConfigs;
use engine::bevy_ecs::system::{Query, Res};
use engine::components::{EngineWindow, Transform2D};
use engine::editor::{EditorPlugin, EditorPluginBuilder};
use engine::{bevy_ecs, Camera, DeferredRenderingPipeline};
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

    app.register_type::<PlayerCharacter>();
    app.renderer().set_early_z_enabled(false);

    app.startup_schedule().add_systems(load_level_system);

    app.update_schedule()
        .add_systems((
            player_input_system,
            player_movement_system.after(player_input_system),
        ))
        .add_systems(camera_system.after(player_movement_system));

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
