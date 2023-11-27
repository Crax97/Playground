mod bitmap_level;

use bitmap_level::{BitmapLevel, BitmapLevelLoader, EntityType};
use engine::app::app_state::app_state;
use engine::bevy_ecs::query::With;
use engine::bevy_ecs::schedule::IntoSystemConfigs;
use engine::bevy_ecs::system::{Query, Res, ResMut};
use engine::components::{DebugName, EngineWindow, SpriteComponent, TestComponent, Transform2D};
use engine::editor::{EditorPlugin, EditorPluginBuilder};
use engine::input::InputState;
use engine::physics::rapier2d::dynamics::RigidBodyBuilder;
use engine::physics::rapier2d::geometry::{ColliderBuilder, SharedShape};
use engine::physics::PhysicsContext2D;
use engine::{bevy_ecs, Camera, CommonResources, DeferredRenderingPipeline, ResourceMap, Texture};
use engine::{
    bevy_ecs::{component::Component, system::Commands},
    BevyEcsApp,
};
use nalgebra::{point, vector};

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

fn camera_system(mut commands: Commands, window: Res<EngineWindow>) {
    let size = window.inner_size().cast::<f32>();
    let aspect = size.width / size.height;
    commands.insert_resource(Camera::new_orthographic(
        10.0 * aspect,
        10.0,
        0.0001,
        1000.0,
    ));
}

fn load_level_system(
    mut resource_map: ResMut<ResourceMap>,
    common_resources: Res<CommonResources>,
    mut commands: Commands,
) {
    let level = resource_map
        .load::<BitmapLevel>(&app_state().gpu, "images/levels/test_level.bmp")
        .unwrap();

    let texture = resource_map
        .load(&app_state().gpu, "images/apple.png")
        .unwrap();

    let level = resource_map.get(&level);
    for entity in &level.entities {
        match entity.ty {
            bitmap_level::EntityType::Player => {}
            bitmap_level::EntityType::Enemy => {}
            bitmap_level::EntityType::Terrain => {}
            bitmap_level::EntityType::Grass => {}
            bitmap_level::EntityType::Star => {}
            bitmap_level::EntityType::Platform => {}
        }

        let mut entity_spawned = commands.spawn((
            Transform2D {
                position: point![entity.x as f32, entity.y as f32] * 0.64,
                layer: 0,
                rotation: 0.0,
                scale: vector![1.0, 1.0],
            },
            SpriteComponent {
                texture: texture.clone(),
                material: common_resources.default_material_transparency.clone(),
                z_layer: 0,
            },
        ));
        if entity.ty == EntityType::Player {
            entity_spawned.insert((Player, DebugName("Player".to_owned())));
        }
    }
}

fn move_player(mut query: Query<(&mut Transform2D, With<Player>)>, input_state: Res<InputState>) {
    let (mut player_transform, _) = query.single_mut();

    if input_state.is_key_pressed(engine::input::Key::Left) {
        player_transform.position.x += 1.0;
    }
    if input_state.is_key_pressed(engine::input::Key::Right) {
        player_transform.position.x -= 1.0;
    }
}
