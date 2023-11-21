use std::collections::HashMap;

use engine::app::app_state::app_state;
use engine::bevy_ecs::query::With;
use engine::bevy_ecs::system::{Query, Res, ResMut};
use engine::components::{SpriteComponent, Transform2D};
use engine::input::InputState;
use engine::{
    bevy_ecs, Camera, CommonResources, DeferredRenderingPipeline, MaterialInstance,
    MaterialInstanceDescription, ResourceMap, Texture,
};
use engine::{
    bevy_ecs::{component::Component, system::Commands},
    BevyEcsApp,
};
use nalgebra::vector;

#[derive(Component)]
pub struct Player;

#[derive(Component, Debug)]
pub struct Name(String);

fn main() -> anyhow::Result<()> {
    let mut app = BevyEcsApp::new()?;

    app.renderer()
        .set_combine_shader(DeferredRenderingPipeline::make_2d_combine_shader(
            &app_state().gpu,
        )?);

    app.renderer().set_early_z_enabled(false);

    app.startup_schedule().add_systems(setup_player_system);

    app.update_schedule()
        .add_systems((camera_system, move_player));

    app.post_update_schedule()
        .add_systems(engine::components::rendering_system_2d);

    app.run()
}

fn camera_system(mut commands: Commands) {
    let size = app_state().window.inner_size().cast::<f32>();
    let aspect = size.width / size.height;
    commands.insert_resource(Camera::new_orthographic(
        10.0 * aspect,
        10.0,
        0.0001,
        1000.0,
    ));
}

fn setup_player_system(
    mut resource_map: ResMut<ResourceMap>,
    common_resources: Res<CommonResources>,
    mut commands: Commands,
) {
    {
        let texture = resource_map
            .load::<Texture>(&app_state().gpu, "images/apple.png")
            .unwrap();

        let mut texture_inputs = HashMap::new();
        texture_inputs.insert("texSampler".to_owned(), texture);

        let material = MaterialInstance::create_instance(
            &app_state().gpu,
            common_resources.default_material_transparency.clone(),
            &resource_map,
            &MaterialInstanceDescription {
                name: "Spriteee",
                texture_inputs,
            },
        )
        .unwrap();

        let material = resource_map.add(material);

        let transform = Transform2D::default();

        commands.spawn((
            SpriteComponent {
                material,
                z_layer: 0,
            },
            transform,
            Player,
        ));
    }

    {
        let texture = resource_map
            .load::<Texture>(&app_state().gpu, "images/texture.jpg")
            .unwrap();

        let mut texture_inputs = HashMap::new();
        texture_inputs.insert("texSampler".to_owned(), texture);

        let material = MaterialInstance::create_instance(
            &app_state().gpu,
            common_resources.default_material.clone(),
            &resource_map,
            &MaterialInstanceDescription {
                name: "Spriteee",
                texture_inputs,
            },
        )
        .unwrap();

        let material = resource_map.add(material);

        let transform = Transform2D {
            position: Default::default(),
            rotation: 0.0,
            scale: vector![5.0, 5.0],
            layer: 100,
        };

        commands.spawn((
            SpriteComponent {
                material,
                z_layer: 100,
            },
            transform,
        ));
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
