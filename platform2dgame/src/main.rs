use std::collections::HashMap;

use engine::app::app_state::app_state;
use engine::bevy_ecs::system::{Res, ResMut};
use engine::components::{SpriteComponent, Transform2D};
use engine::{
    bevy_ecs, Camera, CommonResources, MaterialInstance, MaterialInstanceDescription, ResourceMap,
    SceneRenderingMode, Texture,
};
use engine::{
    bevy_ecs::{component::Component, system::Commands},
    BevyEcsApp,
};

#[derive(Component, Debug)]
pub struct Name(String);

fn main() -> anyhow::Result<()> {
    let mut app = BevyEcsApp::new()?;

    app.renderer()
        .set_scene_rendering_mode(SceneRenderingMode::Mode2D);
    app.startup_schedule().add_systems(setup_player_system);

    app.update_schedule().add_systems(camera_system);

    app.post_update_schedule()
        .add_systems(engine::components::rendering_system_2d);

    app.run()
}

fn camera_system(mut commands: Commands) {
    commands.insert_resource(Camera::new_orthographic(10.0, 10.0, 0.0001, 1000.0));
}

fn setup_player_system(
    mut resource_map: ResMut<ResourceMap>,
    common_resources: Res<CommonResources>,
    mut commands: Commands,
) {
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

    let transform = Transform2D::default();
    println!("Matrix {:?}", &transform.matrix());

    commands.spawn((SpriteComponent { material }, transform));
}
