use engine::app::app_state::app_state;
use engine::app::ConsoleWriter;
use engine::bevy_ecs::entity::Entity;
use engine::bevy_ecs::schedule::IntoSystemConfigs;
use engine::bevy_ecs::system::{Query, Res, ResMut};
use engine::components::{rendering_system, MeshComponent};
use engine::{bevy_ecs, Camera, ResourceMap, Texture, Time};
use engine::{
    bevy_ecs::{component::Component, system::Commands},
    BevyEcsApp,
};

#[derive(Component, Debug)]
pub struct Name(String);

fn main() -> anyhow::Result<()> {
    let mut app = BevyEcsApp::new()?;

    app.startup_schedule()
        .add_systems((hello_system, print_system));

    app.update_schedule().add_systems(greet_system);
    app.post_update_schedule()
        .add_systems((camera_system, rendering_system.after(camera_system)));

    app.run()
}

fn setup_sprite_system(mut resource_map: ResMut<ResourceMap>, mut commands: Commands) {
    // let texture = resource_map.load::<Texture>("images/texture.jpg", ImageLoader);
}

fn print_system(mut console_writer: ResMut<ConsoleWriter>) {
    println!("Hello world!");
    console_writer.write_message("Hello world!").unwrap();
}
fn hello_system(mut commands: Commands) {
    commands.spawn(Name("John".to_owned()));
    commands.spawn(Name("Marc".to_owned()));
    commands.spawn(Name("Jaques".to_owned()));
}

fn camera_system(mut commands: Commands) {
    commands.insert_resource(Camera::new_orthographic(100.0, 100.0, 0.0001, 1000.0));
}

fn greet_system(query: Query<(Entity, &Name)>, mut writer: ResMut<ConsoleWriter>) {
    for (who, name) in query.iter() {
        writer
            .write_message(format!(
                "Greeting {name:?} whose entity id is {who:?} at time {}",
                app_state().time.frames_since_start()
            ))
            .unwrap();
    }
}
