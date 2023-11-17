use engine::app::app_state::app_state;
use engine::app::ConsoleWriter;
use engine::bevy_ecs::entity::Entity;
use engine::bevy_ecs::system::{Query, Res, ResMut};
use engine::{bevy_ecs, Time};
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

    app.run()
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
