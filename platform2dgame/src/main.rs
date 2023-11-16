use engine::BevyEcsApp;

fn main() -> anyhow::Result<()> {
    let (mut bevy_ecs_app, evt) = BevyEcsApp::new()?;

    bevy_ecs_app.run(evt)
}
