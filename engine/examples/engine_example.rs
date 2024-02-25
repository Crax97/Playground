use engine::{
    app::{self},
    game_framework::{systems::RenderingSystem, world::WorldBuilder},
    CvarManager, EngineApp, ResourceMap,
};
use winit::dpi::PhysicalSize;

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1920,
            height: 1080,
        })
        .with_title("Winit App")
        .build(&event_loop)
        .unwrap();

    let app_state = crate::app::app_state::init("Winit App", window).unwrap();

    let mut world = WorldBuilder::default();
    world
        .add_system(RenderingSystem::new(app_state.gpu.clone()))
        .add_resource(ResourceMap::new(app_state.gpu.clone()))
        .add_resource(CvarManager::default());
    let world = world.build();

    let app = EngineApp::new(world);
    crate::app::run(app, event_loop, app_state).unwrap();
}
