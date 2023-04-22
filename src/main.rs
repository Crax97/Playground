mod gpu;

use gpu::{Gpu, GpuConfiguration};
use raw_window_handle::HasRawDisplayHandle;
use winit::event_loop::ControlFlow;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::default();
    let window = winit::window::Window::new(&event_loop)?;

    let gpu = Gpu::new(GpuConfiguration {
        window: Some(&window),
        app_name: "Hello World!",
        engine_name: "Hello Engine!",
        enable_validation_layer: if cfg!(debug) { true } else { false },
        ..Default::default()
    })?;

    let surface = gpu
        .get_presentation_surface()
        .expect("Created a surface without presentation support, somehow");

    event_loop.run(move |event, event_loop, mut control_flow| match event {
        winit::event::Event::NewEvents(_) => {}
        winit::event::Event::WindowEvent { window_id, event } => match event {
            winit::event::WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::ExitWithCode(0)
            }
            _ => {}
        },
        winit::event::Event::DeviceEvent { device_id, event } => {}
        winit::event::Event::UserEvent(_) => {}
        winit::event::Event::Suspended => {}
        winit::event::Event::Resumed => {}
        winit::event::Event::MainEventsCleared => {}
        winit::event::Event::RedrawRequested(_) => {}
        winit::event::Event::RedrawEventsCleared => {}
        winit::event::Event::LoopDestroyed => *control_flow = ControlFlow::ExitWithCode(0),
    })
}
