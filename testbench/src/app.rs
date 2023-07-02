use engine::AppState;
use log::trace;
use winit::{dpi::PhysicalSize, event::Event, event_loop::ControlFlow};

pub trait App {
    fn window_name(&self, app_state: &engine::AppState) -> String;

    fn create(app_state: &AppState) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn input(
        &mut self,
        app_state: &AppState,
        event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()>;
    fn update(&mut self, app_state: &mut AppState) -> anyhow::Result<()>;
    fn draw(&mut self, app_state: &mut AppState) -> anyhow::Result<()>;
}

pub fn app_loop<A: App + 'static>(
    app: &mut A,
    event: Event<'_, ()>,
) -> anyhow::Result<ControlFlow> {
    let app_state_mut = engine::app_state_mut();
    match event {
        winit::event::Event::NewEvents(_) => {}
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::CloseRequested => {
                return Ok(ControlFlow::ExitWithCode(0));
            }
            winit::event::WindowEvent::Resized(_) => {
                app_state_mut
                    .gpu
                    .swapchain_mut()
                    .recreate_swapchain()
                    .unwrap();
            }
            _ => {}
        },
        winit::event::Event::DeviceEvent { event, .. } => {
            app.input(app_state_mut, event)?;
        }
        winit::event::Event::UserEvent(_) => {}
        winit::event::Event::Suspended => {}
        winit::event::Event::Resumed => {}
        winit::event::Event::MainEventsCleared => {
            app_state_mut.gpu.swapchain_mut().window.request_redraw();
        }
        winit::event::Event::RedrawRequested(..) => {
            app_state_mut.begin_frame().unwrap();

            let window_name = app.window_name(&app_state_mut);

            app_state_mut
                .gpu
                .swapchain_mut()
                .window
                .set_title(&window_name);

            app.update(app_state_mut)?;
            app.draw(app_state_mut)?;
            app_state_mut.end_frame().unwrap();
        }
        winit::event::Event::RedrawEventsCleared => {}
        winit::event::Event::LoopDestroyed => {
            app_state_mut.gpu.wait_device_idle().unwrap();
            app_state_mut
                .gpu
                .save_pipeline_cache("pipeline_cache.pso")?;
        }
    }

    Ok(ControlFlow::Poll)
}

pub fn bootstrap<A: App + 'static>() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::default();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1240,
            height: 720,
        })
        .with_title("Winit App")
        .build(&event_loop)?;

    engine::init("Winit App", window)?;

    let app = Box::new(A::create(engine::app_state())?);
    let app = Box::leak(app);

    trace!("Created app");

    event_loop.run(move |event, _, control_flow| match app_loop(app, event) {
        Ok(flow) => {
            *control_flow = flow;
        }
        Err(e) => panic!("In main body of application: {}", e),
    })
}
