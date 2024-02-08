pub mod app_state;
pub mod egui_support;

mod console;

use std::borrow::Cow;

use crate::Backbuffer;
pub use console::*;
use gpu::CommandBuffer;

use log::{info, trace};
use winit::{
    dpi::PhysicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    platform::run_on_demand::EventLoopExtRunOnDemand,
};

use app_state::AppState;
pub trait App {
    fn window_name(&self, app_state: &AppState) -> Cow<str>;

    fn create(app_state: &mut AppState, event_loop: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn on_event(&mut self, _event: &Event<()>, _app_state: &AppState) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_resized(&mut self, _app_state: &AppState, _size: PhysicalSize<u32>) {}

    fn input(
        &mut self,
        _app_state: &AppState,
        _event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    fn on_startup(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn on_shutdown(&mut self, app_state: &mut AppState);

    fn begin_frame(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn update(&mut self, app_state: &mut AppState) -> anyhow::Result<()>;
    fn end_frame(&mut self, _app_state: &AppState) {}

    fn draw<'a>(
        &'a mut self,
        app_state: &'a mut AppState,
        backbuffer: &Backbuffer,
    ) -> anyhow::Result<CommandBuffer>;
}

pub fn app_loop<A: App + 'static>(
    app: &mut A,
    app_state: &mut AppState,
    event: Event<()>,
    target: &EventLoopWindowTarget<()>,
) -> anyhow::Result<ControlFlow> {
    app.on_event(&event, app_state)?;
    match event {
        winit::event::Event::NewEvents(_) => {}
        winit::event::Event::WindowEvent { event, .. } => {
            match event {
                winit::event::WindowEvent::CloseRequested => {
                    // app_state.gpu.on_destroyed();
                    target.exit();
                }
                winit::event::WindowEvent::Resized(new_size) => {
                    app_state.needs_new_swapchain = true;
                    app_state.current_window_size = new_size;
                }
                winit::event::WindowEvent::RedrawRequested => {
                    let sz = app_state.current_window_size;

                    if sz.width > 0 && sz.height > 0 {
                        update_app(app, app_state)?;
                    }
                    app_state.window.request_redraw();
                }
                _ => {}
            };
        }
        winit::event::Event::DeviceEvent { event, .. } => {
            app.input(app_state, event)?;
        }
        winit::event::Event::UserEvent(_) => {}
        winit::event::Event::Suspended => {}
        winit::event::Event::Resumed => {}
        winit::event::Event::AboutToWait => {}

        winit::event::Event::LoopExiting => {
            app_state.gpu.wait_device_idle().unwrap();
            app_state.gpu.save_pipeline_cache("pipeline_cache.pso")?;
        }
        Event::MemoryWarning => {}
    }

    Ok(ControlFlow::Poll)
}

fn update_app(app: &mut dyn App, app_state_mut: &mut AppState) -> anyhow::Result<()> {
    let sz = app_state_mut.current_window_size;
    if app_state_mut.needs_new_swapchain && sz.width > 0 && sz.height > 0 {
        app_state_mut.swapchain_mut().recreate_swapchain()?;

        app_state_mut.needs_new_swapchain = false;
        app.on_resized(app_state_mut, sz);
        return Ok(());
    }

    app_state_mut.begin_frame()?;
    app.begin_frame(app_state_mut)?;

    app.update(app_state_mut)?;

    draw_app(app_state_mut, app)?;

    app.end_frame(app_state_mut);
    app_state_mut.end_frame()?;

    Ok(())
}

fn draw_app(app_state_mut: &mut AppState, app: &mut dyn App) -> Result<(), anyhow::Error> {
    let swapchain_format = app_state_mut.swapchain().present_format();
    let swapchain_extents = app_state_mut.swapchain().extents();
    let (swapchain_image, swapchain_image_view) =
        app_state_mut.swapchain_mut().acquire_next_image()?;
    let backbuffer = Backbuffer {
        size: swapchain_extents,
        format: swapchain_format,
        image: swapchain_image,
        image_view: swapchain_image_view,
    };
    app.draw(app_state_mut, &backbuffer)?;
    Ok(())
}

pub fn create_app<A: App + 'static>() -> anyhow::Result<(A, EventLoop<()>, AppState)> {
    let mut env_logger_builder = env_logger::builder();

    if cfg!(debug_assertions) {
        // Enable all logging in debug configuration
        env_logger_builder.filter(None, log::LevelFilter::Trace);
    }

    env_logger_builder.init();

    info!(
        "Running in {:?}",
        std::env::current_dir().unwrap_or(".".into())
    );
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1920,
            height: 1080,
        })
        .with_title("Winit App")
        .build(&event_loop)?;

    let mut state = crate::app::app_state::init("Winit App", window)?;
    Ok((A::create(&mut state, &event_loop)?, event_loop, state))
}

pub fn run<A: App + 'static>(
    mut app: A,
    mut event_loop: EventLoop<()>,
    mut app_state: AppState,
) -> anyhow::Result<()> {
    trace!("Created app");
    app.on_startup(&mut app_state)?;
    let mut errored = false;
    event_loop.run_on_demand(|event, target| {
        match app_loop(&mut app, &mut app_state, event, target) {
            Err(e) => {
                errored = true;
                panic!(
                    "Backtrace: {}\nDuring app loop: {}",
                    e.backtrace(),
                    e.chain()
                        .fold(String::new(), |s, e| s + &e.to_string() + "\n"),
                );
            }
            _ => {}
        }
    })?;
    let exit_code = if !errored {
        app.on_shutdown(&mut app_state);
        std::mem::drop(app);
        std::mem::drop(app_state);
        0
    } else {
        -1
    };

    std::process::exit(exit_code);
}

pub fn bootstrap<A: App + 'static>() -> anyhow::Result<()> {
    let (app, event_loop, state) = create_app::<A>()?;

    run::<A>(app, event_loop, state)
}
