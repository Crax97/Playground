pub mod app_state;
pub mod egui_support;

mod console;

use crate::Backbuffer;
pub use console::*;
use gpu::{CommandBuffer, CommandBufferSubmitInfo, PipelineStageFlags};

use log::{info, trace};
use winit::{
    dpi::PhysicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use app_state::{app_state_mut, AppState};
pub trait App {
    fn window_name(&self, app_state: &AppState) -> String;

    fn create(
        app_state: &mut AppState,
        event_loop: &EventLoop<()>,
        window: Window,
    ) -> anyhow::Result<Self>
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

    fn begin_frame(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn update(&mut self, app_state: &mut AppState) -> anyhow::Result<()>;
    fn end_frame(&mut self, _app_state: &AppState) {}

    fn draw<'a>(
        &'a mut self,
        app_state: &'a AppState,
        backbuffer: &Backbuffer,
    ) -> anyhow::Result<CommandBuffer>;
}

pub fn app_loop<A: App + 'static>(
    app: &mut A,
    event: Event<'_, ()>,
) -> anyhow::Result<ControlFlow> {
    let app_state_mut = app_state_mut();
    app.on_event(&event, app_state_mut)?;
    match event {
        winit::event::Event::NewEvents(_) => {}
        winit::event::Event::WindowEvent { event, .. } => {
            match event {
                winit::event::WindowEvent::CloseRequested => {
                    return Ok(ControlFlow::ExitWithCode(0));
                }
                winit::event::WindowEvent::Resized(new_size) => {
                    app_state_mut.needs_new_swapchain = true;
                    app_state_mut.current_window_size = new_size;
                }
                _ => {}
            };
        }
        winit::event::Event::DeviceEvent { event, .. } => {
            app.input(app_state_mut, event)?;
        }
        winit::event::Event::UserEvent(_) => {}
        winit::event::Event::Suspended => {}
        winit::event::Event::Resumed => {}
        winit::event::Event::MainEventsCleared => {
            let sz = app_state_mut.current_window_size;
            if app_state_mut.needs_new_swapchain && sz.width > 0 && sz.height > 0 {
                app_state_mut.swapchain_mut().recreate_swapchain()?;

                app_state_mut.needs_new_swapchain = false;
                app.on_resized(&app_state_mut, sz);
            }

            if sz.width > 0 && sz.height > 0 {
                update_app(app, app_state_mut)?;
            }
        }
        winit::event::Event::RedrawRequested(..) => {}
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

fn update_app(app: &mut dyn App, app_state_mut: &mut AppState) -> anyhow::Result<()> {
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
        format: swapchain_format.into(),
        image: swapchain_image,
        image_view: swapchain_image_view,
    };
    let mut command_buffer = app.draw(self::app_state_mut(), &backbuffer)?;
    let frame = app_state_mut.swapchain_mut().get_current_swapchain_frame();
    command_buffer.submit(&CommandBufferSubmitInfo {
        wait_semaphores: &[&frame.image_available_semaphore],
        wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
        signal_semaphores: &[&frame.render_finished_semaphore],
        fence: Some(&frame.in_flight_fence),
    })?;
    Ok(())
}

pub fn create_app<A: App + 'static>() -> anyhow::Result<(A, EventLoop<()>)> {
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
    let event_loop = winit::event_loop::EventLoop::default();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1920,
            height: 1080,
        })
        .with_title("Winit App")
        .build(&event_loop)?;

    crate::app::app_state::init("Winit App", &window)?;
    Ok((A::create(app_state_mut(), &event_loop, window)?, event_loop))
}

pub fn run<A: App + 'static>(app: A, event_loop: EventLoop<()>) -> anyhow::Result<()> {
    let app = Box::new(app);
    let app = Box::leak(app);

    trace!("Created app");

    app.on_startup(app_state_mut())?;
    event_loop.run(move |event, _, control_flow| match app_loop(app, event) {
        Ok(flow) => {
            *control_flow = flow;
        }
        Err(e) => panic!(
            "Backtrace: {}\nDuring app loop: {}",
            e.backtrace(),
            e.chain()
                .fold(String::new(), |s, e| s + &e.to_string() + "\n"),
        ),
    })
}

pub fn bootstrap<A: App + 'static>() -> anyhow::Result<()> {
    let (app, event_loop) = create_app::<A>()?;

    run::<A>(app, event_loop)
}
