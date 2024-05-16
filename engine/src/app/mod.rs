use std::mem::MaybeUninit;

use mgpu::{
    Device, DeviceConfiguration, DeviceFeatures, Extents2D, MgpuResult, Swapchain,
    SwapchainCreationInfo, SwapchainImage,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

use crate::{core::Time, fps_limiter::FpsLimiter, input::InputState};

pub struct AppRunner {}

pub struct AppDescription {
    pub window_size: Extents2D,
    pub initial_title: Option<&'static str>,
    pub app_identifier: &'static str,
}

pub struct AppContext {
    pub device: Device,
    pub swapchain: MaybeUninit<Swapchain>,
    pub window: MaybeUninit<Window>,
    pub input: InputState,
    pub time: Time,
    pub fps_limiter: FpsLimiter,
}
impl AppContext {
    pub fn window(&self) -> &Window {
        unsafe { self.window.assume_init_ref() }
    }
}

pub struct RenderContext {
    pub swapchain_image: SwapchainImage,
}

pub trait App {
    fn create(context: &AppContext) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn on_window_created(&mut self, context: &AppContext) -> anyhow::Result<()> {
        let _ = context;
        Ok(())
    }

    fn handle_window_event(
        &mut self,
        event: &WindowEvent,
        context: &AppContext,
    ) -> anyhow::Result<()> {
        let _ = (event, context);
        Ok(())
    }
    fn handle_device_event(
        &mut self,
        event: &DeviceEvent,
        context: &AppContext,
    ) -> anyhow::Result<()> {
        let _ = (event, context);
        Ok(())
    }
    fn update(&mut self, context: &AppContext) -> anyhow::Result<()>;
    fn render(&mut self, context: &AppContext, render_context: RenderContext)
        -> anyhow::Result<()>;
    fn resized(&mut self, context: &AppContext, new_extents: Extents2D) -> MgpuResult<()>;
    fn shutdown(&mut self, context: &AppContext) -> anyhow::Result<()>;
}

pub fn bootstrap<A: App>(description: AppDescription) -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let device = Device::new(DeviceConfiguration {
        app_name: Some(description.app_identifier),
        features: DeviceFeatures::HAL_DEBUG_LAYERS,
        device_preference: Some(mgpu::DevicePreference::HighPerformance),
        desired_frames_in_flight: 3,
        display_handle: Some(event_loop.display_handle()?.as_raw()),
    })?;

    let context = AppContext {
        device,
        swapchain: MaybeUninit::uninit(),
        window: MaybeUninit::uninit(),
        input: InputState::default(),
        time: Time::default(),
        fps_limiter: FpsLimiter::new(60),
    };

    struct AppRunner<A: App> {
        app: A,
        context: AppContext,
        window_attributes: WindowAttributes,
        description: AppDescription,
    }

    impl<A: App> ApplicationHandler for AppRunner<A> {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            let window = event_loop
                .create_window(self.window_attributes.clone())
                .unwrap();
            let swapchain = self
                .context
                .device
                .create_swapchain(&SwapchainCreationInfo {
                    display_handle: window.display_handle().unwrap(),
                    window_handle: window.window_handle().unwrap(),
                    preferred_format: None,
                    preferred_present_mode: None,
                })
                .expect("Failed to create swapchain");

            self.context.window = MaybeUninit::new(window);
            self.context.swapchain = MaybeUninit::new(swapchain);

            self.app.on_window_created(&self.context).unwrap();
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: winit::event::WindowEvent,
        ) {
            handle_window_event(
                &mut self.app,
                &mut self.context,
                event,
                event_loop,
                &self.description,
            )
            .expect("Failed to handle window event");
        }
    }

    let app = A::create(&context)?;

    event_loop.run_app(&mut AppRunner {
        app,
        context,
        window_attributes: WindowAttributes::default()
            .with_inner_size(PhysicalSize {
                width: description.window_size.width,
                height: description.window_size.height,
            })
            .with_title(description.initial_title.unwrap_or("Engine App")),
        description,
    })?;
    Ok(())
}

fn handle_window_event<A: App>(
    app: &mut A,
    app_context: &mut AppContext,
    event: WindowEvent,
    target: &ActiveEventLoop,
    description: &AppDescription,
) -> anyhow::Result<()> {
    app.handle_window_event(&event, app_context)?;
    app_context.input.update(&event);

    match event {
        winit::event::WindowEvent::Resized(new_size) => {
            let swapchain = unsafe { app_context.swapchain.assume_init_mut() };
            let window = unsafe { app_context.window.assume_init_mut() };
            let new_extents = Extents2D {
                width: new_size.width,
                height: new_size.height,
            };
            swapchain.resized(
                new_extents,
                window.window_handle()?,
                window.display_handle()?,
            )?;
            app.resized(app_context, new_extents)?;
        }

        winit::event::WindowEvent::CloseRequested => target.exit(),
        winit::event::WindowEvent::RedrawRequested => {
            app_context.time.begin_frame();
            app.update(app_context)?;
            let next_image = {
                let swapchain = unsafe { app_context.swapchain.assume_init_mut() };
                swapchain.acquire_next_image()?
            };

            let render_context = RenderContext {
                swapchain_image: next_image,
            };

            app.render(app_context, render_context)?;

            let swapchain = unsafe { app_context.swapchain.assume_init_mut() };
            let window = unsafe { app_context.window.assume_init_mut() };
            swapchain.present()?;
            app_context.device.submit()?;

            let frame_time = app_context.time.delta_from_frame_begin();
            app_context.fps_limiter.update(frame_time);
            app_context.time.end_frame();

            app_context.input.end_frame();
            let fps = 1.0 / app_context.time.delta_seconds();
            let app_title = format!(
                "{} - FPS {}",
                description.initial_title.unwrap_or("Engine App"),
                fps as u64
            );
            window.set_title(&app_title);
            window.request_redraw();
        }
        _ => {}
    };
    Ok(())
}

impl Default for AppDescription {
    fn default() -> Self {
        Self {
            window_size: Extents2D {
                width: 800,
                height: 600,
            },
            initial_title: Default::default(),
            app_identifier: "EngineApp",
        }
    }
}
