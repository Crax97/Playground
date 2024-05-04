use std::mem::MaybeUninit;

use mgpu::{
    Device, DeviceConfiguration, DeviceFeatures, Extents2D, MgpuResult, Swapchain,
    SwapchainCreationInfo, SwapchainImage,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

pub struct AppRunner {}

pub struct AppContext {
    pub device: Device,
    pub swapchain: MaybeUninit<Swapchain>,
    pub window: MaybeUninit<Window>,
}

pub struct RenderContext {
    pub swapchain_image: SwapchainImage,
}

pub trait App {
    fn app_name() -> &'static str;

    fn create(context: &AppContext) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn handle_window_event(&mut self, event: &WindowEvent) -> anyhow::Result<()>;
    fn handle_device_event(&mut self, event: &DeviceEvent) -> anyhow::Result<()>;
    fn update(&mut self, context: &AppContext) -> anyhow::Result<()>;
    fn render(&mut self, context: &AppContext, render_context: RenderContext)
        -> anyhow::Result<()>;
    fn resized(&mut self, context: &AppContext, new_extents: Extents2D) -> MgpuResult<()>;
    fn shutdown(&mut self, context: &AppContext) -> anyhow::Result<()>;
}

pub fn bootstrap<A: App>() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let app_name = A::app_name();
    let device = Device::new(DeviceConfiguration {
        app_name: Some(A::app_name()),
        features: DeviceFeatures::HAL_DEBUG_LAYERS,
        device_preference: Some(mgpu::DevicePreference::HighPerformance),
        desired_frames_in_flight: 3,
        display_handle: Some(event_loop.display_handle()?.as_raw()),
    })?;

    let mut context = AppContext {
        device,
        swapchain: MaybeUninit::uninit(),
        window: MaybeUninit::uninit(),
    };

    struct AppRunner<A: App> {
        app: A,
        context: AppContext,
        window_attributes: WindowAttributes,
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
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: winit::event::WindowEvent,
        ) {
            handle_window_event(&mut self.app, &mut self.context, event, event_loop)
                .expect("Failed to handle window event");
        }
    }

    let app = A::create(&context)?;

    event_loop.run_app(&mut AppRunner {
        app,
        context,
        window_attributes: WindowAttributes::default().with_title(A::app_name()),
    });

    Ok(())
}

fn handle_window_event<A: App>(
    app: &mut A,
    app_context: &mut AppContext,
    event: WindowEvent,
    target: &ActiveEventLoop,
) -> anyhow::Result<()> {
    app.handle_window_event(&event)?;

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
            window.request_redraw();
        }
        _ => {}
    };
    Ok(())
}
