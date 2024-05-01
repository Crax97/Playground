use mgpu::{
    Device, DeviceConfiguration, DeviceFeatures, Extents2D, MgpuResult, Swapchain,
    SwapchainCreationInfo, SwapchainImage,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    event::Event,
    event_loop::{EventLoop, EventLoopWindowTarget},
    window::Window,
};

pub struct AppRunner {}

pub struct AppContext {
    pub device: Device,
    pub swapchain: Swapchain,
    pub window: Window,
}

pub struct RenderContext {
    pub swapchain_image: SwapchainImage,
}

pub trait App {
    fn app_name() -> &'static str;

    fn create(context: &AppContext) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn handle_os_event(&mut self, event: &Event<()>) -> anyhow::Result<()>;
    fn update(&mut self, context: &AppContext) -> anyhow::Result<()>;
    fn render(&mut self, context: &AppContext, render_context: RenderContext)
        -> anyhow::Result<()>;
    fn resized(&mut self, context: &AppContext, new_extents: Extents2D) -> MgpuResult<()>;
    fn shutdown(&mut self, context: &AppContext) -> anyhow::Result<()>;
}

pub fn bootstrap<A: App>() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let window = Window::new(&event_loop)?;
    window.set_title(A::app_name());

    let device = Device::new(DeviceConfiguration {
        app_name: Some(A::app_name()),
        features: DeviceFeatures::HAL_DEBUG_LAYERS,
        device_preference: Some(mgpu::DevicePreference::HighPerformance),
        desired_frames_in_flight: 3,
        display_handle: Some(window.display_handle()?.as_raw()),
    })?;
    let swapchain = device.create_swapchain(&SwapchainCreationInfo {
        display_handle: window.display_handle()?,
        window_handle: window.window_handle()?,
        preferred_format: None,
        preferred_present_mode: None,
    })?;

    let mut context = AppContext {
        device,
        swapchain,
        window,
    };

    let mut app = A::create(&context)?;

    event_loop.run(|event, target| {
        if let Err(e) = handle_event(&mut app, &mut context, event, target) {
            panic!("During app loop: {e:#?}");
        }
    })?;

    Ok(())
}

fn handle_event<A: App>(
    app: &mut A,
    app_context: &mut AppContext,
    event: Event<()>,
    target: &EventLoopWindowTarget<()>,
) -> anyhow::Result<()> {
    app.handle_os_event(&event)?;

    if let Event::WindowEvent { event, .. } = event {
        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                let new_extents = Extents2D {
                    width: new_size.width,
                    height: new_size.height,
                };
                app_context.swapchain.resized(
                    new_extents,
                    app_context.window.window_handle()?,
                    app_context.window.display_handle()?,
                )?;
                app.resized(app_context, new_extents)?;
            }

            winit::event::WindowEvent::CloseRequested => target.exit(),
            winit::event::WindowEvent::RedrawRequested => {
                let next_image = app_context.swapchain.acquire_next_image()?;
                app.update(app_context)?;

                let render_context = RenderContext {
                    swapchain_image: next_image,
                };

                app.render(app_context, render_context)?;

                app_context.swapchain.present()?;
                app_context.device.submit()?;
            }
            _ => {}
        }
    };
    Ok(())
}
