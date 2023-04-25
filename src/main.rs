mod gpu;
mod gpu_extension;

use ash::vk::{Extent2D, PresentModeKHR};
use gpu::{Gpu, GpuConfiguration};
use gpu_extension::{DefaultExtensions, SurfaceParamters, SwapchainExtension};
use raw_window_handle::HasRawDisplayHandle;
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::ControlFlow,
};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::default();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1240,
            height: 720,
        })
        .build(&event_loop)?;

    let mut gpu = Gpu::<SwapchainExtension<DefaultExtensions>>::new(
        GpuConfiguration {
            app_name: "Hello World!",
            engine_name: "Hello Engine!",
            enable_validation_layer: if cfg!(debug_assertions) { true } else { false },
            ..Default::default()
        },
        SurfaceParamters {
            inner_params: (),
            window,
            window_size: Extent2D {
                width: 1240,
                height: 720,
            },
        },
    )?;

    let surface = gpu.presentation_surface();
    gpu.select_present_mode(PresentModeKHR::MAILBOX)?;

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
