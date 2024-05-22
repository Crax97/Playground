use mgpu::{
    Device, DeviceConfiguration, DeviceFeatures, DevicePreference, Extents2D, Swapchain,
    SwapchainCreationInfo,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes};

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let device = mgpu::Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::HAL_DEBUG_LAYERS,
        device_preference: Some(DevicePreference::HighPerformance),
        display_handle: Some(event_loop.display_handle().unwrap().as_raw()),
        desired_frames_in_flight: 3,
    })
    .expect("Failed to create gpu device");

    struct Application {
        device: Device,
        window: Option<Window>,
        swapchain: Option<Swapchain>,
    }

    impl ApplicationHandler for Application {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            let window = event_loop
                .create_window(WindowAttributes::default().with_title("Empty window"))
                .unwrap();
            let swapchain = self
                .device
                .create_swapchain(&SwapchainCreationInfo {
                    display_handle: window.display_handle().unwrap(),
                    window_handle: window.window_handle().unwrap(),
                    preferred_format: None,
                    preferred_present_mode: None,
                    extents: Extents2D {
                        width: window.inner_size().width,
                        height: window.inner_size().height,
                    },
                })
                .expect("Failed to create swapchain");

            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

            self.swapchain = Some(swapchain);
            self.window = Some(window);
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    self.swapchain.as_mut().unwrap().destroy().unwrap();
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    let _ = self
                        .swapchain
                        .as_mut()
                        .unwrap()
                        .acquire_next_image()
                        .unwrap();

                    self.swapchain.as_mut().unwrap().present().unwrap();

                    self.device.submit().unwrap();
                    self.window.as_ref().unwrap().request_redraw();
                }
                WindowEvent::Resized(new_size) => self
                    .swapchain
                    .as_mut()
                    .unwrap()
                    .resized(
                        Extents2D {
                            width: new_size.width,
                            height: new_size.height,
                        },
                        self.window.as_ref().unwrap().window_handle().unwrap(),
                        self.window.as_ref().unwrap().display_handle().unwrap(),
                    )
                    .unwrap(),
                _ => {}
            };
        }
    }

    event_loop
        .run_app(&mut Application {
            device,
            window: None,
            swapchain: None,
        })
        .unwrap();
}
