use egui_mgpu::EguiMgpuIntegration;
use mgpu::{
    Device, DeviceConfiguration, DeviceFeatures, DevicePreference, Extents2D, Graphics, Swapchain,
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
        app_name: Some("egui - Image example"),
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
        egui_integration: Option<EguiMgpuIntegration>,
        name: String,
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
            let egui_integration = EguiMgpuIntegration::new(&self.device, &window).unwrap();
            let ctx = egui_integration.context();

            egui_extras::install_image_loaders(&ctx);

            self.swapchain = Some(swapchain);
            self.window = Some(window);
            self.egui_integration = Some(egui_integration);
        }

        fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
            self.device.wait_idle().unwrap();
            self.egui_integration
                .as_mut()
                .unwrap()
                .destroy(&self.device)
                .unwrap();
            self.swapchain.as_mut().unwrap().destroy().unwrap();
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
            self.egui_integration
                .as_mut()
                .unwrap()
                .on_window_event(self.window.as_ref().unwrap(), &event);
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    let img = self
                        .swapchain
                        .as_mut()
                        .unwrap()
                        .acquire_next_image()
                        .unwrap();
                    {
                        let mut cmd = self.device.create_command_recorder::<Graphics>();
                        cmd.clear_image(img.view, [0.0; 4]);
                        cmd.submit().unwrap();
                    }

                    let egui = self.egui_integration.as_mut().unwrap();
                    egui.begin_frame(self.window.as_ref().unwrap());

                    let ctx = egui.context();
                    egui::CentralPanel::default().show(&ctx, |ui| {
                        ui.label("Hello world!");

                        ui.horizontal(|ui| {
                            ui.label("What's your name?");
                            ui.text_edit_singleline(&mut self.name)
                        });

                        ui.label(format!("Hello, {}", self.name));

                        egui::CollapsingHeader::new("Expand to show an image!").show(ui, |ui| {
                            ui.add(egui::Image::new(egui::include_image!("./images/david.jpg")))
                        });
                    });

                    let output = egui.end_frame();
                    egui.paint_frame(&self.device, img.view, output.textures_delta, output.shapes)
                        .unwrap();

                    egui.handle_platform_output(
                        self.window.as_ref().unwrap(),
                        output.platform_output,
                    );

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
            egui_integration: None,
            name: "Mayo".into(),
        })
        .unwrap();
}
