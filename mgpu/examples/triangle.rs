use mgpu::{
    DeviceConfiguration, DeviceFeatures, DevicePreference, Extents2D, Extents3D, ImageAspect,
    ImageDescription, ImageDimension, ImageFormat, ImageUsageFlags, ImageViewDescription,
    MemoryDomain, SampleCount, SwapchainCreationInfo,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("Triangle")
        .build(&event_loop)
        .unwrap();

    let device = mgpu::Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::DEBUG_FEATURES,
        device_preference: Some(DevicePreference::HighPerformance),
        display_handle: event_loop.display_handle().unwrap().as_raw(),
        desired_frames_in_flight: 3,
    })
    .expect("Failed to create gpu device");
    let mut swapchain = device
        .create_swapchain(&SwapchainCreationInfo {
            display_handle: window.display_handle().unwrap(),
            window_handle: window.window_handle().unwrap(),
            preferred_format: None,
        })
        .expect("Failed to create swapchain");

    let triangle_texture_data = read_image_data();
    let image = device
        .create_image(&ImageDescription {
            label: Some("triangle image"),
            usage_flags: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            initial_data: Some(&triangle_texture_data),
            extents: Extents3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            dimension: ImageDimension::D2,
            mips: 1.try_into().unwrap(),
            array_layers: 1.try_into().unwrap(),
            samples: SampleCount::One,
            format: ImageFormat::Rgba8,
            memory_domain: MemoryDomain::DeviceLocal,
        })
        .expect("Failed to create image");

    // let image_view = device
    //     .create_image_view(&ImageViewDescription {
    //         label: Some("triangle image view"),
    //         format: ImageFormat::Rgba8,
    //         aspect: ImageAspect::Color,
    //         base_mip: 0,
    //         num_mips: 1,
    //         base_array_layer: 0,
    //         num_array_layers: 0,
    //     })
    // .expect("Failed to create image view");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    event_loop
        .run(|event, event_loop| match event {
            Event::NewEvents(_) => {}
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    let _ = swapchain.acquire_next_image().unwrap();
                    swapchain.present().unwrap();

                    device.submit().unwrap();
                    window.request_redraw();
                }
                WindowEvent::Resized(new_size) => swapchain
                    .resized(
                        Extents2D {
                            width: new_size.width,
                            height: new_size.height,
                        },
                        window.window_handle().unwrap(),
                        window.display_handle().unwrap(),
                    )
                    .unwrap(),
                _ => {}
            },
            Event::DeviceEvent { .. } => {}
            Event::UserEvent(_) => {}
            Event::Suspended => {}
            Event::Resumed => {}
            Event::AboutToWait => {}
            Event::LoopExiting => {
                event_loop.exit();
            }
            Event::MemoryWarning => {}
        })
        .unwrap();

    // device.destroy_image_view(image_view);
    device.destroy_image(image).unwrap();
}

fn read_image_data() -> Vec<u8> {
    let image_content = std::fs::read("examples/assets/david.jpg").unwrap();
    image::load_from_memory(&image_content)
        .unwrap()
        .to_rgb8()
        .to_vec()
}
