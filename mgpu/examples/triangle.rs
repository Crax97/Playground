// use winit::event::{Event, WindowEvent};
// use winit::event_loop::EventLoop;
//
// fn main() {
//     let event_loop = EventLoop::new().unwrap();
//     let window = winit::window::WindowBuilder::new().with_title("Triangle")
//         .build(&event_loop).unwrap();
//
//     let device = mgpu::Device::new(DeviceConfiguration {
//         app_name: Some("Triangle Application"),
//         features: DeviceFeatures::DEBUG_LAYER,
//     }).expect("Failed to create gpu device");
//     let swapchain = device.create_swapchain(&SwapchainCreationInfo {
//         window: &window
//     }).expect("Failed to create swapchain");
//
//
//     let triangle_texture_data = read_image_data();
//     let image = device.create_image(&ImageCreationInfo {
//         label: Some("triangle image"),
//         usage_flags: ImageFlags::TRANSFER_DST | ImageFlags::SAMPLED,
//         extents: Extent3D {
//             witdh: 1,
//             height: 1,
//             depth: 1,
//         },
//         dimension: TextureDimension::D2,
//         mips: 1,
//         samples: SampleCount::ONE,
//         format: ImageFormat::Rgba8,
//         memory_domain: MemoryDomain::DeviceLocal,
//     }).expect("Failed to create image");
//
//     let image_view = device.create_image_view(&ImageViewDescription {
//         label: Some("triangle image view"),
//         format: ImageFormat::Rgba8,
//         aspect: ImageAspect::Color,
//         base_mip: 0,
//         num_mips: 1,
//         base_array_layer: 0,
//         num_array_layers: 0,
//     }).expect("Failed to create image view");
//
//
//     event_loop.run(|event, event_loop| {
//         match event {
//             Event::NewEvents(_) => {}
//             Event::WindowEvent { event, .. } => {
//
//                 match event {
//                     WindowEvent::CloseRequested => {
//                         event_loop.exit();
//                     }
//                     WindowEvent::RedrawRequested => {
//
//                     }
//                     _ => {}
//                 }
//             }
//             Event::DeviceEvent { .. } => {}
//             Event::UserEvent(_) => {}
//             Event::Suspended => {}
//             Event::Resumed => {}
//             Event::AboutToWait => {}
//             Event::LoopExiting => {
//                 event_loop.exit();
//             }
//             Event::MemoryWarning => {}
//         }
//     }).unwrap();
// }

fn main() {
    println!("Todo...");
}