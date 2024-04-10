use mgpu::*;
use raw_window_handle::HasDisplayHandle;
use winit::event_loop::EventLoop;
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let device = Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::empty(),
        device_preference: Some(DevicePreference::HighPerformance),
        display_handle: event_loop.display_handle().unwrap().as_raw(),
    })
    .expect("Failed to create gpu device");

    let info = device.get_info();
    println!("Device info: {info}");
}
