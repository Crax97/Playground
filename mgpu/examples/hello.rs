use mgpu::*;
use raw_window_handle::HasDisplayHandle;
use winit::event_loop::EventLoop;
fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let device = Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::DEBUG_FEATURES,
        device_preference: Some(DevicePreference::HighPerformance),
        desired_frames_in_flight: 3,
        display_handle: event_loop.display_handle().unwrap().as_raw(),
    })
    .expect("Failed to create gpu device");

    let info = device.get_info();
    println!("Device info: {info}");
}
