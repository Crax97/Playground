use mgpu::*;
fn main() {
    let device = Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::empty(),
        device_preference: Some(DevicePreference::HighPerformance),
    }).expect("Failed to create gpu device");

    let info = device.get_info();
    println!("Device info: {info}");
}