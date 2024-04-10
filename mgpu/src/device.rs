use std::fmt::Formatter;
use bitflags::bitflags;
use crate::MgpuResult;
bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct DeviceFeatures : u32 {
        const DEBUG_LAYERS = 0b01;
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DevicePreference {
    HighPerformance,
    Software,
    AnyDevice,
}

#[derive(Debug)]
pub struct DeviceConfiguration<'a> {
    pub app_name: Option<&'a str>,
    pub features: DeviceFeatures,
    pub device_preference: Option<DevicePreference>,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
}

#[derive(Clone)]
pub struct Device;

impl Device {
    pub fn new(configuration: DeviceConfiguration) -> MgpuResult<Self> {
        todo!()
    }

    pub fn get_info(&self) -> DeviceInfo {
        todo!()
    }
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Device name: {}", self.name))
    }
}