use crate::hal::Hal;
use crate::{hal, Image, ImageDescription, ImageViewDescription, MgpuResult};
use ash::vk::ImageView;
use bitflags::bitflags;
use std::fmt::Formatter;
use std::sync::Arc;

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

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
    pub desired_frames_in_flight: u32,

    #[cfg(feature = "swapchain")]
    pub display_handle: raw_window_handle::RawDisplayHandle,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub api_description: String,
    pub swapchain_support: bool,
}

#[derive(Clone)]
pub struct Device {
    hal: Arc<dyn Hal>,
    device_info: DeviceInfo,
}

impl Device {
    pub fn new(configuration: DeviceConfiguration) -> MgpuResult<Self> {
        let hal = hal::create(&configuration)?;
        let device_info = hal.device_info();
        Ok(Self { hal, device_info })
    }

    pub fn get_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }

    #[cfg(feature = "swapchain")]
    pub fn create_swapchain(
        &self,
        swapchain_info: &SwapchainCreationInfo,
    ) -> MgpuResult<Swapchain> {
        let pimpl = self.hal.create_swapchain_impl(swapchain_info)?;
        Ok(Swapchain { pimpl })
    }

    pub fn create_image(&self, image_description: &ImageDescription) -> MgpuResult<Image> {
        todo!()
    }

    pub fn destroy_image(&self, image: Image) {
        todo!()
    }

    pub fn create_image_view(
        &self,
        image_view_description: &ImageViewDescription,
    ) -> MgpuResult<ImageView> {
        todo!()
    }

    pub fn destroy_image_view(&self, image_view: ImageView) {
        todo!()
    }
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Device name: {}\n", self.name))?;
        f.write_fmt(format_args!("Api description: {}\n", self.api_description))?;
        f.write_fmt(format_args!(
            "Supports swapchain: {}\n",
            self.swapchain_support
        ))
    }
}
