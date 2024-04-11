use ash::vk;

use crate::{ImageFormat, PresentMode};

pub(crate) trait ToVk {
    type Target;

    fn to_vk(self) -> Self::Target;
    fn from_vk(value: Self::Target) -> Self;
}

impl ToVk for ImageFormat {
    type Target = vk::Format;

    fn to_vk(self) -> Self::Target {
        match self {
            ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        }
    }

    fn from_vk(value: Self::Target) -> Self {
        match value {
            vk::Format::R8G8B8A8_UNORM => ImageFormat::Rgba8,
            _ => unreachable!("Format not known"),
        }
    }
}

#[cfg(feature = "swapchain")]
impl ToVk for PresentMode {
    type Target = vk::PresentModeKHR;

    fn to_vk(self) -> Self::Target {
        match self {
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        }
    }

    fn from_vk(value: Self::Target) -> Self {
        match value {
            vk::PresentModeKHR::IMMEDIATE => PresentMode::Immediate,
            vk::PresentModeKHR::FIFO => PresentMode::Fifo,
            _ => unreachable!("Format not known"),
        }
    }
}
