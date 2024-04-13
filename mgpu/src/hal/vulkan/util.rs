use ash::vk;

use crate::{
    util::{define_resource_resolver, Handle},
    Image, ImageFormat, ImageView, PresentMode, Swapchain,
};

#[cfg(feature = "swapchain")]
use super::swapchain::VulkanSwapchain;

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

#[derive(Clone)]
pub(super) struct VulkanImage {
    pub(super) label: Option<String>,
    pub(super) handle: vk::Image,
    pub(super) external: bool,
}

#[derive(Clone)]
pub(super) struct VulkanImageView {
    pub(super) label: Option<String>,
    pub(super) handle: vk::ImageView,
    pub(super) owner: vk::Image,
    /// if true then the image was created outside of the vulkan instance
    /// e.g it could be a swapchain image
    pub(super) external: bool,
}

define_resource_resolver!(
    VulkanImage => images,
    VulkanImageView => image_views,
    VulkanSwapchain => swapchains
);

pub(super) trait ResolveVulkan<T, H>
where
    Self: Sized,
{
    fn resolve_vulkan(&self, handle: H) -> Option<T>;
}

pub(in crate::hal::vulkan) type VulkanResolver = ResourceResolver;

macro_rules! impl_util_methods {
    ($handle:ty, $object:ty, $vulkan_ty:ty) => {
        impl From<$handle> for Handle<$object> {
            fn from(handle: $handle) -> Handle<$object> {
                unsafe { Handle::from_u64(handle.id) }
            }
        }

        impl<H> ResolveVulkan<$vulkan_ty, H> for VulkanResolver
        where
            H: Into<Handle<$object>>,
        {
            fn resolve_vulkan(&self, handle: H) -> Option<$vulkan_ty> {
                self.get::<$object>()
                    .resolve(handle.into())
                    .map(|v| v.handle)
            }
        }
    };
}

impl_util_methods!(Image, VulkanImage, vk::Image);
impl_util_methods!(ImageView, VulkanImageView, vk::ImageView);
impl_util_methods!(Swapchain, VulkanSwapchain, vk::SwapchainKHR);
