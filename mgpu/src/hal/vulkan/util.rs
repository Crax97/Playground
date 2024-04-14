use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::{
    util::{define_resource_resolver, Handle},
    Extents2D, Extents3D, Image, ImageDimension, ImageFormat, ImageUsageFlags, ImageView,
    PresentMode, SampleCount, Swapchain,
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

impl ToVk for Extents2D {
    type Target = vk::Extent2D;

    fn to_vk(self) -> Self::Target {
        Self::Target {
            width: self.width,
            height: self.height,
        }
    }

    fn from_vk(value: Self::Target) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

impl ToVk for Extents3D {
    type Target = vk::Extent3D;

    fn to_vk(self) -> Self::Target {
        Self::Target {
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }

    fn from_vk(value: Self::Target) -> Self {
        Self {
            width: value.width,
            height: value.height,
            depth: value.depth,
        }
    }
}

impl ToVk for SampleCount {
    type Target = vk::SampleCountFlags;

    fn to_vk(self) -> Self::Target {
        match self {
            SampleCount::One => vk::SampleCountFlags::TYPE_1,
        }
    }

    fn from_vk(value: Self::Target) -> Self {
        match value {
            vk::SampleCountFlags::TYPE_1 => SampleCount::One,
            _ => todo!(),
        }
    }
}

impl ToVk for ImageDimension {
    type Target = vk::ImageType;

    fn to_vk(self) -> Self::Target {
        match self {
            ImageDimension::D1 => vk::ImageType::TYPE_1D,
            ImageDimension::D2 => vk::ImageType::TYPE_2D,
            ImageDimension::D3 => vk::ImageType::TYPE_3D,
        }
    }

    fn from_vk(value: Self::Target) -> Self {
        match value {
            vk::ImageType::TYPE_1D => ImageDimension::D1,
            vk::ImageType::TYPE_2D => ImageDimension::D2,
            vk::ImageType::TYPE_3D => ImageDimension::D3,
            _ => unreachable!(),
        }
    }
}

impl ToVk for ImageUsageFlags {
    type Target = vk::ImageUsageFlags;

    fn to_vk(self) -> Self::Target {
        let mut flags = Self::Target::default();

        if self.contains(Self::TRANSFER_SRC) {
            flags |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
            flags |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if self.contains(Self::SAMPLED) {
            flags |= vk::ImageUsageFlags::SAMPLED;
        }
        if self.contains(Self::STORAGE) {
            flags |= vk::ImageUsageFlags::STORAGE;
        }
        if self.contains(Self::COLOR_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if self.contains(Self::DEPTH_STENCIL_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if self.contains(Self::TRANSIENT_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        }
        if self.contains(Self::INPUT_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::INPUT_ATTACHMENT;
        }
        flags
    }

    fn from_vk(value: Self::Target) -> Self {
        let mut flags = Self::default();

        if value.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
            flags |= Self::TRANSFER_SRC;
        }
        if value.contains(vk::ImageUsageFlags::TRANSFER_DST) {
            flags |= Self::TRANSFER_DST;
        }
        if value.contains(vk::ImageUsageFlags::SAMPLED) {
            flags |= Self::SAMPLED;
        }
        if value.contains(vk::ImageUsageFlags::STORAGE) {
            flags |= Self::STORAGE;
        }
        if value.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
            flags |= Self::COLOR_ATTACHMENT;
        }
        if value.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            flags |= Self::DEPTH_STENCIL_ATTACHMENT;
        }
        if value.contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT) {
            flags |= Self::TRANSIENT_ATTACHMENT;
        }
        if value.contains(vk::ImageUsageFlags::INPUT_ATTACHMENT) {
            flags |= Self::INPUT_ATTACHMENT;
        }
        flags
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

pub(super) struct VulkanImage {
    pub(super) label: Option<String>,
    pub(super) handle: vk::Image,
    pub(super) external: bool,
    pub(super) allocation: Option<Allocation>,
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
