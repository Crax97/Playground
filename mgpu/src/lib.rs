mod device;
mod hal;
mod rdg;
mod swapchain;

#[macro_use]
pub(crate) mod util;

use std::num::NonZeroU32;

use bitflags::bitflags;

pub use device::*;
pub use swapchain::*;

#[cfg(feature = "vulkan")]
use hal::vulkan::VulkanHalError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum MgpuError {
    InvaldImageDescription {
        image_name: Option<String>,
        reason: String,
    },
    InvalidHandle,
    Dynamic(String),

    #[cfg(feature = "vulkan")]
    VulkanError(VulkanHalError),
}

pub type MgpuResult<T> = Result<T, MgpuError>;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub enum MemoryDomain {
    HostVisible,
    HostCoherent,
    DeviceLocal,
}

bitflags! {
    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
    #[cfg_attr(feature="serde", derive(Serialize, Deserialize))]
    pub struct ImageUsageFlags : u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC = 0b1;
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST= 0b10;
        #[doc = "Can be sampled from (SAMPLED_IMAGE and COMBINED_IMAGE_SAMPLER descriptor types)"]
        const SAMPLED= 0b100;
        #[doc = "Can be used as storage image (STORAGE_IMAGE descriptor type)"]
        const STORAGE= 0b1000;
        #[doc = "Can be used as framebuffer color attachment"]
        const COLOR_ATTACHMENT= 0b1_0000;
        #[doc = "Can be used as framebuffer depth/stencil attachment"]
        const DEPTH_STENCIL_ATTACHMENT= 0b10_0000;
        #[doc = "Image data not needed outside of rendering"]
        const TRANSIENT_ATTACHMENT= 0b100_0000;
        #[doc = "Can be used as framebuffer input attachment"]
        const INPUT_ATTACHMENT= 0b1000_0000;
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageFormat {
    Rgba8,
}
impl ImageFormat {
    fn byte_size(&self) -> usize {
        match self {
            ImageFormat::Rgba8 => 4,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageAspect {
    Color,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Extents3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Extents2D {
    pub width: u32,
    pub height: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageDimension {
    D1,
    D2,
    D3,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SampleCount {
    One,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageDescription<'a> {
    pub label: Option<&'a str>,
    pub usage_flags: ImageUsageFlags,
    pub initial_data: Option<&'a [u8]>,
    pub extents: Extents3D,
    pub dimension: ImageDimension,
    pub mips: NonZeroU32,
    pub array_layers: NonZeroU32,
    pub samples: SampleCount,
    pub format: ImageFormat,
    pub memory_domain: MemoryDomain,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageViewDescription<'a> {
    pub label: Option<&'a str>,
    pub format: ImageFormat,
    pub aspect: ImageAspect,
    pub base_mip: u32,
    pub num_mips: u32,
    pub base_array_layer: u32,
    pub num_array_layers: u32,
}

/// An image is a multidimensional buffer of data, with an associated format
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Image {
    id: u64,
}

/// An image view is a view over a portion of an image
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageView {
    owner: Image,
    id: u64,
}

#[derive(Clone)]
pub struct Swapchain {
    id: u64,
    device: crate::Device,
}

impl std::fmt::Display for MgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MgpuError::Dynamic(msg) => f.write_str(msg),
            MgpuError::InvalidHandle => f.write_str("Tried to resolve an invalid handle"),
            MgpuError::InvaldImageDescription { image_name, reason } => f.write_fmt(format_args!(
                "Invalid ImageDescription for image {}: {reason}",
                image_name.as_ref().unwrap_or(&"Unnamed".to_string())
            )),

            #[cfg(feature = "vulkan")]
            MgpuError::VulkanError(error) => f.write_fmt(format_args!("{:?}", error)),
        }
    }
}
impl std::error::Error for MgpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.source()
    }
}
