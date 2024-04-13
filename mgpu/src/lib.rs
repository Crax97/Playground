mod device;
mod hal;
mod rdg;
mod swapchain;

#[macro_use]
pub(crate) mod util;

use bitflags::bitflags;

pub use device::*;
pub use swapchain::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum MgpuError {
    InvalidHandle,
    Dynamic(String),
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
    pub struct ImageFlags : u32 {
        const TRANSFER_DST = 0x00000001;
        const TRANSFER_SRC = 0x00000002;
        const SAMPLED = 0x00000004;
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageFormat {
    Rgba8,
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
    pub depth: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TextureDimension {
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
    pub usage_flags: ImageFlags,
    pub initial_data: Option<&'a [u8]>,
    pub extents: Extents3D,
    pub dimension: TextureDimension,
    pub mips: u32,
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
