use crate::hal::Hal;
use crate::rdg::Rdg;
use crate::MgpuError;
use crate::{hal, Image, ImageDescription, ImageDimension, ImageViewDescription, MgpuResult};
use ash::vk::{self, ImageView};
use bitflags::bitflags;
use std::fmt::Formatter;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct DeviceFeatures : u32 {
        const DEBUG_FEATURES = 0b01;
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
    pub(crate) hal: Arc<dyn Hal>,
    pub(crate) device_info: DeviceInfo,
    pub(crate) rdg: Arc<RwLock<Rdg>>,
}

impl Device {
    pub fn new(configuration: DeviceConfiguration) -> MgpuResult<Self> {
        let hal = hal::create(&configuration)?;
        let device_info = hal.device_info();
        Ok(Self {
            hal,
            device_info,
            rdg: Default::default(),
        })
    }

    pub fn submit(&self) -> MgpuResult<()> {
        let rdg = self.write_rdg().take();
        let compiled = rdg.compile()?;

        self.hal.begin_rendering()?;
        for step in compiled.sequence {
            match step {
                crate::rdg::Step::Barrier { transfers } => {
                    todo!()
                }
                crate::rdg::Step::PassGroup(passes) => {
                    self.hal.execute_passes(passes, &compiled.nodes)?
                }
            }
        }
        self.hal.end_rendering()?;

        Ok(())
    }

    pub fn get_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }

    #[cfg(feature = "swapchain")]
    pub fn create_swapchain(
        &self,
        swapchain_info: &SwapchainCreationInfo,
    ) -> MgpuResult<Swapchain> {
        let swapchain_id = self.hal.create_swapchain_impl(swapchain_info)?;
        Ok(Swapchain {
            device: self.clone(),
            id: swapchain_id,
            current_acquired_image: None,
        })
    }

    pub fn create_image(&self, image_description: &ImageDescription) -> MgpuResult<Image> {
        Self::validate_image_description(image_description)?;
        self.hal.create_image(image_description)
    }

    pub fn destroy_image(&self, image: Image) -> MgpuResult<()> {
        self.hal.destroy_image(image)
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

    pub(crate) fn write_rdg(&self) -> RwLockWriteGuard<'_, Rdg> {
        self.rdg.write().expect("Failed to lock rdg")
    }

    pub(crate) fn read_rdg(&self) -> RwLockReadGuard<'_, Rdg> {
        self.rdg.read().expect("Failed to lock rdg")
    }

    fn validate_image_description(image_description: &ImageDescription) -> MgpuResult<()> {
        let check_condition = |condition: bool, error_message: &str| {
            if !condition {
                Err(MgpuError::InvaldImageDescription {
                    image_name: image_description.label.map(ToOwned::to_owned),
                    reason: error_message.to_string(),
                })
            } else {
                Ok(())
            }
        };

        check_condition(
            image_description.dimension != ImageDimension::D3
                || image_description.extents.depth > 0,
            "If an image is a 3D image, it cannot have a depth of 0",
        )?;

        let total_texels = image_description.extents.width
            * image_description.extents.height
            * image_description.extents.depth;
        check_condition(
            total_texels > 0,
            "The width, height and depth of an image cannot be 0",
        )?;
        let channel_byte_size = image_description.format.byte_size();

        if let Some(initial_data) = image_description.initial_data {
            check_condition(
                 initial_data.len() >= total_texels as usize * channel_byte_size,
            &format!("If an image has some initial data, the initial data byte buffer must have enough bytes for the image. Expected {}, got {}", total_texels as usize * channel_byte_size, initial_data.len()) 
            )?;
        }
        Ok(())
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
