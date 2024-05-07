use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::{Device, Extents2D, Image, ImageFormat, ImageView, MgpuResult, PresentationRequest};

pub struct SwapchainCreationInfo<'a> {
    pub display_handle: DisplayHandle<'a>,
    pub window_handle: WindowHandle<'a>,
    pub preferred_format: Option<ImageFormat>,
    pub preferred_present_mode: Option<PresentMode>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum PresentMode {
    Immediate,
    Fifo,
}

#[derive(Clone)]
pub struct Swapchain {
    pub(crate) device: Device,
    pub(crate) current_acquired_image: Option<SwapchainImage>,
    pub(crate) info: SwapchainInfo,
}

#[derive(Clone, Copy)]
pub struct SwapchainImage {
    pub image: Image,
    pub view: ImageView,
    pub extents: Extents2D,
}

#[derive(Clone, Copy)]
pub(crate) struct SwapchainInfo {
    pub(crate) id: u64,
    pub(crate) format: ImageFormat,
    pub(crate) present_mode: PresentMode,
}

impl Swapchain {
    pub fn set_present_mode(&mut self, present_mode: PresentMode) -> MgpuResult<bool> {
        let supported_present_mode = self
            .device
            .hal
            .try_swapchain_set_present_mode(self.info.id, present_mode)?;
        self.info.present_mode = present_mode;
        Ok(supported_present_mode == present_mode)
    }
    pub fn resized(
        &mut self,
        new_extents: Extents2D,
        window_handle: WindowHandle,
        display_handle: DisplayHandle,
    ) -> MgpuResult<()> {
        self.device.hal.swapchain_on_resized(
            self.info.id,
            new_extents,
            window_handle,
            display_handle,
        )?;

        Ok(())
    }
    pub fn acquire_next_image(&mut self) -> MgpuResult<SwapchainImage> {
        let image = self.device.hal.swapchain_acquire_next_image(self.info.id)?;
        let old_image = self.current_acquired_image.replace(image);
        assert!(old_image.is_none(), "Called acquire without present!");

        Ok(image)
    }
    pub fn current_format(&self) -> ImageFormat {
        self.info.format
    }
    pub fn present(&mut self) -> MgpuResult<()> {
        let image = self
            .current_acquired_image
            .take()
            .expect("Called present without acquire!");
        let mut requests = self
            .device
            .presentation_requests
            .write()
            .expect("Failed to lock presentation requests");

        requests.push(PresentationRequest {
            id: self.info.id,
            image,
        });
        Ok(())
    }

    pub fn destroy(&self) -> MgpuResult<()> {
        self.device.hal.swapchain_destroy(self.info.id)
    }
}
