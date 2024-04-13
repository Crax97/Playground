use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::{Device, Extents2D, Image, ImageFormat, ImageView, MgpuResult};

pub struct SwapchainCreationInfo<'a> {
    pub display_handle: DisplayHandle<'a>,
    pub window_handle: WindowHandle<'a>,
    pub preferred_format: Option<ImageFormat>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum PresentMode {
    Immediate,
    Fifo,
}

#[derive(Clone)]
pub struct Swapchain {
    pub(crate) device: Device,
    pub(crate) id: u64,
    pub(crate) current_acquired_image: Option<SwapchainImage>,
}

#[derive(Clone, Copy)]
pub struct SwapchainImage {
    pub image: Image,
    pub view: ImageView,
}

impl Swapchain {
    pub fn set_present_mode(&mut self, present_mode: PresentMode) -> MgpuResult<bool> {
        todo!()
        // self.pimpl.set_present_mode(present_mode)
    }
    pub fn resized(
        &mut self,
        new_extents: Extents2D,
        window_handle: WindowHandle,
        display_handle: DisplayHandle,
    ) -> MgpuResult<()> {
        self.device
            .hal
            .swapchain_on_resized(self.id, new_extents, window_handle, display_handle)
    }
    pub fn acquire_next_image(&mut self) -> MgpuResult<SwapchainImage> {
        let image = self.device.hal.swapchain_acquire_next_image(self.id)?;
        let old_image = self.current_acquired_image.replace(image);
        assert!(old_image.is_none(), "Called acquire without present!");

        Ok(image)
    }
    pub fn current_format(&self) -> ImageFormat {
        todo!()
        // self.pimpl.current_format()
    }
    pub fn present(&mut self) -> MgpuResult<()> {
        let image = self
            .current_acquired_image
            .take()
            .expect("Called present without acquire!");
        self.device.write_rdg().add_present_pass(image, self.id);
        Ok(())
    }
}
