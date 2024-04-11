use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::{Extents2D, Image, ImageFormat, ImageView, MgpuResult};

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

pub struct Swapchain {
    pub(crate) pimpl: Box<dyn SwapchainImpl>,
}

pub struct SwapchainImage {
    pub image: Image,
    pub view: ImageView,
}

impl Swapchain {
    pub fn set_present_mode(&mut self, present_mode: PresentMode) -> MgpuResult<bool> {
        self.pimpl.set_present_mode(present_mode)
    }
    pub fn resized(&mut self, new_extents: Extents2D) -> MgpuResult<()> {
        self.pimpl.resized(new_extents)
    }
    pub fn acquire_next_image(&mut self) -> MgpuResult<SwapchainImage> {
        self.pimpl.acquire_next_image()
    }
    pub fn current_format(&self) -> ImageFormat {
        self.pimpl.current_format()
    }
    pub fn present(&mut self) -> MgpuResult<()> {
        self.pimpl.present()
    }
}

pub(crate) trait SwapchainImpl: Send + Sync {
    fn set_present_mode(&mut self, present_mode: PresentMode) -> MgpuResult<bool>;
    fn resized(&mut self, new_extents: Extents2D) -> MgpuResult<()>;
    fn acquire_next_image(&mut self) -> MgpuResult<SwapchainImage>;
    fn current_format(&self) -> ImageFormat;
    fn present(&mut self) -> MgpuResult<()>;
}
