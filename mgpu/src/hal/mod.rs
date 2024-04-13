use crate::{
    rdg::{PassGroup, RdgNode},
    swapchain, DeviceConfiguration, DeviceInfo, Extents2D, Image, ImageView, MgpuResult,
};
use std::sync::Arc;

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

pub(crate) trait Hal: Send + Sync {
    fn begin_rendering(&self) -> MgpuResult<()>;
    fn execute_passes(&self, group: PassGroup, passes: &[RdgNode]) -> MgpuResult<()>;
    fn end_rendering(&self) -> MgpuResult<()>;

    #[cfg(feature = "swapchain")]
    fn create_swapchain_impl(&self, swapchain_info: &SwapchainCreationInfo) -> MgpuResult<u64>;

    #[cfg(feature = "swapchain")]
    fn swapchain_acquire_next_image(&self, id: u64) -> MgpuResult<SwapchainImage>;

    #[cfg(feature = "swapchain")]
    fn swapchain_on_resized(
        &self,
        id: u64,
        new_size: crate::Extents2D,
        window_handle: raw_window_handle::WindowHandle,
        display_handle: raw_window_handle::DisplayHandle,
    ) -> MgpuResult<()>;

    fn device_info(&self) -> DeviceInfo;

    fn destroy_image(&self, image: Image) -> MgpuResult<()>;
    fn destroy_image_view(&self, image_view: ImageView) -> MgpuResult<()>;
}

pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
    #[cfg(feature = "vulkan")]
    vulkan::VulkanHal::create(configuration)
}
