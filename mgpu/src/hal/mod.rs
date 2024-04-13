use crate::{swapchain, DeviceConfiguration, DeviceInfo, MgpuResult};
use std::sync::Arc;

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

#[cfg(feature = "vulkan")]
mod vulkan;

pub(crate) trait Hal: Send + Sync {
    #[cfg(feature = "swapchain")]
    fn create_swapchain_impl(&self, swapchain_info: &SwapchainCreationInfo) -> MgpuResult<u64>;

    #[cfg(feature = "swapchain")]
    fn swapchain_acquire_next_image(&self, id: u64) -> MgpuResult<SwapchainImage>;

    fn device_info(&self) -> DeviceInfo;
}

pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
    #[cfg(feature = "vulkan")]
    vulkan::VulkanHal::create(configuration)
}
