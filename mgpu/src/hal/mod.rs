use crate::{DeviceConfiguration, DeviceInfo, MgpuResult};
use std::sync::Arc;

#[cfg(feature = "vulkan")]
mod vulkan;

pub(crate) trait Hal: Send + Sync {
    fn device_info(&self) -> DeviceInfo;
}

pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
    #[cfg(feature = "vulkan")]
    vulkan::VulkanHal::create(configuration)
}
