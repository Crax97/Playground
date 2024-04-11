use ash::{
    khr::{surface, swapchain},
    vk::{self, SurfaceKHR},
};

use crate::{SwapchainCreationInfo, SwapchainImpl};

use super::{VulkanHal, VulkanHalResult};

pub struct VulkanSwapchain;

impl VulkanSwapchain {
    pub(crate) fn create(
        hal: &VulkanHal,
        swapchain_info: &SwapchainCreationInfo,
    ) -> VulkanHalResult<Self> {
        let swapchain_ext = swapchain::Instance::new(&hal.entry, &hal.instance);
        let surface_ext = surface::Instance::new(&hal.entry, &hal.instance);
        let surface = unsafe {
            ash_window::create_surface(
                &hal.entry,
                &hal.instance,
                swapchain_info.display_handle.as_raw(),
                swapchain_info.window_handle.as_raw(),
                super::get_allocation_callbacks(),
            )
        }?;

        let pdevice = hal.physical_device.handle;
        let surface_formats =
            unsafe { surface_ext.get_physical_device_surface_formats(pdevice, surface)? };
        let surface_capabilities =
            unsafe { surface_ext.get_physical_device_surface_capabilities(pdevice, surface)? };
        let present_modes =
            unsafe { surface_ext.get_physical_device_surface_present_modes(pdevice, surface)? };

        todo!()
    }
}

impl SwapchainImpl for VulkanSwapchain {
    fn set_present_mode(&mut self, present_mode: crate::PresentMode) -> crate::MgpuResult<bool> {
        todo!()
    }

    fn resized(&mut self, new_extents: crate::Extents2D) -> crate::MgpuResult<()> {
        todo!()
    }

    fn acquire_next_image(&mut self) -> crate::MgpuResult<crate::SwapchainImage> {
        todo!()
    }

    fn current_format(&self) -> crate::ImageFormat {
        todo!()
    }

    fn present(&mut self) -> crate::MgpuResult<()> {
        todo!()
    }
}
