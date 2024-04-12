use std::sync::atomic::AtomicUsize;

use ash::{
    khr::{surface, swapchain},
    vk,
};
use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::{
    hal::vulkan::{get_allocation_callbacks, util::ToVk},
    Image, ImageFormat, PresentMode, SwapchainCreationInfo, SwapchainImage, SwapchainImpl,
};

use super::{VulkanHal, VulkanHalResult};

pub struct VulkanSwapchain {
    swapchain_instance: swapchain::Instance,
    swapchain_device: swapchain::Device,
    surface_instance: surface::Instance,
    data: SwapchainData,
}

struct SwapchainData {
    capabilities: vk::SurfaceCapabilitiesKHR,
    present_modes: Vec<vk::PresentModeKHR>,
    surface_formats: Vec<vk::SurfaceFormatKHR>,
    current_format: vk::SurfaceFormatKHR,

    images: Vec<SwapchainImage>,
    current_swapchain_image: AtomicUsize,

    swapchain: vk::SwapchainKHR,
    surface: vk::SurfaceKHR,
}

struct SwapchainRecreateParams<'a> {
    hal: &'a VulkanHal,
    surface_instance: &'a surface::Instance,
    swapchain_device: &'a swapchain::Device,
    display_handle: DisplayHandle<'a>,
    window_handle: WindowHandle<'a>,
    preferred_format: Option<ImageFormat>,
    preferred_present_mode: PresentMode,
    old_swapchain: vk::SwapchainKHR,
}

#[derive(Debug)]
pub enum SwapchainError {
    PresentNotSupported,
}

impl VulkanSwapchain {
    pub(crate) fn create(
        hal: &VulkanHal,
        swapchain_info: &SwapchainCreationInfo,
    ) -> VulkanHalResult<Self> {
        let device = &hal.logical_device.handle;
        let swapchain_instance = swapchain::Instance::new(&hal.entry, &hal.instance);
        let swapchain_device = swapchain::Device::new(&hal.instance, device);
        let surface_instance = surface::Instance::new(&hal.entry, &hal.instance);

        let swapchain_create_info = SwapchainRecreateParams {
            hal,
            swapchain_device: &swapchain_device,
            surface_instance: &surface_instance,
            window_handle: swapchain_info.window_handle,
            display_handle: swapchain_info.display_handle,
            preferred_format: swapchain_info.preferred_format,
            preferred_present_mode: PresentMode::Immediate,
            old_swapchain: vk::SwapchainKHR::null(),
        };

        let swapchain_data = Self::recreate(swapchain_create_info)?;

        Ok(Self {
            swapchain_instance,
            swapchain_device,
            surface_instance,
            data: swapchain_data,
        })
    }

    fn recreate(
        swapchain_data_create_info: SwapchainRecreateParams,
    ) -> VulkanHalResult<SwapchainData> {
        let SwapchainRecreateParams {
            hal,
            surface_instance,
            swapchain_device,
            display_handle,
            window_handle,
            preferred_format,
            preferred_present_mode,
            old_swapchain,
        } = swapchain_data_create_info;
        let surface = unsafe {
            ash_window::create_surface(
                &hal.entry,
                &hal.instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                super::get_allocation_callbacks(),
            )
        }?;
        let pdevice = hal.physical_device.handle;
        let device = &hal.logical_device.handle;

        let supported_present = unsafe {
            surface_instance.get_physical_device_surface_support(
                pdevice,
                hal.logical_device.graphics_queue.family_index,
                surface,
            )?
        };

        if !supported_present {
            return Err(super::VulkanHalError::SwapchainError(
                SwapchainError::PresentNotSupported,
            ));
        }

        let surface_formats =
            unsafe { surface_instance.get_physical_device_surface_formats(pdevice, surface)? };
        let surface_capabilities =
            unsafe { surface_instance.get_physical_device_surface_capabilities(pdevice, surface)? };
        let present_modes = unsafe {
            surface_instance.get_physical_device_surface_present_modes(pdevice, surface)?
        };

        let transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let image_format = if let Some(preferred_format) = preferred_format {
            let preferred_format = preferred_format.to_vk();
            if let Some(format) = surface_formats
                .iter()
                .find(|f| f.format == preferred_format)
                .copied()
            {
                format
            } else {
                surface_formats[0]
            }
        } else {
            surface_formats[0]
        };
        let present_mode = if present_modes.contains(&preferred_present_mode.to_vk()) {
            preferred_present_mode.to_vk()
        } else {
            vk::PresentModeKHR::IMMEDIATE
        };
        let image_count = surface_capabilities
            .max_image_count
            .min(hal.configuration.frames_in_flight);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_extent(surface_capabilities.current_extent)
            .image_format(image_format.format)
            .image_color_space(image_format.color_space)
            .present_mode(present_mode)
            .pre_transform(transform)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .old_swapchain(old_swapchain)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .image_array_layers(1)
            .clipped(true);

        let swapchain = unsafe {
            swapchain_device.create_swapchain(&swapchain_create_info, get_allocation_callbacks())?
        };

        let images = unsafe { swapchain_device.get_swapchain_images(swapchain)? };
        let mut swapchain_images = vec![];
        for image in images {
            let mut image_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .format(image_format.format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .layer_count(1)
                        .level_count(1),
                )
                .view_type(vk::ImageViewType::TYPE_2D);
            let mut view =
                unsafe { device.create_image_view(&image_view_info, get_allocation_callbacks())? };
            let image = unsafe { hal.wrap_raw_image(image, Some("Swapchain image"))? };
            let view =
                unsafe { hal.wrap_raw_image_view(image, view, Some("Swapchain image view"))? };

            swapchain_images.push(SwapchainImage { image, view })
        }

        Ok(SwapchainData {
            capabilities: surface_capabilities,
            present_modes,
            surface_formats,
            current_format: image_format,
            surface,
            swapchain,
            images: swapchain_images,
            current_swapchain_image: AtomicUsize::new(0),
        })
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.data.swapchain, get_allocation_callbacks());
            self.surface_instance
                .destroy_surface(self.data.surface, get_allocation_callbacks());
        }
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
