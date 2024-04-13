use std::sync::{atomic::AtomicUsize, Arc};

use ash::{
    khr::{surface, swapchain},
    vk::{self, AccessFlags2, ImageLayout, PipelineStageFlags2},
};
use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::{
    hal::{
        vulkan::{get_allocation_callbacks, util::ToVk},
        Hal,
    },
    Image, ImageFormat, MgpuResult, PresentMode, SwapchainCreationInfo, SwapchainImage,
};

use super::{FramesInFlight, VulkanHal, VulkanHalError, VulkanHalResult};

pub struct VulkanSwapchain {
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) swapchain_instance: swapchain::Instance,
    pub(crate) swapchain_device: swapchain::Device,
    pub(crate) surface_instance: surface::Instance,
    pub(crate) data: SwapchainData,
    pub(crate) frames_in_flight: Arc<FramesInFlight>,
    pub(crate) acquire_fence: vk::Fence,
    pub(crate) current_image_index: Option<u32>,
}

pub(crate) struct SwapchainData {
    pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,
    pub(crate) present_modes: Vec<vk::PresentModeKHR>,
    pub(crate) surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub(crate) current_format: vk::SurfaceFormatKHR,
    pub(crate) images: Vec<SwapchainImage>,
    pub(crate) current_swapchain_image: AtomicUsize,
    pub(crate) surface: vk::SurfaceKHR,
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

        let (handle, swapchain_data) = Self::recreate(swapchain_create_info)?;

        let acquire_fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), get_allocation_callbacks())?
        };

        Ok(Self {
            handle,
            swapchain_instance,
            swapchain_device,
            surface_instance,
            data: swapchain_data,
            frames_in_flight: hal.frames_in_flight.clone(),
            acquire_fence,
            current_image_index: None,
        })
    }

    pub(crate) fn rebuild(
        &mut self,
        hal: &VulkanHal,
        swapchain_info: &SwapchainCreationInfo,
    ) -> MgpuResult<()> {
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.handle, get_allocation_callbacks())
        };
        for image in std::mem::take(&mut self.data.images) {
            hal.destroy_image_view(image.view)?;
            hal.destroy_image(image.image)?;
        }

        let swapchain_create_info = SwapchainRecreateParams {
            hal,
            swapchain_device: &self.swapchain_device,
            surface_instance: &self.surface_instance,
            window_handle: swapchain_info.window_handle,
            display_handle: swapchain_info.display_handle,
            preferred_format: swapchain_info.preferred_format,
            preferred_present_mode: PresentMode::Immediate,
            old_swapchain: vk::SwapchainKHR::null(),
        };
        let (handle, swapchain_data) = Self::recreate(swapchain_create_info)?;
        self.handle = handle;
        self.data = swapchain_data;
        Ok(())
    }

    fn recreate(
        swapchain_data_create_info: SwapchainRecreateParams,
    ) -> VulkanHalResult<(vk::SwapchainKHR, SwapchainData)> {
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
        Self::transition_images_to_present(hal, &images)?;
        let mut swapchain_images = vec![];
        for image in images {
            let image_view_info = vk::ImageViewCreateInfo::default()
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
            let view =
                unsafe { device.create_image_view(&image_view_info, get_allocation_callbacks())? };
            let image = unsafe { hal.wrap_raw_image(image, Some("Swapchain image"))? };
            let view =
                unsafe { hal.wrap_raw_image_view(image, view, Some("Swapchain image view"))? };

            swapchain_images.push(SwapchainImage { image, view })
        }

        let swapchain_data = SwapchainData {
            capabilities: surface_capabilities,
            present_modes,
            surface_formats,
            current_format: image_format,
            surface,
            images: swapchain_images,
            current_swapchain_image: AtomicUsize::new(0),
        };
        Ok((swapchain, swapchain_data))
    }

    fn transition_images_to_present(hal: &VulkanHal, images: &[vk::Image]) -> VulkanHalResult<()> {
        let device = &hal.logical_device.handle;

        let barriers = images
            .iter()
            .map(|image| {
                vk::ImageMemoryBarrier2::default()
                    .image(*image)
                    .src_access_mask(AccessFlags2::empty())
                    .dst_access_mask(AccessFlags2::empty())
                    .src_stage_mask(PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::PRESENT_SRC_KHR)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
            })
            .collect::<Vec<_>>();

        unsafe {
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(hal.logical_device.graphics_queue.family_index),
                get_allocation_callbacks(),
            )?;
            let command_buffer = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0];
            let wait_fence =
                device.create_fence(&vk::FenceCreateInfo::default(), get_allocation_callbacks())?;

            let queue = hal.logical_device.graphics_queue.handle;
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            device.end_command_buffer(command_buffer)?;

            let cb = [command_buffer];
            let submits = vk::SubmitInfo::default().command_buffers(&cb);
            let submits = [submits];
            device.queue_submit(queue, &submits, wait_fence)?;
            device.wait_for_fences(&[wait_fence], true, u64::MAX)?;

            device.destroy_fence(wait_fence, get_allocation_callbacks());
            device.free_command_buffers(command_pool, &[command_buffer]);
            device.destroy_command_pool(command_pool, get_allocation_callbacks());
        }

        Ok(())
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        if self.data.surface == vk::SurfaceKHR::null() {
            return;
        }
        unsafe {
            // self.swapchain_device
            //     .destroy_swapchain(self.handle, get_allocation_callbacks());
            // self.surface_instance
            //     .destroy_surface(self.data.surface, get_allocation_callbacks());
        }
    }
}
