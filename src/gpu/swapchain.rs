use std::num::NonZeroU32;

use ash::{
    extensions::khr::Surface,
    prelude::VkResult,
    vk::{
        self, ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR, Extent2D, Fence,
        FenceCreateFlags, FenceCreateInfo, Format, Image, ImageAspectFlags, ImageSubresourceRange,
        ImageUsageFlags, ImageView, ImageViewCreateFlags, ImageViewCreateInfo, ImageViewType,
        PhysicalDevice, PresentInfoKHR, PresentModeKHR, Queue, Semaphore, SemaphoreCreateFlags,
        SemaphoreCreateInfo, SharingMode, StructureType, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
        SurfaceKHR, SwapchainCreateFlagsKHR, SwapchainCreateInfoKHR, SwapchainKHR,
    },
    Device, Entry, Instance,
};
use log::{info, trace, warn};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

mod util {
    use ash::vk::{PresentModeKHR, SurfaceFormatKHR};

    pub(super) fn stringify_present_mode(mode: PresentModeKHR) -> &'static str {
        match mode {
            PresentModeKHR::FIFO => "FIFO",
            PresentModeKHR::FIFO_RELAXED => "FIFO_RELAXED",
            PresentModeKHR::MAILBOX => "MAILBOX",
            PresentModeKHR::IMMEDIATE => "IMMEDIATE",
            PresentModeKHR::SHARED_CONTINUOUS_REFRESH => "SHARED_CONTINUOUS_REFRESH",
            PresentModeKHR::SHARED_DEMAND_REFRESH => "SHARED_DEMAND_REFRESH",
            _ => unreachable!(),
        }
    }

    pub(super) fn stringify_presentation_format(format: SurfaceFormatKHR) -> String {
        format!(
            "Image format: {:?}, Color space: {:?}",
            format.format, format.color_space
        )
    }
}
pub(crate) struct Swapchain {
    pub surface_extension: Surface,
    pub swapchain_extension: ash::extensions::khr::Swapchain,
    pub surface: SurfaceKHR,
    pub present_mode: PresentModeKHR,
    pub swapchain_image_count: NonZeroU32,
    pub present_extent: Extent2D,
    pub present_format: SurfaceFormatKHR,
    pub supported_present_modes: Vec<PresentModeKHR>,
    pub supported_presentation_formats: Vec<SurfaceFormatKHR>,
    pub surface_capabilities: SurfaceCapabilitiesKHR,
    pub current_swapchain: SwapchainKHR,
    pub current_swapchain_images: Vec<Image>,
    pub current_swapchain_image_views: Vec<ImageView>,
    pub next_image_fence: Fence,
    pub in_flight_fence: Fence,
    pub render_finished_semaphore: Semaphore,
    pub image_available_semaphore: Semaphore,

    current_swapchain_index: u32,

    logical_device: Device,
}

impl Swapchain {
    pub(crate) fn new(
        entry: &Entry,
        instance: &Instance,
        physical_device: &PhysicalDevice,
        logical_device: &Device,
        window: &Window,
    ) -> VkResult<Self> {
        let surface_extension = Surface::new(&entry, &instance);
        let swapchain_extension = ash::extensions::khr::Swapchain::new(&instance, &logical_device);

        let next_image_fence = unsafe {
            let create_info = FenceCreateInfo {
                s_type: StructureType::FENCE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: FenceCreateFlags::empty(),
            };
            logical_device.create_fence(&create_info, None)
        }?;
        let in_flight_fence = unsafe {
            let create_info = FenceCreateInfo {
                s_type: StructureType::FENCE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: FenceCreateFlags::empty(),
            };
            logical_device.create_fence(&create_info, None)
        }?;

        let render_finished_semaphore = unsafe {
            logical_device.create_semaphore(
                &SemaphoreCreateInfo {
                    s_type: StructureType::SEMAPHORE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: SemaphoreCreateFlags::empty(),
                },
                None,
            )?
        };

        let image_available_semaphore = unsafe {
            logical_device.create_semaphore(
                &SemaphoreCreateInfo {
                    s_type: StructureType::SEMAPHORE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: SemaphoreCreateFlags::empty(),
                },
                None,
            )?
        };

        let present_extent = Extent2D {
            width: window.outer_size().width,
            height: window.outer_size().height,
        };

        let mut me = Self {
            surface_extension,
            swapchain_extension,
            surface: SurfaceKHR::null(),
            present_mode: PresentModeKHR::FIFO,
            swapchain_image_count: NonZeroU32::new(3).unwrap(),
            present_extent,
            present_format: SurfaceFormatKHR::builder().build(),
            supported_present_modes: vec![],
            supported_presentation_formats: vec![],
            surface_capabilities: SurfaceCapabilitiesKHR::builder().build(),
            current_swapchain: SwapchainKHR::null(),
            current_swapchain_images: vec![],
            current_swapchain_image_views: vec![],
            current_swapchain_index: 0,
            next_image_fence,
            in_flight_fence,
            render_finished_semaphore,
            image_available_semaphore,
            logical_device: logical_device.clone(),
        };
        me.recreate_swapchain(entry, instance, physical_device, logical_device, window)?;
        me.log_supported_features();
        Ok(me)
    }

    pub(crate) fn acquire_next_image(&mut self, logical_device: &Device) -> VkResult<ImageView> {
        let (next_image, suboptimal) = unsafe {
            self.swapchain_extension.acquire_next_image(
                self.current_swapchain,
                200000,
                self.image_available_semaphore,
                self.next_image_fence,
            )
        }?;
        unsafe {
            logical_device.wait_for_fences(&[self.next_image_fence], true, 200000)?;
            logical_device.reset_fences(&[self.next_image_fence])?;
        }
        if !suboptimal {
            let image_view = self.current_swapchain_image_views.get(next_image as usize);
            self.current_swapchain_index = next_image;
            Ok(image_view.unwrap().clone())
        } else {
            Err(vk::Result::SUBOPTIMAL_KHR)
        }
    }

    pub(crate) fn present(&self, graphics_queue: Queue) -> VkResult<bool> {
        unsafe {
            self.swapchain_extension.queue_present(
                graphics_queue,
                &PresentInfoKHR {
                    s_type: StructureType::PRESENT_INFO_KHR,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &self.render_finished_semaphore as *const Semaphore,
                    swapchain_count: 1,
                    p_swapchains: &self.current_swapchain as *const SwapchainKHR,
                    p_image_indices: &self.current_swapchain_index as *const u32,
                    p_results: std::ptr::null_mut(),
                },
            )
        }
    }

    fn pick_swapchain_format(supported_formats: &Vec<SurfaceFormatKHR>) -> SurfaceFormatKHR {
        for format in supported_formats.iter() {
            if format.format == Format::R8G8B8A8_UNORM {
                return format.clone();
            }
        }

        return supported_formats[0].clone();
    }

    pub(crate) fn recreate_swapchain(
        &mut self,
        entry: &Entry,
        instance: &Instance,
        physical_device: &PhysicalDevice,
        logical_device: &Device,
        window: &Window,
    ) -> VkResult<()> {
        unsafe {
            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None);
            self.surface_extension.destroy_surface(self.surface, None);
        };

        self.surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )?
        };

        self.supported_presentation_formats = unsafe {
            self.surface_extension
                .get_physical_device_surface_formats(*physical_device, self.surface)
        }?;

        self.surface_capabilities = unsafe {
            self.surface_extension
                .get_physical_device_surface_capabilities(*physical_device, self.surface)
        }?;

        self.supported_present_modes = unsafe {
            self.surface_extension
                .get_physical_device_surface_present_modes(*physical_device, self.surface)
        }?;
        self.present_format = Self::pick_swapchain_format(&self.supported_presentation_formats);

        self.validate_selected_swapchain_settings();

        let swapchain_creation_info = SwapchainCreateInfoKHR {
            s_type: StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            flags: SwapchainCreateFlagsKHR::empty(),
            surface: self.surface,
            min_image_count: self.swapchain_image_count.get(),
            image_format: self.present_format.format,
            image_color_space: self.present_format.color_space,
            image_extent: self.present_extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            pre_transform: self.surface_capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: self.present_mode,
            clipped: vk::TRUE,
            old_swapchain: SwapchainKHR::null(),
        };

        unsafe {
            self.current_swapchain = self
                .swapchain_extension
                .create_swapchain(&swapchain_creation_info, None)?;
        };

        trace!(
            "Created a new swapchain with present format {:?}, present mode {:?} and present extents {:?}",
            &self.present_format, &self.present_mode, &self.present_extent
        );
        self.recreate_swapchain_images()?;
        self.recreate_swapchain_image_views(logical_device)?;

        Ok(())
    }
    fn validate_selected_swapchain_settings(&mut self) {
        if !self.supported_present_modes.contains(&self.present_mode) {
            warn!(
                "Device does not support extension mode {:?}, selecting FIFO, which must be supported as per specification",
                self.present_mode
            );
            self.present_mode = PresentModeKHR::FIFO
        };

        if !self
            .supported_presentation_formats
            .contains(&self.present_format)
        {
            warn!(
                "Device does not support present format {:?}, selecting the first available one",
                &self.present_format
            );
            self.present_format = self.supported_presentation_formats[0];
        }

        if self.swapchain_image_count.get() < self.surface_capabilities.min_image_count
            || self.swapchain_image_count.get() > self.surface_capabilities.max_image_count
        {
            warn!(
                "Device does not support less than {} / more than {} swapchain images! Clamping",
                self.surface_capabilities.min_image_count,
                self.surface_capabilities.max_image_count
            );
            self.swapchain_image_count = self.swapchain_image_count.clamp(
                NonZeroU32::new(self.surface_capabilities.min_image_count).unwrap(),
                NonZeroU32::new(self.surface_capabilities.max_image_count).unwrap(),
            );
        }

        let min_exent = self.surface_capabilities.min_image_extent;
        let max_exent = self.surface_capabilities.max_image_extent;
        let current_extent = self.present_extent;
        if current_extent.width < min_exent.width
            || current_extent.height < min_exent.height
            || current_extent.width > max_exent.width
            || current_extent.height > max_exent.height
        {
            warn!(
                "Device does not support extents smaller than {:?} / greather than {:?}! Clamping",
                self.surface_capabilities.min_image_extent,
                self.surface_capabilities.max_image_extent
            );

            self.present_extent = Extent2D {
                width: self.present_extent.width.clamp(
                    self.surface_capabilities.min_image_extent.width,
                    self.surface_capabilities.max_image_extent.width,
                ),
                height: self.present_extent.height.clamp(
                    self.surface_capabilities.min_image_extent.height,
                    self.surface_capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn recreate_swapchain_images(&mut self) -> VkResult<()> {
        let images = unsafe {
            self.swapchain_extension
                .get_swapchain_images(self.current_swapchain)
        }?;

        self.current_swapchain_images = images;
        Ok(())
    }

    fn recreate_swapchain_image_views(&mut self, logical_device: &Device) -> VkResult<()> {
        self.drop_image_views();
        self.current_swapchain_image_views
            .resize(self.current_swapchain_images.len(), ImageView::null());
        for (i, image) in self.current_swapchain_images.iter().enumerate() {
            let view_info = ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ImageViewCreateFlags::empty(),
                image: *image,
                view_type: ImageViewType::TYPE_2D,
                format: self.present_format.format,
                components: ComponentMapping {
                    r: ComponentSwizzle::R,
                    g: ComponentSwizzle::G,
                    b: ComponentSwizzle::B,
                    a: ComponentSwizzle::A,
                },
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let view: ImageView = unsafe { logical_device.create_image_view(&view_info, None)? };
            self.current_swapchain_image_views[i] = view;
        }

        Ok(())
    }

    fn log_supported_features(&self) {
        info!("Device supports the following present modes:");
        for present_mode in &self.supported_present_modes {
            info!("\t{}", util::stringify_present_mode(*present_mode));
        }

        info!("Device supports the following presentation formats:");
        for presentation_format in &self.supported_presentation_formats {
            info!(
                "\t{}",
                util::stringify_presentation_format(*presentation_format)
            );
        }

        info!("Device has the folowing limits:");
        info!(
            "\tMin/Max swapchain images: {}/{}",
            &self.surface_capabilities.min_image_count, self.surface_capabilities.max_image_count
        );
        info!(
            "\tMin/Max swapchain extents: {}x{}/{}x{}",
            self.surface_capabilities.min_image_extent.width,
            self.surface_capabilities.min_image_extent.height,
            self.surface_capabilities.max_image_extent.width,
            self.surface_capabilities.max_image_extent.height
        );
    }

    fn drop_image_views(&self) {
        for view in self.current_swapchain_image_views.iter() {
            unsafe {
                self.logical_device.destroy_image_view(*view, None);
            }
        }
    }

    fn drop_swapchain_structs(&self) {
        unsafe {
            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None);
            self.surface_extension.destroy_surface(self.surface, None);
            self.logical_device
                .destroy_fence(self.in_flight_fence, None);
            self.logical_device
                .destroy_fence(self.next_image_fence, None);
            self.logical_device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.logical_device
                .destroy_semaphore(self.image_available_semaphore, None);
        }
    }
    pub(crate) fn drop_swapchain(&mut self) {
        self.drop_image_views();
        self.drop_swapchain_structs();
    }
}
