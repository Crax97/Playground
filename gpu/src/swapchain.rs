use std::cell::Cell;
use std::{mem::MaybeUninit, num::NonZeroU32, ptr::addr_of, sync::Arc};

use ash::vk::Handle;
use ash::{
    extensions::khr::Surface,
    prelude::VkResult,
    vk::{
        self, ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR, Format, ImageUsageFlags,
        ImageViewCreateFlags, ImageViewType, PresentInfoKHR, PresentModeKHR, SemaphoreCreateFlags,
        SemaphoreCreateInfo, SharingMode, StructureType, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
        SurfaceKHR, SwapchainCreateFlagsKHR, SwapchainCreateInfoKHR, SwapchainKHR,
    },
    Device,
};
use log::{info, trace, warn};
use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use winit::window::Window;

use crate::{
    Extent2D, FenceCreateFlags, FenceCreateInfo, ImageAspectFlags, ImageHandle,
    ImageSubresourceRange, ImageViewHandle, PresentMode, ToVk, VkGpu, VkImage, VkImageView,
};

use super::{GPUFence, GPUSemaphore, GpuThreadSharedState};

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

pub struct SwapchainFrame {
    pub in_flight_fence: GPUFence,
    pub render_finished_semaphore: GPUSemaphore,
    pub image_available_semaphore: GPUSemaphore,
}

impl SwapchainFrame {
    fn new(device: Device) -> anyhow::Result<Self> {
        let in_flight_fence = GPUFence::create(
            device.clone(),
            Some("In Flight Fence"),
            &FenceCreateInfo {
                flags: FenceCreateFlags::SIGNALED,
            },
        )?;

        let render_finished_semaphore = GPUSemaphore::create(
            device.clone(),
            Some("Render finished semaphore"),
            &SemaphoreCreateInfo {
                s_type: StructureType::SEMAPHORE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: SemaphoreCreateFlags::empty(),
            },
        )?;

        let image_available_semaphore = GPUSemaphore::create(
            device,
            Some("Image available semaphore"),
            &SemaphoreCreateInfo {
                s_type: StructureType::SEMAPHORE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: SemaphoreCreateFlags::empty(),
            },
        )?;
        Ok(Self {
            in_flight_fence,
            render_finished_semaphore,
            image_available_semaphore,
        })
    }
}

pub struct VkSwapchain {
    pub(super) surface_extension: Surface,
    pub swapchain_extension: ash::extensions::khr::Swapchain,
    pub(super) surface: SurfaceKHR,
    pub(super) present_mode: PresentModeKHR,
    pub(super) swapchain_image_count: NonZeroU32,
    pub(super) present_extent: Extent2D,
    pub present_format: SurfaceFormatKHR,
    pub(super) supported_present_modes: Vec<PresentModeKHR>,
    pub(super) supported_presentation_formats: Vec<SurfaceFormatKHR>,
    pub(super) surface_capabilities: SurfaceCapabilitiesKHR,
    pub current_swapchain: SwapchainKHR,
    pub(super) current_swapchain_images: Vec<ImageHandle>,
    pub(super) current_swapchain_image_views: Vec<ImageViewHandle>,
    pub(super) frames_in_flight: Vec<SwapchainFrame>,

    current_swapchain_index: Cell<u32>,
    state: Arc<GpuThreadSharedState>,
    pub current_frame: Cell<usize>,
    pub next_image_fence: GPUFence,

    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
}

impl VkSwapchain {
    pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

    pub fn new(gpu: &VkGpu, window: &Window) -> anyhow::Result<Self> {
        let state = gpu.state.clone();
        let surface_extension = Surface::new(&state.entry, &state.instance);
        let swapchain_extension =
            ash::extensions::khr::Swapchain::new(&state.instance, &state.logical_device);

        let next_image_fence = GPUFence::create(
            state.logical_device.clone(),
            Some("Next image fence"),
            &FenceCreateInfo {
                flags: FenceCreateFlags::empty(),
            },
        )?;

        let present_extent = Extent2D {
            width: window.outer_size().width,
            height: window.outer_size().height,
        };

        let mut frames_in_flight = vec![];
        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let swapchain_frame = SwapchainFrame::new(state.logical_device.clone())?;
            frames_in_flight.push(swapchain_frame);
        }

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
            current_swapchain_index: Cell::new(0),
            frames_in_flight,
            next_image_fence,
            current_frame: Cell::new(0),
            state,
            window_handle: window.raw_window_handle(),
            display_handle: window.raw_display_handle(),
        };
        me.recreate_swapchain()?;
        me.log_supported_features();
        Ok(me)
    }

    pub fn acquire_next_image(&mut self) -> anyhow::Result<(ImageHandle, ImageViewHandle)> {
        let current_frame = &self.frames_in_flight[self.current_frame.get()];
        let wait_semaphore = current_frame.image_available_semaphore.inner;

        unsafe {
            self.state
                .logical_device
                .wait_for_fences(&[current_frame.in_flight_fence.inner], true, u64::MAX)
                .unwrap();
            self.state
                .logical_device
                .reset_fences(&[current_frame.in_flight_fence.inner])
                .unwrap();
        }
        let next_image_fence = self.next_image_fence.inner;
        loop {
            let (next_image, suboptimal) = unsafe {
                self.swapchain_extension.acquire_next_image(
                    self.current_swapchain,
                    u64::MAX,
                    wait_semaphore,
                    next_image_fence,
                )
            }?;
            unsafe {
                self.state
                    .logical_device
                    .wait_for_fences(&[next_image_fence], true, u64::MAX)?;
                self.state
                    .logical_device
                    .reset_fences(&[next_image_fence])?;
            }
            if !suboptimal {
                let image_view = self
                    .current_swapchain_image_views
                    .get(next_image as usize)
                    .unwrap();
                self.current_swapchain_index.replace(next_image);
                let image_idx = self.current_swapchain_index.get() as usize;
                return Ok(unsafe {
                    (
                        self.current_swapchain_images[image_idx].clone(),
                        image_view.clone(),
                    )
                });
            } else {
                self.recreate_swapchain()?;
            }
        }
    }

    pub fn get_current_swapchain_frame(&self) -> &SwapchainFrame {
        &self.frames_in_flight[self.current_frame.get()]
    }

    pub fn present(&self) -> anyhow::Result<bool> {
        unsafe {
            let current_frame = self.get_current_swapchain_frame();
            let wait_semaphore = current_frame.render_finished_semaphore.inner;
            let result = self.swapchain_extension.queue_present(
                self.state.graphics_queue,
                &PresentInfoKHR {
                    s_type: StructureType::PRESENT_INFO_KHR,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 1,
                    p_wait_semaphores: addr_of!(wait_semaphore),
                    swapchain_count: 1,
                    p_swapchains: &self.current_swapchain as *const SwapchainKHR,
                    p_image_indices: self.current_swapchain_index.as_ptr(),
                    p_results: std::ptr::null_mut(),
                },
            );

            if let Err(e) = result {
                if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                    return Ok(false);
                } else {
                    anyhow::bail!(e)
                }
            }
        }

        self.current_frame
            .replace((self.current_frame.get() + 1) % Self::MAX_FRAMES_IN_FLIGHT);
        Ok(true)
    }

    fn pick_swapchain_format(supported_formats: &[SurfaceFormatKHR]) -> SurfaceFormatKHR {
        for format in supported_formats.iter() {
            if format.format == Format::R8G8B8A8_SRGB {
                return *format;
            }
        }

        supported_formats[0]
    }

    pub fn recreate_swapchain(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.state.logical_device.device_wait_idle()?;
            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None);
            self.surface_extension.destroy_surface(self.surface, None);
        };

        self.surface = unsafe {
            ash_window::create_surface(
                &self.state.entry,
                &self.state.instance,
                self.display_handle,
                self.window_handle,
                None,
            )?
        };

        self.supported_presentation_formats = unsafe {
            self.surface_extension.get_physical_device_surface_formats(
                self.state.physical_device.physical_device,
                self.surface,
            )
        }?;

        self.surface_capabilities = unsafe {
            self.surface_extension
                .get_physical_device_surface_capabilities(
                    self.state.physical_device.physical_device,
                    self.surface,
                )
        }?;

        self.supported_present_modes = unsafe {
            self.surface_extension
                .get_physical_device_surface_present_modes(
                    self.state.physical_device.physical_device,
                    self.surface,
                )
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
            image_extent: self.present_extent.to_vk(),
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
        self.recreate_swapchain_image_views()?;

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

        self.current_swapchain_images = images
            .iter()
            .map(|i| {
                VkImage::wrap(
                    self.state.logical_device.clone(),
                    Some("Swapchain image"),
                    *i,
                    self.extents(),
                    self.present_format().into(),
                )
            })
            .map(|img| {
                let handle =
                    <ImageHandle as crate::Handle>::new(img.inner.as_raw(), self.state.clone());
                self.state
                    .allocated_resources
                    .borrow_mut()
                    .insert(&handle, img);
                handle
            })
            .collect();
        Ok(())
    }

    fn recreate_swapchain_image_views(&mut self) -> VkResult<()> {
        self.current_swapchain_image_views.clear();
        self.current_swapchain_image_views
            .resize_with(self.current_swapchain_images.len(), || {
                <ImageViewHandle as crate::Handle>::null()
            });
        for (i, image) in self.current_swapchain_images.iter().enumerate() {
            let vk_image = self
                .state
                .allocated_resources
                .borrow()
                .resolve::<VkImage>(&image)
                .inner;
            let view_info = ash::vk::ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ImageViewCreateFlags::empty(),
                image: vk_image,
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
                }
                .to_vk(),
            };
            let view = VkImageView::create(
                self.state.logical_device.clone(),
                Some("Swapchain Image View"),
                &view_info,
                view_info.format.into(),
                image.clone(),
                self.present_extent,
            )?;
            let view_handle =
                <ImageViewHandle as crate::Handle>::new(view.as_raw(), self.state.clone());
            self.state
                .allocated_resources
                .borrow_mut()
                .insert(&view_handle, view);
            self.current_swapchain_image_views[i] = view_handle;
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

    fn drop_swapchain_structs(&self) {
        unsafe {
            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None);
            self.surface_extension.destroy_surface(self.surface, None);
        }
    }
    pub(crate) fn drop_swapchain(&mut self) {
        self.drop_swapchain_structs();
    }

    pub fn select_present_mode(&mut self, present_mode: PresentMode) -> anyhow::Result<()> {
        self.present_mode = present_mode.to_vk();
        self.recreate_swapchain()
    }

    pub fn extents(&self) -> Extent2D {
        self.present_extent
    }

    pub fn present_format(&self) -> Format {
        self.present_format.format
    }
}

impl ToVk for PresentMode {
    type Inner = PresentModeKHR;

    fn to_vk(&self) -> Self::Inner {
        match self {
            PresentMode::Immediate => PresentModeKHR::IMMEDIATE,
            PresentMode::Fifo => PresentModeKHR::FIFO,
            PresentMode::Mailbox => PresentModeKHR::MAILBOX,
        }
    }
}

impl Drop for VkSwapchain {
    fn drop(&mut self) {
        self.drop_swapchain()
    }
}
