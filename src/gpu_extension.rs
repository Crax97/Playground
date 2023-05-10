use std::{
    ffi::CStr,
    num::NonZeroU32,
    ops::{Deref, DerefMut},
    ptr::null,
    sync::Arc,
};

use ash::{
    extensions::khr::{Surface, Swapchain},
    prelude::VkResult,
    vk::{
        self, AttachmentDescription, AttachmentDescriptionFlags, AttachmentLoadOp,
        AttachmentReference, AttachmentStoreOp, ComponentMapping, ComponentSwizzle,
        CompositeAlphaFlagsKHR, Extent2D, Fence, FenceCreateFlags, Format, Framebuffer,
        FramebufferCreateFlags, FramebufferCreateInfo, Image, ImageAspectFlags, ImageLayout,
        ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateFlags,
        ImageViewCreateInfo, ImageViewType, PhysicalDevice, PipelineBindPoint, PresentInfoKHR,
        PresentModeKHR, RenderPass, RenderPassCreateFlags, RenderPassCreateInfo, SampleCountFlags,
        Semaphore, SemaphoreCreateFlags, SemaphoreCreateInfo, SharingMode, StructureType,
        SubpassDescription, SubpassDescriptionFlags, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
        SurfaceKHR, SwapchainCreateFlagsKHR, SwapchainCreateInfoKHR, SwapchainKHR, TRUE,
    },
    Entry, Instance,
};
use log::{info, trace, warn};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

pub struct GpuParameters<'a> {
    pub entry: &'a Entry,
    pub instance: &'a Instance,
}

pub trait GpuExtension {
    type SetupParameters;

    fn new(parameters: Self::SetupParameters, gpu_info: SharedGpuInfo) -> VkResult<Self>
    where
        Self: Sized;

    fn post_init(&mut self) -> VkResult<()> {
        Ok(())
    }

    fn accepts_queue_families(
        &self,
        selected_queues: QueueFamilies,
        physical_device: PhysicalDevice,
    ) -> VkResult<bool>;
    fn get_instance_extensions(parameters: &Self::SetupParameters) -> Vec<String>;
    fn get_device_extensions(
        parameters: &Self::SetupParameters,
        gpu_parameters: &GpuParameters,
    ) -> Vec<String>;
}

pub type DefaultExtensions = ();
impl GpuExtension for DefaultExtensions {
    type SetupParameters = ();

    fn new(_: Self::SetupParameters, _: SharedGpuInfo) -> VkResult<Self>
    where
        Self: Sized,
    {
        Ok(())
    }

    fn accepts_queue_families(&self, _: QueueFamilies, _: PhysicalDevice) -> VkResult<bool> {
        Ok(true)
    }

    fn get_instance_extensions(_: &Self::SetupParameters) -> Vec<String> {
        vec![]
    }

    fn get_device_extensions(_: &Self::SetupParameters, _: &GpuParameters) -> Vec<String> {
        vec![]
    }
}

macro_rules! define_gpu_extension {
    ($name:ident { $($mem:ident : $memty:ty,)* }
        $param_name:ident { $($param_mem:ident : $param_ty:ty,)* } ) => {
        pub struct $param_name<T: GpuExtension> {
            pub inner_params: T::SetupParameters,
            $(pub $param_mem : $param_ty,)*
        }

        pub struct $name<T: GpuExtension> {
            inner_extension: T,
            $($mem : $memty,)*
        }

        impl<T: GpuExtension> Deref for $name<T> {
            type Target = T;

            fn deref(&self) -> &Self::Target {
                &self.inner_extension
            }
        }

        impl<T: GpuExtension> DerefMut for $name<T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner_extension
            }
        }
    };
}

define_gpu_extension!(
    SwapchainExtension {
        gpu_info: SharedGpuInfo,
        extension_surface: Surface,
        swapchain_extension: Swapchain,
        khr_surface: SurfaceKHR,
        present_mode: PresentModeKHR,
        swapchain_image_count: NonZeroU32,
        present_extent: Extent2D,
        present_format: SurfaceFormatKHR,

        supported_present_modes: Vec<PresentModeKHR>,
        supported_presentation_formats: Vec<SurfaceFormatKHR>,
        device_capabilities: SurfaceCapabilitiesKHR,
        render_pass: RenderPass,
        window: Window,
        current_swapchain: SwapchainKHR,
        current_swapchain_images: Vec<Image>,
        current_swapchain_image_views: Vec<ImageView>,
        current_swapchain_framebuffers: Vec<Framebuffer>,

        next_image_fence: Fence,
        in_flight_fence: Fence,
        render_finished_semaphore: Semaphore,
    }

    SurfaceParamters {
        window: Window,
        window_size: Extent2D,
    }
);

impl<T: GpuExtension> GpuExtension for SwapchainExtension<T> {
    type SetupParameters = SurfaceParamters<T>;

    fn new(parameters: Self::SetupParameters) -> VkResult<Self>
    where
        Self: Sized,
    {
        let inner_extension = T::new(parameters.inner_params, gpu_info.clone())?;
        let khr_surface = unsafe {
            ash_window::create_surface(
                &gpu_info.entry,
                &gpu_info.instance,
                parameters.window.raw_display_handle(),
                parameters.window.raw_window_handle(),
                None,
            )
        }
        .unwrap();
        let swapchain_extension = Swapchain::new(&gpu_info.instance, &gpu_info.logical_device);

        let extension_surface = Surface::new(&gpu_info.entry, &gpu_info.instance);

        let (supported_formats, device_capabilities, supported_present_modes) = unsafe {
            let supported_formats = extension_surface
                .get_physical_device_surface_formats(gpu_info.physical_device, khr_surface)?;
            let device_capabilities = extension_surface
                .get_physical_device_surface_capabilities(gpu_info.physical_device, khr_surface)?;
            let supported_present_modes = extension_surface
                .get_physical_device_surface_present_modes(gpu_info.physical_device, khr_surface)?;
            (
                supported_formats,
                device_capabilities,
                supported_present_modes,
            )
        };

        let swapchain_format = supported_formats[0];

        let next_image_fence = unsafe {
            let create_info = vk::FenceCreateInfo {
                s_type: StructureType::FENCE_CREATE_INFO,
                p_next: null(),
                flags: FenceCreateFlags::empty(),
            };
            gpu_info.logical_device.create_fence(&create_info, None)
        }?;
        let in_flight_fence = unsafe {
            let create_info = vk::FenceCreateInfo {
                s_type: StructureType::FENCE_CREATE_INFO,
                p_next: null(),
                flags: FenceCreateFlags::empty(),
            };
            gpu_info.logical_device.create_fence(&create_info, None)
        }?;

        let render_finished_semaphore = unsafe {
            gpu_info.logical_device.create_semaphore(
                &SemaphoreCreateInfo {
                    s_type: StructureType::SEMAPHORE_CREATE_INFO,
                    p_next: null(),
                    flags: SemaphoreCreateFlags::empty(),
                },
                None,
            )?
        };

        let pass_info = RenderPassCreateInfo {
            s_type: StructureType::RENDER_PASS_CREATE_INFO,
            p_next: null(),
            flags: RenderPassCreateFlags::empty(),
            attachment_count: 1,
            p_attachments: &[AttachmentDescription {
                flags: AttachmentDescriptionFlags::empty(),
                format: swapchain_format.format,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }] as *const AttachmentDescription,
            subpass_count: 1,
            p_subpasses: &[SubpassDescription {
                flags: SubpassDescriptionFlags::empty(),
                pipeline_bind_point: PipelineBindPoint::GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: null(),
                color_attachment_count: 1,
                p_color_attachments: &[AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }] as *const AttachmentReference,
                p_resolve_attachments: null(),
                p_depth_stencil_attachment: null(),
                preserve_attachment_count: vk::FALSE,
                p_preserve_attachments: null(),
            }] as *const SubpassDescription,
            dependency_count: 0,
            p_dependencies: null(),
        };
        let render_pass = unsafe {
            gpu_info
                .logical_device
                .create_render_pass(&pass_info, None)?
        };

        Ok(Self {
            inner_extension,
            gpu_info,
            window: parameters.window,
            extension_surface,
            swapchain_extension,

            supported_present_modes,
            supported_presentation_formats: supported_formats,
            device_capabilities,

            present_mode: PresentModeKHR::FIFO,
            khr_surface,
            render_pass,
            present_extent: parameters.window_size,
            present_format: swapchain_format,
            swapchain_image_count: NonZeroU32::new(2).unwrap(),
            current_swapchain: SwapchainKHR::null(),
            current_swapchain_images: vec![],
            current_swapchain_image_views: vec![],
            current_swapchain_framebuffers: vec![],
            next_image_fence,
            in_flight_fence,
            render_finished_semaphore,
        })
    }

    fn post_init(&mut self) -> VkResult<()> {
        self.inner_extension.post_init()?;
        self.log_supported_features();
        self.recreate_swapchain()
    }

    fn get_instance_extensions(parameters: &Self::SetupParameters) -> Vec<String> {
        let mut inner_instance_extensions = T::get_instance_extensions(&parameters.inner_params);
        let mut my_extensions =
            ash_window::enumerate_required_extensions(parameters.window.raw_display_handle())
                .unwrap()
                .into_iter()
                .map(|c_ext| unsafe { CStr::from_ptr(*c_ext) })
                .map(|c_str| c_str.to_string_lossy().to_string())
                .collect();
        inner_instance_extensions.append(&mut my_extensions);
        inner_instance_extensions
    }

    fn get_device_extensions(
        parameters: &Self::SetupParameters,
        gpu_parameters: &GpuParameters,
    ) -> Vec<String> {
        let mut inner_device_extensions =
            T::get_device_extensions(&parameters.inner_params, gpu_parameters);
        let mut my_extensions = vec!["VK_KHR_swapchain".into()];
        inner_device_extensions.append(&mut my_extensions);

        inner_device_extensions
    }

    fn accepts_queue_families(
        &self,
        selected_queues: QueueFamilies,
        physical_device: PhysicalDevice,
    ) -> VkResult<bool> {
        let inner_supported = self
            .inner_extension
            .accepts_queue_families(selected_queues, physical_device);

        if inner_supported.is_err() {
            return inner_supported;
        }
        let graphics_queue_supported = unsafe {
            self.extension_surface.get_physical_device_surface_support(
                physical_device,
                selected_queues.graphics_family.index,
                self.khr_surface,
            )
        };
        graphics_queue_supported
    }
}

impl<T: GpuExtension> SwapchainExtension<T> {
    pub fn presentation_surface(&self) -> SurfaceKHR {
        self.khr_surface
    }

    pub fn swapchain_format(&self) -> Format {
        self.present_format.format
    }

    pub(crate) fn in_flight_fence(&self) -> vk::Fence {
        self.in_flight_fence
    }

    pub fn render_finished_semaphore(&self) -> &Semaphore {
        &self.render_finished_semaphore
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
            &self.device_capabilities.min_image_count, self.device_capabilities.max_image_count
        );
        info!(
            "\tMin/Max swapchain extents: {}x{}/{}x{}",
            self.device_capabilities.min_image_extent.width,
            self.device_capabilities.min_image_extent.height,
            self.device_capabilities.max_image_extent.width,
            self.device_capabilities.max_image_extent.height
        );
    }

    pub fn select_present_mode(&mut self, present_mode: PresentModeKHR) -> VkResult<()> {
        self.present_mode = present_mode;
        self.recreate_swapchain()
    }

    pub fn get_next_swapchain_image(&mut self) -> VkResult<(u32, Framebuffer, ImageView)> {
        loop {
            let (next_image, suboptimal) = unsafe {
                self.swapchain_extension.acquire_next_image(
                    self.current_swapchain,
                    200000,
                    Semaphore::null(),
                    self.next_image_fence,
                )
            }?;
            unsafe {
                self.gpu_info.logical_device.wait_for_fences(
                    &[self.next_image_fence],
                    true,
                    200000,
                )?;
                self.gpu_info
                    .logical_device
                    .reset_fences(&[self.next_image_fence])?;
            }
            if !suboptimal {
                let image_view = self.current_swapchain_image_views.get(next_image as usize);
                let framebuffer = self.current_swapchain_framebuffers[next_image as usize];
                return Ok((next_image, framebuffer, *image_view.unwrap()));
            }
            self.recreate_swapchain()?;
        }
    }

    pub(crate) fn render_pass(&self) -> &vk::RenderPass {
        &self.render_pass
    }

    pub(crate) fn extents(&self) -> Extent2D {
        self.present_extent.clone()
    }

    pub(crate) fn present(&self, index: u32) -> VkResult<bool> {
        unsafe {
            let mut result = ash::vk::Result::SUCCESS;
            self.swapchain_extension.queue_present(
                self.gpu_info.graphics_queue,
                &PresentInfoKHR {
                    s_type: StructureType::PRESENT_INFO_KHR,
                    p_next: null(),
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &self.render_finished_semaphore as *const Semaphore,
                    swapchain_count: 1,
                    p_swapchains: &self.current_swapchain as *const SwapchainKHR,
                    p_image_indices: &index as *const u32,
                    p_results: &mut result as *mut ash::vk::Result,
                },
            )
        }
    }

    fn recreate_swapchain(&mut self) -> VkResult<()> {
        unsafe {
            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None);
        }

        let khr_surface = unsafe {
            self.extension_surface
                .destroy_surface(self.khr_surface, None);
            ash_window::create_surface(
                &self.gpu_info.entry,
                &self.gpu_info.instance,
                self.window.raw_display_handle(),
                self.window.raw_window_handle(),
                None,
            )
        }
        .unwrap();
        self.khr_surface = khr_surface;

        let (supported_formats, device_capabilities, supported_present_modes) = unsafe {
            let supported_formats = self
                .extension_surface
                .get_physical_device_surface_formats(self.gpu_info.physical_device, khr_surface)?;
            let device_capabilities = self
                .extension_surface
                .get_physical_device_surface_capabilities(
                    self.gpu_info.physical_device,
                    khr_surface,
                )?;
            let supported_present_modes = self
                .extension_surface
                .get_physical_device_surface_present_modes(
                    self.gpu_info.physical_device,
                    khr_surface,
                )?;
            (
                supported_formats,
                device_capabilities,
                supported_present_modes,
            )
        };

        self.supported_presentation_formats = supported_formats;
        self.device_capabilities = device_capabilities;
        self.supported_present_modes = supported_present_modes;
        self.present_format = self.supported_presentation_formats[0];
        self.validate_selected_swapchain_settings();

        let swapchain_creation_info = self.make_swapchain_creation_info();
        let swapchain = unsafe {
            self.swapchain_extension
                .create_swapchain(&swapchain_creation_info, None)?
        };
        self.current_swapchain = swapchain;
        trace!(
            "Created a new swapchain with present format {:?}, present mode {:?} and present extents {:?}",
            &self.present_format, &self.present_mode, &self.present_extent
        );

        self.recreate_swapchain_images()?;
        self.recreate_swapchain_image_views()?;
        self.recreate_swapchain_framebuffers()?;

        Ok(())
    }

    fn recreate_swapchain_images(&mut self) -> VkResult<()> {
        let images = unsafe {
            self.swapchain_extension
                .get_swapchain_images(self.current_swapchain)
        }?;
        self.current_swapchain_images = images;
        Ok(())
    }

    fn recreate_swapchain_image_views(&mut self) -> VkResult<()> {
        self.current_swapchain_image_views
            .resize(self.current_swapchain_images.len(), ImageView::null());
        for (i, image) in self.current_swapchain_images.iter().enumerate() {
            let view_info = ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: null(),
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
            let view = unsafe {
                self.gpu_info
                    .logical_device
                    .create_image_view(&view_info, None)
            }?;
            self.current_swapchain_image_views[i] = view;
        }

        Ok(())
    }

    fn recreate_swapchain_framebuffers(&mut self) -> VkResult<()> {
        self.current_swapchain_framebuffers
            .resize(self.current_swapchain_images.len(), Framebuffer::null());
        for (i, image_view) in self.current_swapchain_image_views.iter().enumerate() {
            let create_info = FramebufferCreateInfo {
                s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: null(),
                flags: FramebufferCreateFlags::empty(),
                render_pass: self.render_pass,
                attachment_count: 1,
                p_attachments: image_view as *const ImageView,
                width: self.present_extent.width,
                height: self.present_extent.height,
                layers: 1,
            };

            let fb = unsafe {
                self.gpu_info
                    .logical_device
                    .create_framebuffer(&create_info, None)
            }?;
            self.current_swapchain_framebuffers[i] = fb;
        }

        Ok(())
    }

    fn make_swapchain_creation_info(&self) -> SwapchainCreateInfoKHR {
        let swapchain_creation_info = SwapchainCreateInfoKHR {
            s_type: StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: null(),
            flags: SwapchainCreateFlagsKHR::default(),
            surface: self.khr_surface,
            min_image_count: self.swapchain_image_count.get(),
            image_format: self.present_format.format,
            image_color_space: self.present_format.color_space,
            image_extent: self.present_extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: null(),
            pre_transform: self.device_capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: self.present_mode,
            clipped: TRUE,
            old_swapchain: SwapchainKHR::null(),
        };
        swapchain_creation_info
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

        if self.swapchain_image_count.get() < self.device_capabilities.min_image_count
            || self.swapchain_image_count.get() > self.device_capabilities.max_image_count
        {
            warn!(
                "Device does not support less than {} / more than {} swapchain images! Clamping",
                self.device_capabilities.min_image_count, self.device_capabilities.max_image_count
            );
            self.swapchain_image_count = self.swapchain_image_count.clamp(
                NonZeroU32::new(self.device_capabilities.min_image_count).unwrap(),
                NonZeroU32::new(self.device_capabilities.max_image_count).unwrap(),
            );
        }

        let min_exent = self.device_capabilities.min_image_extent;
        let max_exent = self.device_capabilities.max_image_extent;
        let current_extent = self.present_extent;
        if current_extent.width < min_exent.width
            || current_extent.height < min_exent.height
            || current_extent.width > max_exent.width
            || current_extent.height > max_exent.height
        {
            warn!(
                "Device does not support extents smaller than {:?} / greather than {:?}! Clamping",
                self.device_capabilities.min_image_extent,
                self.device_capabilities.max_image_extent
            );

            self.present_extent = Extent2D {
                width: self.present_extent.width.clamp(
                    self.device_capabilities.min_image_extent.width,
                    self.device_capabilities.max_image_extent.width,
                ),
                height: self.present_extent.height.clamp(
                    self.device_capabilities.min_image_extent.height,
                    self.device_capabilities.max_image_extent.height,
                ),
            }
        }
    }
}

impl<T: GpuExtension> Drop for SwapchainExtension<T> {
    fn drop(&mut self) {
        unsafe {
            for image_view in &self.current_swapchain_image_views {
                self.gpu_info
                    .logical_device
                    .destroy_image_view(*image_view, None);
            }
            self.current_swapchain_image_views.clear();

            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None);
            self.extension_surface
                .destroy_surface(self.khr_surface, None);
        }
    }
}
