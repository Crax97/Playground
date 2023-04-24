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
        ColorSpaceKHR, CompositeAlphaFlagsKHR, Extent2D, Format, ImageUsageFlags, PhysicalDevice,
        PresentModeKHR, SharingMode, StructureType, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
        SurfaceKHR, SwapchainCreateFlagsKHR, SwapchainCreateInfoKHR, SwapchainKHR, TRUE,
    },
    Entry, Instance,
};
use log::{trace, warn};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

use crate::gpu::{self, GpuInfo, QueueFamilies, SharedGpuInfo};

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
        extension_surface: Surface,
        swapchain_extension: Swapchain,
        gpu_info: SharedGpuInfo,
        khr_surface: SurfaceKHR,
        present_mode: PresentModeKHR,
        swapchain_image_count: NonZeroU32,
        present_extent: Extent2D,
        present_format: SurfaceFormatKHR,

        supported_present_modes: Vec<PresentModeKHR>,
        supported_formats: Vec<SurfaceFormatKHR>,
        device_capabilities: SurfaceCapabilitiesKHR,

        window: Window,
        current_swapchain: SwapchainKHR,
    }

    SurfaceParamters {
        window: Window,
    }
);

impl<T: GpuExtension> GpuExtension for SwapchainExtension<T> {
    type SetupParameters = SurfaceParamters<T>;

    fn new(parameters: Self::SetupParameters, gpu_info: SharedGpuInfo) -> VkResult<Self>
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

        Ok(Self {
            inner_extension,
            gpu_info,
            window: parameters.window,
            extension_surface,
            swapchain_extension,

            supported_present_modes,
            supported_formats,
            device_capabilities,

            present_mode: PresentModeKHR::FIFO,
            khr_surface,
            present_extent: Extent2D {
                width: 800,
                height: 600,
            },
            present_format: swapchain_format,
            swapchain_image_count: NonZeroU32::new(2).unwrap(),
            current_swapchain: SwapchainKHR::null(),
        })
    }

    fn post_init(&mut self) -> VkResult<()> {
        let inner = self.inner_extension.post_init();
        if inner.is_err() {
            return inner;
        }
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

    pub fn select_present_mode(&mut self, present_mode: PresentModeKHR) -> VkResult<()> {
        let physical_device = self.gpu_info.physical_device;
        let supported_present_modes = unsafe {
            self.extension_surface
                .get_physical_device_surface_present_modes(physical_device, self.khr_surface)
        }?;
        self.present_mode = present_mode;
        self.recreate_swapchain()
    }

    fn recreate_swapchain(&mut self) -> VkResult<()> {
        self.validate_selected_swapchain_settings();

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
            old_swapchain: self.current_swapchain,
        };

        let swapchain = unsafe {
            self.swapchain_extension
                .create_swapchain(&swapchain_creation_info, None)?
        };

        self.current_swapchain = swapchain;
        trace!(
            "Created a new swapchain with present format {:?}, present mode {:?} and present extents {:?}",
            &self.present_format, &self.present_mode, &self.present_extent
        );
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

        if !self.supported_formats.contains(&self.present_format) {
            warn!(
                "Device does not support present format {:?}, selecting the first available one",
                &self.present_format
            );
            self.present_format = self.supported_formats[0];
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
            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None)
        }
    }
}
