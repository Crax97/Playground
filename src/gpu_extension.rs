use std::{
    ffi::{CStr, CString},
    ops::{Deref, DerefMut},
};

use ash::{
    extensions::khr::Surface,
    prelude::VkResult,
    vk::{PhysicalDevice, QueueFamilyProperties, QueueFlags, SurfaceKHR},
    Entry, Instance,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

use crate::gpu::QueueFamilies;

pub struct GpuParameters<'a> {
    pub entry: &'a Entry,
    pub instance: &'a Instance,
}

pub trait GpuExtension {
    type SetupParameters;

    fn new(parameters: Self::SetupParameters, gpu_parameters: &GpuParameters) -> Self;

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

    fn new(_: Self::SetupParameters, _: &GpuParameters) -> Self {
        ()
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
        khr_surface: SurfaceKHR,
        window: Window,
    }

    SurfaceParamters {
        window: Window,
    }
);

impl<T: GpuExtension> GpuExtension for SwapchainExtension<T> {
    type SetupParameters = SurfaceParamters<T>;

    fn new(parameters: Self::SetupParameters, gpu_parameters: &GpuParameters) -> Self {
        let inner_extension = T::new(parameters.inner_params, gpu_parameters);
        let khr_surface = unsafe {
            ash_window::create_surface(
                gpu_parameters.entry,
                gpu_parameters.instance,
                parameters.window.raw_display_handle(),
                parameters.window.raw_window_handle(),
                None,
            )
        }
        .unwrap();
        // let surface = Surface::new(gpu_parameters.entry, gpu_parameters.instance);
        Self {
            inner_extension,
            window: parameters.window,
            extension_surface: Surface::new(gpu_parameters.entry, gpu_parameters.instance),
            khr_surface,
        }
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
    pub fn presentation_surface(&self) -> &SurfaceKHR {
        &self.khr_surface
    }
}
