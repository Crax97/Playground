use std::ops::Deref;

use crate::gpu::Gpu;
use ash::{
    prelude::*,
    vk::{
        self, AllocationCallbacks, DeviceCreateInfo, FenceCreateInfo, InstanceCreateInfo,
        PhysicalDevice, SemaphoreCreateInfo,
    },
    Instance,
};
use std::sync::Arc;

pub fn get_allocation_callbacks() -> Option<&'static AllocationCallbacks> {
    None
}

macro_rules! define_raii_wrapper {
    (($name:ident,  $vk_type:ty, $drop_fn:path) {($($arg_name:ident : $arg_typ:ty,)*) => $create_impl_block:tt}) => {
        pub struct $name {
            gpu: Arc<Gpu>,
            inner: $vk_type,
        }

        impl $name {
            pub fn create(gpu: Arc<Gpu>, $($arg_name : $arg_typ,)*) -> VkResult<Self> {

                let inner = $create_impl_block(&gpu)?;
                Ok(Self {
                    gpu: gpu.clone(),
                    inner
                })
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    $drop_fn(&self.gpu.logical_device, self.inner, self::get_allocation_callbacks());
                }
            }
        }

        impl Deref for $name {
            type Target = $vk_type;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }
    };
}

define_raii_wrapper!((GPUSemaphore, vk::Semaphore, ash::Device::destroy_semaphore) {
    (create_info: &SemaphoreCreateInfo,) => {
        |gpu: &Gpu| { unsafe { gpu.logical_device.create_semaphore(create_info, get_allocation_callbacks()) }}
    }
});

define_raii_wrapper!((GPUFence, vk::Fence, ash::Device::destroy_fence) {
    (create_info: &FenceCreateInfo,) => {
        |gpu: &Gpu| { unsafe { gpu.logical_device.create_fence(create_info, get_allocation_callbacks()) }}
    }
});

// pub struct Semaphore<'d> {
//     owning_device: &'d Device,
//     inner: vk::Semaphore,
// }
//
// impl<'d> Semaphore<'d> {
//     pub fn create(device: &'d Device, create_info: &SemaphoreCreateInfo) -> VkResult<Self> {
//         let inner = unsafe { device.create_semaphore(create_info, get_allocation_callbacks()) }?;
//         Ok(Self {
//             owning_device: device,
//             inner,
//         })
//     }
// }
//
// impl<'d> Drop for Semaphore<'d> {
//     fn drop(&mut self) {
//         unsafe {
//             self.owning_device
//                 .destroy_semaphore(self.inner, get_allocation_callbacks())
//         }
//     }
// }
//
// impl<'d> Deref for Semaphore<'d> {
//     type Target = ash::vk::Semaphore;
//
//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }
