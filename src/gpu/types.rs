use std::ops::Deref;

use crate::gpu::Gpu;
use ash::{
    prelude::*,
    vk::{
        self, AllocationCallbacks, Buffer, BufferCreateInfo, DeviceCreateInfo, FenceCreateInfo,
        InstanceCreateInfo, MappedMemoryRange, MemoryMapFlags, PhysicalDevice, SemaphoreCreateInfo,
        StructureType,
    },
    Instance,
};
use std::sync::Arc;

use super::MemoryAllocation;

pub fn get_allocation_callbacks() -> Option<&'static AllocationCallbacks> {
    None
}

macro_rules! define_raii_wrapper {
    ((struct $name:ident { $($mem_name:ident : $mem_ty : ty,)* }, $vk_type:ty, $drop_fn:path) {($arg_name:ident : $arg_typ:ty,) => $create_impl_block:tt}) => {
        pub struct $name {
            device: ash::Device,
            pub(crate) inner: $vk_type,
            $(pub(crate) $mem_name : $mem_ty,)*
        }

        impl $name {

            pub fn create<A: crate::gpu::allocator::GpuAllocator>(gpu: &Gpu<A>, $arg_name : $arg_typ, $($mem_name : $mem_ty,)*) -> VkResult<Self> {

                let inner = $create_impl_block(&gpu)?;
                Ok(Self {
                    device: gpu.logical_device.clone(),
                    inner,
                    $($mem_name),*
                })
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    $drop_fn(&self.device, self.inner, self::get_allocation_callbacks());
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

define_raii_wrapper!((struct GPUSemaphore {}, vk::Semaphore, ash::Device::destroy_semaphore) {
    (create_info: &SemaphoreCreateInfo,) => {
        |gpu: &Gpu<A>| { unsafe {
            gpu.logical_device.create_semaphore(create_info, get_allocation_callbacks())
        }}
    }
});

define_raii_wrapper!((struct GPUFence {}, vk::Fence, ash::Device::destroy_fence) {
    (create_info: &FenceCreateInfo,) => {
        |gpu: &Gpu<A>| { unsafe { gpu.logical_device.create_fence(create_info, get_allocation_callbacks()) }}
    }
});

define_raii_wrapper!((struct GpuBuffer {
    allocation: MemoryAllocation,
}, vk::Buffer, ash::Device::destroy_buffer) {
    (buffer: Buffer,) => {
        |gpu: &Gpu<A>| { Ok(buffer) }
    }
});

impl GpuBuffer {
    pub fn write_data<I: Sized + Copy>(&self, data: &[I]) {
        let length = data.len() as u64;
        let size_bytes = length * std::mem::size_of::<I>() as u64;
        let address = unsafe {
            self.device
                .map_memory(
                    self.allocation.device_memory,
                    self.allocation.offset,
                    size_bytes,
                    MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory!")
        };
        let address = address as *mut I;
        let address = unsafe { std::slice::from_raw_parts_mut(address, 3) };

        address.copy_from_slice(data);

        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[MappedMemoryRange {
                    s_type: StructureType::MAPPED_MEMORY_RANGE,
                    p_next: std::ptr::null(),
                    memory: self.allocation.device_memory,
                    offset: 0,
                    size: size_bytes,
                }])
                .expect("Failed to flush memory ranges")
        };

        unsafe { self.device.unmap_memory(self.allocation.device_memory) };
    }
}

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
