use std::{cell::RefCell, ops::Deref, sync::Arc};

use crate::{gpu::allocator::GpuAllocator, gpu::Gpu};
use ash::{
    prelude::*,
    vk::{
        self, AllocationCallbacks, Buffer, FenceCreateInfo, ImageAspectFlags,
        ImageSubresourceRange, ImageViewCreateFlags, ImageViewType, MappedMemoryRange,
        MemoryMapFlags, SamplerCreateInfo, SemaphoreCreateInfo, StructureType,
    },
};

use super::{resource::Resource, MemoryAllocation, MemoryDomain};

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

            pub fn create(gpu: &Gpu, $arg_name : $arg_typ, $($mem_name : $mem_ty,)*) -> VkResult<Self> {

                let inner = $create_impl_block(&gpu)?;
                Ok(Self {
                    device: gpu.state.logical_device.clone(),
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

        impl Resource for $name {

        }
    };
}

define_raii_wrapper!((struct GPUSemaphore {}, vk::Semaphore, ash::Device::destroy_semaphore) {
    (create_info: &SemaphoreCreateInfo,) => {
        |gpu: &Gpu| { unsafe {
            gpu.state.logical_device.create_semaphore(create_info, get_allocation_callbacks())
        }}
    }
});

define_raii_wrapper!((struct GPUFence {}, vk::Fence, ash::Device::destroy_fence) {
    (create_info: &FenceCreateInfo,) => {
        |gpu: &Gpu| { unsafe { gpu.state.logical_device.create_fence(create_info, get_allocation_callbacks()) }}
    }
});

pub struct GpuBuffer {
    device: ash::Device,
    pub(crate) inner: vk::Buffer,
    pub(crate) memory_domain: MemoryDomain,
    pub(crate) allocation: MemoryAllocation,
    pub(crate) allocator: Arc<RefCell<dyn GpuAllocator>>,
}
impl GpuBuffer {
    pub fn create(
        device: ash::Device,
        buffer: Buffer,
        memory_domain: MemoryDomain,
        allocation: MemoryAllocation,
        allocator: Arc<RefCell<dyn GpuAllocator>>,
    ) -> VkResult<Self> {
        Ok(Self {
            device,
            inner: buffer,
            memory_domain,
            allocation,
            allocator,
        })
    }
}
impl Drop for GpuBuffer {
    fn drop(&mut self) {
        self.allocator
            .borrow_mut()
            .deallocate(&self.device, &self.allocation);
        unsafe {
            ash::Device::destroy_buffer(&self.device, self.inner, self::get_allocation_callbacks());
        }
    }
}
impl Deref for GpuBuffer {
    type Target = vk::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl Resource for GpuBuffer {}

impl GpuBuffer {
    pub fn write_data<I: Sized + Copy>(&self, data: &[I]) {
        let address = unsafe {
            self.device
                .map_memory(
                    self.allocation.device_memory,
                    self.allocation.offset,
                    vk::WHOLE_SIZE,
                    MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory!")
        };
        let address = address as *mut I;
        let address = unsafe { std::slice::from_raw_parts_mut(address, data.len()) };

        address.copy_from_slice(data);

        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[MappedMemoryRange {
                    s_type: StructureType::MAPPED_MEMORY_RANGE,
                    p_next: std::ptr::null(),
                    memory: self.allocation.device_memory,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }])
                .expect("Failed to flush memory ranges")
        };

        unsafe { self.device.unmap_memory(self.allocation.device_memory) };
    }
}

pub struct GpuImage {
    device: ash::Device,
    pub(crate) inner: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) allocation: MemoryAllocation,
    pub(crate) allocator: Arc<RefCell<dyn GpuAllocator>>,
}
impl GpuImage {
    pub fn create(
        gpu: &Gpu,
        image: vk::Image,
        format: vk::Format,
        allocation: MemoryAllocation,
        allocator: Arc<RefCell<dyn GpuAllocator>>,
    ) -> VkResult<Self> {
        let view = unsafe {
            gpu.state.logical_device.create_image_view(
                &vk::ImageViewCreateInfo {
                    s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: ImageViewCreateFlags::empty(),
                    image,
                    view_type: ImageViewType::TYPE_2D,
                    format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                },
                get_allocation_callbacks(),
            )
        }?;

        Ok(Self {
            device: gpu.state.logical_device.clone(),
            inner: image,
            view,
            allocation,
            allocator,
        })
    }
}
impl Drop for GpuImage {
    fn drop(&mut self) {
        self.allocator
            .borrow_mut()
            .deallocate(&self.device, &self.allocation);
        unsafe {
            self.device
                .destroy_image_view(self.view, get_allocation_callbacks());
            self.device
                .destroy_image(self.inner, self::get_allocation_callbacks());
        }
    }
}
impl Deref for GpuImage {
    type Target = vk::Image;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl Resource for GpuImage {}

define_raii_wrapper!((struct GpuSampler {}, vk::Sampler, ash::Device::destroy_sampler) {
    (create_info: &SamplerCreateInfo,) => {
        |gpu: &Gpu| { unsafe { gpu.state.logical_device.create_sampler(create_info, get_allocation_callbacks()) }}
    }
});

// impl GpuImage {
//     pub fn write_data<I: Sized + Copy>(&self, data: &[I]) {
//         let address = unsafe {
//             self.device
//                 .map_memory(
//                     self.allocation.device_memory,
//                     self.allocation.offset,
//                     vk::WHOLE_SIZE,
//                     MemoryMapFlags::empty(),
//                 )
//                 .expect("Failed to map memory!")
//         };
//         let address = address as *mut I;
//         let address = unsafe { std::slice::from_raw_parts_mut(address, data.len()) };
//
//         address.copy_from_slice(data);
//
//         unsafe {
//             self.device
//                 .flush_mapped_memory_ranges(&[MappedMemoryRange {
//                     s_type: StructureType::MAPPED_MEMORY_RANGE,
//                     p_next: std::ptr::null(),
//                     memory: self.allocation.device_memory,
//                     offset: 0,
//                     size: vk::WHOLE_SIZE,
//                 }])
//                 .expect("Failed to flush memory ranges")
//         };
//
//         unsafe { self.device.unmap_memory(self.allocation.device_memory) };
//     }
// }