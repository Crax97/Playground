use std::{cell::RefCell, ops::Deref, sync::Arc};

use super::{allocator::GpuAllocator, gpu::Gpu};
use ash::{
    prelude::*,
    vk::{
        self, AllocationCallbacks, Buffer, FenceCreateInfo, FramebufferCreateFlags,
        ImageAspectFlags, ImageSubresourceRange, ImageView, ImageViewCreateFlags, ImageViewType,
        MappedMemoryRange, MemoryMapFlags, SamplerCreateInfo, SemaphoreCreateInfo,
        ShaderModuleCreateInfo, StructureType,
    },
};

use super::{
    descriptor_set::{DescriptorSetAllocation, DescriptorSetAllocator},
    resource::Resource,
    MemoryAllocation, MemoryDomain, RenderPass,
};

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

            pub fn create(device: ash::Device, $arg_name : $arg_typ, $($mem_name : $mem_ty,)*) -> VkResult<Self> {

                let inner = $create_impl_block(&device)?;
                Ok(Self {
                    device,
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
        |device: &ash::Device| { unsafe {
            device.create_semaphore(create_info, get_allocation_callbacks())
        }}
    }
});

define_raii_wrapper!((struct GPUFence {}, vk::Fence, ash::Device::destroy_fence) {
    (create_info: &FenceCreateInfo,) => {
        |device: &ash::Device| { unsafe { device.create_fence(create_info, get_allocation_callbacks()) }}
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

pub struct GpuDescriptorSet {
    pub(crate) allocation: DescriptorSetAllocation,
    pub(crate) allocator: Arc<RefCell<dyn DescriptorSetAllocator>>,
}
impl GpuDescriptorSet {
    pub fn create(
        allocation: DescriptorSetAllocation,
        allocator: Arc<RefCell<dyn DescriptorSetAllocator>>,
    ) -> VkResult<Self> {
        Ok(Self {
            allocation,
            allocator,
        })
    }
}
impl Drop for GpuDescriptorSet {
    fn drop(&mut self) {
        self.allocator
            .borrow_mut()
            .deallocate(&self.allocation)
            .expect("Failed to deallocate descriptor set");
    }
}
impl Deref for GpuDescriptorSet {
    type Target = vk::DescriptorSet;
    fn deref(&self) -> &Self::Target {
        &self.allocation.descriptor_set
    }
}
impl Resource for GpuDescriptorSet {}

define_raii_wrapper!((struct GpuSampler {}, vk::Sampler, ash::Device::destroy_sampler) {
    (create_info: &SamplerCreateInfo,) => {
        |device: &ash::Device| { unsafe { device.create_sampler(create_info, get_allocation_callbacks()) }}
    }
});
define_raii_wrapper!((struct GpuShaderModule {}, vk::ShaderModule, ash::Device::destroy_shader_module) {
    (create_info: &ShaderModuleCreateInfo,) => {
        |device: &ash::Device| { unsafe { device.create_shader_module(create_info, get_allocation_callbacks()) }}
    }
});

pub struct FramebufferCreateInfo<'a> {
    pub render_pass: &'a RenderPass,
    pub attachments: &'a [&'a ImageView],
    pub width: u32,
    pub height: u32,
}

define_raii_wrapper!((struct GpuFramebuffer {}, vk::Framebuffer, ash::Device::destroy_framebuffer) {
    (create_info: &FramebufferCreateInfo,) => {
        |device: &ash::Device| { unsafe {
                    let create_info = vk::FramebufferCreateInfo {
                    s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: FramebufferCreateFlags::empty(),
                    render_pass: create_info.render_pass.inner,

                    attachment_count: create_info.attachments.len() as _,
                    p_attachments: *create_info.attachments.as_ptr(),
                    width: create_info.width,
                    height: create_info.height,

                    // We only support one single framebuffer
                    layers: 1,
            };
            device.create_framebuffer(&create_info, get_allocation_callbacks()) }}
    }
});
