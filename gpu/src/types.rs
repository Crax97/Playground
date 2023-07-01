use std::{cell::RefCell, ops::Deref, sync::Arc};

use super::{allocator::GpuAllocator, gpu::Gpu};
use ash::{
    prelude::*,
    vk::{
        self, AllocationCallbacks, Buffer, Extent2D, FenceCreateInfo, MappedMemoryRange,
        MemoryMapFlags, SamplerCreateInfo, SemaphoreCreateInfo, ShaderModuleCreateInfo,
        StructureType,
    },
};

use super::{
    descriptor_set::{DescriptorSetAllocation, DescriptorSetAllocator},
    MemoryAllocation, MemoryDomain,
};

pub fn get_allocation_callbacks() -> Option<&'static AllocationCallbacks> {
    None
}

pub trait ToVk {
    type Inner;

    fn to_vk(&self) -> Self::Inner;
}

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImageFormat {
    Rgba8,
    Rgb8,
    RgbaFloat,
    Depth,
}

impl ToVk for ImageFormat {
    type Inner = vk::Format;
    fn to_vk(&self) -> Self::Inner {
        match self {
            ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            ImageFormat::Rgb8 => vk::Format::R8G8B8_UNORM,
            ImageFormat::RgbaFloat => vk::Format::R32G32B32A32_SFLOAT,
            ImageFormat::Depth => vk::Format::D16_UNORM,
        }
    }
}

impl From<&vk::Format> for ImageFormat {
    fn from(value: &vk::Format) -> Self {
        match *value {
            vk::Format::R8G8B8A8_UNORM => ImageFormat::Rgba8,
            vk::Format::R8G8B8_UNORM => ImageFormat::Rgb8,
            vk::Format::D16_UNORM => ImageFormat::Depth,
            vk::Format::R32G32B32A32_SFLOAT => ImageFormat::RgbaFloat,
            _ => panic!("ImageFormat::from(vk::Format): cannot convert {:?} to ImageFormat, most likely a bug: report it", value)
        }
    }
}

impl From<vk::Format> for ImageFormat {
    fn from(value: vk::Format) -> Self {
        From::<&vk::Format>::from(&value)
    }
}

macro_rules! impl_raii_wrapper_hash {
    ($name:ident) => {
        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
                self.inner.hash(hasher)
            }
        }
    };
}
macro_rules! impl_raii_wrapper_to_vk {
    ($name:ident, $inner:ty) => {
        impl ToVk for $name {
            type Inner = $inner;
            fn to_vk(&self) -> Self::Inner {
                self.inner
            }
        }
    };
}
macro_rules! define_raii_wrapper {
    ((struct $name:ident { $($mem_name:ident : $mem_ty : ty,)* }, $vk_type:ty, $drop_fn:path) {($arg_name:ident : $arg_typ:ty,) => $create_impl_block:tt}) => {
        pub struct $name {
            device: ash::Device,
            pub(super) inner: $vk_type,
            $(pub(super) $mem_name : $mem_ty,)*
        }

        impl $name {

            pub(super) fn create(device: ash::Device, $arg_name : $arg_typ, $($mem_name : $mem_ty,)*) -> VkResult<Self> {

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

        impl_raii_wrapper_hash!($name);
        impl_raii_wrapper_to_vk!($name, $vk_type);

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
    pub(super) inner: vk::Buffer,
    pub(super) memory_domain: MemoryDomain,
    pub(super) allocation: MemoryAllocation,
    pub(super) allocator: Arc<RefCell<dyn GpuAllocator>>,
}

impl GpuBuffer {
    pub(super) fn create(
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

impl GpuBuffer {
    pub fn write_data<I: Sized + Copy>(&self, offset: u64, data: &[I]) {
        let data_length = (std::mem::size_of::<I>() * data.len()) as u64;
        assert!(
            data_length > 0,
            "Cannot write on a buffer with 0 data length!"
        );
        assert!(offset < self.allocation.size);
        assert!(data_length + offset <= self.allocation.size);

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
        let address = unsafe { address.add(offset as _) } as *mut I;
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

impl_raii_wrapper_hash!(GpuBuffer);
impl_raii_wrapper_to_vk!(GpuBuffer, vk::Buffer);

pub struct GpuImage {
    device: ash::Device,
    pub(super) inner: vk::Image,
    pub(super) allocation: Option<MemoryAllocation>,
    pub(super) allocator: Option<Arc<RefCell<dyn GpuAllocator>>>,
    pub(super) extents: Extent2D,
    pub(super) format: ImageFormat,
}
impl GpuImage {
    pub(super) fn create(
        gpu: &Gpu,
        image: vk::Image,
        allocation: MemoryAllocation,
        allocator: Arc<RefCell<dyn GpuAllocator>>,
        extents: Extent2D,
        format: ImageFormat,
    ) -> VkResult<Self> {
        Ok(Self {
            device: gpu.state.logical_device.clone(),
            inner: image,
            allocation: Some(allocation),
            allocator: Some(allocator),
            extents,
            format,
        })
    }

    pub(super) fn wrap(
        device: ash::Device,
        inner: vk::Image,
        extents: Extent2D,
        format: ImageFormat,
    ) -> Self {
        Self {
            device,
            inner,
            allocation: None,
            allocator: None,
            extents,
            format,
        }
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn extents(&self) -> Extent2D {
        self.extents
    }
}
impl Drop for GpuImage {
    fn drop(&mut self) {
        match (&self.allocator, &self.allocation) {
            (Some(allocator), Some(allocation)) => {
                allocator.borrow_mut().deallocate(&self.device, &allocation);
                unsafe {
                    self.device
                        .destroy_image(self.inner, self::get_allocation_callbacks());
                }
            }
            _ => {
                // this is a wrapped image
            }
        };
    }
}
impl Deref for GpuImage {
    type Target = vk::Image;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl_raii_wrapper_hash!(GpuImage);
impl_raii_wrapper_to_vk!(GpuImage, vk::Image);

define_raii_wrapper!((struct GpuImageView{
    format: ImageFormat,
    extents: Extent2D,
}, vk::ImageView, ash::Device::destroy_image_view) {
    (create_info: &vk::ImageViewCreateInfo,) => {
        |device: &ash::Device| {
            unsafe {
                device.create_image_view(create_info, get_allocation_callbacks())
            }
        }
    }
});

impl GpuImageView {
    pub fn inner_image_view(&self) -> vk::ImageView {
        self.inner
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn extents(&self) -> Extent2D {
        self.extents
    }
}

pub struct GpuDescriptorSet {
    pub(super) inner: vk::DescriptorSet,
    pub(super) allocation: DescriptorSetAllocation,
    pub(super) allocator: Arc<RefCell<dyn DescriptorSetAllocator>>,
}
impl GpuDescriptorSet {
    pub fn create(
        allocation: DescriptorSetAllocation,
        allocator: Arc<RefCell<dyn DescriptorSetAllocator>>,
    ) -> VkResult<Self> {
        Ok(Self {
            inner: allocation.descriptor_set,
            allocation,
            allocator,
        })
    }
}
impl Drop for GpuDescriptorSet {
    fn drop(&mut self) {
        println!("Dropped descriptor set!");
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
impl_raii_wrapper_hash!(GpuDescriptorSet);
impl_raii_wrapper_to_vk!(GpuDescriptorSet, vk::DescriptorSet);

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

define_raii_wrapper!((struct GpuFramebuffer {}, vk::Framebuffer, ash::Device::destroy_framebuffer) {
    (create_info: &vk::FramebufferCreateInfo,) => {
        |device: &ash::Device| {
            unsafe {
                device.create_framebuffer(create_info, get_allocation_callbacks()) }}
            }
        }
);
