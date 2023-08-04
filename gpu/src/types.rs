use std::{cell::RefCell, ops::Deref, sync::Arc};

use super::{allocator::GpuAllocator, gpu::Gpu};
use ash::vk::{
    BufferUsageFlags as VkBufferUsageFlags, ImageAspectFlags, ImageLayout, ImageUsageFlags,
    StructureType,
};
use ash::{
    prelude::*,
    vk::{
        self, AllocationCallbacks, Buffer, Extent2D, FenceCreateInfo as VkFenceCreateInfo,
        SamplerCreateInfo, SemaphoreCreateInfo, ShaderModuleCreateInfo,
    },
};
use bitflags::bitflags;

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

macro_rules! case {
    ($value: expr, $target:expr, $outer:expr, $inner:expr) => {
        if $value.contains($outer) {
            $target |= $inner
        }
    };
}

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImageFormat {
    Rgba8,
    Bgra8,
    SRgba8,
    Rgb8,
    RgbaFloat,
    Depth,
}

impl ImageFormat {
    pub fn is_color(&self) -> bool {
        match self {
            ImageFormat::Rgba8
            | ImageFormat::Bgra8
            | ImageFormat::SRgba8
            | ImageFormat::Rgb8
            | ImageFormat::RgbaFloat => true,
            ImageFormat::Depth => false,
        }
    }

    pub fn is_depth(&self) -> bool {
        ImageFormat::Depth == *self
    }
    pub fn default_usage_flags(&self) -> ImageUsageFlags {
        if self.is_color() {
            ImageUsageFlags::COLOR_ATTACHMENT
        } else if self.is_depth() {
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            unreachable!()
        }
    }
    pub fn aspect_mask(&self) -> ImageAspectFlags {
        if self.is_color() {
            ImageAspectFlags::COLOR
        } else if self.is_depth() {
            ImageAspectFlags::DEPTH
        } else {
            unreachable!()
        }
    }
    pub fn preferred_attachment_read_layout(&self) -> ImageLayout {
        if self.is_color() {
            ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else if self.is_depth() {
            ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        } else {
            unreachable!()
        }
    }
    pub fn preferred_attachment_write_layout(&self) -> ImageLayout {
        if self.is_color() {
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else if self.is_depth() {
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        } else {
            unreachable!()
        }
    }

    pub fn preferred_shader_read_layout(&self) -> vk::ImageLayout {
        if self.is_color() {
            ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else if self.is_depth() {
            ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        } else {
            unreachable!()
        }
    }
    pub fn preferred_shader_write_layout(&self) -> ImageLayout {
        if self.is_color() {
            ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else if self.is_depth() {
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        } else {
            unreachable!()
        }
    }
}

impl ToVk for ImageFormat {
    type Inner = vk::Format;
    fn to_vk(&self) -> Self::Inner {
        match self {
            ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            ImageFormat::SRgba8 => vk::Format::R8G8B8A8_SRGB,
            ImageFormat::Rgb8 => vk::Format::R8G8B8_UNORM,
            ImageFormat::RgbaFloat => vk::Format::R32G32B32A32_SFLOAT,
            ImageFormat::Depth => vk::Format::D32_SFLOAT,
            ImageFormat::Bgra8 => vk::Format::B8G8R8A8_UNORM,
        }
    }
}

impl From<&vk::Format> for ImageFormat {
    fn from(value: &vk::Format) -> Self {
        match *value {
            vk::Format::R8G8B8A8_UNORM => ImageFormat::Rgba8,
            vk::Format::R8G8B8A8_SRGB => ImageFormat::SRgba8,
            vk::Format::R8G8B8_UNORM => ImageFormat::Rgb8,
            vk::Format::D32_SFLOAT => ImageFormat::Depth,
            vk::Format::R32G32B32A32_SFLOAT => ImageFormat::RgbaFloat,
            vk::Format::B8G8R8A8_UNORM => ImageFormat::Bgra8,
            _ => panic!("ImageFormat::from(vk::Format): cannot convert {:?} to ImageFormat, most likely a bug: report it", value)
        }
    }
}

impl From<vk::Format> for ImageFormat {
    fn from(value: vk::Format) -> Self {
        From::<&vk::Format>::from(&value)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CompareOp {
    #[default]
    Always,
    Never,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreatereEqual,
}

impl ToVk for CompareOp {
    type Inner = vk::CompareOp;

    fn to_vk(&self) -> Self::Inner {
        match self {
            CompareOp::Always => vk::CompareOp::ALWAYS,
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::GreatereEqual => vk::CompareOp::GREATER_OR_EQUAL,
        }
    }
}

impl From<vk::CompareOp> for CompareOp {
    fn from(value: vk::CompareOp) -> Self {
        match value {
            vk::CompareOp::ALWAYS => CompareOp::Always,
            vk::CompareOp::NEVER => CompareOp::Never,
            vk::CompareOp::EQUAL => CompareOp::Equal,
            vk::CompareOp::NOT_EQUAL => CompareOp::NotEqual,
            vk::CompareOp::LESS => CompareOp::Less,
            vk::CompareOp::LESS_OR_EQUAL => CompareOp::LessEqual,
            vk::CompareOp::GREATER => CompareOp::Greater,
            vk::CompareOp::GREATER_OR_EQUAL => CompareOp::GreatereEqual,
            _ => unreachable!(),
        }
    }
}

bitflags! {
#[repr(transparent)]
#[derive(Default, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct BufferUsageFlags : u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC = 0b1;
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST = 0b10;
        #[doc = "Can be used as TBO"]
        const UNIFORM_TEXEL_BUFFER = 0b100;
        #[doc = "Can be used as IBO"]
        const STORAGE_TEXEL_BUFFER = 0b1000;
        #[doc = "Can be used as UBO"]
        const UNIFORM_BUFFER = 0b1_0000;
        #[doc = "Can be used as SSBO"]
        const STORAGE_BUFFER = 0b10_0000;
        #[doc = "Can be used as source of fixed-function index fetch (index buffer)"]
        const INDEX_BUFFER = 0b100_0000;
        #[doc = "Can be used as source of fixed-function vertex fetch (VBO)"]
        const VERTEX_BUFFER = 0b1000_0000;
        #[doc = "Can be the source of indirect parameters (e.g. indirect buffer, parameter buffer)"]
        const INDIRECT_BUFFER = 0b1_0000_0000;
    }
}

impl ToVk for BufferUsageFlags {
    type Inner = VkBufferUsageFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut inner = Self::Inner::empty();

        case!(self, inner, Self::TRANSFER_SRC, Self::Inner::TRANSFER_SRC);
        case!(self, inner, Self::TRANSFER_DST, Self::Inner::TRANSFER_DST);
        case!(
            self,
            inner,
            Self::UNIFORM_TEXEL_BUFFER,
            Self::Inner::UNIFORM_TEXEL_BUFFER
        );
        case!(
            self,
            inner,
            Self::STORAGE_TEXEL_BUFFER,
            Self::Inner::STORAGE_TEXEL_BUFFER
        );
        case!(
            self,
            inner,
            Self::UNIFORM_BUFFER,
            Self::Inner::UNIFORM_BUFFER
        );
        case!(
            self,
            inner,
            Self::STORAGE_BUFFER,
            Self::Inner::STORAGE_BUFFER
        );
        case!(self, inner, Self::INDEX_BUFFER, Self::Inner::INDEX_BUFFER);
        case!(self, inner, Self::VERTEX_BUFFER, Self::Inner::VERTEX_BUFFER);
        case!(
            self,
            inner,
            Self::INDIRECT_BUFFER,
            Self::Inner::INDIRECT_BUFFER
        );

        inner
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

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct FenceCreateFlags : u32 {
        const SIGNALED = 0b00000001;
    }
}

impl ToVk for FenceCreateFlags {
    type Inner = vk::FenceCreateFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut inner = Self::Inner::empty();
        if self.contains(FenceCreateFlags::SIGNALED) {
            inner |= Self::Inner::SIGNALED;
        }

        inner
    }
}

pub struct FenceCreateInfo {
    pub flags: FenceCreateFlags,
}

define_raii_wrapper!((struct GPUFence {}, vk::Fence, ash::Device::destroy_fence) {
    (create_info: &FenceCreateInfo,) => {
        |device: &ash::Device| {
        let vk_fence_create_info = VkFenceCreateInfo {
            s_type: StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: create_info.flags.to_vk()

        };
        unsafe { device.create_fence(&vk_fence_create_info, get_allocation_callbacks()) }}
    }
});

impl GPUFence {
    pub fn new(gpu: &Gpu, create_info: &FenceCreateInfo) -> VkResult<Self> {
        Self::create(gpu.vk_logical_device(), create_info)
    }
}

impl GPUSemaphore {
    pub fn new(gpu: &Gpu, create_info: &SemaphoreCreateInfo) -> VkResult<Self> {
        Self::create(gpu.vk_logical_device(), create_info)
    }
}

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
        self.allocator.borrow_mut().deallocate(&self.allocation);
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
        let data_length = std::mem::size_of_val(data) as u64;
        assert!(
            data_length > 0,
            "Cannot write on a buffer with 0 data length!"
        );
        assert!(offset < self.allocation.size);
        assert!(data_length + offset <= self.allocation.size);

        let address = unsafe {
            self.allocation
                .persistent_ptr
                .expect("Tried to write to a buffer without a persistent ptr!")
                .as_ptr()
                .add(offset as _)
        } as *mut I;
        let address = unsafe { std::slice::from_raw_parts_mut(address, data.len()) };

        address.copy_from_slice(data);
    }

    pub fn read<T: Copy + Sized>(&self, offset: u64) -> T {
        let data_length = std::mem::size_of::<T>();
        assert!(
            data_length > 0,
            "Cannot write on a buffer with 0 data length!"
        );
        assert!(offset < self.allocation.size);
        assert!(data_length as u64 + offset <= self.allocation.size);

        let address = unsafe {
            self.allocation
                .persistent_ptr
                .expect("Tried to write to a buffer without a persistent ptr!")
                .as_ptr()
                .add(offset as _)
        } as *mut T;
        let address = unsafe { std::slice::from_raw_parts_mut(address, data_length) };

        address[0]
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
        if let (Some(allocator), Some(allocation)) = (&self.allocator, &self.allocation) {
            allocator.borrow_mut().deallocate(allocation);
            unsafe {
                self.device
                    .destroy_image(self.inner, self::get_allocation_callbacks());
            }
        } else {
            // this is a wrapped image
        }
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
    owner_image: vk::Image,
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
    pub fn inner_image(&self) -> vk::Image {
        self.owner_image
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

impl PartialEq for GpuDescriptorSet {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for GpuDescriptorSet {}

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
