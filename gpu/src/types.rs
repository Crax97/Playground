use std::{cell::RefCell, ops::Deref, sync::Arc};

use super::gpu::*;
use crate::{
    Extent2D, Filter, IndexType, Offset2D, PipelineBindPoint, Rect2D, SamplerAddressMode,
    SamplerCreateInfo, StencilOp, StencilOpState, *,
};
use ash::vk::{
    BufferUsageFlags as VkBufferUsageFlags, SamplerCreateFlags, SamplerMipmapMode, StructureType,
};
use ash::{
    prelude::*,
    vk::{
        self, AllocationCallbacks, Buffer, FenceCreateInfo as VkFenceCreateInfo,
        SemaphoreCreateInfo, ShaderModuleCreateInfo,
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

impl ToVk for bool {
    type Inner = vk::Bool32;

    fn to_vk(&self) -> Self::Inner {
        if *self {
            vk::TRUE
        } else {
            vk::FALSE
        }
    }
}

macro_rules! case {
    ($value: expr, $target:expr, $outer:expr, $inner:expr) => {
        if $value.contains($outer) {
            $target |= $inner
        }
    };
}

impl ToVk for IndexType {
    type Inner = vk::IndexType;

    fn to_vk(&self) -> Self::Inner {
        match self {
            IndexType::Uint16 => Self::Inner::UINT16,
            IndexType::Uint32 => Self::Inner::UINT32,
            IndexType::Uint64 => todo!(),
        }
    }
}

impl ToVk for StencilOp {
    type Inner = vk::StencilOp;

    fn to_vk(&self) -> Self::Inner {
        match self {
            StencilOp::Keep => Self::Inner::KEEP,
            StencilOp::Zero => Self::Inner::ZERO,
            StencilOp::Replace(_) => Self::Inner::REPLACE,
            StencilOp::ClampedIncrement => Self::Inner::INCREMENT_AND_CLAMP,
            StencilOp::ClampedDecrement => Self::Inner::DECREMENT_AND_CLAMP,
            StencilOp::Invert => Self::Inner::INVERT,
            StencilOp::WrappedIncrement => Self::Inner::INCREMENT_AND_WRAP,
            StencilOp::WrappedDecrement => Self::Inner::DECREMENT_AND_WRAP,
        }
    }
}

impl ToVk for StencilOpState {
    type Inner = vk::StencilOpState;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            fail_op: self.fail.to_vk(),
            pass_op: self.pass.to_vk(),
            depth_fail_op: self.depth_fail.to_vk(),
            compare_op: self.compare.to_vk(),
            compare_mask: self.compare_mask,
            write_mask: self.write_mask,
            reference: self.reference,
        }
    }
}

impl ToVk for ShaderStage {
    type Inner = vk::ShaderStageFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut result = Self::Inner::default();
        case!(self, result, Self::VERTEX, Self::Inner::VERTEX);
        case!(
            self,
            result,
            Self::TESSELLATION_CONTROL,
            Self::Inner::TESSELLATION_CONTROL
        );
        case!(
            self,
            result,
            Self::TESSELLATION_EVALUATION,
            Self::Inner::TESSELLATION_EVALUATION
        );
        case!(self, result, Self::GEOMETRY, Self::Inner::GEOMETRY);
        case!(self, result, Self::FRAGMENT, Self::Inner::FRAGMENT);
        case!(self, result, Self::COMPUTE, Self::Inner::COMPUTE);
        case!(self, result, Self::ALL_GRAPHICS, Self::Inner::ALL_GRAPHICS);
        case!(self, result, Self::ALL, Self::Inner::ALL);
        result
    }
}

impl ToVk for PushConstantRange {
    type Inner = vk::PushConstantRange;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            stage_flags: self.stage_flags.to_vk(),
            offset: self.offset,
            size: self.size,
        }
    }
}

impl ToVk for ImageViewType {
    type Inner = vk::ImageViewType;

    fn to_vk(&self) -> Self::Inner {
        match self {
            ImageViewType::Type1D => Self::Inner::TYPE_1D,
            ImageViewType::Type2D => Self::Inner::TYPE_2D,
            ImageViewType::Type3D => Self::Inner::TYPE_3D,
            ImageViewType::Cube => Self::Inner::CUBE,
            ImageViewType::Type1DArray => Self::Inner::TYPE_1D_ARRAY,
            ImageViewType::Type2DArray => Self::Inner::TYPE_2D_ARRAY,
            ImageViewType::TypeCubeArray => Self::Inner::CUBE_ARRAY,
        }
    }
}

impl From<vk::ImageViewType> for ImageViewType {
    fn from(value: vk::ImageViewType) -> Self {
        match value {
            vk::ImageViewType::TYPE_1D => ImageViewType::Type1D,
            vk::ImageViewType::TYPE_2D => ImageViewType::Type2D,
            vk::ImageViewType::TYPE_3D => ImageViewType::Type3D,
            vk::ImageViewType::CUBE => ImageViewType::Cube,
            vk::ImageViewType::TYPE_1D_ARRAY => ImageViewType::Type1DArray,
            vk::ImageViewType::TYPE_2D_ARRAY => ImageViewType::Type2DArray,
            vk::ImageViewType::CUBE_ARRAY => ImageViewType::TypeCubeArray,
            _ => unreachable!(),
        }
    }
}

impl ToVk for ComponentSwizzle {
    type Inner = vk::ComponentSwizzle;

    fn to_vk(&self) -> Self::Inner {
        match self {
            ComponentSwizzle::Identity => Self::Inner::IDENTITY,
            ComponentSwizzle::Zero => Self::Inner::ZERO,
            ComponentSwizzle::One => Self::Inner::ONE,
            ComponentSwizzle::R => Self::Inner::R,
            ComponentSwizzle::G => Self::Inner::G,
            ComponentSwizzle::B => Self::Inner::B,
            ComponentSwizzle::A => Self::Inner::A,
        }
    }
}

impl ToVk for ComponentMapping {
    type Inner = vk::ComponentMapping;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            r: self.r.to_vk(),
            g: self.g.to_vk(),
            b: self.b.to_vk(),
            a: self.a.to_vk(),
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

impl ToVk for ImageAspectFlags {
    type Inner = vk::ImageAspectFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut inner = Self::Inner::empty();
        case!(self, inner, Self::COLOR, Self::Inner::COLOR);
        case!(self, inner, Self::DEPTH, Self::Inner::DEPTH);
        case!(self, inner, Self::STENCIL, Self::Inner::STENCIL);
        inner
    }
}

impl ToVk for ImageFormat {
    type Inner = vk::Format;
    fn to_vk(&self) -> Self::Inner {
        match self {
            ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            ImageFormat::SRgba8 => vk::Format::R8G8B8A8_SRGB,
            ImageFormat::Rgb8 => vk::Format::R8G8B8_UNORM,
            ImageFormat::RFloat16 => vk::Format::R16_SFLOAT,
            ImageFormat::RgFloat16 => vk::Format::R16G16_SFLOAT,
            ImageFormat::RgbFloat16 => vk::Format::R16G16B16_SFLOAT,
            ImageFormat::RgbaFloat16 => vk::Format::R16G16B16A16_SFLOAT,
            ImageFormat::RFloat32 => vk::Format::R32_SFLOAT,
            ImageFormat::RgFloat32 => vk::Format::R32G32_SFLOAT,
            ImageFormat::RgbFloat32 => vk::Format::R32G32B32_SFLOAT,
            ImageFormat::RgbaFloat32 => vk::Format::R32G32B32A32_SFLOAT,
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
            vk::Format::R32_SFLOAT => ImageFormat::RFloat32,
            vk::Format::R32G32_SFLOAT => ImageFormat::RgFloat32,
            vk::Format::R32G32B32_SFLOAT => ImageFormat::RgbFloat32,
            vk::Format::D32_SFLOAT => ImageFormat::Depth,
            vk::Format::R32G32B32A32_SFLOAT => ImageFormat::RgbaFloat32,
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

impl ToVk for ImageLayout {
    type Inner = vk::ImageLayout;
    fn to_vk(&self) -> Self::Inner {
        match self {
            ImageLayout::Undefined => Self::Inner::UNDEFINED,
            ImageLayout::General => Self::Inner::GENERAL,
            ImageLayout::ColorAttachment => Self::Inner::COLOR_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilAttachment => Self::Inner::DEPTH_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilReadOnly => Self::Inner::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ImageLayout::ShaderReadOnly => Self::Inner::SHADER_READ_ONLY_OPTIMAL,
            ImageLayout::TransferSrc => Self::Inner::TRANSFER_SRC_OPTIMAL,
            ImageLayout::TransferDst => Self::Inner::TRANSFER_DST_OPTIMAL,
            ImageLayout::PreinitializedByCpu => Self::Inner::PREINITIALIZED,

            ImageLayout::PresentSrc => Self::Inner::PRESENT_SRC_KHR,
        }
    }
}

impl ToVk for AccessFlags {
    type Inner = vk::AccessFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut inner = Self::Inner::empty();

        case!(self, inner, Self::NONE, Self::Inner::NONE);
        case!(
            self,
            inner,
            Self::INDIRECT_COMMAND_READ,
            Self::Inner::INDIRECT_COMMAND_READ
        );
        case!(self, inner, Self::INDEX_READ, Self::Inner::INDEX_READ);
        case!(
            self,
            inner,
            Self::VERTEX_ATTRIBUTE_READ,
            Self::Inner::VERTEX_ATTRIBUTE_READ
        );
        case!(self, inner, Self::UNIFORM_READ, Self::Inner::UNIFORM_READ);
        case!(
            self,
            inner,
            Self::INPUT_ATTACHMENT_READ,
            Self::Inner::INPUT_ATTACHMENT_READ
        );
        case!(self, inner, Self::SHADER_READ, Self::Inner::SHADER_READ);
        case!(self, inner, Self::SHADER_WRITE, Self::Inner::SHADER_WRITE);
        case!(
            self,
            inner,
            Self::COLOR_ATTACHMENT_READ,
            Self::Inner::COLOR_ATTACHMENT_READ
        );
        case!(
            self,
            inner,
            Self::COLOR_ATTACHMENT_WRITE,
            Self::Inner::COLOR_ATTACHMENT_WRITE
        );
        case!(
            self,
            inner,
            Self::DEPTH_STENCIL_ATTACHMENT_READ,
            Self::Inner::DEPTH_STENCIL_ATTACHMENT_READ
        );
        case!(
            self,
            inner,
            Self::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Self::Inner::DEPTH_STENCIL_ATTACHMENT_WRITE
        );
        case!(self, inner, Self::TRANSFER_READ, Self::Inner::TRANSFER_READ);
        case!(
            self,
            inner,
            Self::TRANSFER_WRITE,
            Self::Inner::TRANSFER_WRITE
        );
        case!(self, inner, Self::HOST_READ, Self::Inner::HOST_READ);
        case!(self, inner, Self::HOST_WRITE, Self::Inner::HOST_WRITE);
        case!(self, inner, Self::MEMORY_READ, Self::Inner::MEMORY_READ);
        case!(self, inner, Self::MEMORY_WRITE, Self::Inner::MEMORY_WRITE);

        inner
    }
}

impl ToVk for PipelineBindPoint {
    type Inner = vk::PipelineBindPoint;

    fn to_vk(&self) -> Self::Inner {
        match self {
            PipelineBindPoint::Graphics => Self::Inner::GRAPHICS,
            PipelineBindPoint::Compute => Self::Inner::COMPUTE,
        }
    }
}

impl ToVk for PipelineStageFlags {
    type Inner = vk::PipelineStageFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut inner = Self::Inner::empty();

        case!(self, inner, Self::TOP_OF_PIPE, Self::Inner::TOP_OF_PIPE);
        case!(self, inner, Self::DRAW_INDIRECT, Self::Inner::DRAW_INDIRECT);
        case!(self, inner, Self::VERTEX_SHADER, Self::Inner::VERTEX_SHADER);
        case!(
            self,
            inner,
            Self::TESSELLATION_CONTROL_SHADER,
            Self::Inner::TESSELLATION_CONTROL_SHADER
        );
        case!(
            self,
            inner,
            Self::TESSELLATION_EVALUATION_SHADER,
            Self::Inner::TESSELLATION_EVALUATION_SHADER
        );
        case!(
            self,
            inner,
            Self::GEOMETRY_SHADER,
            Self::Inner::GEOMETRY_SHADER
        );
        case!(
            self,
            inner,
            Self::FRAGMENT_SHADER,
            Self::Inner::FRAGMENT_SHADER
        );
        case!(
            self,
            inner,
            Self::EARLY_FRAGMENT_TESTS,
            Self::Inner::EARLY_FRAGMENT_TESTS
        );
        case!(
            self,
            inner,
            Self::LATE_FRAGMENT_TESTS,
            Self::Inner::LATE_FRAGMENT_TESTS
        );
        case!(
            self,
            inner,
            Self::COLOR_ATTACHMENT_OUTPUT,
            Self::Inner::COLOR_ATTACHMENT_OUTPUT
        );
        case!(
            self,
            inner,
            Self::COMPUTE_SHADER,
            Self::Inner::COMPUTE_SHADER
        );
        case!(self, inner, Self::TRANSFER, Self::Inner::TRANSFER);
        case!(
            self,
            inner,
            Self::BOTTOM_OF_PIPE,
            Self::Inner::BOTTOM_OF_PIPE
        );
        case!(self, inner, Self::HOST, Self::Inner::HOST);
        case!(self, inner, Self::ALL_GRAPHICS, Self::Inner::ALL_GRAPHICS);
        case!(self, inner, Self::ALL_COMMANDS, Self::Inner::ALL_COMMANDS);

        inner
    }
}

impl ToVk for ImageUsageFlags {
    type Inner = vk::ImageUsageFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut inner = Self::Inner::empty();

        case!(self, inner, Self::TRANSFER_SRC, Self::Inner::TRANSFER_SRC);
        case!(self, inner, Self::TRANSFER_DST, Self::Inner::TRANSFER_DST);
        case!(self, inner, Self::SAMPLED, Self::Inner::SAMPLED);
        case!(
            self,
            inner,
            Self::COLOR_ATTACHMENT,
            Self::Inner::COLOR_ATTACHMENT
        );
        case!(
            self,
            inner,
            Self::DEPTH_STENCIL_ATTACHMENT,
            Self::Inner::DEPTH_STENCIL_ATTACHMENT
        );
        case!(
            self,
            inner,
            Self::TRANSIENT_ATTACHMENT,
            Self::Inner::TRANSIENT_ATTACHMENT
        );
        case!(
            self,
            inner,
            Self::INPUT_ATTACHMENT,
            Self::Inner::INPUT_ATTACHMENT
        );

        inner
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
        #[derive(Clone)]
        pub struct $name {
            pub(super) inner: $vk_type,
            pub(super) label: Option<String>,
            $(pub(super) $mem_name : $mem_ty,)*
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_fmt(format_args!("{:?}", &self.label))
            }
        }

        impl $name {

            pub(super) fn create(device: ash::Device, label: Option<&str>, $arg_name : $arg_typ, $($mem_name : $mem_ty,)*) -> VkResult<Self> {

                let inner = $create_impl_block(&device)?;
                Ok(Self {
                    inner,
                    label: label.map(|s| s.to_owned()),
                    $($mem_name),*
                })
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
    pub fn new(gpu: &VkGpu, create_info: &FenceCreateInfo) -> VkResult<Self> {
        Self::create(gpu.vk_logical_device(), None, create_info)
    }
}

impl GPUSemaphore {
    pub fn new(gpu: &VkGpu, create_info: &SemaphoreCreateInfo) -> VkResult<Self> {
        Self::create(gpu.vk_logical_device(), None, create_info)
    }
}

pub struct VkCommandPool {
    device: ash::Device,
    pub inner: vk::CommandPool,
    pub(super) associated_queue: QueueType,
}

impl Drop for VkCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_command_pool(self.inner, get_allocation_callbacks())
        };
    }
}

impl VkCommandPool {
    pub(super) fn new(
        device: ash::Device,
        families: &QueueFamilies,
        create_info: &CommandPoolCreateInfo,
    ) -> VkResult<Self> {
        let inner = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    // Always allow command buffers to be resettable
                    flags: create_info.flags.to_vk()
                        | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: create_info.queue_type.get_vk_queue_index(families),
                },
                get_allocation_callbacks(),
            )?
        };

        Ok(Self {
            inner,
            device,
            associated_queue: create_info.queue_type,
        })
    }
}

#[derive(Clone)]
pub struct VkBuffer {
    pub(super) label: Option<String>,
    pub(super) inner: vk::Buffer,
    pub(super) memory_domain: MemoryDomain,
    pub(super) allocation: MemoryAllocation,
}

impl std::fmt::Debug for VkBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Buffer - {:?}", &self.label))
    }
}

impl VkBuffer {
    pub(super) fn create(
        label: Option<&str>,
        buffer: Buffer,
        memory_domain: MemoryDomain,
        allocation: MemoryAllocation,
    ) -> VkResult<Self> {
        Ok(Self {
            label: label.map(|s| s.to_owned()),
            inner: buffer,
            memory_domain,
            allocation,
        })
    }

    pub(crate) fn size(&self) -> usize {
        self.allocation.size as _
    }
}
impl Deref for VkBuffer {
    type Target = vk::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VkBuffer {
    pub fn write_data<I: Sized + Copy>(&self, offset: u64, data: &[I]) {
        let data_length_bytes = std::mem::size_of_val(data) as u64;
        assert!(
            data_length_bytes > 0,
            "Cannot write on a buffer with 0 data length!"
        );
        assert!(offset < self.allocation.size);
        assert!(data_length_bytes + offset <= self.allocation.size);

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

impl_raii_wrapper_hash!(VkBuffer);
impl_raii_wrapper_to_vk!(VkBuffer, vk::Buffer);

#[derive(Clone)]
pub struct VkImage {
    pub(super) label: Option<String>,
    pub(super) inner: vk::Image,
    pub(super) allocation: Option<MemoryAllocation>,
    pub(super) extents: Extent2D,
    pub(super) format: ImageFormat,
}
impl std::fmt::Debug for VkImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Image - {:?}", &self.label))
    }
}
impl VkImage {
    pub(super) fn create(
        image: vk::Image,
        label: Option<&str>,
        allocation: MemoryAllocation,
        extents: Extent2D,
        format: ImageFormat,
    ) -> VkResult<Self> {
        Ok(Self {
            label: label.map(|s| s.to_owned()),
            inner: image,
            allocation: Some(allocation),
            extents,
            format,
        })
    }

    pub(super) fn wrap(
        _device: ash::Device,
        label: Option<&str>,
        inner: vk::Image,
        extents: Extent2D,
        format: ImageFormat,
    ) -> Self {
        Self {
            inner,
            label: label.map(|s| s.to_owned()),
            allocation: None,
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

impl Deref for VkImage {
    type Target = vk::Image;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl_raii_wrapper_hash!(VkImage);
impl_raii_wrapper_to_vk!(VkImage, vk::Image);

define_raii_wrapper!((struct VkImageView{
    format: ImageFormat,
    owner_image: ImageHandle,
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

impl VkImageView {
    pub fn inner_image_view(&self) -> vk::ImageView {
        self.inner
    }
    pub fn owner_image_handle(&self) -> ImageHandle {
        self.owner_image.clone()
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn extents(&self) -> Extent2D {
        self.extents
    }
}

pub struct VkDescriptorSet {
    pub(super) inner: vk::DescriptorSet,
    pub(super) allocation: DescriptorSetAllocation,
}

impl PartialEq for VkDescriptorSet {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for VkDescriptorSet {}

impl VkDescriptorSet {
    pub fn create(allocation: DescriptorSetAllocation) -> VkResult<Self> {
        Ok(Self {
            inner: allocation.descriptor_set,
            allocation,
        })
    }
}
impl Deref for VkDescriptorSet {
    type Target = vk::DescriptorSet;
    fn deref(&self) -> &Self::Target {
        &self.allocation.descriptor_set
    }
}
impl_raii_wrapper_hash!(VkDescriptorSet);
impl_raii_wrapper_to_vk!(VkDescriptorSet, vk::DescriptorSet);

define_raii_wrapper!((struct VkSampler {}, vk::Sampler, ash::Device::destroy_sampler) {
    (create_info: &SamplerCreateInfo,) => {
        |device: &ash::Device| { unsafe { device.create_sampler(&create_info.to_vk(), get_allocation_callbacks()) }}
    }
});
define_raii_wrapper!((struct VkShaderModule {}, vk::ShaderModule, ash::Device::destroy_shader_module) {
    (create_info: &ShaderModuleCreateInfo,) => {
        |device: &ash::Device| { unsafe { device.create_shader_module(create_info, get_allocation_callbacks()) }}
    }
});

define_raii_wrapper!((struct VkFramebuffer {}, vk::Framebuffer, ash::Device::destroy_framebuffer) {
    (create_info: &vk::FramebufferCreateInfo,) => {
        |device: &ash::Device| {
            unsafe {
                device.create_framebuffer(create_info, get_allocation_callbacks()) }}
            }
        }
);

impl ToVk for Offset2D {
    type Inner = vk::Offset2D;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            x: self.x,
            y: self.y,
        }
    }
}

impl ToVk for Extent2D {
    type Inner = vk::Extent2D;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            width: self.width,
            height: self.height,
        }
    }
}

impl ToVk for Rect2D {
    type Inner = vk::Rect2D;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            offset: self.offset.to_vk(),
            extent: self.extent.to_vk(),
        }
    }
}

impl ToVk for Offset3D {
    type Inner = vk::Offset3D;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl ToVk for Extent3D {
    type Inner = vk::Extent3D;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }
}

impl ToVk for Filter {
    type Inner = vk::Filter;

    fn to_vk(&self) -> Self::Inner {
        match self {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear => vk::Filter::LINEAR,
        }
    }
}

impl ToVk for SamplerAddressMode {
    type Inner = vk::SamplerAddressMode;
    fn to_vk(&self) -> Self::Inner {
        match self {
            SamplerAddressMode::Repeat => Self::Inner::REPEAT,
            SamplerAddressMode::MirroredRepeat => Self::Inner::MIRRORED_REPEAT,
            SamplerAddressMode::ClampToEdge => Self::Inner::CLAMP_TO_EDGE,
            SamplerAddressMode::ClampToBorder => Self::Inner::CLAMP_TO_BORDER,
        }
    }
}

impl ToVk for SamplerCreateInfo {
    type Inner = vk::SamplerCreateInfo;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            s_type: StructureType::SAMPLER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: SamplerCreateFlags::empty(),
            mag_filter: self.mag_filter.to_vk(),
            min_filter: self.min_filter.to_vk(),
            mipmap_mode: SamplerMipmapMode::default(),
            address_mode_u: self.address_u.to_vk(),
            address_mode_v: self.address_v.to_vk(),
            address_mode_w: self.address_w.to_vk(),
            mip_lod_bias: self.mip_lod_bias,
            anisotropy_enable: vk::FALSE,
            max_anisotropy: 0.0,
            compare_enable: if self.compare_function.is_some() {
                vk::TRUE
            } else {
                vk::FALSE
            },
            compare_op: self.compare_function.unwrap_or_default().to_vk(),
            min_lod: self.min_lod,
            max_lod: self.max_lod,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE, // TODO: use VK_EXT_custom_border_color
            unnormalized_coordinates: vk::FALSE,
        }
    }
}

impl ToVk for super::BlendMode {
    type Inner = vk::BlendFactor;

    fn to_vk(&self) -> Self::Inner {
        match self {
            Self::Zero => Self::Inner::ZERO,
            Self::One => Self::Inner::ONE,
            Self::SrcColor => Self::Inner::SRC_COLOR,
            Self::OneMinusSrcColor => Self::Inner::ONE_MINUS_SRC_COLOR,
            Self::DstColor => Self::Inner::DST_COLOR,
            Self::OneMinusDstColor => Self::Inner::ONE_MINUS_DST_COLOR,
            Self::SrcAlpha => Self::Inner::SRC_ALPHA,
            Self::OneMinusSrcAlpha => Self::Inner::ONE_MINUS_SRC_ALPHA,
            Self::DstAlpha => Self::Inner::DST_ALPHA,
            Self::OneMinusDstAlpha => Self::Inner::ONE_MINUS_DST_ALPHA,
            Self::ConstantColor => Self::Inner::CONSTANT_COLOR,
            Self::OneMinusConstantColor => Self::Inner::ONE_MINUS_CONSTANT_COLOR,
            Self::ConstantAlpha => Self::Inner::CONSTANT_ALPHA,
            Self::OneMinusConstantAlpha => Self::Inner::ONE_MINUS_CONSTANT_ALPHA,
            Self::SrcAlphaSaturate => Self::Inner::SRC_ALPHA_SATURATE,
            Self::Src1Color => Self::Inner::SRC1_COLOR,
            Self::OneMinusSrc1Color => Self::Inner::ONE_MINUS_SRC1_COLOR,
            Self::Src1Alpha => Self::Inner::SRC1_ALPHA,
            Self::OneMinusSrc1Alpha => Self::Inner::ONE_MINUS_SRC1_ALPHA,
        }
    }
}

impl ToVk for super::BlendOp {
    type Inner = vk::BlendOp;

    fn to_vk(&self) -> Self::Inner {
        match self {
            Self::Add => Self::Inner::ADD,
            Self::Subtract => Self::Inner::SUBTRACT,
            Self::ReverseSubtract => Self::Inner::REVERSE_SUBTRACT,
            Self::Min => Self::Inner::MIN,
            Self::Max => Self::Inner::MAX,
        }
    }
}

impl ToVk for super::ColorComponentFlags {
    type Inner = vk::ColorComponentFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut res = vk::ColorComponentFlags::default();
        case!(self, res, Self::R, Self::Inner::R);
        case!(self, res, Self::G, Self::Inner::G);
        case!(self, res, Self::B, Self::Inner::B);
        case!(self, res, Self::A, Self::Inner::A);
        res
    }
}

impl ToVk for super::AttachmentReference {
    type Inner = vk::AttachmentReference;
    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            attachment: self.attachment,
            layout: self.layout.to_vk(),
        }
    }
}

impl ToVk for super::SampleCount {
    type Inner = vk::SampleCountFlags;

    fn to_vk(&self) -> Self::Inner {
        match self {
            super::SampleCount::Sample1 => Self::Inner::TYPE_1,
            super::SampleCount::Sample2 => Self::Inner::TYPE_2,
            super::SampleCount::Sample4 => Self::Inner::TYPE_4,
            super::SampleCount::Sample8 => Self::Inner::TYPE_8,
            super::SampleCount::Sample16 => Self::Inner::TYPE_16,
            super::SampleCount::Sample32 => Self::Inner::TYPE_32,
            super::SampleCount::Sample64 => Self::Inner::TYPE_64,
        }
    }
}

impl ToVk for CullMode {
    type Inner = vk::CullModeFlags;

    fn to_vk(&self) -> Self::Inner {
        match self {
            CullMode::Back => Self::Inner::BACK,
            CullMode::Front => Self::Inner::FRONT,
            CullMode::None => Self::Inner::NONE,
            CullMode::FrontAndBack => Self::Inner::FRONT_AND_BACK,
        }
    }
}

impl ToVk for PolygonMode {
    type Inner = vk::PolygonMode;

    fn to_vk(&self) -> Self::Inner {
        match self {
            PolygonMode::Fill => vk::PolygonMode::FILL,
            PolygonMode::Line(_) => vk::PolygonMode::LINE,
            PolygonMode::Point => vk::PolygonMode::POINT,
        }
    }
}

impl ToVk for FrontFace {
    type Inner = vk::FrontFace;

    fn to_vk(&self) -> Self::Inner {
        match self {
            FrontFace::CounterClockWise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::ClockWise => vk::FrontFace::CLOCKWISE,
        }
    }
}

impl ToVk for MemoryBarrier {
    type Inner = vk::MemoryBarrier;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            s_type: StructureType::MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask: self.src_access_mask.to_vk(),
            dst_access_mask: self.dst_access_mask.to_vk(),
        }
    }
}

impl ToVk for ImageSubresourceRange {
    type Inner = vk::ImageSubresourceRange;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            aspect_mask: self.aspect_mask.to_vk(),
            base_mip_level: self.base_mip_level,
            level_count: self.level_count,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        }
    }
}

impl ToVk for ColorLoadOp {
    type Inner = vk::AttachmentLoadOp;

    fn to_vk(&self) -> Self::Inner {
        use ColorLoadOp::{Clear, DontCare, Load};
        match self {
            DontCare => Self::Inner::DONT_CARE,
            Load => Self::Inner::LOAD,
            Clear(_) => Self::Inner::CLEAR,
        }
    }
}

impl ToVk for DepthLoadOp {
    type Inner = vk::AttachmentLoadOp;

    fn to_vk(&self) -> Self::Inner {
        use DepthLoadOp::{Clear, DontCare, Load};
        match self {
            DontCare => Self::Inner::DONT_CARE,
            Load => Self::Inner::LOAD,
            Clear(_) => Self::Inner::CLEAR,
        }
    }
}

impl ToVk for StencilLoadOp {
    type Inner = vk::AttachmentLoadOp;

    fn to_vk(&self) -> Self::Inner {
        use StencilLoadOp::{Clear, DontCare, Load};
        match self {
            DontCare => Self::Inner::DONT_CARE,
            Load => Self::Inner::LOAD,
            Clear(_) => Self::Inner::CLEAR,
        }
    }
}

impl ToVk for AttachmentStoreOp {
    type Inner = vk::AttachmentStoreOp;

    fn to_vk(&self) -> Self::Inner {
        match self {
            AttachmentStoreOp::DontCare => Self::Inner::DONT_CARE,
            AttachmentStoreOp::Store => Self::Inner::STORE,
        }
    }
}

impl ToVk for Viewport {
    type Inner = vk::Viewport;

    fn to_vk(&self) -> Self::Inner {
        vk::Viewport {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        }
    }
}

impl ToVk for CommandPoolCreateFlags {
    type Inner = vk::CommandPoolCreateFlags;

    fn to_vk(&self) -> Self::Inner {
        let mut result = vk::CommandPoolCreateFlags::default();

        case!(self, result, Self::TRANSIENT, Self::Inner::TRANSIENT);

        result
    }
}

impl ToVk for LogicOp {
    type Inner = vk::LogicOp;

    fn to_vk(&self) -> Self::Inner {
        match self {
            LogicOp::Clear => Self::Inner::CLEAR,
            LogicOp::And => Self::Inner::AND,
            LogicOp::AndReverse => Self::Inner::AND_REVERSE,
            LogicOp::Copy => Self::Inner::COPY,
            LogicOp::AndInverted => Self::Inner::AND_INVERTED,
            LogicOp::NoOp => Self::Inner::NO_OP,
            LogicOp::Xor => Self::Inner::XOR,
            LogicOp::Or => Self::Inner::OR,
            LogicOp::Nor => Self::Inner::NOR,
            LogicOp::Equivalent => Self::Inner::EQUIVALENT,
            LogicOp::Invert => Self::Inner::INVERT,
            LogicOp::OrReverse => Self::Inner::OR_REVERSE,
            LogicOp::CopyInverted => Self::Inner::COPY_INVERTED,
            LogicOp::OrInverted => Self::Inner::OR_INVERTED,
            LogicOp::Nand => Self::Inner::NAND,
            LogicOp::Set => Self::Inner::SET,
        }
    }
}

impl ToVk for PipelineColorBlendAttachmentState {
    type Inner = vk::PipelineColorBlendAttachmentState;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            blend_enable: self.blend_enable.to_vk(),
            src_color_blend_factor: self.src_color_blend_factor.to_vk(),
            dst_color_blend_factor: self.dst_color_blend_factor.to_vk(),
            color_blend_op: self.color_blend_op.to_vk(),
            src_alpha_blend_factor: self.src_alpha_blend_factor.to_vk(),
            dst_alpha_blend_factor: self.dst_alpha_blend_factor.to_vk(),
            alpha_blend_op: self.alpha_blend_op.to_vk(),
            color_write_mask: self.color_write_mask.to_vk(),
        }
    }
}

impl ToVk for PrimitiveTopology {
    type Inner = vk::PrimitiveTopology;

    fn to_vk(&self) -> Self::Inner {
        match self {
            PrimitiveTopology::TriangleList => Self::Inner::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => Self::Inner::TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => Self::Inner::TRIANGLE_FAN,
            PrimitiveTopology::PointList => Self::Inner::POINT_LIST,
            PrimitiveTopology::LineList => Self::Inner::LINE_LIST,
            PrimitiveTopology::LineStrip => Self::Inner::LINE_STRIP,
        }
    }
}
