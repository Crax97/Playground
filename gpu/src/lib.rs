mod allocator;
mod command_buffer;
mod descriptor_set;
mod gpu;
mod pipeline;
mod swapchain;
mod types;

pub use bitflags::bitflags;
pub use crate::gpu::*;
pub use allocator::*;
pub use command_buffer::*;
pub use pipeline::*;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
pub use swapchain::Swapchain;
pub use types::*;

pub const WHOLE_SIZE: u64 = u64::MAX;
pub const QUEUE_FAMILY_IGNORED: u32 = u32::MAX;

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndexType {
    #[default]
    Uint16,
    Uint32,
    Uint64,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum StencilOp {
   #[default]
    Keep,
    Zero,
    Replace(u32),
    ClampedIncrement,
    ClampedDecrement,
    Invert,
    WrappedIncrement,
    WrappedDecrement,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct StencilOpState {
    fail: StencilOp,
    pass: StencilOp,
    depth_fail: StencilOp,
    compare: CompareOp,
    compare_mask: u32,
    write_mask: u32,
    reference: u32,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum SamplerAddressMode {
    #[default]
    ClampToBorder,
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Filter {
    #[default]
    Linear,
    Nearest,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlendMode {
    #[default]
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

#[derive(Default, Clone, Debug, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum BlendOp {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Default, Clone, Debug, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum SampleCount {
    #[default]
    Sample1,
    Sample2,
    Sample4,
    Sample8,
    Sample16,
    Sample32,
    Sample64
}

bitflags! {
#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColorComponentFlags : u32 {
    const R = 0b1;
    const G = 0b10;
    const B = 0b100;
    const A = 0b1000;
}}

impl ColorComponentFlags {
    pub const RGBA : Self = Self::from_bits_truncate(Self::R.bits() | Self::G.bits() | Self::B.bits() | Self::A.bits());
}

#[derive(Default, Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct SamplerCreateInfo {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub address_u: SamplerAddressMode,
    pub address_v: SamplerAddressMode,
    pub address_w: SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub compare_function: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: [f32; 4],
}

#[derive(Default, Clone, Debug, Copy, PartialEq, PartialOrd)]
pub enum PipelineBindPoint {
    #[default]
    Graphics,
    Compute,
}

#[derive(Default)]
pub enum QueueType {
    #[default]
    Graphics,
    AsyncCompute,
    Transfer,
}
impl QueueType {
    fn get_vk_command_pool(&self, gpu: &Gpu) -> ash::vk::CommandPool {
        let thread_local_state = &gpu.thread_local_state;
        match self {
            QueueType::Graphics => thread_local_state.graphics_command_pool,
            QueueType::AsyncCompute => thread_local_state.compute_command_pool,
            QueueType::Transfer => thread_local_state.transfer_command_pool,
        }
    }
    fn get_vk_queue(&self, gpu: &Gpu) -> ash::vk::Queue {
        match self {
            QueueType::Graphics => gpu.state.graphics_queue,
            QueueType::AsyncCompute => gpu.state.async_compute_queue,
            QueueType::Transfer => gpu.state.transfer_queue,
        }
    }
}

#[derive(Clone, Hash)]
pub struct BufferRange<'a> {
    pub handle: &'a GpuBuffer,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Hash)]
pub struct SamplerState<'a> {
    pub sampler: &'a GpuSampler,
    pub image_view: &'a GpuImageView,
    pub image_layout: ImageLayout,
}

#[derive(Clone)]
pub enum DescriptorType<'a> {
    UniformBuffer(BufferRange<'a>),
    StorageBuffer(BufferRange<'a>),
    Sampler(SamplerState<'a>),
    CombinedImageSampler(SamplerState<'a>),
}

impl<'a> std::hash::Hash for DescriptorType<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl<'a> Debug for DescriptorType<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", core::mem::discriminant(self)))
    }
}

#[derive(Clone, Hash, Debug)]
pub struct DescriptorInfo<'a> {
    pub binding: u32,
    pub element_type: DescriptorType<'a>,
    pub binding_stage: ShaderStage,
}

#[derive(Clone, Hash)]
pub struct DescriptorSetInfo<'a> {
    pub descriptors: &'a [DescriptorInfo<'a>],
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AttachmentReference {
    pub attachment: u32,
    pub layout: ImageLayout,
}
#[derive(Clone, Copy, Hash, Debug, PartialEq, Ord, PartialOrd, Eq)]
pub enum PresentMode {
    #[doc = "Present images immediately"]
    Immediate,
    #[doc = "Wait for next vsync, new images are discarded"]
    Fifo,
    #[doc = "May not be supported, wait for next vsync, discard old images in queue in favour for new images"]
    Mailbox,
}
