mod allocator;
mod command_buffer;
mod descriptor_set;
mod gpu;
mod pipeline;
mod swapchain;
mod types;

pub use crate::gpu::*;
pub use allocator::*;
pub use command_buffer::*;
pub use pipeline::*;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
pub use swapchain::Swapchain;
pub use types::*;

pub const WHOLE_SIZE: u64 = u64::MAX;

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

#[derive(Clone, Copy, Hash, Debug, PartialEq, Ord, PartialOrd, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    VertexFragment,
    Compute,
    All,
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

#[derive(Clone, Copy, Hash, Debug, PartialEq, Ord, PartialOrd, Eq)]
pub enum PresentMode {
    #[doc = "Present images immediately"]
    Immediate,
    #[doc = "Wait for next vsync, new images are discarded"]
    Fifo,
    #[doc = "May not be supported, wait for next vsync, discard old images in queue in favour for new images"]
    Mailbox,
}
