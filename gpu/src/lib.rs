mod allocator;
mod command_buffer;
mod descriptor_set;
mod gpu;
mod pipeline;
mod swapchain;
mod types;

pub use allocator::*;
use ash::vk::ImageLayout;
pub use command_buffer::*;
pub use gpu::*;
pub use pipeline::*;
pub use swapchain::Swapchain;
pub use types::*;

#[derive(Default)]
pub enum QueueType {
    #[default]
    Graphics,
    AsyncCompute,
    Transfer,
}
impl QueueType {
    fn get_vk_command_pool(&self, gpu: &Gpu) -> ash::vk::CommandPool {
        match self {
            QueueType::Graphics => gpu.thread_local_state.graphics_command_pool,
            QueueType::AsyncCompute => gpu.thread_local_state.compute_command_pool,
            QueueType::Transfer => gpu.thread_local_state.transfer_command_pool,
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

#[derive(Clone, Copy, Hash, Debug)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Hash)]
pub struct DescriptorInfo<'a> {
    pub binding: u32,
    pub element_type: DescriptorType<'a>,
    pub binding_stage: ShaderStage,
}

#[derive(Clone, Hash)]
pub struct DescriptorSetInfo<'a> {
    pub descriptors: &'a [DescriptorInfo<'a>],
}
