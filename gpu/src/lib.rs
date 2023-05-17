mod allocator;
mod command_buffer;
mod descriptor_set;
mod gpu;
mod material;
mod resource;
mod swapchain;
mod types;

pub use allocator::*;
use ash::vk::{self, ImageLayout};
pub use command_buffer::*;
pub use gpu::*;
pub use material::*;
pub use resource::*;
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
    fn get_vk_queue(&self, gpu: &Gpu) -> ash::vk::CommandPool {
        match self {
            QueueType::Graphics => gpu.thread_local_state.graphics_command_pool,
            QueueType::AsyncCompute => gpu.thread_local_state.compute_command_pool,
            QueueType::Transfer => gpu.thread_local_state.transfer_command_pool,
        }
    }
}

#[derive(Clone, Hash)]
pub struct BufferRange {
    pub handle: ResourceHandle<GpuBuffer>,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Hash)]
pub struct SamplerState {
    pub sampler: ResourceHandle<GpuSampler>,
    pub image_view: ResourceHandle<GpuImage>,
    pub image_layout: ImageLayout,
}

#[derive(Clone)]
pub enum DescriptorType {
    UniformBuffer(BufferRange),
    StorageBuffer(BufferRange),
    Sampler(SamplerState),
    CombinedImageSampler(SamplerState),
}

impl std::hash::Hash for DescriptorType {
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
pub struct DescriptorInfo {
    pub binding: u32,
    pub element_type: DescriptorType,
    pub binding_stage: ShaderStage,
}

#[derive(Clone, Hash)]
pub struct DescriptorSetInfo<'a> {
    pub descriptors: &'a [DescriptorInfo],
}
