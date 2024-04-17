use crate::{
    BindingSet, Buffer, BufferDescription, BufferWriteParams, DeviceConfiguration, DeviceInfo,
    GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription, ImageView, MgpuResult,
    RenderPassInfo, ShaderModule, ShaderModuleDescription,
};
use std::sync::Arc;

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

#[derive(Clone, Copy)]
pub struct CommandRecorderAllocator {
    id: u64,
}

#[derive(Clone, Copy)]
pub struct CommandRecorder {
    id: u64,
}

#[derive(Clone)]
pub struct SubmitInfo {
    pub graphics_command_recorders: Vec<CommandRecorder>,
    pub async_compute_command_recorders: Vec<CommandRecorder>,
}

pub struct RenderState {
    pub graphics_compute_allocator: CommandRecorderAllocator,
    pub async_compute_allocator: CommandRecorderAllocator,
}

pub(crate) trait Hal: Send + Sync {
    unsafe fn begin_rendering(&self) -> MgpuResult<RenderState>;
    unsafe fn request_command_recorder(
        &self,
        allocator: CommandRecorderAllocator,
    ) -> MgpuResult<CommandRecorder>;
    unsafe fn finalize_command_recorder(&self, command_buffer: CommandRecorder) -> MgpuResult<()>;

    unsafe fn begin_render_pass(
        &self,
        command_recorder: CommandRecorder,
        render_pass_info: &RenderPassInfo,
    ) -> MgpuResult<()>;

    unsafe fn bind_graphics_pipeline(
        &self,
        command_recorder: CommandRecorder,
        pipeline: GraphicsPipeline,
    ) -> MgpuResult<()>;

    unsafe fn set_vertex_buffers(
        &self,
        command_recorder: CommandRecorder,
        vertex_buffers: &[Buffer],
    ) -> MgpuResult<()>;

    unsafe fn set_binding_sets(
        &self,
        command_recorder: CommandRecorder,
        binding_sets: &[BindingSet],
    ) -> MgpuResult<()>;

    unsafe fn draw(
        &self,
        command_recorder: CommandRecorder,
        vertices: usize,
        indices: usize,
        first_vertex: usize,
        first_index: usize,
    ) -> MgpuResult<()>;

    unsafe fn advance_to_next_step(&self, command_recorder: CommandRecorder) -> MgpuResult<()>;
    unsafe fn end_render_pass(&self, command_recorder: CommandRecorder) -> MgpuResult<()>;
    unsafe fn present_image(&self, swapchain_id: u64, image: Image) -> MgpuResult<()>;
    unsafe fn submit(&self, end_rendering_info: SubmitInfo) -> MgpuResult<()>;
    unsafe fn end_rendering(&self) -> MgpuResult<()>;

    #[cfg(feature = "swapchain")]
    fn create_swapchain_impl(&self, swapchain_info: &SwapchainCreationInfo) -> MgpuResult<u64>;

    #[cfg(feature = "swapchain")]
    fn swapchain_acquire_next_image(&self, id: u64) -> MgpuResult<SwapchainImage>;

    #[cfg(feature = "swapchain")]
    fn swapchain_on_resized(
        &self,
        id: u64,
        new_size: crate::Extents2D,
        window_handle: raw_window_handle::WindowHandle,
        display_handle: raw_window_handle::DisplayHandle,
    ) -> MgpuResult<()>;

    fn device_info(&self) -> DeviceInfo;

    fn create_image(&self, image_description: &ImageDescription) -> MgpuResult<Image>;
    fn image_name(&self, image: Image) -> MgpuResult<Option<String>>;
    fn destroy_image(&self, image: Image) -> MgpuResult<()>;

    fn create_buffer(&self, buffer_description: &BufferDescription) -> MgpuResult<Buffer>;
    fn buffer_name(&self, buffer: Buffer) -> MgpuResult<Option<String>>;
    fn destroy_buffer(&self, buffer: Buffer) -> MgpuResult<()>;

    fn create_graphics_pipeline(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<GraphicsPipeline>;

    fn destroy_graphics_pipeline(&self, graphics_pipeline: GraphicsPipeline) -> MgpuResult<()>;

    fn create_shader_module(
        &self,
        shader_module_description: &ShaderModuleDescription,
    ) -> MgpuResult<ShaderModule>;
    fn destroy_shader_module(&self, shader_module: ShaderModule) -> MgpuResult<()>;

    fn destroy_image_view(&self, image_view: ImageView) -> MgpuResult<()>;

    unsafe fn write_host_visible_buffer(
        &self,
        buffer: Buffer,
        params: &BufferWriteParams,
    ) -> MgpuResult<()>;
}

pub(crate) fn create(configuration: &DeviceConfiguration) -> MgpuResult<Arc<dyn Hal>> {
    #[cfg(feature = "vulkan")]
    vulkan::VulkanHal::create(configuration)
}
