use crate::{
    BindingSet, BindingSetLayout, Buffer, BufferDescription, BufferWriteParams, CullMode,
    DepthStencilState, DepthStencilTargetInfo, DeviceConfiguration, DeviceInfo, FrontFace,
    GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription, ImageView, MgpuResult,
    MultisampleState, PolygonMode, PrimitiveTopology, RenderPassInfo, RenderTargetInfo,
    ShaderModule, ShaderModuleDescription, ShaderModuleLayout, VertexInputDescription,
};
use std::sync::Arc;

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

#[derive(Clone, Hash)]
pub struct OwnedVertexStageInfo {
    pub shader: ShaderModule,
    pub entry_point: String,
    pub vertex_inputs: Vec<VertexInputDescription>,
}

#[derive(Clone, Hash)]
pub struct OwnedFragmentStageInfo {
    pub shader: ShaderModule,
    pub entry_point: String,
    pub render_targets: Vec<RenderTargetInfo>,
    pub depth_stencil_target: Option<DepthStencilTargetInfo>,
}

#[derive(Clone, Hash)]
pub struct GraphicsPipelineLayout {
    pub label: Option<String>,
    pub binding_sets: Vec<BindingSetLayout>,
    pub vertex_stage: OwnedVertexStageInfo,
    pub fragment_stage: Option<OwnedFragmentStageInfo>,
    pub primitive_restart_enabled: bool,
    pub primitive_topology: PrimitiveTopology,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub multisample_state: Option<MultisampleState>,
    pub depth_stencil_state: DepthStencilState,
}

#[derive(Clone, Copy)]
pub struct CommandRecorderAllocator {
    id: u64,
    queue_type: QueueType,
}

#[derive(Clone, Copy)]
pub struct CommandRecorder {
    id: u64,
    queue_type: QueueType,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum QueueType {
    // This queue can execute both graphics commands (such as drawing) and sync compute commands
    Graphics,

    // This queue can execute only compute commands, and it runs asynchronously from the Graphics and Transfer queue
    AsyncCompute,

    // This queue can execute only transfer commands, and it runs asynchronously from the Graphics and Compute queue
    AsyncTransfer,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum AttachmentType {
    Color,
    DepthStencil,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ResourceAccessMode {
    AttachmentRead(AttachmentType),
    AttachmentWrite(AttachmentType),

    VertexInput,

    ShaderRead,
    ShaderWrite,

    TransferSrc,
    TransferDst,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Resource {
    Image {
        image: Image,
    },
    ImageView {
        view: ImageView,
    },
    Buffer {
        buffer: Buffer,
        offset: usize,
        size: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ResourceInfo {
    pub resource: Resource,
    pub access_mode: ResourceAccessMode,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SynchronizationResource {
    pub ty: Resource,
    pub source_access_mode: ResourceAccessMode,
    pub destination_access_mode: ResourceAccessMode,
}

#[derive(Clone)]
pub struct SynchronizationInfo {
    pub source_queue: QueueType,
    pub source_command_recorder: CommandRecorder,
    pub destination_queue: QueueType,
    pub destination_command_recorder: CommandRecorder,
    pub resources: Vec<ResourceInfo>,
}

#[derive(Clone, Default)]
pub struct SubmissionGroup {
    pub command_recorders: Vec<CommandRecorder>,
}

#[derive(Clone)]
pub struct SubmitInfo {
    pub submission_groups: Vec<SubmissionGroup>,
}

pub struct RenderState {
    pub graphics_compute_allocator: CommandRecorderAllocator,
    pub async_compute_allocator: CommandRecorderAllocator,
    pub async_transfer_allocator: CommandRecorderAllocator,
}

pub(crate) trait Hal: Send + Sync {
    fn device_wait_idle(&self) -> MgpuResult<()>;
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

    unsafe fn cmd_copy_buffer_to_buffer(
        &self,
        command_buffer: CommandRecorder,
        source: Buffer,
        dest: Buffer,
        source_offset: usize,
        dest_offset: usize,
        size: usize,
    ) -> MgpuResult<()>;

    unsafe fn enqueue_synchronization(&self, infos: &[SynchronizationInfo]) -> MgpuResult<()>;

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
    fn get_graphics_pipeline_layout(
        &self,
        graphics_pipeline: GraphicsPipeline,
    ) -> MgpuResult<GraphicsPipelineLayout>;
    fn destroy_graphics_pipeline(&self, graphics_pipeline: GraphicsPipeline) -> MgpuResult<()>;

    fn create_shader_module(
        &self,
        shader_module_description: &ShaderModuleDescription,
    ) -> MgpuResult<ShaderModule>;
    fn get_shader_module_layout(
        &self,
        shader_module: ShaderModule,
    ) -> MgpuResult<ShaderModuleLayout>;
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
