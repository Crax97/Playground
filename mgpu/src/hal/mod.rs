use crate::{
    BindingSet, BindingSetDescription, BindingSetLayout, Buffer, BufferDescription,
    BufferWriteParams, ComputePipeline, ComputePipelineDescription, CullMode, DepthStencilState,
    DepthStencilTargetInfo, DeviceConfiguration, DeviceInfo, FilterMode, FrontFace,
    GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription, ImageRegion,
    ImageSubresource, ImageView, ImageViewDescription, MgpuResult, MultisampleState,
    OwnedBindingSetLayoutInfo, PolygonMode, PrimitiveTopology, PushConstantInfo, RenderPassInfo,
    RenderTargetInfo, Sampler, SamplerDescription, ShaderModule, ShaderModuleDescription,
    ShaderModuleLayout, ShaderStageFlags, VertexInputDescription,
};
use std::sync::Arc;

pub(crate) mod dummy;

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
    pub binding_sets_infos: Vec<OwnedBindingSetLayoutInfo>,
    pub vertex_stage: OwnedVertexStageInfo,
    pub fragment_stage: Option<OwnedFragmentStageInfo>,
    pub primitive_restart_enabled: bool,
    pub primitive_topology: PrimitiveTopology,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub multisample_state: Option<MultisampleState>,
    pub depth_stencil_state: DepthStencilState,
    pub push_constant_range: Option<PushConstantInfo>,
}

#[derive(Clone, Hash)]
pub struct ComputePipelineLayout {
    pub label: Option<String>,
    pub binding_sets_infos: Vec<OwnedBindingSetLayoutInfo>,
    pub shader: ShaderModule,
    pub entry_point: String,
    pub push_constant_range: Option<PushConstantInfo>,
}

#[derive(Clone, Copy)]
pub struct CommandRecorderAllocator {
    id: u64,
    queue_type: QueueType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct CommandRecorder {
    id: u64,
    queue_type: QueueType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum ResourceAccessMode {
    #[default]
    Undefined,
    AttachmentRead(AttachmentType),
    AttachmentWrite(AttachmentType),

    VertexInput,

    ShaderRead(ShaderStageFlags),
    #[allow(dead_code)]
    ShaderWrite(ShaderStageFlags),

    TransferSrc,
    TransferDst,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Resource {
    Image {
        image: Image,
        subresource: ImageSubresource,
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ResourceTransition {
    pub resource: Resource,
    pub old_usage: ResourceAccessMode,
    pub new_usage: ResourceAccessMode,
}

#[derive(Clone)]
pub struct SynchronizationInfo {
    pub source_queue: QueueType,
    pub source_command_recorder: Option<CommandRecorder>,
    pub destination_queue: QueueType,
    pub destination_command_recorder: CommandRecorder,
    pub resources: Vec<ResourceTransition>,
}

#[derive(Clone, Default, Debug)]
pub struct SubmissionGroup {
    pub command_recorders: Vec<CommandRecorder>,
}

#[derive(Clone, Debug)]
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
    fn device_wait_queue(&self, queue: QueueType) -> MgpuResult<()>;
    unsafe fn prepare_next_frame(&self) -> MgpuResult<()>;
    unsafe fn begin_rendering(&self) -> MgpuResult<RenderState>;
    unsafe fn request_oneshot_command_recorder(
        &self,
        queue_type: QueueType,
    ) -> MgpuResult<CommandRecorder>;

    unsafe fn submit_command_recorder_immediate(
        &self,
        command_recorder: CommandRecorder,
    ) -> MgpuResult<()>;
    unsafe fn request_command_recorder(
        &self,
        allocator: CommandRecorderAllocator,
    ) -> MgpuResult<CommandRecorder>;
    unsafe fn finalize_command_recorder(&self, command_buffer: CommandRecorder) -> MgpuResult<()>;

    unsafe fn set_graphics_push_constant(
        &self,
        command_recorder: CommandRecorder,
        graphics_pipeline: GraphicsPipeline,
        data: &[u8],
        visibility: ShaderStageFlags,
    ) -> MgpuResult<()>;

    unsafe fn set_compute_push_constant(
        &self,
        command_recorder: CommandRecorder,
        compute_pipeline: ComputePipeline,
        data: &[u8],
        visibility: ShaderStageFlags,
    ) -> MgpuResult<()>;

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

    unsafe fn set_index_buffer(
        &self,
        command_recorder: CommandRecorder,
        index_buffer: Buffer,
    ) -> MgpuResult<()>;

    unsafe fn bind_graphics_binding_sets(
        &self,
        command_recorder: CommandRecorder,
        binding_sets: &[BindingSet],
        graphics_pipeline: GraphicsPipeline,
    ) -> MgpuResult<()>;

    unsafe fn draw(
        &self,
        command_recorder: CommandRecorder,
        vertices: usize,
        indices: usize,
        first_vertex: usize,
        first_index: usize,
    ) -> MgpuResult<()>;
    unsafe fn draw_indexed(
        &self,
        command_recorder: CommandRecorder,
        indices: usize,
        instances: usize,
        first_index: usize,
        vertex_offset: i32,
        first_instance: usize,
    ) -> MgpuResult<()>;

    unsafe fn advance_to_next_step(&self, command_recorder: CommandRecorder) -> MgpuResult<()>;
    unsafe fn end_render_pass(&self, command_recorder: CommandRecorder) -> MgpuResult<()>;
    unsafe fn present_image(&self, swapchain_id: u64, image: Image) -> MgpuResult<()>;
    unsafe fn submit(&self, end_rendering_info: SubmitInfo) -> MgpuResult<()>;
    unsafe fn end_rendering(&self) -> MgpuResult<()>;

    unsafe fn bind_compute_pipeline(
        &self,
        command_recorder: CommandRecorder,
        pipeline: ComputePipeline,
    ) -> MgpuResult<()>;
    unsafe fn bind_compute_binding_sets(
        &self,
        command_recorder: CommandRecorder,
        binding_sets: &[BindingSet],
        pipeline: ComputePipeline,
    ) -> MgpuResult<()>;

    unsafe fn dispatch(
        &self,
        command_recorder: CommandRecorder,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> MgpuResult<()>;

    unsafe fn cmd_copy_buffer_to_buffer(
        &self,
        command_buffer: CommandRecorder,
        source: Buffer,
        dest: Buffer,
        source_offset: usize,
        dest_offset: usize,
        size: usize,
    ) -> MgpuResult<()>;

    unsafe fn cmd_copy_buffer_to_image(
        &self,
        command_buffer: CommandRecorder,
        source: Buffer,
        dest: Image,
        source_offset: usize,
        dest_region: ImageRegion,
    ) -> MgpuResult<()>;

    unsafe fn cmd_blit_image(
        &self,
        command_buffer: CommandRecorder,
        source: Image,
        source_region: ImageRegion,
        dest: Image,
        dest_region: ImageRegion,
        filter: FilterMode,
    ) -> MgpuResult<()>;

    unsafe fn enqueue_synchronization(&self, infos: &[SynchronizationInfo]) -> MgpuResult<()>;
    fn transition_resources(
        &self,
        command_recorder: CommandRecorder,
        resources: &[ResourceTransition],
    ) -> MgpuResult<()>;

    #[cfg(feature = "swapchain")]
    fn create_swapchain_impl(
        &self,
        swapchain_info: &SwapchainCreationInfo,
    ) -> MgpuResult<SwapchainInfo>;

    #[cfg(feature = "swapchain")]
    fn swapchain_destroy(&self, id: u64) -> MgpuResult<()>;

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

    #[cfg(feature = "swapchain")]
    fn try_swapchain_set_present_mode(
        &self,
        id: u64,
        present_mode: PresentMode,
    ) -> MgpuResult<PresentMode>;

    fn device_info(&self) -> DeviceInfo;

    fn create_image(&self, image_description: &ImageDescription) -> MgpuResult<Image>;
    fn image_name(&self, image: Image) -> MgpuResult<Option<String>>;
    fn destroy_image(&self, image: Image) -> MgpuResult<()>;

    fn create_image_view(
        &self,
        image_view_description: &ImageViewDescription,
    ) -> MgpuResult<ImageView>;

    fn create_buffer(&self, buffer_description: &BufferDescription) -> MgpuResult<Buffer>;
    fn buffer_name(&self, buffer: Buffer) -> MgpuResult<Option<String>>;
    fn destroy_buffer(&self, buffer: Buffer) -> MgpuResult<()>;

    fn create_graphics_pipeline(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<GraphicsPipeline>;

    fn create_compute_pipeline(
        &self,
        compute_pipeline_description: &ComputePipelineDescription,
    ) -> MgpuResult<ComputePipeline>;

    fn get_graphics_pipeline_layout(
        &self,
        graphics_pipeline: GraphicsPipeline,
    ) -> MgpuResult<GraphicsPipelineLayout>;
    fn get_compute_pipeline_layout(
        &self,
        compute_pipeline: ComputePipeline,
    ) -> MgpuResult<ComputePipelineLayout>;
    fn destroy_graphics_pipeline(&self, graphics_pipeline: GraphicsPipeline) -> MgpuResult<()>;
    fn destroy_compute_pipeline(&self, pipeline: ComputePipeline) -> MgpuResult<()>;

    fn create_shader_module(
        &self,
        shader_module_description: &ShaderModuleDescription,
    ) -> MgpuResult<ShaderModule>;
    fn get_shader_module_layout(
        &self,
        shader_module: ShaderModule,
    ) -> MgpuResult<ShaderModuleLayout>;
    fn destroy_shader_module(&self, shader_module: ShaderModule) -> MgpuResult<()>;

    fn create_sampler(&self, sampler_description: &SamplerDescription) -> MgpuResult<Sampler>;
    fn destroy_sampler(&self, sampler: Sampler) -> MgpuResult<()>;

    fn create_binding_set(
        &self,
        description: &BindingSetDescription,
        layout: &BindingSetLayout,
    ) -> MgpuResult<BindingSet>;
    fn destroy_binding_set(&self, binding_set: BindingSet) -> MgpuResult<()>;

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
