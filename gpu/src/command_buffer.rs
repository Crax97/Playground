use std::cell::RefCell;
use std::ffi::CString;
use std::rc::Rc;
use std::sync::Arc;

use ash::vk::{
    self, ClearColorValue, ClearDepthStencilValue, ClearValue, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags, DebugUtilsLabelEXT, DependencyFlags, ImageMemoryBarrier2, ImageView, PipelineBindPoint, PipelineInputAssemblyStateCreateFlags, RenderingAttachmentInfo, RenderingInfo, StructureType, SubmitInfo
};
use ash::{extensions::ext::DebugUtils, prelude::VkResult};
use log::warn;

use crate::*;

use super::{QueueType, VkBuffer, VkGpu};

const RENDER_PASSS_LABEL_COLOR: [f32; 4] = [0.6, 0.6, 0.6, 1.0];
const SUBPASS_LABEL_COLOR: [f32; 4] = [0.373, 0.792, 0.988, 1.0];

struct VkCommandBufferState {
    inner_command_buffer: vk::CommandBuffer,
    has_recorded_anything: bool,
    has_been_submitted: bool,
    target_queue: vk::Queue,
    descriptor_state: DescriptorSetState,
    push_constant_data: Vec<Vec<u8>>,
}

impl VkCommandBufferState {}

pub struct VkCommandBuffer {
    state: Arc<GpuThreadSharedState>,
    command_buffer_state: Rc<RefCell<VkCommandBufferState>>,
    vk_command_buffer: vk::CommandBuffer,
}

pub struct VkRenderPassCommand {
    command_buffer: vk::CommandBuffer,
    state: Arc<GpuThreadSharedState>,
    command_buffer_state: Rc<RefCell<VkCommandBufferState>>,
    has_draw_command: bool,
    viewport_area: Option<Viewport>,
    depth_bias_setup: Option<(f32, f32, f32)>,

    pipeline_state: GraphicsPipelineState,
    pub render_pass: vk::RenderPass,
    render_pass_label: ScopedDebugLabel,
    subpass_label: Option<ScopedDebugLabel>,

    subpasses: Vec<SubpassDescription>,
}

#[derive(Hash)]
pub(crate) struct ComputePipelineState {
    pub(crate) shader: ShaderModuleHandle,
}
impl ComputePipelineState {
    fn new() -> ComputePipelineState {
        Self {
            shader: ShaderModuleHandle::null(),
        }
    }
}

pub struct VkComputePassCommand {
    pipeline_state: ComputePipelineState,
    command_buffer: vk::CommandBuffer,
    command_buffer_state: Rc<RefCell<VkCommandBufferState>>,
    state: Arc<GpuThreadSharedState>,
}

#[derive(Hash)]
pub(crate) struct GraphicsPipelineState {
    pub(crate) fragment_shader: ShaderModuleHandle,
    pub(crate) vertex_shader: ShaderModuleHandle,
    pub(crate) scissor_area: Option<Rect2D>,
    pub(crate) front_face: FrontFace,
    pub(crate) cull_mode: CullMode,
    // If vulkan complains about a missing shader frament shader, check that these two bastards are
    // enabled
    pub(crate) enable_depth_test: bool,
    pub(crate) depth_clamp_enabled: bool,
    pub(crate) depth_write_enabled: bool,

    pub(crate) early_discard_enabled: bool,

    pub(crate) depth_compare_op: CompareOp,
    pub(crate) render_area: Rect2D,
    pub(crate) vertex_inputs: Vec<VertexBindingInfo>,
    pub(crate) color_blend_states: Vec<PipelineColorBlendAttachmentState>,
    pub(crate) primitive_topology: PrimitiveTopology,
    pub(crate) polygon_mode: PolygonMode,
    pub(crate) color_output_enabled: bool,

    pub(crate) current_subpass: u32,
}

#[derive(Hash, Default)]
pub(crate) struct GraphicsPipelineState2 {
    pub(crate) fragment_shader: ShaderModuleHandle,
    pub(crate) vertex_shader: ShaderModuleHandle,
    pub(crate) scissor_area: Option<Rect2D>,
    pub(crate) front_face: FrontFace,
    pub(crate) cull_mode: CullMode,
    // If vulkan complains about a missing shader frament shader, check that these two bastards are
    // enabled
    pub(crate) enable_depth_test: bool,
    pub(crate) depth_clamp_enabled: bool,
    pub(crate) depth_write_enabled: bool,

    pub(crate) early_discard_enabled: bool,

    pub(crate) depth_compare_op: CompareOp,
    pub(crate) vertex_inputs: Vec<VertexBindingInfo>,
    pub(crate) primitive_topology: PrimitiveTopology,
    pub(crate) polygon_mode: PolygonMode,
    pub(crate) color_output_enabled: bool,
}

impl GraphicsPipelineState {
    fn new(info: &BeginRenderPassInfo) -> Self {
        // TODO: put this into pipeline state
        let color_blend_states = info
            .color_attachments
            .iter()
            .map(|_| PipelineColorBlendAttachmentState {
                blend_enable: true,
                src_color_blend_factor: BlendMode::One,
                dst_color_blend_factor: BlendMode::OneMinusSrcAlpha,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendMode::One,
                dst_alpha_blend_factor: BlendMode::OneMinusSrcAlpha,
                alpha_blend_op: BlendOp::Add,
                color_write_mask: ColorComponentFlags::RGBA,
            })
            .collect();
        Self {
            fragment_shader: Handle::null(),
            vertex_shader: Handle::null(),
            scissor_area: None,
            front_face: FrontFace::default(),
            cull_mode: CullMode::default(),
            render_area: info.render_area,
            enable_depth_test: false,
            depth_clamp_enabled: false,
            depth_write_enabled: false,
            depth_compare_op: CompareOp::Never,
            primitive_topology: PrimitiveTopology::TriangleList,
            polygon_mode: PolygonMode::Fill,
            early_discard_enabled: false,

            vertex_inputs: vec![],
            color_blend_states,
            color_output_enabled: true,
            current_subpass: 0,
        }
    }

    pub(crate) fn input_assembly_state(&self) -> vk::PipelineInputAssemblyStateCreateInfo {
        vk::PipelineInputAssemblyStateCreateInfo {
            s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: PipelineInputAssemblyStateCreateFlags::empty(),
            topology: self.primitive_topology.to_vk(),
            primitive_restart_enable: vk::FALSE,
        }
    }

    pub(crate) fn get_vertex_inputs_description(
        &self,
    ) -> (
        Vec<vk::VertexInputBindingDescription>,
        Vec<vk::VertexInputAttributeDescription>,
    ) {
        let mut inputs = vec![];
        let mut attributes = vec![];

        for (index, input) in self.vertex_inputs.iter().enumerate() {
            inputs.push(vk::VertexInputBindingDescription {
                binding: index as _,
                stride: input.stride,
                input_rate: input.input_rate.to_vk(),
            });
        }
        for (index, input) in self.vertex_inputs.iter().enumerate() {
            attributes.push(vk::VertexInputAttributeDescription {
                location: input.location,
                binding: index as _,
                format: input.format.to_vk(),
                offset: input.offset,
            });
        }

        (inputs, attributes)
    }

    pub(crate) fn rasterization_state(&self) -> vk::PipelineRasterizationStateCreateInfo {
        vk::PipelineRasterizationStateCreateInfo {
            s_type: StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: self.depth_clamp_enabled.to_vk(),
            rasterizer_discard_enable: self.early_discard_enabled.to_vk(),
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: self.cull_mode.to_vk(),
            front_face: self.front_face.to_vk(),
            depth_bias_enable: vk::TRUE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        }
    }

    pub(crate) fn multisample_state(&self) -> vk::PipelineMultisampleStateCreateInfo {
        vk::PipelineMultisampleStateCreateInfo {
            s_type: StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: SampleCount::Sample1.to_vk(),
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
        }
    }

    pub(crate) fn depth_stencil_state(&self) -> vk::PipelineDepthStencilStateCreateInfo {
        vk::PipelineDepthStencilStateCreateInfo {
            s_type: StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: self.enable_depth_test.to_vk(),
            depth_write_enable: self.depth_write_enabled.to_vk(),
            depth_compare_op: self.depth_compare_op.to_vk(),
            depth_bounds_test_enable: false.to_vk(),
            stencil_test_enable: false.to_vk(),
            front: StencilOpState::default().to_vk(),
            back: StencilOpState::default().to_vk(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        }
    }

    pub(crate) fn dynamic_state(&self) -> vk::PipelineDynamicStateCreateInfo {
        static DYNAMIC_STATES: &[vk::DynamicState] = &[
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::DEPTH_TEST_ENABLE,
        ];
        vk::PipelineDynamicStateCreateInfo {
            s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineDynamicStateCreateFlags::empty(),
            dynamic_state_count: DYNAMIC_STATES.len() as _,
            p_dynamic_states: DYNAMIC_STATES.as_ptr() as *const _,
        }
    }
}

#[derive(Hash, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum DescriptorBindingType {
    UniformBuffer {
        handle: BufferHandle,
        offset: u64,
        range: usize,
    },

    StorageBuffer {
        handle: BufferHandle,
        offset: u64,
        range: usize,
    },
    ImageView {
        image_view_handle: ImageViewHandle,
        sampler_handle: SamplerHandle,
        layout: ImageLayout,
    },
    InputAttachment {
        image_view_handle: ImageViewHandle,
        layout: ImageLayout,
    },
}

impl Default for DescriptorBindingType {
    fn default() -> Self {
        Self::UniformBuffer {
            handle: BufferHandle::null(),
            offset: 0,
            range: 0,
        }
    }
}

#[derive(Hash, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum DescriptorBindingType2 {
    UniformBuffer {
        handle: BufferHandle,
        offset: u64,
        range: u64,
    },

    StorageBuffer {
        handle: BufferHandle,
        offset: u64,
        range: u64,
    },
    ImageView {
        image_view_handle: ImageViewHandle,
        sampler_handle: SamplerHandle,
    },
}

impl Default for DescriptorBindingType2 {
    fn default() -> Self {
        Self::UniformBuffer {
            handle: BufferHandle::null(),
            offset: 0,
            range: 0,
        }
    }
}

impl DescriptorSetLayoutDescription {
    pub(crate) fn vk_set_layout_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        let mut descriptor_set_bindings = vec![];
        for (binding_index, binding) in self.elements.iter().enumerate() {
            let stage_flags = binding.stage.to_vk();
            let descriptor_type = match binding.binding_type {
                BindingType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                BindingType::Storage => vk::DescriptorType::STORAGE_BUFFER,
                BindingType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                BindingType::Sampler => vk::DescriptorType::SAMPLER,
                BindingType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
            };
            let binding = vk::DescriptorSetLayoutBinding {
                binding: binding_index as _,
                descriptor_type,
                descriptor_count: 1,
                stage_flags,
                p_immutable_samplers: std::ptr::null(),
            };
            descriptor_set_bindings.push(binding);
        }
        descriptor_set_bindings
    }
}

impl VkCommandBuffer {
    pub(crate) fn new(
        gpu: &VkGpu,
        command_pool: &VkCommandPool,
        target_queue: QueueType,
    ) -> VkResult<Self> {
        assert_eq!(command_pool.associated_queue, target_queue);

        let device = gpu.vk_logical_device();
        let inner_command_buffer = unsafe {
            device.allocate_command_buffers(&CommandBufferAllocateInfo {
                s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                command_pool: command_pool.inner,
                level: CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
            })
        }?[0];

        unsafe {
            device.begin_command_buffer(
                inner_command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )
        }?;

        Ok(Self {
            state: gpu.state.clone(),
            vk_command_buffer: inner_command_buffer,
            command_buffer_state: Rc::new(RefCell::new(VkCommandBufferState {
                inner_command_buffer,
                has_recorded_anything: false,
                has_been_submitted: false,
                target_queue: target_queue.get_vk_queue(gpu),
                descriptor_state: DescriptorSetState::default(),
                push_constant_data: vec![],
            })),
        })
    }

    pub fn begin_render_pass(&mut self, info: &BeginRenderPassInfo) -> VkRenderPassCommand {
        VkRenderPassCommand::new(self, info)
    }

    pub fn begin_compute_pass(&mut self) -> VkComputePassCommand {
        VkComputePassCommand::new(self)
    }

    pub fn copy_buffer(
        &mut self,
        src_buffer: &VkBuffer,
        dst_buffer: &VkBuffer,
        dst_offset: u64,
        size: usize,
    ) -> VkResult<()> {
        self.command_buffer_state.borrow_mut().has_recorded_anything = true;
        unsafe {
            self.state.logical_device.cmd_copy_buffer(
                self.command_buffer_state.borrow().inner_command_buffer,
                src_buffer.inner,
                dst_buffer.inner,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: dst_offset as _,
                    size: size as _,
                }],
            );

            Ok(())
        }
    }

    pub fn copy_buffer_to_image(&mut self, info: &BufferImageCopyInfo) -> VkResult<()> {
        self.command_buffer_state.borrow_mut().has_recorded_anything = true;
        let source = self.state.resolve_resource::<VkBuffer>(&info.source).inner;
        let image = self.state.resolve_resource::<VkImage>(&info.dest).inner;
        unsafe {
            self.state.logical_device.cmd_copy_buffer_to_image(
                self.command_buffer_state.borrow().inner_command_buffer,
                source,
                image,
                info.dest_layout.to_vk(),
                &[vk::BufferImageCopy {
                    buffer_offset: info.buffer_offset,
                    buffer_row_length: info.buffer_row_length,
                    buffer_image_height: info.buffer_image_height,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: ImageAspectFlags::COLOR.to_vk(),
                        mip_level: info.mip_level,
                        layer_count: info.num_layers,
                        base_array_layer: info.base_layer,
                    },
                    image_offset: info.image_offset.to_vk(),
                    image_extent: info.image_extent.to_vk(),
                }],
            );

            Ok(())
        }
    }

    pub fn push_constants(
        &mut self,
        index: u32,
        offset: u32,
        data: &[u8],
        shader_stage: ShaderStage,
    ) {
        // Ensure enough push constant range descriptions are allocated
        let mut state = self.command_buffer_state.borrow_mut();
        if state.descriptor_state.push_constant_range.len() <= (index as _) {
            state
                .descriptor_state
                .push_constant_range
                .resize(index as usize + 1, PushConstantRange::default());
            state.push_constant_data.resize(index as usize + 1, vec![]);
        }

        state.push_constant_data[index as usize] = data.to_vec();

        state.descriptor_state.push_constant_range[index as usize] = PushConstantRange {
            stage_flags: shader_stage,
            offset,
            size: std::mem::size_of_val(data) as _,
        }
    }

    pub fn bind_resources(&mut self, set: u32, bindings: &[Binding]) {
        let mut state = self.command_buffer_state.borrow_mut();
        if state.descriptor_state.sets.len() <= (set as _) {
            state
                .descriptor_state
                .sets
                .resize(set as usize + 1, DescriptorSetInfo2::default());
        }

        state.descriptor_state.sets[set as usize].bindings = bindings.to_vec();
    }

    pub fn pipeline_barrier(&mut self, barrier_info: &PipelineBarrierInfo) {
        self.command_buffer_state.borrow_mut().has_recorded_anything = true;
        let device = &self.state.logical_device;
        let memory_barriers: Vec<_> = barrier_info
            .memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        let buffer_memory_barriers: Vec<_> = barrier_info
            .buffer_memory_barriers
            .iter()
            .map(|b| vk::BufferMemoryBarrier {
                s_type: StructureType::BUFFER_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: b.src_access_mask.to_vk(),
                dst_access_mask: b.dst_access_mask.to_vk(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                buffer: self
                    .state
                    .allocated_resources
                    .read()
                    .unwrap()
                    .resolve::<VkBuffer>(&b.buffer)
                    .inner,
                offset: b.offset,
                size: b.size,
            })
            .collect();
        let image_memory_barriers: Vec<_> = barrier_info
            .image_memory_barriers
            .iter()
            .map(|b| vk::ImageMemoryBarrier {
                s_type: StructureType::IMAGE_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: b.src_access_mask.to_vk(),
                dst_access_mask: b.dst_access_mask.to_vk(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                old_layout: b.old_layout.to_vk(),
                new_layout: b.new_layout.to_vk(),
                image: self.state.resolve_resource::<VkImage>(&b.image).inner,
                subresource_range: b.subresource_range.to_vk(),
            })
            .collect();
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer_state.borrow().inner_command_buffer,
                barrier_info.src_stage_mask.to_vk(),
                barrier_info.dst_stage_mask.to_vk(),
                DependencyFlags::empty(),
                &memory_barriers,
                &buffer_memory_barriers,
                &image_memory_barriers,
            )
        };
    }

    pub fn submit(&mut self, submit_info: &CommandBufferSubmitInfo) -> VkResult<()> {
        let mut state = self.command_buffer_state.borrow_mut();
        state.has_been_submitted = true;
        if !state.has_recorded_anything {
            return Ok(());
        }

        let device = &self.state.logical_device;
        unsafe {
            device
                .end_command_buffer(state.inner_command_buffer)
                .expect("Failed to end inner command buffer");
            let target_queue = state.target_queue;

            let wait_semaphores: Vec<_> = submit_info
                .wait_semaphores
                .iter()
                .map(|s| self.state.resolve_resource::<VkSemaphore>(s).inner)
                .collect();

            let signal_semaphores: Vec<_> = submit_info
                .signal_semaphores
                .iter()
                .map(|s| self.state.resolve_resource::<VkSemaphore>(s).inner)
                .collect();

            let stage_masks: Vec<_> = submit_info.wait_stages.iter().map(|v| v.to_vk()).collect();

            device.queue_submit(
                target_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: wait_semaphores.len() as _,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    p_wait_dst_stage_mask: stage_masks.as_ptr(),
                    command_buffer_count: 1,
                    p_command_buffers: [state.inner_command_buffer].as_ptr(),
                    signal_semaphore_count: signal_semaphores.len() as _,
                    p_signal_semaphores: signal_semaphores.as_ptr(),
                }],
                if let Some(fence) = &submit_info.fence {
                    self.state.resolve_resource::<VkFence>(fence).inner
                } else {
                    vk::Fence::null()
                },
            )?;
            Ok(())
        }
    }

    pub fn vk_command_buffer(&self) -> vk::CommandBuffer {
        self.vk_command_buffer
    }
}

impl CommandBufferPassBegin for VkCommandBuffer {
    fn create_render_pass_impl(
        &mut self,
        info: &BeginRenderPassInfo,
    ) -> Box<dyn render_pass::Impl> {
        Box::new(self.begin_render_pass(info))
    }

    fn create_compute_pass_impl(&mut self) -> Box<dyn compute_pass::Impl> {
        Box::new(self.begin_compute_pass())
    }

    fn create_render_pass_2_impl(
        &mut self,
        info: &BeginRenderPassInfo2,
    ) -> Box<dyn render_pass_2::Impl> {
        Box::new(VkRenderPass2::new(self, &self.state, info))
    }
}

impl command_buffer_2::Impl for VkCommandBuffer {
    fn push_constants(&mut self, index: u32, offset: u32, data: &[u8], shader_stage: ShaderStage) {
        let mut state = self.command_buffer_state.borrow_mut();
        // Ensure enough push constant range descriptions are allocated
        if state.descriptor_state.push_constant_range.len() <= (index as _) {
            state
                .descriptor_state
                .push_constant_range
                .resize(index as usize + 1, PushConstantRange::default());
            state.push_constant_data.resize(index as usize + 1, vec![]);
        }

        state.push_constant_data[index as usize] = data.to_vec();

        state.descriptor_state.push_constant_range[index as usize] = PushConstantRange {
            stage_flags: shader_stage,
            offset,
            size: std::mem::size_of_val(data) as _,
        }
    }

    fn bind_resources(&mut self, set: u32, bindings: &[Binding]) {
        let mut state = self.command_buffer_state.borrow_mut();
        if state.descriptor_state.sets.len() <= (set as _) {
            state
                .descriptor_state
                .sets
                .resize(set as usize + 1, DescriptorSetInfo2::default());
        }

        state.descriptor_state.sets[set as usize].bindings = bindings.to_vec();
    }

    fn pipeline_barrier(&mut self, barrier_info: &PipelineBarrierInfo) {
        let mut state = self.command_buffer_state.borrow_mut();
        state.has_recorded_anything = true;
        let device = &self.state.logical_device;
        let memory_barriers: Vec<_> = barrier_info
            .memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        let buffer_memory_barriers: Vec<_> = barrier_info
            .buffer_memory_barriers
            .iter()
            .map(|b| vk::BufferMemoryBarrier {
                s_type: StructureType::BUFFER_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: b.src_access_mask.to_vk(),
                dst_access_mask: b.dst_access_mask.to_vk(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                buffer: self
                    .state
                    .allocated_resources
                    .read()
                    .unwrap()
                    .resolve::<VkBuffer>(&b.buffer)
                    .inner,
                offset: b.offset,
                size: b.size,
            })
            .collect();
        let image_memory_barriers: Vec<_> = barrier_info
            .image_memory_barriers
            .iter()
            .map(|b| vk::ImageMemoryBarrier {
                s_type: StructureType::IMAGE_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: b.src_access_mask.to_vk(),
                dst_access_mask: b.dst_access_mask.to_vk(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                old_layout: b.old_layout.to_vk(),
                new_layout: b.new_layout.to_vk(),
                image: self.state.resolve_resource::<VkImage>(&b.image).inner,
                subresource_range: b.subresource_range.to_vk(),
            })
            .collect();
        unsafe {
            device.cmd_pipeline_barrier(
                state.inner_command_buffer,
                barrier_info.src_stage_mask.to_vk(),
                barrier_info.dst_stage_mask.to_vk(),
                DependencyFlags::empty(),
                &memory_barriers,
                &buffer_memory_barriers,
                &image_memory_barriers,
            )
        };
    }

    fn submit(&mut self, submit_info: &CommandBufferSubmitInfo) -> anyhow::Result<()> {
        let mut state = self.command_buffer_state.borrow_mut();
        state.has_been_submitted = true;
        if !state.has_recorded_anything {
            return Ok(());
        }

        let device = &self.state.logical_device;
        unsafe {
            device
                .end_command_buffer(state.inner_command_buffer)
                .expect("Failed to end inner command buffer");
            let target_queue = state.target_queue;

            let wait_semaphores: Vec<_> = submit_info
                .wait_semaphores
                .iter()
                .map(|s| self.state.resolve_resource::<VkSemaphore>(s).inner)
                .collect();

            let signal_semaphores: Vec<_> = submit_info
                .signal_semaphores
                .iter()
                .map(|s| self.state.resolve_resource::<VkSemaphore>(s).inner)
                .collect();

            let stage_masks: Vec<_> = submit_info.wait_stages.iter().map(|v| v.to_vk()).collect();

            device.queue_submit(
                target_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: wait_semaphores.len() as _,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    p_wait_dst_stage_mask: stage_masks.as_ptr(),
                    command_buffer_count: 1,
                    p_command_buffers: [state.inner_command_buffer].as_ptr(),
                    signal_semaphore_count: signal_semaphores.len() as _,
                    p_signal_semaphores: signal_semaphores.as_ptr(),
                }],
                if let Some(fence) = &submit_info.fence {
                    self.state.resolve_resource::<VkFence>(fence).inner
                } else {
                    vk::Fence::null()
                },
            )?;
            Ok(())
        }
    }

    fn insert_debug_label(&self, label: &str, color: [f32; 4]) {
        if let Some(debug_utils) = &self.state.debug_utilities {
            unsafe {
                let c_label = CString::new(label).unwrap();
                debug_utils.cmd_insert_debug_utils_label(
                    self.command_buffer_state.borrow().inner_command_buffer,
                    &DebugUtilsLabelEXT {
                        s_type: StructureType::DEBUG_UTILS_LABEL_EXT,
                        p_next: std::ptr::null(),
                        p_label_name: c_label.as_ptr(),
                        color,
                    },
                );
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn bind_resources_2(&mut self, set:u32, bindings: &[Binding2],) {
        todo!()
    }
}
// Debug utilities

pub struct ScopedDebugLabelInner {
    debug_utils: DebugUtils,
    command_buffer: vk::CommandBuffer,
}

impl ScopedDebugLabelInner {
    fn new(
        label: &str,
        color: [f32; 4],
        debug_utils: DebugUtils,
        command_buffer: vk::CommandBuffer,
    ) -> Self {
        unsafe {
            let c_label = CString::new(label).unwrap();
            debug_utils.cmd_begin_debug_utils_label(
                command_buffer,
                &DebugUtilsLabelEXT {
                    s_type: StructureType::DEBUG_UTILS_LABEL_EXT,
                    p_next: std::ptr::null(),
                    p_label_name: c_label.as_ptr(),
                    color,
                },
            );
        }
        Self {
            debug_utils,
            command_buffer,
        }
    }
    fn end(&self) {
        unsafe {
            self.debug_utils
                .cmd_end_debug_utils_label(self.command_buffer);
        }
    }
}

pub struct ScopedDebugLabel {
    inner: Option<ScopedDebugLabelInner>,
}

impl ScopedDebugLabel {
    pub fn end(mut self) {
        if let Some(label) = self.inner.take() {
            label.end();
        }
    }
    fn end_from_render_pass(&mut self) {
        if let Some(label) = self.inner.take() {
            label.end();
        }
    }
}

impl Drop for ScopedDebugLabel {
    fn drop(&mut self) {
        if let Some(label) = self.inner.take() {
            label.end();
        }
    }
}

impl VkCommandBuffer {
    pub fn begin_debug_region(&self, label: &str, color: [f32; 4]) -> ScopedDebugLabel {
        ScopedDebugLabel {
            inner: self.state.debug_utilities.as_ref().map(|debug_utils| {
                ScopedDebugLabelInner::new(
                    label,
                    color,
                    debug_utils.clone(),
                    self.command_buffer_state.borrow().inner_command_buffer,
                )
            }),
        }
    }
}

impl Drop for VkCommandBuffer {
    fn drop(&mut self) {
        if !self.command_buffer_state.borrow().has_been_submitted {
            warn!("CommandBuffer::submit has not been called!")
        }
    }
}

impl VkRenderPassCommand {
    fn new(command_buffer: &mut VkCommandBuffer, render_pass_info: &BeginRenderPassInfo) -> Self {
        assert!(!render_pass_info.subpasses.is_empty());
        let render_pass = command_buffer.state.get_render_pass(
            &Self::get_attachments(&command_buffer.state, render_pass_info),
            render_pass_info.label,
        );

        let framebuffer = command_buffer
            .state
            .get_framebuffer(render_pass_info, render_pass);

        let mut clear_colors = render_pass_info
            .color_attachments
            .iter()
            .map(|at| vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: match at.load_op {
                        ColorLoadOp::Clear(c) => c,
                        _ => [0.0; 4],
                    },
                },
            })
            .collect::<Vec<_>>();
        if let Some(ref attch) = render_pass_info.depth_attachment {
            clear_colors.push(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: match attch.load_op {
                        DepthLoadOp::Clear(v) => v,
                        _ => 1.0,
                    },
                    stencil: 0,
                },
            });
        }
        let begin_render_pass_info = vk::RenderPassBeginInfo {
            s_type: StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: std::ptr::null(),
            render_pass,
            framebuffer,
            render_area: render_pass_info.render_area.to_vk(),
            clear_value_count: clear_colors.len() as _,
            p_clear_values: clear_colors.as_ptr(),
        };

        unsafe {
            command_buffer.state.logical_device.cmd_begin_render_pass(
                command_buffer
                    .command_buffer_state
                    .borrow()
                    .inner_command_buffer,
                &begin_render_pass_info,
                vk::SubpassContents::default(),
            );
        };

        let render_pass_label = command_buffer.begin_debug_region(
            render_pass_info.label.unwrap_or("Graphics render pass"),
            RENDER_PASSS_LABEL_COLOR,
        );

        let subpass_label = render_pass_info.subpasses[0]
            .label
            .as_ref()
            .map(|l| command_buffer.begin_debug_region(l, SUBPASS_LABEL_COLOR));

        let state = command_buffer.state.clone();
        Self {
            command_buffer: command_buffer
                .command_buffer_state
                .borrow()
                .inner_command_buffer,
            command_buffer_state: command_buffer.command_buffer_state.clone(),
            state,
            has_draw_command: false,
            viewport_area: None,
            depth_bias_setup: None,
            pipeline_state: GraphicsPipelineState::new(render_pass_info),
            render_pass,
            render_pass_label,
            subpass_label,
            subpasses: render_pass_info.subpasses.to_vec(),
        }
    }

    fn prepare_draw(&mut self) -> anyhow::Result<()> {
        #[cfg(debug_assertions)]
        self.validate_state()?;

        let mut state = self.command_buffer_state.borrow_mut();
        self.has_draw_command = true;
        state.has_recorded_anything = true;

        let layout = self.find_matching_pipeline_layout(&state.descriptor_state);
        let pipeline = self.find_matching_graphics_pipeline(layout, self.render_pass);
        let device = &self.state.logical_device;
        {
            unsafe {
                for (idx, constant_range) in state
                    .descriptor_state
                    .push_constant_range
                    .iter()
                    .enumerate()
                {
                    device.cmd_push_constants(
                        self.command_buffer,
                        layout,
                        constant_range.stage_flags.to_vk(),
                        constant_range.offset,
                        &state.push_constant_data[idx],
                    );
                }
                if !self.pipeline_state.vertex_inputs.is_empty() {
                    let buffers = self
                        .pipeline_state
                        .vertex_inputs
                        .iter()
                        .map(|b| {
                            self.state
                                .allocated_resources()
                                .resolve::<VkBuffer>(&b.handle)
                                .inner
                        })
                        .collect::<Vec<_>>();
                    let offsets = self
                        .pipeline_state
                        .vertex_inputs
                        .iter()
                        .map(|_| 0)
                        .collect::<Vec<_>>();
                    device.cmd_bind_vertex_buffers(self.command_buffer, 0, &buffers, &offsets);
                }
            }
        }

        if !state.descriptor_state.sets.is_empty() {
            let descriptors = self.find_matching_descriptor_sets(&state.descriptor_state);
            unsafe {
                device.cmd_bind_descriptor_sets(
                    self.command_buffer,
                    PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &descriptors,
                    &[],
                );
            }
        }

        unsafe {
            device.cmd_bind_pipeline(self.command_buffer, PipelineBindPoint::GRAPHICS, pipeline);
        }
        // Negate height because of Khronos brain farts
        let height = self.pipeline_state.render_area.extent.height as f32;
        let viewport = match self.viewport_area {
            Some(viewport) => viewport,
            None => Viewport {
                x: 0 as f32,
                y: height,
                width: self.pipeline_state.render_area.extent.width as f32,
                height: -height,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        };
        let scissor = match self.pipeline_state.scissor_area {
            Some(scissor) => scissor,
            None => Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: self.pipeline_state.render_area.extent,
            },
        };
        let (depth_constant, depth_clamp, depth_slope) = match self.depth_bias_setup {
            Some((depth_constant, depth_clamp, depth_slope)) => {
                (depth_constant, depth_clamp, depth_slope)
            }
            _ => (0.0, 0.0, 0.0),
        };
        unsafe {
            device.cmd_set_depth_bias_enable(self.command_buffer, true);
            device.cmd_set_depth_bias(
                self.command_buffer,
                depth_constant,
                depth_clamp,
                depth_slope,
            );
            device.cmd_set_viewport(self.command_buffer, 0, &[viewport.to_vk()]);
            device.cmd_set_scissor(self.command_buffer, 0, &[scissor.to_vk()]);
            device.cmd_set_depth_test_enable(
                self.command_buffer,
                self.pipeline_state.enable_depth_test,
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn validate_state(&self) -> anyhow::Result<()> {
        use anyhow::anyhow;
        macro_rules! validate {
            ($cond:expr, $err:expr) => {
                if !($cond) {
                    return Err(anyhow!($err));
                }
            };
        }
        
        let command_buffer_state = 
                self.command_buffer_state.borrow();

        validate!(
            !self.pipeline_state.early_discard_enabled || self.pipeline_state.fragment_shader.is_valid(),
            "Primitive early discard is enabled, but no valid fragment shader has been set"
        );
        validate!(
            {
                command_buffer_state.descriptor_state.sets.iter().all(|s| {
                    s.bindings.iter().all(|b| 
                    !matches!(b.ty, DescriptorBindingType::InputAttachment { .. }) || b.binding_stage == ShaderStage::FRAGMENT
                    )
                })
            },
            "If there's an InputAttachment in the bindings, it must only be visible to the Fragment stage"       
        );
        Ok(())
    }

    fn get_attachments(
        state: &GpuThreadSharedState,
        render_pass_info: &BeginRenderPassInfo<'_>,
    ) -> RenderPassAttachments {
         RenderPassAttachments {
            color_attachments: render_pass_info
            .color_attachments
            .iter()
            .map(|att| RenderPassAttachment {
                format: state
                    .resolve_resource::<VkImageView>(&att.image_view)
                    .format,
                samples: SampleCount::Sample1,
                load_op: att.load_op,
                store_op: att.store_op,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: att.initial_layout,
                final_layout: att.final_layout,
                // TODO: Add blend state to framebuffer attachments
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::OneMinusSrcAlpha,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::OneMinusSrcAlpha,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            })
            .collect(),
        depth_attachment :
                    render_pass_info
                        .depth_attachment
                        .as_ref()
                        .map(|att| RenderPassAttachment {
                            format: state
                                .resolve_resource::<VkImageView>(&att.image_view)
                                .format,
                            samples: SampleCount::Sample1,
                            load_op: match att.load_op {
                                DepthLoadOp::DontCare => ColorLoadOp::DontCare,
                                DepthLoadOp::Load => ColorLoadOp::Load,
                                DepthLoadOp::Clear(_) => ColorLoadOp::Clear([0.0; 4]),
                            },
                            store_op: att.store_op,
                            stencil_load_op: StencilLoadOp::DontCare,
                            stencil_store_op: AttachmentStoreOp::DontCare,
                            initial_layout: att.initial_layout,
                            final_layout: att.final_layout,
                            // TODO: Add blend state to framebuffer attachments
                            blend_state: BlendState {
                                blend_enable: true,
                                src_color_blend_factor: BlendMode::One,
                                dst_color_blend_factor: BlendMode::OneMinusSrcAlpha,
                                color_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendMode::One,
                                dst_alpha_blend_factor: BlendMode::OneMinusSrcAlpha,
                                alpha_blend_op: BlendOp::Add,
                                color_write_mask: ColorComponentFlags::RGBA,
                            },
                    }),
        subpasses : render_pass_info.subpasses.to_vec(),
        dependencies : render_pass_info.dependencies.to_vec(),
        ..Default::default()
        }

        }
    fn find_matching_graphics_pipeline(
        &self,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
    ) -> vk::Pipeline {
        self.state.get_graphics_pipeline(
            &self.pipeline_state,
            layout,
            render_pass,
            &self.subpasses[self.pipeline_state.current_subpass as usize],
        )
    }

    fn find_matching_pipeline_layout(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> vk::PipelineLayout {
        self.state.get_pipeline_layout(descriptor_state)
    }

    fn find_matching_descriptor_sets(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> Vec<vk::DescriptorSet> {
        self.state.get_descriptor_sets(descriptor_state)
    }

    pub fn begin_debug_region(&self, label: &str, color: [f32; 4]) -> ScopedDebugLabel {
        ScopedDebugLabel {
            inner: self.state.debug_utilities.as_ref().map(|debug_utils| {
                ScopedDebugLabelInner::new(label, color, debug_utils.clone(), self.command_buffer)
            }),
        }
    }
}

impl render_pass::Impl for VkRenderPassCommand {
    fn set_primitive_topology(&mut self, new_topology: PrimitiveTopology) {
        self.pipeline_state.primitive_topology = new_topology;
    }

    fn set_vertex_shader(&mut self, vertex_shader: ShaderModuleHandle) {
        self.pipeline_state.vertex_shader = vertex_shader;
    }

    fn set_fragment_shader(&mut self, fragment_shader: ShaderModuleHandle) {
        self.pipeline_state.fragment_shader = fragment_shader;
    }

    fn set_vertex_buffers(&mut self, bindings: &[VertexBindingInfo]) {
        self.pipeline_state.vertex_inputs = bindings.to_vec();
    }

    fn set_color_output_enabled(&mut self, color_output_enabled: bool) {
        self.pipeline_state.color_output_enabled = color_output_enabled;
    }

    fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport_area = Some(viewport);
    }

    fn set_depth_bias(&mut self, constant: f32, slope: f32) {
        self.depth_bias_setup = Some((constant, 0.0, slope));
    }

    fn set_front_face(&mut self, front_face: FrontFace) {
        self.pipeline_state.front_face = front_face;
    }

    fn set_polygon_mode(&mut self, polygon_mode: PolygonMode) {
        self.pipeline_state.polygon_mode = polygon_mode;
    }

    fn set_cull_mode(&mut self, cull_mode: CullMode) {
        self.pipeline_state.cull_mode = cull_mode;
    }

    fn set_enable_depth_test(&mut self, enable_depth_test: bool) {
        self.pipeline_state.enable_depth_test = enable_depth_test;
    }

    fn set_depth_write_enabled(&mut self, depth_write_enabled: bool) {
        self.pipeline_state.depth_write_enabled = depth_write_enabled;
    }

    fn set_depth_compare_op(&mut self, depth_compare_op: CompareOp) {
        self.pipeline_state.depth_compare_op = depth_compare_op;
    }

    fn advance_to_next_subpass(&mut self) {
        self.pipeline_state.current_subpass += 1;
        if self.pipeline_state.current_subpass as usize == self.subpasses.len() {
            return;
        }
        if self.pipeline_state.current_subpass as usize > self.subpasses.len() {
            panic!(
                "Tried to start subpass {} but there are {} subpasses!",
                self.pipeline_state.current_subpass,
                self.subpasses.len(),
            );
        }
        self.subpass_label.take();
        if let Some(ref label) = self.subpasses[self.pipeline_state.current_subpass as usize].label
        {
            self.subpass_label = Some(self.begin_debug_region(label, SUBPASS_LABEL_COLOR));
        }
        unsafe {
            self.state
                .logical_device
                .cmd_next_subpass(self.command_buffer, vk::SubpassContents::default());
        }
    }

    fn draw_indexed(
        &mut self,
        num_indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> anyhow::Result<()> {
        self.prepare_draw()?;
        unsafe {
            self.state.logical_device.cmd_draw_indexed(
                self.command_buffer,
                num_indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
        Ok(())
    }

    fn draw(
        &mut self,
        num_vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> anyhow::Result<()> {
        self.prepare_draw()?;
        unsafe {
            self.state.logical_device.cmd_draw(
                self.command_buffer,
                num_vertices,
                instances,
                first_vertex,
                first_instance,
            );
        }
        Ok(())
    }

    fn set_index_buffer(&self, index_buffer: BufferHandle, index_type: IndexType, offset: usize) {
        assert!(!index_buffer.is_null());
        let index_buffer = self
            .state
            .allocated_resources()
            .resolve::<VkBuffer>(&index_buffer);
        let device = &self.state.logical_device;
        unsafe {
            device.cmd_bind_index_buffer(
                self.command_buffer,
                index_buffer.inner,
                offset as _,
                index_type.to_vk(),
            );
        }
    }

    /* If enabled, fragments may be discarded after the vertex shader stage,
    before any fragment shader is executed.
    When enabled, a valid fragment shader must be set */
    fn set_early_discard_enabled(&mut self, allow_early_discard: bool) {
        self.pipeline_state.early_discard_enabled = allow_early_discard;
    }

    fn set_color_attachment_blend_state(&mut self, attachment:usize, blend_state:PipelineColorBlendAttachmentState,) {
        self.pipeline_state.color_blend_states[attachment] = blend_state
    }

    fn set_enable_depth_clamp(&mut self, enable_depth_clamp:bool) {
        self.pipeline_state.depth_clamp_enabled = enable_depth_clamp;
    }
}

impl Drop for VkRenderPassCommand {
    fn drop(&mut self) {
        self.subpass_label.take();
        self.render_pass_label.end_from_render_pass();
        unsafe {
            self.state
                .logical_device
                .cmd_end_render_pass(self.command_buffer);
        }
    }
}

impl VkComputePassCommand {
    fn new(command_buffer: &mut VkCommandBuffer) -> Self {
        Self {
            command_buffer: command_buffer
                .command_buffer_state
                .borrow()
                .inner_command_buffer,
            state: command_buffer.state.clone(),
            command_buffer_state: command_buffer.command_buffer_state.clone(),
            pipeline_state: ComputePipelineState::new(),
        }
    }

    fn find_matching_compute_pipeline(&mut self, layout: vk::PipelineLayout) -> vk::Pipeline {
        self.state
            .get_compute_pipeline(&self.pipeline_state, layout)
    }

    fn find_matching_pipeline_layout(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> vk::PipelineLayout {
        self.state.get_pipeline_layout(descriptor_state)
    }

    fn find_matching_descriptor_sets(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> Vec<vk::DescriptorSet> {
        self.state.get_descriptor_sets(descriptor_state)
    }
}

impl compute_pass::Impl for VkComputePassCommand {
    fn set_compute_shader(&mut self, compute_shader: ShaderModuleHandle) {
        self.pipeline_state.shader = compute_shader;
    }

    fn dispatch(&mut self, group_size_x: u32, group_size_y: u32, group_size_z: u32) {
        assert!(self.pipeline_state.shader.is_valid());
        let pipeline_layout = {
            let state = self.command_buffer_state.borrow();
            self.find_matching_pipeline_layout(&state.descriptor_state)
        };
        let pipeline = self.find_matching_compute_pipeline(pipeline_layout);
        let mut state = self.command_buffer_state.borrow_mut();
        let device = &self.state.logical_device;
        unsafe {
            device.cmd_bind_pipeline(self.command_buffer, PipelineBindPoint::COMPUTE, pipeline);

            let sets = self.find_matching_descriptor_sets(&state.descriptor_state);
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &sets,
                &[],
            );
            device.cmd_dispatch(
                self.command_buffer,
                group_size_x,
                group_size_y,
                group_size_z,
            );
        }
        state.has_recorded_anything = true;
    }
}

pub(crate) struct VkRenderPass2 {
    device: ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline_state: GraphicsPipelineState2,
    
}

impl VkRenderPass2 {
    pub(crate) fn new(command_buffer: &VkCommandBuffer, state: &GpuThreadSharedState, info: &BeginRenderPassInfo2) -> Self {
        let device = state.logical_device.clone();
        Self::insert_pipeline_transitions(command_buffer.vk_command_buffer, state, info);
        let color_attachments = info.color_attachments.iter()
            .map(|attachment| {
                RenderingAttachmentInfo::builder()
                    .image_view(state.resolve_resource::<VkImageView>(&attachment.image_view).inner)
                    .clear_value(match attachment.load_op {
                        ColorLoadOp::DontCare => ClearValue::default(),
                        ColorLoadOp::Load => ClearValue::default(),
                        ColorLoadOp::Clear(value) => ClearValue{ color: ClearColorValue {float32: value }},
                    })
                    .load_op(attachment.load_op.to_vk())
                    .store_op(attachment.store_op.to_vk())
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .build()
            }).collect::<Vec<_>>();
            let depth_attachment = if let Some(attach) = info.depth_attachment {
                RenderingAttachmentInfo::builder()
                    .image_view(state.resolve_resource::<VkImageView>(&attach.image_view).inner)
                    .image_layout(match attach.load_op {
                        DepthLoadOp::DontCare |
                        DepthLoadOp::Clear(_) => vk::ImageLayout::UNDEFINED,
                        DepthLoadOp::Load => match attach.store_op {
                            AttachmentStoreOp::DontCare => vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
                            AttachmentStoreOp::Store => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                        },
                    })
                    .clear_value(match attach.load_op {
                        DepthLoadOp::DontCare |
                        DepthLoadOp::Load => ClearValue::default(),
                        DepthLoadOp::Clear(value) => ClearValue {depth_stencil: ClearDepthStencilValue { depth: value, stencil: 0 }},
                    })
                    .load_op(attach.load_op.to_vk())
                    .store_op(attach.store_op.to_vk())
                    .build()
            } else {
                RenderingAttachmentInfo::default()
            };
            unsafe {
                state.dynamic_rendering.cmd_begin_rendering(command_buffer.vk_command_buffer, &RenderingInfo::builder()
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment)
                .render_area(info.render_area.to_vk())
                .build()
            );
        };
        Self {
                device,
                command_buffer: command_buffer.vk_command_buffer,
                pipeline_state: GraphicsPipelineState2::default(),
        }
    }

    fn insert_pipeline_transitions(command_buffer: vk::CommandBuffer, state: &GpuThreadSharedState, info: &BeginRenderPassInfo2<'_>) {
        let device = state.logical_device.clone();
        let mut image_layout_transitions = vec![];

        record_necessary_color_attachment_transitions(info, state, &mut image_layout_transitions);

        if let Some(attach) = info.depth_attachment {
            let vk_image_view = state.resolve_resource::<VkImageView>(&attach.image_view);
            let vk_image = state.resolve_resource::<VkImage>(&vk_image_view.owner_image);
            if vk_image.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                image_layout_transitions.push(ImageMemoryBarrier2 {
                    s_type: StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: std::ptr::null(),
                    src_stage_mask: vk_image.current_stage_mask,
                    src_access_mask: vk_image.current_access_mask,
                    dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    old_layout: vk_image.layout,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: vk_image.inner,
                    subresource_range: vk_image_view.whole_subresource(),
                })
            }
        }

        unsafe {
            device.cmd_pipeline_barrier2(command_buffer, &vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                p_next: std::ptr::null(),
                dependency_flags: vk::DependencyFlags::BY_REGION,
                memory_barrier_count: 0,
                p_memory_barriers: std::ptr::null(),
                buffer_memory_barrier_count: 0,
                p_buffer_memory_barriers: std::ptr::null(),
                image_memory_barrier_count: image_layout_transitions.len() as _,
                p_image_memory_barriers: image_layout_transitions.as_ptr(),
            });
        }
    }
}

fn record_necessary_color_attachment_transitions(info: &BeginRenderPassInfo2<'_>, state: &GpuThreadSharedState, image_layout_transitions: &mut Vec<ImageMemoryBarrier2>) {
    info.color_attachments.iter().for_each(|attach| {
        let vk_image_view = state.resolve_resource::<VkImageView>(&attach.image_view);
        let vk_image = state.resolve_resource::<VkImage>(&vk_image_view.owner_image);
        if vk_image.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            let barrier = state.record_image_transition(vk_image_view.owner_image, 
                vk_image_view.whole_subresource(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            );
            image_layout_transitions.push(barrier);
        }
    });
}

impl crate::render_pass_2::Impl for VkRenderPass2 {
    fn set_primitive_topology(&mut self, new_topology:PrimitiveTopology,) {
        todo!()
    }

    fn set_vertex_shader(&mut self, vertex_shader:ShaderModuleHandle,) {
        self.pipeline_state.vertex_shader = vertex_shader;
    }

    fn set_fragment_shader(&mut self, fragment_shader:ShaderModuleHandle,) {
        self.pipeline_state.fragment_shader = fragment_shader;
    }

    fn set_vertex_buffers(&mut self,bindings:&[VertexBindingInfo],) {
        todo!()
    }

    fn set_color_output_enabled(&mut self,color_output_enabled:bool,) {
        self.pipeline_state.color_output_enabled = color_output_enabled;
    }

    fn set_viewport(&mut self,viewport:Viewport,) {
        todo!()
    }

    fn set_depth_bias(&mut self,constant:f32,slope:f32,) {
        todo!()
    }

    fn set_front_face(&mut self,front_face:FrontFace,) {
        self.pipeline_state.front_face = front_face;
    }

    fn set_polygon_mode(&mut self,polygon_mode:PolygonMode,) {
        self.pipeline_state.polygon_mode = polygon_mode;
    }

    fn set_cull_mode(&mut self,cull_mode:CullMode,) {
        self.pipeline_state.cull_mode = cull_mode;
    }

    fn set_enable_depth_test(&mut self,enable_depth_test:bool,) {
        self.pipeline_state.enable_depth_test = enable_depth_test;
    }

    fn set_enable_depth_clamp(&mut self,enable_depth_clamp:bool,) {
        self.pipeline_state.depth_clamp_enabled = enable_depth_clamp;
    }

    fn set_depth_write_enabled(&mut self,depth_write_enabled:bool,) {
        self.pipeline_state.depth_write_enabled = depth_write_enabled;
    }

    fn set_depth_compare_op(&mut self,depth_compare_op:CompareOp,) {
        self.pipeline_state.depth_compare_op = depth_compare_op;
    }

    fn set_color_attachment_blend_state(&mut self,attachment:usize,blend_state:PipelineColorBlendAttachmentState,) {
        todo!()
    }

    fn set_early_discard_enabled(&mut self,allow_early_discard:bool,) {
        self.pipeline_state.early_discard_enabled = allow_early_discard;
    }

    fn set_index_buffer(&self,index_buffer:BufferHandle,index_type:IndexType,offset:usize,) {
        todo!()
    }

    fn draw_indexed(&mut self,num_indices:u32,instances:u32,first_index:u32,vertex_offset:i32,first_instance:u32,) -> anyhow::Result<()>  {
        todo!()
    }

    fn draw(&mut self,num_vertices:u32,instances:u32,first_vertex:u32,first_instance:u32,) -> anyhow::Result<()>  {
        todo!()
    }

}

impl Drop for VkRenderPass2 {
    fn drop(&mut self) {
        unsafe {
            self.device.cmd_end_rendering(self.command_buffer);
        }
    }
}