use std::cell::RefCell;
use std::ffi::CString;
use std::rc::Rc;

use anyhow::Context;
use ash::vk::{
    self, DebugUtilsLabelEXT, DependencyFlags, PipelineInputAssemblyStateCreateFlags, StructureType,
};
use ash::{extensions::ext::DebugUtils, prelude::VkResult};

use crate::vulkan::render_graph::{
    self, AttachmentMode, DrawCommand, DrawCommandType, DynamicState, GraphicsPassInfo,
    GraphicsSubpass, IndexBuffer, PushConstant, RenderGraphImage, ShaderInput, VertexBuffer,
};

use crate::vulkan::*;
use crate::*;

use self::gpu::{GpuError, GpuResult};

use super::gpu::GpuThreadSharedState;

struct VkCommandBufferState {
    inner_command_buffer: vk::CommandBuffer,
    has_recorded_anything: bool,
    descriptor_state: DescriptorSetState,
    push_constant_data: Vec<Vec<u8>>,
}

impl VkCommandBufferState {}

pub struct VkCommandBuffer {
    state: Arc<GpuThreadSharedState>,
    command_buffer_state: Rc<RefCell<VkCommandBufferState>>,
    vk_command_buffer: vk::CommandBuffer,
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
#[derive(Hash)]
pub(crate) struct GraphicsPipelineTraditional {
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
    pub(crate) index_buffer: Option<IndexBuffer>,
    pub(crate) primitive_topology: PrimitiveTopology,
    pub(crate) polygon_mode: PolygonMode,
    pub(crate) color_output_enabled: bool,
    pub(crate) color_attachments: Vec<crate::ColorAttachment>,
    pub(crate) depth_format: vk::Format,
    pub(crate) blend_states: Vec<PipelineColorBlendAttachmentState>,
}

impl GraphicsPipelineState2 {
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

// Reason: in the near future the render graph will use traditional render passes
#[allow(dead_code)]
impl GraphicsPipelineTraditional {
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
                BindingType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
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
        state: Arc<GpuThreadSharedState>,
        command_buffer: vk::CommandBuffer,
    ) -> VkResult<Self> {
        Ok(Self {
            state,
            vk_command_buffer: command_buffer,
            command_buffer_state: Rc::new(RefCell::new(VkCommandBufferState {
                inner_command_buffer: command_buffer,
                has_recorded_anything: false,
                descriptor_state: DescriptorSetState::default(),
                push_constant_data: vec![],
            })),
        })
    }

    pub fn begin_compute_pass(&mut self, info: &BeginComputePassInfo) -> VkComputePassCommand {
        VkComputePassCommand::new(self, info)
    }

    pub fn pipeline_barrier(&mut self, barrier_info: &PipelineBarrierInfo) -> anyhow::Result<()> {
        self.command_buffer_state.borrow_mut().has_recorded_anything = true;
        let device = &self.state.logical_device;
        let memory_barriers: Vec<_> = barrier_info
            .memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        let vk_buffers: Vec<_> = barrier_info
            .buffer_memory_barriers
            .iter()
            .map(|b| self.state.resolve_resource::<VkBuffer>(&b.buffer))
            .collect::<GpuResult<Vec<_>>>()?;

        let vk_images: Vec<_> = barrier_info
            .image_memory_barriers
            .iter()
            .map(|b| self.state.resolve_resource::<VkImage>(&b.image))
            .collect::<GpuResult<Vec<_>>>()?;

        let buffer_memory_barriers = barrier_info
            .buffer_memory_barriers
            .iter()
            .zip(vk_buffers)
            .map(|(b, buffer)| vk::BufferMemoryBarrier {
                s_type: StructureType::BUFFER_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: b.src_access_mask.to_vk(),
                dst_access_mask: b.dst_access_mask.to_vk(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                buffer: buffer.inner,
                offset: b.offset,
                size: b.size,
            })
            .collect::<Vec<_>>();
        let image_memory_barriers: Vec<_> = barrier_info
            .image_memory_barriers
            .iter()
            .zip(vk_images)
            .map(|(b, image)| vk::ImageMemoryBarrier {
                s_type: StructureType::IMAGE_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: b.src_access_mask.to_vk(),
                dst_access_mask: b.dst_access_mask.to_vk(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                old_layout: b.old_layout.to_vk(),
                new_layout: b.new_layout.to_vk(),
                image: image.inner,
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
        Ok(())
    }

    pub fn vk_command_buffer(&self) -> vk::CommandBuffer {
        self.vk_command_buffer
    }
}

impl CommandBufferPassBegin for VkCommandBuffer {
    fn create_compute_pass_impl(
        &mut self,
        info: &BeginComputePassInfo,
    ) -> anyhow::Result<Box<dyn compute_pass::Impl>> {
        Ok(Box::new(self.begin_compute_pass(info)))
    }

    fn create_render_pass_2_impl(
        &mut self,
        info: &BeginRenderPassInfo2,
    ) -> anyhow::Result<Box<dyn render_pass_2::Impl>> {
        Ok(Box::new(VkRenderPass2::new(
            self,
            self.state.clone(),
            info,
        )?))
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

    fn bind_resources(&mut self, set: u32, bindings: &[Binding]) -> anyhow::Result<()> {
        let mut state = self.command_buffer_state.borrow_mut();
        if state.descriptor_state.sets.len() <= (set as _) {
            state
                .descriptor_state
                .sets
                .resize(set as usize + 1, DescriptorSetInfo2::default());
        }

        state.descriptor_state.sets[set as usize].bindings = bindings.to_vec();
        Ok(())
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
}

// Debug utilities

pub struct ScopedDebugLabelInner {
    debug_utils: DebugUtils,
    command_buffer: vk::CommandBuffer,
}

impl ScopedDebugLabelInner {
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
impl Drop for ScopedDebugLabel {
    fn drop(&mut self) {
        if let Some(label) = self.inner.take() {
            label.end();
        }
    }
}

pub struct VkComputePassCommand {
    pipeline_state: ComputePipelineState,
    command_buffer_state: Rc<RefCell<VkCommandBufferState>>,
    state: Arc<GpuThreadSharedState>,
    pass_info: ComputePassInfo,
}

impl VkComputePassCommand {
    fn new(command_buffer: &mut VkCommandBuffer, info: &BeginComputePassInfo) -> Self {
        Self {
            state: command_buffer.state.clone(),
            command_buffer_state: command_buffer.command_buffer_state.clone(),
            pipeline_state: ComputePipelineState::new(),
            pass_info: ComputePassInfo {
                label: info.label.map(|s| s.to_owned()),
                flags: info.flags.into(),
                ..Default::default()
            },
        }
    }

    fn find_matching_compute_pipeline(
        &mut self,
        layout: vk::PipelineLayout,
    ) -> anyhow::Result<vk::Pipeline> {
        self.state
            .get_compute_pipeline(&self.pipeline_state, layout)
    }

    fn find_matching_pipeline_layout(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> anyhow::Result<vk::PipelineLayout> {
        self.state.get_pipeline_layout(descriptor_state)
    }

    fn find_matching_descriptor_sets(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
        self.state.get_descriptor_sets(descriptor_state)
    }
}

impl Drop for VkComputePassCommand {
    fn drop(&mut self) {
        self.state
            .render_graph
            .write()
            .expect("Failed to lock render graph")
            .add_compute(std::mem::take(&mut self.pass_info));
    }
}

impl compute_pass::Impl for VkComputePassCommand {
    fn set_compute_shader(&mut self, compute_shader: ShaderModuleHandle) {
        self.pipeline_state.shader = compute_shader;
    }

    fn bind_resources_2(&mut self, set: usize, resources: &[Binding2]) -> anyhow::Result<()> {
        let bindings = resources
            .iter()
            .enumerate()
            .filter(|(_i, r)| !r.write)
            .map(|(i, r)| {
                Ok(match r.ty {
                    DescriptorBindingType2::UniformBuffer { handle, .. }
                    | DescriptorBindingType2::StorageBuffer { handle, .. } => ShaderInput {
                        resource: Resource::Buffer(
                            self.state
                                .resolve_resource::<VkBuffer>(&handle)
                                .context(format!("Binding location {i}"))?
                                .inner,
                        ),
                    },
                    DescriptorBindingType2::ImageView {
                        image_view_handle, ..
                    } => {
                        let view = self
                            .state
                            .resolve_resource::<VkImageView>(&image_view_handle)
                            .context(format!("Binding location {i}"))?;
                        let handle = view.owner_image;
                        let image = self
                            .state
                            .resolve_resource::<VkImage>(&handle)
                            .context("Binding location {i}")?;

                        ShaderInput {
                            resource: Resource::Image(
                                image.inner,
                                view.inner,
                                image.format.aspect_mask().to_vk(),
                            ),
                        }
                    }
                    DescriptorBindingType2::StorageImage { handle } => {
                        let view = self
                            .state
                            .resolve_resource::<VkImageView>(&handle)
                            .context(format!("Binding location {i}"))?;
                        let handle = view.owner_image;
                        let image = self.state.resolve_resource::<VkImage>(&handle)?;

                        ShaderInput {
                            resource: Resource::Image(
                                image.inner,
                                view.inner,
                                image.format.aspect_mask().to_vk(),
                            ),
                        }
                    }
                })
            })
            .collect::<GpuResult<Vec<_>>>()?;
        let written = resources
            .iter()
            .enumerate()
            .filter(|(_, r)| r.write)
            .map(|(i, r)| {
                Ok(match r.ty {
                    DescriptorBindingType2::UniformBuffer { handle, .. }
                    | DescriptorBindingType2::StorageBuffer { handle, .. } => ShaderInput {
                        resource: Resource::Buffer(
                            self.state
                                .resolve_resource::<VkBuffer>(&handle)
                                .context(format!("Binding location {i}"))?
                                .inner,
                        ),
                    },
                    DescriptorBindingType2::ImageView {
                        image_view_handle, ..
                    } => {
                        let view = self
                            .state
                            .resolve_resource::<VkImageView>(&image_view_handle)
                            .context(format!("Binding location {i}"))?;
                        let handle = view.owner_image;
                        let image = self
                            .state
                            .resolve_resource::<VkImage>(&handle)
                            .context("Binding location {i}")?;

                        ShaderInput {
                            resource: Resource::Image(
                                image.inner,
                                view.inner,
                                image.format.aspect_mask().to_vk(),
                            ),
                        }
                    }
                    DescriptorBindingType2::StorageImage { handle } => {
                        let view = self
                            .state
                            .resolve_resource::<VkImageView>(&handle)
                            .context(format!("Binding location {i}"))?;
                        let handle = view.owner_image;
                        let image = self.state.resolve_resource::<VkImage>(&handle)?;

                        ShaderInput {
                            resource: Resource::Image(
                                image.inner,
                                view.inner,
                                image.format.aspect_mask().to_vk(),
                            ),
                        }
                    }
                })
            })
            .collect::<GpuResult<Vec<_>>>()?;
        self.pass_info
            .all_read_resources
            .extend(bindings.into_iter());
        self.pass_info
            .all_written_resources
            .extend(written.into_iter());
        let mut state = self.command_buffer_state.borrow_mut();
        if state.descriptor_state.sets.len() <= (set as _) {
            state
                .descriptor_state
                .sets
                .resize(set + 1, DescriptorSetInfo2::default());
        }

        state.descriptor_state.sets[set].bindings = resources
            .iter()
            .enumerate()
            .map(|(i, b)| Binding {
                ty: match b.ty {
                    DescriptorBindingType2::UniformBuffer {
                        handle,
                        offset,
                        range,
                    } => DescriptorBindingType::UniformBuffer {
                        handle,
                        offset,
                        range: range as usize,
                    },
                    DescriptorBindingType2::StorageBuffer {
                        handle,
                        offset,
                        range,
                    } => DescriptorBindingType::StorageBuffer {
                        handle,
                        offset,
                        range: range as usize,
                    },
                    DescriptorBindingType2::ImageView {
                        image_view_handle,
                        sampler_handle,
                    } => DescriptorBindingType::ImageView {
                        image_view_handle,
                        sampler_handle,
                        layout: ImageLayout::ShaderReadOnly,
                    },

                    DescriptorBindingType2::StorageImage { handle } => {
                        DescriptorBindingType::StorageImage { handle }
                    }
                },
                binding_stage: ShaderStage::COMPUTE,
                location: i as _,
            })
            .collect();
        Ok(())
    }

    fn dispatch(
        &mut self,
        group_size_x: u32,
        group_size_y: u32,
        group_size_z: u32,
    ) -> anyhow::Result<()> {
        let pipeline_layout = {
            let state = self.command_buffer_state.borrow();
            self.find_matching_pipeline_layout(&state.descriptor_state)
        }?;
        let pipeline = self.find_matching_compute_pipeline(pipeline_layout)?;
        let mut state = self.command_buffer_state.borrow_mut();

        let sets = self.find_matching_descriptor_sets(&state.descriptor_state)?;
        self.pass_info.dispatches.push(DispatchCommand {
            pipeline,
            pipeline_layout,
            descriptor_sets: sets,
            push_constants: vec![],
            group_count: [group_size_x, group_size_y, group_size_z],
        });

        state.has_recorded_anything = true;
        Ok(())
    }
}

pub(crate) struct VkRenderPass2 {
    pipeline_state: GraphicsPipelineState2,
    gpu_state: Arc<GpuThreadSharedState>,
    command_buffer_state: Rc<RefCell<VkCommandBufferState>>,
    viewport_area: Viewport,
    depth_bias_config: Option<(f32, f32, f32)>,

    graphics_pass_info: GraphicsPassInfo,

    current_subpass: usize,
    vertex_offsets: Vec<u64>,
}

impl VkRenderPass2 {
    pub(crate) fn new(
        command_buffer: &VkCommandBuffer,
        state: Arc<GpuThreadSharedState>,
        info: &BeginRenderPassInfo2,
    ) -> anyhow::Result<Self> {
        let _device = state.logical_device.clone();

        let color_attachments = info
            .color_attachments
            .iter()
            .enumerate()
            .map(|(index, attachment)| render_graph::SubpassColorAttachment {
                index,
                attachment_mode: match attachment.store_op {
                    AttachmentStoreOp::DontCare => AttachmentMode::AttachmentRead,
                    AttachmentStoreOp::Store => AttachmentMode::AttachmentWrite,
                },
            })
            .collect::<Vec<_>>();
        let depth_attachment = info
            .depth_attachment
            .map(|_| AttachmentMode::AttachmentRead);
        let stencil_attachment = info
            .stencil_attachment
            .map(|_| AttachmentMode::AttachmentRead);

        Ok(Self {
            vertex_offsets: vec![],
            pipeline_state: GraphicsPipelineState2 {
                color_attachments: info.color_attachments.to_vec(),
                depth_format: if info.depth_attachment.is_some() {
                    ImageFormat::Depth.to_vk()
                } else {
                    vk::Format::UNDEFINED
                },
                ..Default::default()
            },
            gpu_state: state.clone(),
            command_buffer_state: command_buffer.command_buffer_state.clone(),
            viewport_area: Viewport {
                x: info.render_area.offset.x as f32,
                y: info.render_area.offset.y as f32,
                width: info.render_area.extent.width as f32,
                height: info.render_area.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            depth_bias_config: Default::default(),
            graphics_pass_info: GraphicsPassInfo {
                label: info.label.map(|s| s.to_owned()),
                all_color_attachments: info
                    .color_attachments
                    .iter()
                    .map(|att| {
                        let view = state.resolve_resource::<VkImageView>(&att.image_view)?;
                        Ok(render_graph::ColorAttachment {
                            render_image: {
                                RenderGraphImage {
                                    view: view.inner,
                                    image: state
                                        .resolve_resource::<VkImage>(&view.owner_image)?
                                        .inner,
                                }
                            },
                            load_op: att.load_op,
                            store_op: att.store_op,
                            flags: view.flags.into(),
                        })
                    })
                    .collect::<GpuResult<Vec<_>>>()?,
                depth_attachment: info
                    .depth_attachment
                    .map(|att| {
                        GpuResult::Ok(render_graph::DepthAttachment {
                            render_image: {
                                let view =
                                    state.resolve_resource::<VkImageView>(&att.image_view)?;
                                RenderGraphImage {
                                    view: view.inner,
                                    image: state
                                        .resolve_resource::<VkImage>(&view.owner_image)?
                                        .inner,
                                }
                            },
                            load_op: att.load_op,
                            store_op: att.store_op,
                        })
                    })
                    .transpose()?,
                stencil_attachment: info
                    .stencil_attachment
                    .map(|att| {
                        GpuResult::Ok(render_graph::StencilAttachment {
                            render_image: {
                                let view =
                                    state.resolve_resource::<VkImageView>(&att.image_view)?;
                                RenderGraphImage {
                                    view: view.inner,
                                    image: state
                                        .resolve_resource::<VkImage>(&view.owner_image)?
                                        .inner,
                                }
                            },
                            load_op: att.load_op,
                            store_op: att.store_op,
                        })
                    })
                    .transpose()?,
                sub_passes: vec![GraphicsSubpass {
                    color_attachments,
                    depth_attachment,
                    stencil_attachment,
                    render_area: info.render_area.to_vk(),

                    draw_commands: vec![],
                    shader_reads: vec![],
                    shader_writes: vec![],
                }],
            },

            current_subpass: 0,
        })
    }

    fn validate_bindings(&self, bindings: &[Binding2]) -> anyhow::Result<()> {
        for (i, binding) in bindings.iter().enumerate() {
            match binding.ty {
                DescriptorBindingType2::UniformBuffer { handle, .. } => {
                    if handle.is_null() {
                        return Err(GpuError::NullHandle(HandleType::Buffer))
                            .with_context(|| format!("binding {i}"));
                    }
                }
                DescriptorBindingType2::StorageBuffer { handle, .. } => {
                    if handle.is_null() {
                        return Err(GpuError::NullHandle(HandleType::Buffer))
                            .with_context(|| format!("binding {i}"));
                    }
                }
                DescriptorBindingType2::StorageImage { handle } => {
                    if handle.is_null() {
                        return Err(GpuError::NullHandle(HandleType::ImageView))
                            .with_context(|| format!("binding {i}"));
                    }
                }
                DescriptorBindingType2::ImageView {
                    image_view_handle,
                    sampler_handle,
                } => {
                    if image_view_handle.is_null() {
                        return Err(GpuError::NullHandle(HandleType::ImageView))
                            .with_context(|| format!("binding {i}"));
                    }

                    if sampler_handle.is_null() {
                        return Err(GpuError::NullHandle(HandleType::Sampler))
                            .with_context(|| format!("binding {i}"));
                    }
                }
            }
        }
        Ok(())
    }
}

impl crate::render_pass_2::Impl for VkRenderPass2 {
    fn set_primitive_topology(&mut self, new_topology: PrimitiveTopology) {
        self.pipeline_state.primitive_topology = new_topology;
    }

    fn set_vertex_shader(&mut self, vertex_shader: ShaderModuleHandle) {
        self.pipeline_state.vertex_shader = vertex_shader;
    }

    fn set_fragment_shader(&mut self, fragment_shader: ShaderModuleHandle) {
        self.pipeline_state.fragment_shader = fragment_shader;
    }

    fn set_vertex_buffers(&mut self, bindings: &[VertexBindingInfo], offsets: &[u64]) {
        assert!(bindings.len() == offsets.len());
        self.pipeline_state.vertex_inputs = bindings.to_vec();
        self.vertex_offsets = offsets.to_vec();
    }

    fn set_color_output_enabled(&mut self, color_output_enabled: bool) {
        self.pipeline_state.color_output_enabled = color_output_enabled;
    }

    fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport_area = viewport;
    }

    fn set_depth_bias(&mut self, constant: f32, slope: f32) {
        self.depth_bias_config = Some((constant, 0.0, slope));
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

    fn set_enable_depth_clamp(&mut self, enable_depth_clamp: bool) {
        self.pipeline_state.depth_clamp_enabled = enable_depth_clamp;
    }

    fn set_depth_write_enabled(&mut self, depth_write_enabled: bool) {
        self.pipeline_state.depth_write_enabled = depth_write_enabled;
    }

    fn set_depth_compare_op(&mut self, depth_compare_op: CompareOp) {
        self.pipeline_state.depth_compare_op = depth_compare_op;
    }

    fn set_color_attachment_blend_state(
        &mut self,
        attachment: usize,
        blend_state: PipelineColorBlendAttachmentState,
    ) {
        if self.pipeline_state.blend_states.len() <= attachment {
            self.pipeline_state
                .blend_states
                .resize(attachment + 1, PipelineColorBlendAttachmentState::default());
        }

        self.pipeline_state.blend_states[attachment] = blend_state;
    }

    fn set_early_discard_enabled(&mut self, allow_early_discard: bool) {
        self.pipeline_state.early_discard_enabled = allow_early_discard;
    }

    fn set_index_buffer(
        &mut self,
        index_buffer: BufferHandle,
        index_type: IndexType,
        offset: usize,
    ) {
        if index_buffer.is_null() {
            self.pipeline_state.index_buffer = None;
        } else {
            let index_buffer = self
                .gpu_state
                .resolve_resource::<VkBuffer>(&index_buffer)
                .unwrap();
            self.pipeline_state.index_buffer = Some(IndexBuffer {
                buffer: index_buffer.inner,
                index_type: index_type.to_vk(),
                offset: offset as _,
            })
        }
    }
    fn bind_resources_2(&mut self, set: u32, bindings: &[Binding2]) -> anyhow::Result<()> {
        self.validate_bindings(bindings)
            .with_context(|| format!("In descriptor set {set}"))?;
        let mut state = self.command_buffer_state.borrow_mut();
        if state.descriptor_state.sets.len() <= (set as _) {
            state
                .descriptor_state
                .sets
                .resize(set as usize + 1, DescriptorSetInfo2::default());
        }

        let current_subpass = &mut self.graphics_pass_info.sub_passes[self.current_subpass];
        for binding in state.descriptor_state.sets[set as usize].bindings.iter() {
            if let DescriptorBindingType::ImageView {
                image_view_handle, ..
            } = binding.ty
            {
                let image_view = self
                    .gpu_state
                    .resolve_resource::<VkImageView>(&image_view_handle)?;
                let image = self
                    .gpu_state
                    .resolve_resource::<VkImage>(&image_view.owner_image)?;
                current_subpass.shader_reads.retain(|im| match im.resource {
                    Resource::Image(im, ..) => im != image.inner,
                    _ => true,
                });
                current_subpass
                    .shader_writes
                    .retain(|im| match im.resource {
                        Resource::Image(im, ..) => im != image.inner,
                        _ => true,
                    });
            }
        }

        for binding in bindings.iter() {
            if let DescriptorBindingType2::ImageView {
                image_view_handle, ..
            } = binding.ty
            {
                let read = true; // TODO
                let image_view = self
                    .gpu_state
                    .resolve_resource::<VkImageView>(&image_view_handle)?;
                let image = self
                    .gpu_state
                    .resolve_resource::<VkImage>(&image_view.owner_image)?;
                if read {
                    current_subpass.shader_reads.push(ShaderInput {
                        resource: Resource::Image(
                            image.inner,
                            image_view.inner,
                            image_view.format.aspect_mask().to_vk(),
                        ),
                    });
                } else {
                    current_subpass.shader_writes.push(ShaderInput {
                        resource: Resource::Image(
                            image.inner,
                            image_view.inner,
                            image_view.format.aspect_mask().to_vk(),
                        ),
                    });
                }
            }
        }

        state.descriptor_state.sets[set as usize].bindings = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| Binding {
                ty: match b.ty {
                    DescriptorBindingType2::UniformBuffer {
                        handle,
                        offset,
                        range,
                    } => DescriptorBindingType::UniformBuffer {
                        handle,
                        offset,
                        range: range as usize,
                    },
                    DescriptorBindingType2::StorageBuffer {
                        handle,
                        offset,
                        range,
                    } => DescriptorBindingType::StorageBuffer {
                        handle,
                        offset,
                        range: range as usize,
                    },
                    DescriptorBindingType2::ImageView {
                        image_view_handle,
                        sampler_handle,
                    } => DescriptorBindingType::ImageView {
                        image_view_handle,
                        sampler_handle,
                        layout: ImageLayout::ShaderReadOnly,
                    },
                    DescriptorBindingType2::StorageImage { handle } => {
                        DescriptorBindingType::StorageImage { handle }
                    }
                },
                binding_stage: ShaderStage::ALL_GRAPHICS,
                location: i as _,
            })
            .collect();
        Ok(())
    }
    fn draw_indexed(
        &mut self,
        num_indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> anyhow::Result<()> {
        let state = self.command_buffer_state.borrow_mut();

        let pipeline_layout = self.find_matching_pipeline_layout(&state.descriptor_state)?;
        let pipeline = self.find_matching_graphics_pipeline(pipeline_layout)?;
        let descriptors = self.find_matching_descriptor_sets(&state.descriptor_state)?;
        let push_constants = state
            .descriptor_state
            .push_constant_range
            .iter()
            .enumerate()
            .map(|(i, range)| PushConstant {
                data: state.push_constant_data[i].clone(),
                stage_flags: range.stage_flags.to_vk(),
            })
            .collect();

        let viewport = flip_viewport(self.viewport_area.to_vk());

        let dynamic_state = DynamicState {
            viewport,
            scissor_rect: self
                .pipeline_state
                .scissor_area
                .unwrap_or(Rect2D {
                    offset: Offset2D::make(
                        self.viewport_area.x as i32,
                        self.viewport_area.y as i32,
                    ),
                    extent: Extent2D {
                        width: self.viewport_area.width as u32,
                        height: self.viewport_area.height as u32,
                    },
                })
                .to_vk(),
            depth_bias: self.depth_bias_config,
            depth_test_enable: self.pipeline_state.enable_depth_test,
        };

        if self.pipeline_state.depth_write_enabled {
            if let Some(da) =
                &mut self.graphics_pass_info.sub_passes[self.current_subpass].depth_attachment
            {
                *da = AttachmentMode::AttachmentWrite;
            }
        }

        let draw_comand = DrawCommand {
            pipeline,
            pipeline_layout,
            descriptor_sets: descriptors,
            push_constants,
            dynamic_state,
            vertex_buffers: self
                .pipeline_state
                .vertex_inputs
                .iter()
                .zip(self.vertex_offsets.iter())
                .map(|(buffer, &offset)| {
                    let buffer = self
                        .gpu_state
                        .resolve_resource::<VkBuffer>(&buffer.handle)?
                        .inner;
                    Ok(VertexBuffer { buffer, offset })
                })
                .into_iter()
                .collect::<GpuResult<Vec<_>>>()?,
            index_buffer: self.pipeline_state.index_buffer,
            command_type: DrawCommandType::DrawIndexed(
                num_indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            ),
        };

        self.graphics_pass_info.sub_passes[self.current_subpass]
            .draw_commands
            .push(draw_comand);

        Ok(())
    }

    fn draw(
        &mut self,
        num_vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> anyhow::Result<()> {
        let state = self.command_buffer_state.borrow_mut();

        let pipeline_layout = self.find_matching_pipeline_layout(&state.descriptor_state)?;
        let pipeline = self.find_matching_graphics_pipeline(pipeline_layout)?;
        let descriptors = self.find_matching_descriptor_sets(&state.descriptor_state)?;
        let push_constants = state
            .descriptor_state
            .push_constant_range
            .iter()
            .enumerate()
            .map(|(i, range)| PushConstant {
                data: state.push_constant_data[i].clone(),
                stage_flags: range.stage_flags.to_vk(),
            })
            .collect();

        let viewport = flip_viewport(self.viewport_area.to_vk());
        let dynamic_state = DynamicState {
            viewport,
            scissor_rect: self
                .pipeline_state
                .scissor_area
                .unwrap_or(Rect2D {
                    offset: Offset2D::make(
                        self.viewport_area.x as i32,
                        self.viewport_area.y as i32,
                    ),
                    extent: Extent2D {
                        width: self.viewport_area.width as u32,
                        height: self.viewport_area.height as u32,
                    },
                })
                .to_vk(),
            depth_bias: self.depth_bias_config,
            depth_test_enable: self.pipeline_state.enable_depth_test,
        };

        if self.pipeline_state.depth_write_enabled {
            if let Some(da) =
                &mut self.graphics_pass_info.sub_passes[self.current_subpass].depth_attachment
            {
                *da = AttachmentMode::AttachmentWrite;
            }
        }

        self.graphics_pass_info.sub_passes[self.current_subpass]
            .draw_commands
            .push(DrawCommand {
                pipeline,
                pipeline_layout,
                descriptor_sets: descriptors,
                push_constants,
                dynamic_state,
                vertex_buffers: self
                    .pipeline_state
                    .vertex_inputs
                    .iter()
                    .zip(self.vertex_offsets.iter())
                    .map(|(buffer, &offset)| {
                        let buffer = self
                            .gpu_state
                            .resolve_resource::<VkBuffer>(&buffer.handle)?
                            .inner;
                        Ok(VertexBuffer { buffer, offset })
                    })
                    .into_iter()
                    .collect::<GpuResult<Vec<_>>>()?,
                index_buffer: self.pipeline_state.index_buffer,
                command_type: DrawCommandType::Draw(
                    num_vertices,
                    instances,
                    first_vertex,
                    first_instance,
                ),
            });

        Ok(())
    }

    fn set_scissor_rect(&mut self, scissor_rect: Rect2D) {
        self.pipeline_state.scissor_area = Some(scissor_rect);
    }
}

fn flip_viewport(viewport: vk::Viewport) -> vk::Viewport {
    let height = viewport.height;
    vk::Viewport {
        y: height - viewport.y,
        height: -height,
        ..viewport
    }
}

impl VkRenderPass2 {
    fn find_matching_pipeline_layout(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> anyhow::Result<vk::PipelineLayout> {
        self.gpu_state.get_pipeline_layout(descriptor_state)
    }

    fn find_matching_descriptor_sets(
        &self,
        descriptor_state: &DescriptorSetState,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
        self.gpu_state.get_descriptor_sets(descriptor_state)
    }

    fn find_matching_graphics_pipeline(
        &self,
        layout: vk::PipelineLayout,
    ) -> anyhow::Result<vk::Pipeline> {
        self.gpu_state
            .get_graphics_pipeline_dynamic(&self.pipeline_state, layout)
    }
}

impl Drop for VkRenderPass2 {
    fn drop(&mut self) {
        self.gpu_state
            .render_graph
            .write()
            .expect("Failed to lock render graph")
            .add_graphics(std::mem::take(&mut self.graphics_pass_info));
    }
}
