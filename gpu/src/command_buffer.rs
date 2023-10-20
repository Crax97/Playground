use core::panic;
use std::{ffi::CString, ops::Deref};

use ash::vk::{
    self, ClearDepthStencilValue, CommandBufferAllocateInfo, CommandBufferBeginInfo,
    CommandBufferLevel, CommandBufferUsageFlags, DebugUtilsLabelEXT, DependencyFlags,
    PipelineBindPoint, PipelineInputAssemblyStateCreateFlags, RenderingAttachmentInfoKHR,
    RenderingFlags, RenderingInfoKHR, ResolveModeFlags, StructureType, SubmitInfo,
};
use ash::{extensions::ext::DebugUtils, prelude::VkResult, RawPtr};

use crate::pipeline::VkPipelineInfo;
use crate::*;

use super::{QueueType, VkBuffer, VkDescriptorSet, VkGpu, VkGraphicsPipeline};

pub struct VkCommandBuffer<'g> {
    gpu: &'g VkGpu,
    inner_command_buffer: vk::CommandBuffer,
    has_recorded_anything: bool,
    has_been_submitted: bool,
    target_queue: vk::Queue,
}

pub struct VkRenderPassCommand<'c, 'g>
where
    'g: 'c,
{
    command_buffer: &'c mut VkCommandBuffer<'g>,
    has_draw_command: bool,
    viewport_area: Option<Viewport>,
    depth_bias_setup: Option<(f32, f32, f32)>,

    pipeline_state: PipelineState,
    descriptor_state: DescriptorSetState,

    push_constant_data: Vec<Vec<u8>>,
}

pub struct VkComputePassCommand<'c, 'g>
where
    'g: 'c,
{
    command_buffer: &'c mut VkCommandBuffer<'g>,
}

#[derive(Hash)]
pub(crate) struct PipelineState {
    pub(crate) fragment_shader: ShaderModuleHandle,
    pub(crate) vertex_shader: ShaderModuleHandle,
    pub(crate) scissor_area: Option<Rect2D>,
    pub(crate) front_face: FrontFace,
    pub(crate) cull_mode: CullMode,
    pub(crate) enable_depth_test: bool,
    pub(crate) render_area: Rect2D,
    pub(crate) vertex_inputs: Vec<VertexBindingInfo>,
    pub(crate) color_blend_states: Vec<PipelineColorBlendAttachmentState>,
}

impl PipelineState {
    fn new(info: &BeginRenderPassInfo) -> Self {
        // TODO: put this into pipeline state
        let color_blend_states = info
            .color_attachments
            .iter()
            .map(|_| PipelineColorBlendAttachmentState {
                blend_enable: true,
                src_color_blend_factor: BlendMode::One,
                dst_color_blend_factor: BlendMode::One,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendMode::One,
                dst_alpha_blend_factor: BlendMode::One,
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
            enable_depth_test: true,

            vertex_inputs: vec![],
            color_blend_states,
        }
    }

    pub(crate) fn input_assembly_state(&self) -> vk::PipelineInputAssemblyStateCreateInfo {
        vk::PipelineInputAssemblyStateCreateInfo {
            s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: PipelineInputAssemblyStateCreateFlags::empty(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
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
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: self.cull_mode.to_vk(),
            front_face: self.front_face.to_vk(),
            depth_bias_enable: vk::FALSE,
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
            depth_write_enable: true.to_vk(),
            depth_compare_op: CompareOp::Always.to_vk(),
            depth_bounds_test_enable: false.to_vk(),
            stencil_test_enable: false.to_vk(),
            front: StencilOpState::default().to_vk(),
            back: StencilOpState::default().to_vk(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        }
    }

    pub(crate) fn dynamic_state(&self) -> vk::PipelineDynamicStateCreateInfo {
        const DYNAMIC_STATES: &'static [vk::DynamicState] = &[
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::CULL_MODE,
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

#[derive(Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum DescriptorBindingType {
    BufferRange {
        handle: BufferHandle,
        offset: u64,
        range: usize,
    },

    ImageView {
        image_view_handle: ImageViewHandle,
        sampler_handle: SamplerHandle,
    },
}

impl Default for DescriptorBindingType {
    fn default() -> Self {
        Self::BufferRange {
            handle: BufferHandle::null(),
            offset: 0,
            range: 0,
        }
    }
}

#[derive(Default, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Binding {
    pub ty: DescriptorBindingType,
    pub binding_stage: ShaderStage,
}

#[derive(Default, Hash, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) struct DescriptorBindings {
    pub(crate) locations: Vec<Binding>,
}

#[derive(Hash, Clone, Default, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) struct DescriptorSetState {
    pub(crate) bindings: Vec<DescriptorBindings>,
    pub(crate) push_constant_range: Vec<PushConstantRange>,
}

mod inner {
    use ash::vk::ShaderStageFlags;

    pub(super) fn push_constant<T: Copy + Sized>(
        command_buffer: &crate::VkCommandBuffer,
        pipeline: &crate::VkGraphicsPipeline,
        data: &T,
        offset: u32,
    ) {
        let device = command_buffer.gpu.vk_logical_device();
        unsafe {
            let ptr: *const u8 = data as *const T as *const u8;
            let slice = std::slice::from_raw_parts(ptr, std::mem::size_of::<T>());
            device.cmd_push_constants(
                command_buffer.inner_command_buffer,
                pipeline.pipeline_layout,
                ShaderStageFlags::ALL,
                offset,
                slice,
            );
        }
    }
}

impl<'g> VkCommandBuffer<'g> {
    pub(crate) fn new(
        gpu: &'g VkGpu,
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
            gpu,
            inner_command_buffer,
            has_recorded_anything: false,
            has_been_submitted: false,
            target_queue: target_queue.get_vk_queue(gpu),
        })
    }
    pub fn begin_render_pass<'p>(
        &'p mut self,
        info: &BeginRenderPassInfo<'p>,
    ) -> VkRenderPassCommand<'p, 'g> {
        VkRenderPassCommand::<'p, 'g>::new(self, info)
    }

    pub fn begin_compute_pass<'p>(&'p mut self) -> VkComputePassCommand<'p, 'g> {
        VkComputePassCommand::<'p, 'g>::new(self)
    }

    pub fn pipeline_barrier(&mut self, barrier_info: &PipelineBarrierInfo) {
        self.has_recorded_anything = true;
        let device = self.gpu.vk_logical_device();
        let memory_barriers: Vec<_> = barrier_info
            .memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        let buffer_memory_barriers: Vec<_> = barrier_info
            .buffer_memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        let image_memory_barriers: Vec<_> = barrier_info
            .image_memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        unsafe {
            device.cmd_pipeline_barrier(
                self.inner_command_buffer,
                barrier_info.src_stage_mask.to_vk(),
                barrier_info.dst_stage_mask.to_vk(),
                DependencyFlags::empty(),
                &memory_barriers,
                &buffer_memory_barriers,
                &image_memory_barriers,
            )
        };
    }

    pub fn bind_descriptor_sets<T: VkPipelineInfo>(
        &self,
        pipeline: &T,
        first_index: u32,
        descriptor_sets: &[&VkDescriptorSet],
    ) {
        let descriptor_sets: Vec<_> = descriptor_sets
            .iter()
            .map(|d| d.allocation.descriptor_set)
            .collect();
        unsafe {
            self.gpu.vk_logical_device().cmd_bind_descriptor_sets(
                self.inner_command_buffer,
                T::bind_point().to_vk(),
                pipeline.vk_pipeline_layout(),
                first_index,
                &descriptor_sets,
                &[],
            );
        }
    }

    pub fn submit(mut self, submit_info: &CommandBufferSubmitInfo) -> VkResult<()> {
        self.has_been_submitted = true;
        if !self.has_recorded_anything {
            return Ok(());
        }

        let device = self.gpu.vk_logical_device();
        unsafe {
            device
                .end_command_buffer(self.inner())
                .expect("Failed to end inner command buffer");
            let target_queue = self.target_queue;

            let wait_semaphores: Vec<_> = submit_info
                .wait_semaphores
                .iter()
                .map(|s| s.inner)
                .collect();

            let signal_semaphores: Vec<_> = submit_info
                .signal_semaphores
                .iter()
                .map(|s| s.inner)
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
                    p_command_buffers: [self.inner_command_buffer].as_ptr(),
                    signal_semaphore_count: signal_semaphores.len() as _,
                    p_signal_semaphores: signal_semaphores.as_ptr(),
                }],
                if let Some(fence) = &submit_info.fence {
                    fence.inner
                } else {
                    vk::Fence::null()
                },
            )
        }
    }

    pub fn inner(&self) -> vk::CommandBuffer {
        self.inner_command_buffer
    }

    pub fn copy_buffer(
        &mut self,
        src_buffer: &VkBuffer,
        dst_buffer: &VkBuffer,
        dst_offset: u64,
        size: usize,
    ) -> VkResult<()> {
        self.has_recorded_anything = true;
        unsafe {
            self.gpu.vk_logical_device().cmd_copy_buffer(
                self.inner(),
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
        self.has_recorded_anything = true;
        unsafe {
            self.gpu.vk_logical_device().cmd_copy_buffer_to_image(
                self.inner(),
                info.source.inner,
                info.dest.inner,
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
}

impl Drop for ScopedDebugLabel {
    fn drop(&mut self) {
        if let Some(label) = self.inner.take() {
            label.end();
        }
    }
}

impl<'g> VkCommandBuffer<'g> {
    pub fn begin_debug_region(&self, label: &str, color: [f32; 4]) -> ScopedDebugLabel {
        ScopedDebugLabel {
            inner: self.gpu.state.debug_utilities.as_ref().map(|debug_utils| {
                ScopedDebugLabelInner::new(label, color, debug_utils.clone(), self.inner())
            }),
        }
    }

    pub fn insert_debug_label(&self, label: &str, color: [f32; 4]) {
        if let Some(debug_utils) = &self.gpu.state.debug_utilities {
            unsafe {
                let c_label = CString::new(label).unwrap();
                debug_utils.cmd_insert_debug_utils_label(
                    self.inner(),
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
}

impl<'g> Drop for VkCommandBuffer<'g> {
    fn drop(&mut self) {
        if !self.has_been_submitted {
            panic!("CommandBuffer::submit hasn't been called!");
        }
    }
}

impl<'c, 'g> VkRenderPassCommand<'c, 'g> {
    fn new(command_buffer: &'c mut VkCommandBuffer<'g>, info: &BeginRenderPassInfo<'c>) -> Self {
        let color_attachments: Vec<_> = info
            .color_attachments
            .iter()
            .map(|attch| RenderingAttachmentInfoKHR {
                s_type: StructureType::RENDERING_ATTACHMENT_INFO,
                p_next: std::ptr::null(),
                image_view: attch.image_view.inner,
                image_layout: attch.initial_layout.to_vk(),
                resolve_mode: ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(),
                resolve_image_layout: ImageLayout::Undefined.to_vk(),
                load_op: attch.load_op.to_vk(),
                store_op: attch.store_op.to_vk(),
                clear_value: match attch.load_op {
                    ColorLoadOp::Clear(color) => ash::vk::ClearValue {
                        color: ash::vk::ClearColorValue { float32: color },
                    },
                    _ => ash::vk::ClearValue::default(),
                },
            })
            .collect();

        let depth_attachment = info
            .depth_attachment
            .map(|attch| RenderingAttachmentInfoKHR {
                s_type: StructureType::RENDERING_ATTACHMENT_INFO,
                p_next: std::ptr::null(),
                image_view: attch.image_view.inner,
                image_layout: attch.initial_layout.to_vk(),
                resolve_mode: ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(),
                resolve_image_layout: ImageLayout::Undefined.to_vk(),
                load_op: attch.load_op.to_vk(),
                store_op: attch.store_op.to_vk(),
                clear_value: match attch.load_op {
                    DepthLoadOp::Clear(d) => ash::vk::ClearValue {
                        depth_stencil: ClearDepthStencilValue {
                            depth: d,
                            stencil: 255,
                        },
                    },
                    _ => ash::vk::ClearValue::default(),
                },
            });

        let stencil_attachment = info
            .stencil_attachment
            .map(|attch| RenderingAttachmentInfoKHR {
                s_type: StructureType::RENDERING_ATTACHMENT_INFO,
                p_next: std::ptr::null(),
                image_view: attch.image_view.inner,
                image_layout: attch.initial_layout.to_vk(),
                resolve_mode: ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(),
                resolve_image_layout: ImageLayout::Undefined.to_vk(),
                load_op: attch.load_op.to_vk(),
                store_op: attch.store_op.to_vk(),
                clear_value: match attch.load_op {
                    StencilLoadOp::Clear(s) => ash::vk::ClearValue {
                        depth_stencil: ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: s as _,
                        },
                    },
                    _ => ash::vk::ClearValue::default(),
                },
            });

        let create_info = RenderingInfoKHR {
            s_type: StructureType::RENDERING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: RenderingFlags::empty(),
            layer_count: 1,
            view_mask: 0,
            render_area: info.render_area.to_vk(),
            color_attachment_count: color_attachments.len() as _,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: depth_attachment.as_ref().as_raw_ptr(),
            p_stencil_attachment: stencil_attachment.as_ref().as_raw_ptr(),
        };
        unsafe {
            command_buffer
                .gpu
                .state
                .dynamic_rendering
                .cmd_begin_rendering(command_buffer.inner_command_buffer, &create_info);
        };

        Self {
            command_buffer,
            has_draw_command: false,
            viewport_area: None,
            depth_bias_setup: None,
            pipeline_state: PipelineState::new(info),
            descriptor_state: DescriptorSetState::default(),
            push_constant_data: vec![],
        }
    }

    #[deprecated(note = "Use the new, higher-level, api")]
    pub fn bind_pipeline(&mut self, material: &VkGraphicsPipeline) {
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_bind_pipeline(
                self.command_buffer.inner_command_buffer,
                PipelineBindPoint::GRAPHICS,
                material.pipeline,
            )
        }
    }

    pub fn set_vertex_shader(&mut self, vertex_shader: ShaderModuleHandle) {
        self.pipeline_state.vertex_shader = vertex_shader;
    }

    pub fn set_fragment_shader(&mut self, fragment_shader: ShaderModuleHandle) {
        self.pipeline_state.fragment_shader = fragment_shader;
    }

    pub fn set_vertex_buffers(&mut self, bindings: &[VertexBindingInfo]) {
        self.pipeline_state.vertex_inputs = bindings.to_vec();
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.prepare_draw();

        self.has_draw_command = true;
        self.command_buffer.has_recorded_anything = true;
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_draw_indexed(
                self.command_buffer.inner(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.prepare_draw();
        self.has_draw_command = true;
        self.command_buffer.has_recorded_anything = true;
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_draw(
                self.command_buffer.inner(),
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport_area = Some(viewport);
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.depth_bias_setup = Some((constant, clamp, slope));
    }

    pub fn set_front_face(&mut self, front_face: FrontFace) {
        self.pipeline_state.front_face = front_face;
    }

    pub fn set_cull_mode(&mut self, cull_mode: CullMode) {
        self.pipeline_state.cull_mode = cull_mode;
    }

    pub fn set_enable_depth_test(&mut self, enable_depth_test: bool) {
        self.pipeline_state.enable_depth_test = enable_depth_test;
    }

    fn prepare_draw(&self) {
        let device = self.command_buffer.gpu.vk_logical_device();

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
            device.cmd_set_depth_bias_enable(self.command_buffer.inner(), true);
            device.cmd_set_depth_bias(
                self.command_buffer.inner(),
                depth_constant,
                depth_clamp,
                depth_slope,
            );
            device.cmd_set_viewport(self.command_buffer.inner(), 0, &[viewport.to_vk()]);
            device.cmd_set_scissor(self.command_buffer.inner(), 0, &[scissor.to_vk()]);
            device.cmd_set_front_face(
                self.command_buffer.inner(),
                self.pipeline_state.front_face.to_vk(),
            );
            device.cmd_set_cull_mode(
                self.command_buffer.inner(),
                self.pipeline_state.cull_mode.to_vk(),
            );
            device.cmd_set_depth_test_enable(
                self.command_buffer.inner(),
                self.pipeline_state.enable_depth_test,
            );
        }
    }

    pub fn bind_index_buffer(&self, buffer: &VkBuffer, offset: u64, index_type: IndexType) {
        let device = self.command_buffer.gpu.vk_logical_device();
        let index_buffer = buffer.inner;
        unsafe {
            device.cmd_bind_index_buffer(
                self.command_buffer.inner_command_buffer,
                index_buffer,
                offset,
                index_type.to_vk(),
            );
        }
    }
    pub fn bind_vertex_buffer(&self, first_binding: u32, buffers: &[&VkBuffer], offsets: &[u64]) {
        assert!(buffers.len() == offsets.len());
        let device = self.command_buffer.gpu.vk_logical_device();
        let vertex_buffers: Vec<_> = buffers.iter().map(|b| b.inner).collect();
        unsafe {
            device.cmd_bind_vertex_buffers(
                self.command_buffer.inner_command_buffer,
                first_binding,
                &vertex_buffers,
                offsets,
            );
        }
    }

    pub fn push_constant<T: Copy + Sized>(
        &self,
        pipeline: &VkGraphicsPipeline,
        data: &T,
        offset: u32,
    ) {
        inner::push_constant(self, pipeline, data, offset)
    }

    pub fn draw_indexed_handle(
        &mut self,
        num_indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> anyhow::Result<()> {
        self.has_draw_command = true;
        let layout = self.find_matching_pipeline_layout();
        let pipeline = self.find_matching_pipeline(layout);
        let device = self.gpu.vk_logical_device();
        {
            let buffers = self
                .pipeline_state
                .vertex_inputs
                .iter()
                .map(|b| self.gpu.resolve_buffer(b.handle).inner)
                .collect::<Vec<_>>();
            let offsets = self
                .pipeline_state
                .vertex_inputs
                .iter()
                .map(|_| 0)
                .collect::<Vec<_>>();

            unsafe {
                for (idx, constant_range) in
                    self.descriptor_state.push_constant_range.iter().enumerate()
                {
                    device.cmd_push_constants(
                        self.inner(),
                        layout,
                        constant_range.stage_flags.to_vk(),
                        constant_range.offset,
                        &self.push_constant_data[idx],
                    );
                }
                device.cmd_bind_vertex_buffers(self.inner(), 0, &buffers, &offsets);
            }
        }

        if self.descriptor_state.bindings.len() > 0 {
            let descriptors = self.find_matching_descriptor_sets();
            unsafe {
                device.cmd_bind_descriptor_sets(
                    self.inner(),
                    PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &descriptors,
                    &[],
                );
            }
        }

        unsafe {
            device.cmd_bind_pipeline(self.inner(), PipelineBindPoint::GRAPHICS, pipeline);
        }
        self.prepare_draw();
        unsafe {
            device.cmd_draw_indexed(
                self.inner(),
                num_indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            );
        }

        Ok(())
    }

    fn find_matching_pipeline(&mut self, pipeline_layout: vk::PipelineLayout) -> vk::Pipeline {
        self.gpu.get_pipeline(&self.pipeline_state, pipeline_layout)
    }

    fn find_matching_pipeline_layout(&self) -> vk::PipelineLayout {
        self.gpu.get_pipeline_layout(&self.descriptor_state)
    }

    fn find_matching_descriptor_sets(&self) -> Vec<vk::DescriptorSet> {
        self.gpu
            .get_descriptor_sets(&self.descriptor_state.bindings)
    }
    pub fn set_index_buffer(
        &self,
        index_buffer: BufferHandle,
        index_type: IndexType,
        offset: usize,
    ) {
        assert!(!index_buffer.is_null());
        let index_buffer = self.gpu.resolve_buffer(index_buffer);
        let device = self.gpu.vk_logical_device();
        unsafe {
            device.cmd_bind_index_buffer(
                self.inner(),
                index_buffer.inner,
                offset as _,
                index_type.to_vk(),
            );
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
        if self.descriptor_state.push_constant_range.len() <= (index as _) {
            self.descriptor_state
                .push_constant_range
                .resize(index as usize + 1, PushConstantRange::default());
            self.push_constant_data.resize(index as usize + 1, vec![]);
        }

        self.push_constant_data[index as usize] = data.to_vec();

        self.descriptor_state.push_constant_range[index as usize] = PushConstantRange {
            stage_flags: shader_stage,
            offset,
            size: std::mem::size_of_val(data) as _,
        }
    }

    pub fn bind_resources(&mut self, set: u32, bindings: &[Binding]) {
        if self.descriptor_state.bindings.len() <= (set as _) {
            self.descriptor_state
                .bindings
                .resize(set as usize + 1, DescriptorBindings::default());
        }

        self.descriptor_state.bindings[set as usize].locations = bindings.to_vec();
    }
}

impl<'c, 'g> AsRef<VkCommandBuffer<'g>> for VkRenderPassCommand<'c, 'g> {
    fn as_ref(&self) -> &VkCommandBuffer<'g> {
        self.command_buffer
    }
}

impl<'c, 'g> Deref for VkRenderPassCommand<'c, 'g> {
    type Target = VkCommandBuffer<'g>;

    fn deref(&self) -> &Self::Target {
        self.command_buffer
    }
}

impl<'c, 'g> Drop for VkRenderPassCommand<'c, 'g> {
    fn drop(&mut self) {
        unsafe {
            self.command_buffer
                .gpu
                .state
                .dynamic_rendering
                .cmd_end_rendering(self.command_buffer.inner_command_buffer)
        };
    }
}

impl<'c, 'g> VkComputePassCommand<'c, 'g> {
    fn new(command_buffer: &'c mut VkCommandBuffer<'g>) -> Self {
        Self { command_buffer }
    }

    pub fn bind_pipeline(&mut self, pipeline: &VkComputePipeline) {
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_bind_pipeline(
                self.command_buffer.inner_command_buffer,
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            )
        }
    }

    pub fn dispatch(&mut self, group_size_x: u32, group_size_y: u32, group_size_z: u32) {
        unsafe {
            self.command_buffer.gpu.vk_logical_device().cmd_dispatch(
                self.command_buffer.inner_command_buffer,
                group_size_x,
                group_size_y,
                group_size_z,
            );
        }
        self.command_buffer.has_recorded_anything = true;
    }
}

impl<'c, 'g> AsRef<VkCommandBuffer<'g>> for VkComputePassCommand<'c, 'g> {
    fn as_ref(&self) -> &VkCommandBuffer<'g> {
        self.command_buffer
    }
}

impl<'c, 'g> Deref for VkComputePassCommand<'c, 'g> {
    type Target = VkCommandBuffer<'g>;

    fn deref(&self) -> &Self::Target {
        self.command_buffer
    }
}
