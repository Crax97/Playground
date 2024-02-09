use bitflags::bitflags;

use std::{collections::HashMap, ffi::CString};

use ash::{
    extensions::ext::DebugUtils,
    vk::{self, ClearColorValue, ClearDepthStencilValue, ClearValue},
};

use crate::vulkan::ToVk;
use crate::{AttachmentStoreOp, ColorLoadOp, DepthLoadOp, StencilLoadOp};

pub enum DrawCommandType {
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, u32, i32, u32),
}

pub struct PushConstant {
    pub data: Vec<u8>,
    pub stage_flags: vk::ShaderStageFlags,
}

#[derive(Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub enum AttachmentMode {
    AttachmentRead,
    AttachmentWrite,
    ShaderRead,
    ShaderWrite,
}

pub struct SubpassColorAttachment {
    pub index: usize,
    pub attachment_mode: AttachmentMode,
}

pub struct DynamicState {
    pub viewport: vk::Viewport,
    pub scissor_rect: vk::Rect2D,
    pub depth_bias: Option<(f32, f32, f32)>,
    pub depth_test_enable: bool,
}

bitflags! {
    #[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
    pub struct AttachmentFlags: u32 {
        const SWAPCHAIN_IMAGE = 1;
    }
}

#[derive(Clone, Copy, Hash)]
pub struct ColorAttachment {
    pub render_image: RenderGraphImage,
    pub load_op: ColorLoadOp,
    pub store_op: AttachmentStoreOp,
    pub flags: AttachmentFlags,
}

#[derive(Copy, Clone, Hash)]
pub struct RenderGraphImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
}
#[derive(Clone, Copy, Hash)]
pub struct DepthAttachment {
    pub render_image: RenderGraphImage,
    pub load_op: DepthLoadOp,
    pub store_op: AttachmentStoreOp,
}

#[derive(Clone, Copy, Hash)]
pub struct StencilAttachment {
    pub render_image: RenderGraphImage,
    pub load_op: StencilLoadOp,
    pub store_op: AttachmentStoreOp,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum Resource {
    Image(vk::Image, vk::ImageAspectFlags),
    Buffer(vk::Buffer),
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct ShaderInput {
    pub resource: Resource,
}

pub struct GraphicsSubpass {
    pub color_attachments: Vec<SubpassColorAttachment>,
    pub depth_attachment: Option<AttachmentMode>,
    pub stencil_attachment: Option<AttachmentMode>,
    pub render_area: vk::Rect2D,

    pub draw_commands: Vec<DrawCommand>,

    pub shader_reads: Vec<ShaderInput>,
    pub shader_writes: Vec<ShaderInput>,
}

#[derive(Default, Hash, Eq, PartialEq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct IndexBuffer {
    pub buffer: vk::Buffer,
    pub index_type: vk::IndexType,
    pub offset: vk::DeviceSize,
}

pub struct VertexBuffer {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
}

pub struct DrawCommand {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub push_constants: Vec<PushConstant>,
    pub dynamic_state: DynamicState,

    pub vertex_buffers: Vec<VertexBuffer>,
    pub index_buffer: Option<IndexBuffer>,

    pub command_type: DrawCommandType,
}

#[derive(Default)]
pub struct GraphicsPassInfo {
    pub label: Option<String>,
    pub all_color_attachments: Vec<ColorAttachment>,
    pub depth_attachment: Option<DepthAttachment>,
    pub stencil_attachment: Option<StencilAttachment>,
    pub sub_passes: Vec<GraphicsSubpass>,
}

pub struct DispatchCommand {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub push_constants: Vec<PushConstant>,
    pub group_count: [u32; 3],
}

bitflags! {
    #[derive(Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
    pub struct ComputePassFlags : u32 {
        const ASYNC = 1;
    }
}

#[derive(Default)]
pub struct ComputePassInfo {
    pub label: Option<String>,
    pub flags: ComputePassFlags,
    pub dispatches: Vec<DispatchCommand>,
    pub all_read_resources: Vec<ShaderInput>,
    pub all_written_resources: Vec<ShaderInput>,
}

#[derive(Clone, Copy)]
pub enum RenderPass {
    Graphics(usize),
    Compute(usize),
}

pub struct RenderGraph {
    device: ash::Device,
    all_graphics_attachments: Vec<vk::Image>,
    graphics_passes: Vec<GraphicsPassInfo>,
    compute_passes: Vec<ComputePassInfo>,
    debug_utils: Option<DebugUtils>,

    sequence: Vec<RenderPass>,
}

impl RenderGraph {
    pub fn new(device: ash::Device, debug_utils: Option<DebugUtils>) -> Self {
        Self {
            device,
            all_graphics_attachments: vec![],
            graphics_passes: vec![],
            compute_passes: vec![],
            debug_utils,

            sequence: vec![],
        }
    }

    fn validate_graph(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn pick_label_color_graphics(&self, _info: &GraphicsPassInfo) -> [f32; 4] {
        [0.3, 0.5, 0.2, 1.0]
    }

    fn pick_label_color_compute(&self, _info: &ComputePassInfo) -> [f32; 4] {
        [0.1, 0.6, 0.5, 1.0]
    }

    fn find_starting_render_pass(&self) -> Option<&RenderPass> {
        self.sequence.last()
    }

    fn find_execution_plan(&self) -> Vec<RenderPass> {
        self.sequence.clone()
    }
}

pub struct RenderGraphExecutionContext {
    pub graphics_command_buffer: vk::CommandBuffer,
    pub async_compute_command_buffer: vk::CommandBuffer,
}

#[derive(Default)]
pub struct LayoutInfo {
    pub layout: vk::ImageLayout,
    pub access_flags: vk::AccessFlags2,
    pub pipeline_flags: vk::PipelineStageFlags2,
}

impl RenderGraph {
    pub fn add_graphics(&mut self, pass: GraphicsPassInfo) {
        self.all_graphics_attachments.extend(
            pass.all_color_attachments
                .iter()
                .map(|att| att.render_image.image),
        );
        self.all_graphics_attachments.extend(
            pass.depth_attachment
                .iter()
                .map(|att| att.render_image.image),
        );
        self.all_graphics_attachments.extend(
            pass.stencil_attachment
                .iter()
                .map(|att| att.render_image.image),
        );
        self.sequence
            .push(RenderPass::Graphics(self.graphics_passes.len()));
        self.graphics_passes.push(pass);
    }

    pub fn add_compute(&mut self, pass: ComputePassInfo) {
        self.sequence
            .push(RenderPass::Compute(self.compute_passes.len()));
        self.compute_passes.push(pass);
    }

    pub fn execute(
        &mut self,
        execution_context: &RenderGraphExecutionContext,
    ) -> anyhow::Result<()> {
        self.validate_graph()?;
        let execution_plan = self.find_execution_plan();
        let mut previous_layout = HashMap::<vk::Image, LayoutInfo>::new();
        for pass in execution_plan {
            match pass {
                RenderPass::Graphics(index) => {
                    let info = &self.graphics_passes[index];
                    self.execute_graphics_pass_dynamic_rendering(
                        info,
                        execution_context,
                        &mut previous_layout,
                    );
                }
                RenderPass::Compute(index) => {
                    let info = &self.compute_passes[index];
                    self.execute_compute_pass(info, execution_context, &mut previous_layout);
                }
            }
        }

        if let Some(swapchain_image) = self.find_swapchain_image() {
            self.make_sure_swapchain_is_presentable(
                execution_context,
                swapchain_image,
                &mut previous_layout,
            );
        }
        self.reset();
        Ok(())
    }
}

impl RenderGraph {
    fn find_swapchain_image(&self) -> Option<vk::Image> {
        for gp in self.graphics_passes.iter().rev() {
            for attachment in &gp.all_color_attachments {
                if attachment.flags.contains(AttachmentFlags::SWAPCHAIN_IMAGE) {
                    return Some(attachment.render_image.image);
                }
            }
        }

        None
    }

    fn execute_graphics_pass_dynamic_rendering(
        &self,
        info: &GraphicsPassInfo,
        execution_context: &RenderGraphExecutionContext,
        previous_layouts: &mut HashMap<vk::Image, LayoutInfo>,
    ) {
        if let Some(utils) = &self.debug_utils {
            let label_string = info.label.as_deref().unwrap_or("Graphics Pass");
            let label_pointer = CString::new(label_string).unwrap();
            let label_color = self.pick_label_color_graphics(info);
            unsafe {
                utils.cmd_begin_debug_utils_label(
                    execution_context.graphics_command_buffer,
                    &vk::DebugUtilsLabelEXT {
                        p_label_name: label_pointer.as_ptr(),
                        color: label_color,
                        ..Default::default()
                    },
                );
            }
        }
        for (i, subpass) in info.sub_passes.iter().enumerate() {
            if let Some(utils) = &self.debug_utils {
                let label_string = format!("Subpass #{i}");
                let label_pointer = CString::new(label_string).unwrap();
                let label_color = self.pick_label_color_graphics(info);
                unsafe {
                    utils.cmd_begin_debug_utils_label(
                        execution_context.graphics_command_buffer,
                        &vk::DebugUtilsLabelEXT {
                            p_label_name: label_pointer.as_ptr(),
                            color: label_color,
                            ..Default::default()
                        },
                    );
                }
            }
            let mut image_barriers = vec![];
            for attachment_reference in &subpass.color_attachments {
                let attachment = &info.all_color_attachments[attachment_reference.index];
                push_image_transition(
                    previous_layouts,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    match attachment_reference.attachment_mode {
                        AttachmentMode::AttachmentRead => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                        AttachmentMode::AttachmentWrite => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        _ => unreachable!(),
                    },
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    attachment.render_image.image,
                    &mut image_barriers,
                    vk::ImageAspectFlags::COLOR,
                );
            }
            for (image, aspect_mask) in
                subpass
                    .shader_reads
                    .iter()
                    .filter_map(|res| match res.resource {
                        Resource::Image(image, aspect_mask) => Some((image, aspect_mask)),
                        Resource::Buffer(_) => None,
                    })
            {
                try_push_image_transition(
                    previous_layouts,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    image,
                    &mut image_barriers,
                    aspect_mask,
                );
            }

            for (image, aspect_mask) in
                subpass
                    .shader_writes
                    .iter()
                    .filter_map(|res| match res.resource {
                        Resource::Image(image, aspect_mask) => Some((image, aspect_mask)),
                        Resource::Buffer(_) => None,
                    })
            {
                try_push_image_transition(
                    previous_layouts,
                    vk::ImageLayout::GENERAL,
                    vk::AccessFlags2::SHADER_WRITE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    image,
                    &mut image_barriers,
                    aspect_mask,
                );
            }

            if let Some(attachment_mode) = subpass.depth_attachment {
                let attachment = info.depth_attachment.unwrap();
                push_image_transition(
                    previous_layouts,
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    match attachment_mode {
                        AttachmentMode::AttachmentRead => {
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        }
                        AttachmentMode::AttachmentWrite => {
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        }
                        _ => unreachable!(),
                    },
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    attachment.render_image.image,
                    &mut image_barriers,
                    vk::ImageAspectFlags::DEPTH,
                );
            }

            if let Some(attachment_mode) = subpass.stencil_attachment {
                let attachment = info.stencil_attachment.unwrap();
                push_image_transition(
                    previous_layouts,
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    match attachment_mode {
                        AttachmentMode::AttachmentRead => {
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        }
                        AttachmentMode::AttachmentWrite => {
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        }
                        _ => unreachable!(),
                    },
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    attachment.render_image.image,
                    &mut image_barriers,
                    vk::ImageAspectFlags::STENCIL,
                );
            }

            unsafe {
                self.device.cmd_pipeline_barrier2(
                    execution_context.graphics_command_buffer,
                    &vk::DependencyInfo::builder()
                        .image_memory_barriers(&image_barriers)
                        .build(),
                );
            }

            let color_attachments = subpass
                .color_attachments
                .iter()
                .map(|attachment| info.all_color_attachments[attachment.index])
                .map(|attachment| {
                    vk::RenderingAttachmentInfo::builder()
                        .image_view(attachment.render_image.view)
                        .clear_value(match attachment.load_op {
                            ColorLoadOp::DontCare => ClearValue::default(),
                            ColorLoadOp::Load => ClearValue::default(),
                            ColorLoadOp::Clear(value) => ClearValue {
                                color: ClearColorValue { float32: value },
                            },
                        })
                        .load_op(attachment.load_op.to_vk())
                        .store_op(attachment.store_op.to_vk())
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .build()
                })
                .collect::<Vec<_>>();
            let depth_attachment = if let Some(attach) = info.depth_attachment {
                vk::RenderingAttachmentInfo::builder()
                    .image_view(attach.render_image.view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .clear_value(match attach.load_op {
                        DepthLoadOp::DontCare | DepthLoadOp::Load => ClearValue::default(),
                        DepthLoadOp::Clear(value) => ClearValue {
                            depth_stencil: ClearDepthStencilValue {
                                depth: value,
                                stencil: 0,
                            },
                        },
                    })
                    .load_op(attach.load_op.to_vk())
                    .store_op(attach.store_op.to_vk())
                    .build()
            } else {
                vk::RenderingAttachmentInfo::default()
            };
            let stencil_attachment = if let Some(_attach) = info.stencil_attachment {
                todo!()
            } else {
                vk::RenderingAttachmentInfo::default()
            };
            unsafe {
                self.device.cmd_begin_rendering(
                    execution_context.graphics_command_buffer,
                    &vk::RenderingInfo {
                        render_area: subpass.render_area,
                        layer_count: 1,
                        view_mask: 0,
                        color_attachment_count: color_attachments.len() as _,
                        p_color_attachments: color_attachments.as_ptr(),
                        p_depth_attachment: std::ptr::addr_of!(depth_attachment),
                        p_stencil_attachment: std::ptr::addr_of!(stencil_attachment),
                        ..Default::default()
                    },
                );

                self.execute_draw_command(subpass, execution_context);

                self.device
                    .cmd_end_rendering(execution_context.graphics_command_buffer);

                if let Some(utils) = &self.debug_utils {
                    utils.cmd_end_debug_utils_label(execution_context.graphics_command_buffer);
                }
            }
        }
        if let Some(utils) = &self.debug_utils {
            unsafe {
                utils.cmd_end_debug_utils_label(execution_context.graphics_command_buffer);
            }
        }
    }

    fn execute_compute_pass(
        &self,
        info: &ComputePassInfo,
        execution_context: &RenderGraphExecutionContext,
        previous_layout: &mut HashMap<vk::Image, LayoutInfo>,
    ) {
        if let Some(utils) = &self.debug_utils {
            let label_string = info.label.as_deref().unwrap_or("Graphics Pass");
            let label_pointer = CString::new(label_string).unwrap();
            let label_color = self.pick_label_color_compute(info);
            unsafe {
                utils.cmd_begin_debug_utils_label(
                    execution_context.graphics_command_buffer,
                    &vk::DebugUtilsLabelEXT {
                        p_label_name: label_pointer.as_ptr(),
                        color: label_color,
                        ..Default::default()
                    },
                );
            }
        }

        let command_buffer = if info.flags.contains(ComputePassFlags::ASYNC) {
            execution_context.async_compute_command_buffer
        } else {
            execution_context.graphics_command_buffer
        };
        unsafe {
            for dispatch in &info.dispatches {
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    dispatch.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    dispatch.pipeline_layout,
                    0,
                    &dispatch.descriptor_sets,
                    &[],
                );
                for push_constant in &dispatch.push_constants {
                    self.device.cmd_push_constants(
                        execution_context.graphics_command_buffer,
                        dispatch.pipeline_layout,
                        push_constant.stage_flags,
                        0,
                        &push_constant.data,
                    );
                }
                self.device.cmd_dispatch(
                    command_buffer,
                    dispatch.group_count[0],
                    dispatch.group_count[1],
                    dispatch.group_count[2],
                )
            }
        }

        if let Some(utils) = &self.debug_utils {
            unsafe {
                utils.cmd_end_debug_utils_label(execution_context.graphics_command_buffer);
            }
        }
    }

    unsafe fn execute_draw_command(
        &self,
        subpass: &GraphicsSubpass,
        execution_context: &RenderGraphExecutionContext,
    ) {
        let mut current_pipeline = vk::Pipeline::null();

        for command in &subpass.draw_commands {
            if current_pipeline != command.pipeline {
                self.device.cmd_bind_pipeline(
                    execution_context.graphics_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    command.pipeline,
                );
                current_pipeline = command.pipeline;
            }
            self.device.cmd_bind_descriptor_sets(
                execution_context.graphics_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                command.pipeline_layout,
                0,
                &command.descriptor_sets,
                &[],
            );
            let buffers = command
                .vertex_buffers
                .iter()
                .map(|b| b.buffer)
                .collect::<Vec<_>>();
            let offsets = command
                .vertex_buffers
                .iter()
                .map(|b| b.offset)
                .collect::<Vec<_>>();
            if !command.vertex_buffers.is_empty() {
                self.device.cmd_bind_vertex_buffers(
                    execution_context.graphics_command_buffer,
                    0,
                    &buffers,
                    // TODO:  Add support for offsets
                    &offsets,
                )
            }

            if let Some(index_buffer) = command.index_buffer {
                self.device.cmd_bind_index_buffer(
                    execution_context.graphics_command_buffer,
                    index_buffer.buffer,
                    index_buffer.offset,
                    index_buffer.index_type,
                )
            }

            for push_constant in &command.push_constants {
                self.device.cmd_push_constants(
                    execution_context.graphics_command_buffer,
                    command.pipeline_layout,
                    push_constant.stage_flags,
                    0,
                    &push_constant.data,
                );
            }

            command
                .dynamic_state
                .apply(&self.device, execution_context.graphics_command_buffer);

            match command.command_type {
                DrawCommandType::Draw(
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                ) => self.device.cmd_draw(
                    execution_context.graphics_command_buffer,
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                ),
                DrawCommandType::DrawIndexed(
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                ) => self.device.cmd_draw_indexed(
                    execution_context.graphics_command_buffer,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                ),
            }
        }
    }

    fn make_sure_swapchain_is_presentable(
        &self,
        execution_context: &RenderGraphExecutionContext,
        swapchain_image: vk::Image,
        previous_layouts: &mut HashMap<vk::Image, LayoutInfo>,
    ) {
        let mut image_barriers = vec![];
        push_image_transition(
            previous_layouts,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            swapchain_image,
            &mut image_barriers,
            vk::ImageAspectFlags::COLOR,
        );
        unsafe {
            self.device.cmd_pipeline_barrier2(
                execution_context.graphics_command_buffer,
                &vk::DependencyInfo::builder()
                    .image_memory_barriers(&image_barriers)
                    .build(),
            );
        }
    }

    fn reset(&mut self) {
        self.all_graphics_attachments.clear();
        self.graphics_passes.clear();
        self.compute_passes.clear();
        self.sequence.clear();
    }
}
impl DynamicState {
    unsafe fn apply(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let (constant, clamp, slope) = self.depth_bias.unwrap_or_default();

        device.cmd_set_viewport(command_buffer, 0, &[self.viewport]);
        device.cmd_set_scissor(command_buffer, 0, &[self.scissor_rect]);
        device.cmd_set_depth_bias_enable(command_buffer, self.depth_bias.is_some());
        device.cmd_set_depth_bias(command_buffer, constant, clamp, slope);
        device.cmd_set_depth_test_enable(command_buffer, self.depth_test_enable);
    }
}

fn push_image_transition(
    previous_layouts: &mut HashMap<vk::Image, LayoutInfo>,
    new_layout: vk::ImageLayout,
    dst_access_mask: vk::AccessFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    attachment: vk::Image,
    image_barriers: &mut Vec<vk::ImageMemoryBarrier2>,
    aspect_mask: vk::ImageAspectFlags,
) {
    let previous_layout = previous_layouts.entry(attachment).or_default();
    let new_layout = LayoutInfo {
        layout: new_layout,
        access_flags: dst_access_mask,
        pipeline_flags: dst_stage_mask,
    };

    image_barriers.push(vk::ImageMemoryBarrier2 {
        src_access_mask: previous_layout.access_flags,
        dst_access_mask: new_layout.access_flags,
        old_layout: previous_layout.layout,
        new_layout: new_layout.layout,
        src_stage_mask: previous_layout.pipeline_flags,
        dst_stage_mask: new_layout.pipeline_flags,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: attachment,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },

        ..Default::default()
    });

    *previous_layout = new_layout;
}

fn try_push_image_transition(
    previous_layouts: &mut HashMap<vk::Image, LayoutInfo>,
    new_layout: vk::ImageLayout,
    dst_access_mask: vk::AccessFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    attachment: vk::Image,
    image_barriers: &mut Vec<vk::ImageMemoryBarrier2>,
    aspect_mask: vk::ImageAspectFlags,
) {
    if let Some(previous_layout) = previous_layouts.get_mut(&attachment) {
        let new_layout = LayoutInfo {
            layout: new_layout,
            access_flags: dst_access_mask,
            pipeline_flags: dst_stage_mask,
        };

        image_barriers.push(vk::ImageMemoryBarrier2 {
            src_access_mask: previous_layout.access_flags,
            dst_access_mask: new_layout.access_flags,
            old_layout: previous_layout.layout,
            new_layout: new_layout.layout,
            src_stage_mask: previous_layout.pipeline_flags,
            dst_stage_mask: new_layout.pipeline_flags,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: attachment,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },

            ..Default::default()
        });

        *previous_layout = new_layout;
    }
}
