use std::marker::PhantomData;

use crate::{
    hal::QueueType, rdg::Node, util::check, AttachmentStoreOp, BindingSet, Buffer,
    BufferUsageFlags, ComputePassDescription, ComputePipeline, DepthStencilTarget, Device,
    Extents2D, GraphicsPipeline, ImageUsageFlags, MgpuResult, Rect2D, RenderPassDescription,
    RenderTarget,
};

pub struct CommandRecorder<T: CommandRecorderType> {
    pub(crate) _ph: PhantomData<T>,
    pub(crate) device: Device,
    pub(crate) binding_sets: Vec<BindingSet>,
    pub(crate) new_nodes: Vec<Node>,
}

pub struct RenderPass<'c> {
    command_recorder: &'c mut CommandRecorder<Graphics>,
    info: RenderPassInfo,
    pipeline: Option<GraphicsPipeline>,
    vertex_buffers: Vec<Buffer>,
    index_buffer: Option<Buffer>,
}

pub struct ComputePass<'c, C: ComputeCommandRecorder> {
    command_recorder: &'c mut CommandRecorder<C>,
    info: ComputePassInfo,
    pipeline: Option<ComputePipeline>,
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AttachmentAccessMode {
    Read,
    Write,
}

#[derive(Debug, Hash)]
pub(crate) struct RenderAttachmentReference {
    pub index: usize,
    pub access_mode: AttachmentAccessMode,
}

#[derive(Debug, Hash)]
pub(crate) struct DepthStencilAttachmentReference {
    pub access_mode: AttachmentAccessMode,
}

#[derive(Debug, Hash)]
pub(crate) enum DrawType {
    Draw {
        vertices: usize,
        instances: usize,
        first_vertex: usize,
        first_instance: usize,
    },

    DrawIndexed {
        indices: usize,
        instances: usize,
        first_index: usize,
        vertex_offset: i32,
        first_instance: usize,
    },
}

#[derive(Debug, Hash)]
pub(crate) enum DispatchType {
    Dispatch(u32, u32, u32),
}

#[derive(Debug, Hash)]
pub(crate) struct DrawCommand {
    pub(crate) pipeline: GraphicsPipeline,
    pub(crate) vertex_buffers: Vec<Buffer>,
    pub(crate) index_buffer: Option<Buffer>,
    pub(crate) binding_sets: Vec<BindingSet>,
    pub(crate) draw_type: DrawType,
}

#[derive(Debug, Hash)]
pub(crate) struct DispatchCommand {
    pub(crate) pipeline: ComputePipeline,
    pub(crate) binding_sets: Vec<BindingSet>,
    pub(crate) dispatch_type: DispatchType,
}

#[derive(Debug, Hash)]
pub(crate) struct RenderStep {
    pub(crate) color_attachments: Vec<RenderAttachmentReference>,
    pub(crate) depth_stencil_attachment: Option<DepthStencilAttachmentReference>,
    pub(crate) commands: Vec<DrawCommand>,
}

#[derive(Debug, Hash, Default)]
pub(crate) struct ComputeStep {
    pub(crate) commands: Vec<DispatchCommand>,
}

#[derive(Default, Debug, Hash)]
pub struct Framebuffer {
    pub(crate) render_targets: Vec<RenderTarget>,
    pub(crate) depth_stencil_target: Option<DepthStencilTarget>,
    pub(crate) extents: Extents2D,
}

#[derive(Default, Debug, Hash)]
pub struct RenderPassInfo {
    pub(crate) label: Option<String>,
    pub(crate) framebuffer: Framebuffer,
    pub(crate) render_area: Rect2D,
    pub(crate) steps: Vec<RenderStep>,
}

#[derive(Default, Debug, Hash)]
pub struct ComputePassInfo {
    pub(crate) label: Option<String>,
    pub(crate) steps: Vec<ComputeStep>,
}

// The commands recorded on this command recorder will be put on the graphics queue
pub struct Graphics;

// The commands recorded on this command recorder will be put on the async compute queue
pub struct AsyncCompute;

impl<T: CommandRecorderType> CommandRecorder<T> {
    pub fn submit(self) -> MgpuResult<()> {
        let mut rdg = self.device.write_rdg();
        match T::PREFERRED_QUEUE_TYPE {
            QueueType::Graphics => {
                for node in self.new_nodes {
                    rdg.add_graphics_pass(node);
                }
            }
            QueueType::AsyncCompute => {
                for node in self.new_nodes {
                    rdg.add_async_compute_pass(node);
                }
            }
            QueueType::AsyncTransfer => {
                for node in self.new_nodes {
                    rdg.add_async_transfer_node(node);
                }
            }
        }

        Ok(())
    }

    fn validate_render_pass_description(
        &self,
        render_pass_description: &RenderPassDescription<'_>,
    ) {
        check!(
            !render_pass_description.render_targets.is_empty()
                || render_pass_description.depth_stencil_attachment.is_some(),
            "Neither a render target and a depth stencil target is attached to the render pass!"
        );
        let framebuffer_size = if let Some(rt) = render_pass_description.render_targets.first() {
            rt.view.owner.extents
        } else {
            let rt = render_pass_description.depth_stencil_attachment.unwrap();
            rt.view.owner.extents
        };
        for rt in render_pass_description.render_targets {
            let image_name = self
                .device
                .hal
                .image_name(rt.view.owner)
                .expect("Invalid image handle");
            let image_name = image_name.as_deref().unwrap_or("Unknown");
            let error_message = format!(
                    "Render Target image '{}''s size is different than the expected framebuffer size! Expected {:?}, got {:?}", image_name,
                    framebuffer_size,
                    rt.view.owner.extents
                );
            check!(
                rt.view
                    .owner
                    .usage_flags
                    .contains(ImageUsageFlags::COLOR_ATTACHMENT),
                "Attachment does not contain COLOR_ATTACHMENT flag"
            );
            check!(
                rt.view.owner.extents.width == framebuffer_size.width
                    && rt.view.owner.extents.height == framebuffer_size.height,
                &error_message
            );
        }

        if let Some(rt) = &render_pass_description.depth_stencil_attachment {
            let image_name = self
                .device
                .hal
                .image_name(rt.view.owner)
                .expect("Invalid image handle");
            let image_name = image_name.as_deref().unwrap_or("Unknown");
            let error_message = format!(
                    "Depth Stencil Target image '{}''s size is different than the expected framebuffer size! Expected {:?}, got {:?}", image_name,
                    framebuffer_size,
                    rt.view.owner.extents
                );
            check!(
                rt.view
                    .owner
                    .usage_flags
                    .contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                "Depth Attachment does not contain DEPTH_STENCIL_ATTACHMENT flag"
            );
            check!(
                rt.view.owner.extents.width == framebuffer_size.width
                    && rt.view.owner.extents.height == framebuffer_size.height,
                &error_message
            );
        }

        check!(
            render_pass_description.render_area.offset.x as u32
                + render_pass_description.render_area.extents.width
                <= framebuffer_size.width
                && render_pass_description.render_area.offset.y as u32
                    + render_pass_description.render_area.extents.height
                    <= framebuffer_size.height,
            &format!(
                "The render area is too smaller/outflows the framebuffer! Got {:?}, expected at least {:?}",
                render_pass_description.render_area, framebuffer_size
            )
        );
    }

    fn set_binding_sets(&mut self, binding_sets: &[BindingSet]) {
        self.binding_sets = binding_sets.to_vec();
    }
}

impl CommandRecorder<Graphics> {
    pub fn begin_render_pass(
        &mut self,
        render_pass_description: &RenderPassDescription,
    ) -> MgpuResult<RenderPass> {
        #[cfg(debug_assertions)]
        self.validate_render_pass_description(render_pass_description);
        let label = render_pass_description.label.map(ToOwned::to_owned);

        let first_step = RenderStep {
            color_attachments: render_pass_description
                .render_targets
                .iter()
                .enumerate()
                .map(|(i, rt)| {
                    let access_mode = if rt.store_op == AttachmentStoreOp::Store {
                        AttachmentAccessMode::Write
                    } else {
                        AttachmentAccessMode::Read
                    };
                    RenderAttachmentReference {
                        access_mode,
                        index: i,
                    }
                })
                .collect(),
            depth_stencil_attachment: render_pass_description.depth_stencil_attachment.map(
                |attachment| {
                    let access_mode = if attachment.store_op == AttachmentStoreOp::Store {
                        AttachmentAccessMode::Write
                    } else {
                        AttachmentAccessMode::Read
                    };
                    DepthStencilAttachmentReference { access_mode }
                },
            ),
            commands: vec![],
        };
        let framebuffer_size = if let Some(rt) = render_pass_description.render_targets.first() {
            rt.view.owner.extents
        } else {
            render_pass_description
                .depth_stencil_attachment
                .unwrap()
                .view
                .owner
                .extents
        };
        let framebuffer_size = Extents2D {
            width: framebuffer_size.width,
            height: framebuffer_size.height,
        };
        Ok(RenderPass {
            command_recorder: self,
            info: RenderPassInfo {
                label,
                framebuffer: Framebuffer {
                    render_targets: render_pass_description.render_targets.to_vec(),
                    depth_stencil_target: render_pass_description
                        .depth_stencil_attachment
                        .map(ToOwned::to_owned),
                    extents: framebuffer_size,
                },
                steps: vec![first_step],
                render_area: render_pass_description.render_area,
            },
            pipeline: None,
            vertex_buffers: Default::default(),
            index_buffer: Default::default(),
        })
    }
}

impl<C: ComputeCommandRecorder> CommandRecorder<C> {
    pub fn begin_compute_pass(
        &mut self,
        compute_pass_description: &ComputePassDescription,
    ) -> ComputePass<C> {
        ComputePass {
            command_recorder: self,
            info: ComputePassInfo {
                label: compute_pass_description.label.map(ToOwned::to_owned),
                steps: vec![ComputeStep::default()],
            },
            pipeline: None,
        }
    }
}

impl<'c> RenderPass<'c> {
    pub fn set_pipeline(&mut self, pipeline: GraphicsPipeline) {
        self.pipeline = Some(pipeline);
    }

    pub fn set_vertex_buffers(&mut self, vertex_buffers: impl IntoIterator<Item = Buffer>) {
        self.vertex_buffers = vertex_buffers.into_iter().collect()
    }

    pub fn set_index_buffer(&mut self, index_buffer: Buffer) {
        self.index_buffer = Some(index_buffer);
    }

    pub fn set_binding_sets(&mut self, binding_sets: &[BindingSet]) {
        self.command_recorder.set_binding_sets(binding_sets);
    }

    pub fn draw(
        &mut self,
        vertices: usize,
        instances: usize,
        first_vertex: usize,
        first_instance: usize,
    ) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_state(false)?;

        let step = self.info.steps.last_mut().unwrap();
        step.commands.push(DrawCommand {
            pipeline: self.pipeline.unwrap(),
            vertex_buffers: self.vertex_buffers.clone(),
            index_buffer: self.index_buffer,
            binding_sets: self.command_recorder.binding_sets.clone(),
            draw_type: DrawType::Draw {
                vertices,
                instances,
                first_vertex,
                first_instance,
            },
        });
        Ok(())
    }

    pub fn draw_indexed(
        &mut self,
        indices: usize,
        instances: usize,
        first_index: usize,
        vertex_offset: i32,
        first_instance: usize,
    ) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_state(true)?;

        let step = self.info.steps.last_mut().unwrap();
        step.commands.push(DrawCommand {
            pipeline: self.pipeline.unwrap(),
            vertex_buffers: self.vertex_buffers.clone(),
            index_buffer: self.index_buffer,
            binding_sets: self.command_recorder.binding_sets.clone(),
            draw_type: DrawType::DrawIndexed {
                indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            },
        });
        Ok(())
    }

    fn validate_state(&self, check_index_buffer: bool) -> MgpuResult<()> {
        check!(
            self.pipeline.is_some(),
            "Issued a draw call without a pipeline set"
        );
        let pipeline = self.pipeline.unwrap();
        let pipeline_layout = self
            .command_recorder
            .device
            .hal
            .get_graphics_pipeline_layout(pipeline)?;

        for input in &pipeline_layout.vertex_stage.vertex_inputs {
            let vertex_buffer = self.vertex_buffers.get(input.location);
            check!(
                vertex_buffer.is_some(),
                &format!(
                    "Pipeline '{}' expects a vertex buffer at location {}, but none was bound",
                    pipeline_layout.label.as_deref().unwrap_or("Unknown"),
                    input.location
                )
            );
            let vertex_buffer = vertex_buffer.unwrap();
            check!(vertex_buffer.usage_flags.contains(BufferUsageFlags::VERTEX_BUFFER),
            &format!("Bound a vertex buffer at location {} that doesn't have the VERTEX_BUFFER usage flag", input.location));
        }

        if check_index_buffer {
            check!(
                self.index_buffer.is_some(),
                "Tried executing a draw_indexed without an index buffer"
            );
        }

        for set in &pipeline_layout.binding_sets_infos {
            let bound_set = self.command_recorder.binding_sets.get(set.set);
            check!(
                bound_set.is_some(),
                &format!(
                    "Pipeline '{}' expects a binding set at location {}, but none was bound",
                    pipeline_layout.label.as_deref().unwrap_or("Unknown"),
                    set.set
                )
            );
        }

        Ok(())
    }
}

impl<'c, C: ComputeCommandRecorder> ComputePass<'c, C> {
    pub fn set_pipeline(&mut self, pipeline: ComputePipeline) {
        self.pipeline = Some(pipeline);
    }

    pub fn set_binding_sets(&mut self, binding_sets: &[BindingSet]) {
        self.command_recorder.set_binding_sets(binding_sets);
    }

    pub fn dispatch(
        &mut self,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> MgpuResult<()> {
        self.validate_state()?;
        let last_command_step = self.info.steps.last_mut().unwrap();
        last_command_step.commands.push(DispatchCommand {
            pipeline: self.pipeline.unwrap(),
            dispatch_type: DispatchType::Dispatch(group_count_x, group_count_y, group_count_z),
            binding_sets: self.command_recorder.binding_sets.to_vec(),
        });
        Ok(())
    }

    fn validate_state(&self) -> MgpuResult<()> {
        check!(
            self.pipeline.is_some(),
            "Issued a dispatch call without a pipeline set"
        );
        let pipeline = self.pipeline.unwrap();
        let pipeline_layout = self
            .command_recorder
            .device
            .hal
            .get_compute_pipeline_layout(pipeline)?;
        for set in &pipeline_layout.binding_sets_infos {
            let bound_set = self.command_recorder.binding_sets.get(set.set);
            check!(
                bound_set.is_some(),
                &format!(
                    "Pipeline '{}' expects a binding set at location {}, but none was bound",
                    pipeline_layout.label.as_deref().unwrap_or("Unknown"),
                    set.set
                )
            );
        }

        Ok(())
    }
}
impl<'c> Drop for RenderPass<'c> {
    fn drop(&mut self) {
        if self.info.steps.is_empty() {
            return;
        }
        self.command_recorder.new_nodes.push(Node::RenderPass {
            info: std::mem::take(&mut self.info),
        })
    }
}

impl<'c, C: ComputeCommandRecorder> Drop for ComputePass<'c, C> {
    fn drop(&mut self) {
        if self.info.steps.is_empty() {
            return;
        }
        self.command_recorder.new_nodes.push(Node::ComputePass {
            info: std::mem::take(&mut self.info),
        })
    }
}

pub trait CommandRecorderType {
    const PREFERRED_QUEUE_TYPE: QueueType;
}

impl CommandRecorderType for Graphics {
    const PREFERRED_QUEUE_TYPE: QueueType = QueueType::Graphics;
}
impl CommandRecorderType for AsyncCompute {
    const PREFERRED_QUEUE_TYPE: QueueType = QueueType::AsyncCompute;
}

pub trait ComputeCommandRecorder: CommandRecorderType {}
impl ComputeCommandRecorder for Graphics {}
impl ComputeCommandRecorder for AsyncCompute {}
