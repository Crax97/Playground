use std::marker::PhantomData;

use crate::{
    hal::QueueType, rdg::Node, util::check, AttachmentStoreOp, BindingSet, Buffer,
    BufferUsageFlags, DepthStencilTarget, Device, Extents2D, GraphicsPipeline, MgpuResult, Rect2D,
    RenderPassDescription, RenderTarget,
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
pub(crate) struct RenderStep {
    pub(crate) color_attachments: Vec<RenderAttachmentReference>,
    pub(crate) depth_stencil_attachment: Option<DepthStencilAttachmentReference>,
    pub(crate) commands: Vec<DrawCommand>,
}

#[derive(Default, Debug, Hash)]
pub(crate) struct Framebuffer {
    pub(crate) render_targets: Vec<RenderTarget>,
    pub(crate) depth_stencil_target: Option<DepthStencilTarget>,
    pub(crate) extents: Extents2D,
}

#[derive(Default, Debug, Hash)]
pub(crate) struct RenderPassInfo {
    pub(crate) label: Option<String>,
    pub(crate) framebuffer: Framebuffer,
    pub(crate) render_area: Rect2D,
    pub(crate) steps: Vec<RenderStep>,
}

// The commands recorded on this command recorder will be put on the graphics queue
pub struct Graphics;

// The commands recorded on this command recorder will be put on the async compute queue
pub struct Compute;

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

impl<'c> RenderPass<'c> {
    pub fn set_vertex_buffers(&mut self, vertex_buffers: impl IntoIterator<Item = Buffer>) {
        self.vertex_buffers = vertex_buffers.into_iter().collect()
    }

    pub fn set_pipeline(&mut self, pipeline: GraphicsPipeline) {
        self.pipeline = Some(pipeline);
    }

    pub fn draw(
        &mut self,
        vertices: usize,
        instances: usize,
        first_vertex: usize,
        first_instance: usize,
    ) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_state()?;

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

    fn validate_state(&self) -> MgpuResult<()> {
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

        for set in &pipeline_layout.binding_sets {
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

pub trait CommandRecorderType {
    const PREFERRED_QUEUE_TYPE: QueueType;
}

pub(crate) struct ComputeParams {
    label: Option<String>,
}

impl CommandRecorderType for Graphics {
    const PREFERRED_QUEUE_TYPE: QueueType = QueueType::Graphics;
}
impl CommandRecorderType for Compute {
    const PREFERRED_QUEUE_TYPE: QueueType = QueueType::Graphics;
}
impl CommandRecorderType for AsyncCompute {
    const PREFERRED_QUEUE_TYPE: QueueType = QueueType::AsyncCompute;
}
