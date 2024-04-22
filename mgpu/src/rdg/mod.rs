use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    hal::{AttachmentType, QueueType, Resource, ResourceAccessMode, ResourceInfo},
    util::check,
    AttachmentStoreOp, Buffer, MgpuResult, RenderPassInfo, SwapchainImage,
};

#[derive(Debug)]
pub enum Node {
    RenderPass {
        info: RenderPassInfo,
    },
    CopyBufferToBuffer {
        source: Buffer,
        dest: Buffer,
        source_offset: usize,
        dest_offset: usize,
        size: usize,
    },
}

#[derive(Debug)]
pub struct RdgNode {
    // The resources read by this node
    pub(crate) reads: HashSet<ResourceInfo>,

    // The resources written by this node
    pub(crate) writes: HashSet<ResourceInfo>,

    // The operation executed by this node
    pub(crate) ty: Node,

    // A monotonically increasing index that identifies this pass
    pub(crate) global_index: usize,

    // The queue where this node should be placed on
    pub(crate) queue: QueueType,
}

#[derive(Default, Debug)]
pub struct Rdg {
    nodes: Vec<RdgNode>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct OwnershipTransfer {
    pub source: QueueType,
    pub destination: QueueType,
    pub resource: ResourceInfo,
}

#[derive(Default, Debug)]
pub struct PassGroup {
    pub graphics_nodes: Vec<usize>,
    pub compute_nodes: Vec<usize>,
    pub transfer_nodes: Vec<usize>,
}

#[derive(Debug)]
pub enum Step {
    OwnershipTransfer { transfers: Vec<OwnershipTransfer> },
    ExecutePasses(PassGroup),
}

#[derive(Default, Debug)]
pub struct RdgCompiledGraph {
    pub sequence: Vec<Step>,
    pub nodes: Vec<RdgNode>,
    pub adjacency_list: Vec<HashSet<usize>>,
}

impl Rdg {
    pub fn add_graphics_pass(&mut self, node: Node) {
        Self::add_on_queue(QueueType::Graphics, &mut self.nodes, node);
    }
    pub fn add_async_compute_pass(&mut self, node: Node) {
        check!(
            !matches!(node, Node::RenderPass { .. }),
            "Cannot add a render pass node on a transfer queue!"
        );
        check!(
            !matches!(node, Node::CopyBufferToBuffer { .. }),
            "Cannot add a copy pass node on a transfer queue!"
        );
        Self::add_on_queue(QueueType::AsyncCompute, &mut self.nodes, node);
    }

    pub fn add_async_transfer_node(&mut self, node: Node) {
        check!(
            !matches!(node, Node::RenderPass { .. }),
            "Cannot add a render pass node on a transfer queue!"
        );
        Self::add_on_queue(QueueType::AsyncTransfer, &mut self.nodes, node);
    }

    pub fn inform_present(&mut self, swapchain_image: SwapchainImage, _swapchain_id: u64) {
        for node_info in self.nodes.iter_mut().rev() {
            let mut stop = false;
            match &mut node_info.ty {
                Node::RenderPass { info } => info.steps.iter_mut().for_each(|step| {
                    step.color_attachments.iter_mut().for_each(|rta| {
                        if info.framebuffer.render_targets[rta.index].view == swapchain_image.view {
                            stop = true;
                            info.framebuffer.render_targets[rta.index].store_op =
                                AttachmentStoreOp::Present;
                        }
                    });
                }),
                Node::CopyBufferToBuffer { .. } => {}
            }
            if stop {
                return;
            };
        }
    }

    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    fn add_on_queue(queue: QueueType, nodes: &mut Vec<RdgNode>, pass: Node) {
        let toi = nodes.len();
        let node = pass.into_rdg(toi, queue);

        nodes.push(node);
    }

    pub fn compile(mut self) -> MgpuResult<RdgCompiledGraph> {
        if self.nodes.is_empty() {
            return Ok(RdgCompiledGraph::default());
        }
        let adjacency_list = self.create_adjacency_list();
        let topological_sorting = self.do_topological_sorting(&adjacency_list)?;
        self.create_compiled_graph(adjacency_list, &topological_sorting)
    }

    fn create_adjacency_list(&self) -> Vec<HashSet<usize>> {
        let mut adjacency_list = Vec::with_capacity(self.nodes.len());
        adjacency_list.resize(self.nodes.len(), HashSet::default());

        // Helper map to track which node currently owns a resource
        let mut ownerships = HashMap::<Resource, usize>::new();

        for node in &self.nodes {
            let mut inner_ownerships = HashMap::<Resource, usize>::new();
            for written_resource in &node.writes {
                ownerships.insert(written_resource.resource, node.global_index);
                inner_ownerships.insert(written_resource.resource, node.global_index);

                for other in &self.nodes {
                    if other.reads.iter().any(|r| r.resource == written_resource.resource)
                        // Avoid incorrectly adding a dependency when a resource changes in ownership
                        && inner_ownerships
                            .get(&written_resource.resource)
                            .is_some_and(|o| *o == node.global_index)
                    {
                        adjacency_list[node.global_index].insert(other.global_index);
                    }

                    for other_written in &other.writes {
                        inner_ownerships.insert(other_written.resource, other.global_index);
                    }
                }
            }
        }

        adjacency_list
    }

    fn do_topological_sorting(&self, adjacency_list: &[HashSet<usize>]) -> MgpuResult<Vec<usize>> {
        let mut visited = Vec::with_capacity(adjacency_list.len());
        visited.resize(adjacency_list.len(), false);

        let mut on_stack = Vec::with_capacity(adjacency_list.len());
        on_stack.resize(adjacency_list.len(), false);

        let mut sorted = Vec::with_capacity(adjacency_list.len());

        let mut to_visit = VecDeque::new();
        to_visit.push_front(0usize);

        while let Some(node) = to_visit.pop_front() {
            on_stack[node] = true;
            if visited[node] {
                on_stack[node] = false;
                sorted.push(node);
            } else {
                to_visit.push_front(node);
                let children = &adjacency_list[node];
                for &child in children {
                    if on_stack[child] {
                        panic!("Loop!");
                    }

                    if !visited[child] {
                        to_visit.push_front(child);
                    }
                }
                visited[node] = true;
            }
        }

        sorted.reverse();

        Ok(sorted)
    }

    fn create_compiled_graph(
        &mut self,
        adjacency_list: Vec<HashSet<usize>>,
        topological_sorting: &[usize],
    ) -> Result<RdgCompiledGraph, crate::MgpuError> {
        struct QueueOwnership {
            queue: QueueType,
            node: usize,
        }
        let mut ownerships = HashMap::<Resource, QueueOwnership>::new();
        let mut steps = Vec::<Step>::new();

        let mut barrier_info = Vec::<OwnershipTransfer>::default();
        let mut current_group = PassGroup::default();
        for &node in topological_sorting {
            let node_info = &self.nodes[node];
            for &read_resource in &node_info.reads {
                let ownership =
                    ownerships
                        .entry(read_resource.resource)
                        .or_insert(QueueOwnership {
                            queue: node_info.queue,
                            node: node_info.global_index,
                        });

                if ownership.queue != node_info.queue {
                    barrier_info.push(OwnershipTransfer {
                        source: ownership.queue,
                        destination: node_info.queue,
                        resource: read_resource,
                    });
                    ownership.node = node_info.global_index;
                    ownership.queue = node_info.queue;
                }
            }

            for &written_resource in &node_info.writes {
                let ownership =
                    ownerships
                        .entry(written_resource.resource)
                        .or_insert(QueueOwnership {
                            queue: node_info.queue,
                            node: node_info.global_index,
                        });

                ownership.node = node_info.global_index;
                ownership.queue = node_info.queue;
            }

            if !barrier_info.is_empty() {
                steps.push(Step::ExecutePasses(std::mem::take(&mut current_group)));
                steps.push(Step::OwnershipTransfer {
                    transfers: std::mem::take(&mut barrier_info),
                });
            }
            match node_info.queue {
                QueueType::Graphics => current_group.graphics_nodes.push(node_info.global_index),
                QueueType::AsyncCompute => current_group.compute_nodes.push(node_info.global_index),
                QueueType::AsyncTransfer => {
                    current_group.transfer_nodes.push(node_info.global_index)
                }
            }
        }

        if !current_group.compute_nodes.is_empty()
            || !current_group.graphics_nodes.is_empty()
            || !current_group.transfer_nodes.is_empty()
        {
            steps.push(Step::ExecutePasses(std::mem::take(&mut current_group)));
        }

        Ok(RdgCompiledGraph {
            sequence: steps,
            nodes: std::mem::take(&mut self.nodes),
            adjacency_list,
        })
    }
}

impl Node {
    fn into_rdg(self, toi: usize, queue: QueueType) -> RdgNode {
        let (read, write) = match &self {
            Node::RenderPass { info } => {
                let all_buffers_read = info
                    .steps
                    .iter()
                    .flat_map(|step| {
                        step.commands.iter().flat_map(|command| {
                            command
                                .vertex_buffers
                                .iter()
                                .chain(command.index_buffer.iter())
                                .copied()
                        })
                    })
                    .map(|buffer| ResourceInfo {
                        access_mode: ResourceAccessMode::VertexInput,
                        resource: Resource::Buffer {
                            buffer,
                            offset: 0,
                            size: buffer.size,
                        },
                    });

                // let all_buffers_written =
                //     info.steps
                //         .iter()
                //         .flat_map(|step| {})
                //         .map(|buffer| Resource::Buffer {
                //             buffer,
                //             offset: 0,
                //             size: buffer.size,
                //         });

                let attachments_read = info
                    .framebuffer
                    .render_targets
                    .iter()
                    .filter_map(|rt| match rt.store_op {
                        AttachmentStoreOp::DontCare => Some(ResourceInfo {
                            resource: Resource::ImageView { view: rt.view },
                            access_mode: ResourceAccessMode::AttachmentRead(AttachmentType::Color),
                        }),
                        AttachmentStoreOp::Store | AttachmentStoreOp::Present => None,
                    })
                    .chain(
                        info.framebuffer
                            .depth_stencil_target
                            .iter()
                            .filter_map(|dt| match dt.store_op {
                                AttachmentStoreOp::DontCare => Some(ResourceInfo {
                                    resource: Resource::ImageView { view: dt.view },
                                    access_mode: ResourceAccessMode::AttachmentRead(
                                        AttachmentType::DepthStencil,
                                    ),
                                }),
                                AttachmentStoreOp::Store | AttachmentStoreOp::Present => None,
                            }),
                    );
                let attachments_written = info
                    .framebuffer
                    .render_targets
                    .iter()
                    .filter_map(|rt| match rt.store_op {
                        AttachmentStoreOp::DontCare => None,
                        AttachmentStoreOp::Store | AttachmentStoreOp::Present => {
                            Some(ResourceInfo {
                                resource: Resource::ImageView { view: rt.view },
                                access_mode: ResourceAccessMode::AttachmentWrite(
                                    AttachmentType::Color,
                                ),
                            })
                        }
                    })
                    .chain(
                        info.framebuffer
                            .depth_stencil_target
                            .iter()
                            .filter_map(|dt| match dt.store_op {
                                AttachmentStoreOp::DontCare => None,
                                AttachmentStoreOp::Store | AttachmentStoreOp::Present => {
                                    Some(ResourceInfo {
                                        resource: Resource::ImageView { view: dt.view },
                                        access_mode: ResourceAccessMode::AttachmentWrite(
                                            AttachmentType::DepthStencil,
                                        ),
                                    })
                                }
                            }),
                    );
                (
                    attachments_read.chain(all_buffers_read).collect(),
                    attachments_written
                        // .chain(all_buffers_written)
                        .collect(),
                )
            }
            Node::CopyBufferToBuffer {
                source,
                dest,
                source_offset,
                dest_offset,
                size,
            } => (
                [ResourceInfo {
                    access_mode: ResourceAccessMode::TransferSrc,
                    resource: Resource::Buffer {
                        buffer: *source,
                        offset: *source_offset,
                        size: *size,
                    },
                }]
                .into(),
                [ResourceInfo {
                    access_mode: ResourceAccessMode::TransferDst,
                    resource: Resource::Buffer {
                        buffer: *dest,
                        offset: *dest_offset,
                        size: *size,
                    },
                }]
                .into(),
            ),
        };

        RdgNode {
            reads: read,
            writes: write,
            ty: self,
            global_index: toi,
            queue,
        }
    }
}

impl RdgCompiledGraph {
    pub fn dump_dot(&self) -> String {
        let mut nodes: String = "".into();
        let mut edges: String = "".into();

        for (node, children) in self.adjacency_list.iter().enumerate() {
            let info = &self.nodes[node];
            let node_label = match &info.ty {
                Node::RenderPass { info } => {
                    format!(
                        "RenderPass '{}'",
                        info.label.as_deref().unwrap_or("Unknown")
                    )
                }
                Node::CopyBufferToBuffer {
                    source,
                    dest,
                    source_offset,
                    dest_offset,
                    size,
                } => format!(
                    "Copy from {:?}:{:?} to {:?}:{:?} {} bytes",
                    source.id, source_offset, dest.id, dest_offset, size
                ),
            };
            let label = format!("\t{} [label = \"{}\"];\n", node, node_label);
            nodes += label.as_str();

            let mut line: String = format!("\t{}", node);
            for child in children {
                line += &format!("-> {}", child)
            }
            line += ";\n";
            edges += line.as_str();
        }
        let mut result = "digraph rdg {\n".into();

        result += nodes.as_str();
        result += edges.as_str();
        result += "}";

        result
    }
}

#[cfg(test)]
mod tests {

    use crate::{Image, ImageView};

    use super::Rdg;

    #[test]
    fn present() {
        let mut rdg = Rdg::default();

        let image_0 = Image {
            id: 0,
            usage_flags: Default::default(),
            extents: Default::default(),
            dimension: crate::ImageDimension::D1,
            mips: 1.try_into().unwrap(),
            array_layers: 1.try_into().unwrap(),
            samples: crate::SampleCount::One,
            format: crate::ImageFormat::Rgba8,
        };
        let view_0 = ImageView {
            owner: image_0,
            id: 0,
        };
        rdg.inform_present(
            crate::SwapchainImage {
                image: image_0,
                view: view_0,
                extents: Default::default(),
            },
            0,
        );
        let compiled = rdg.compile().unwrap();
        assert_eq!(compiled.sequence.len(), 0);
        let step_0 = &compiled.sequence[0];
        match step_0 {
            crate::rdg::Step::OwnershipTransfer { .. } => panic!("Barrier with only a present"),
            crate::rdg::Step::ExecutePasses(..) => {
                panic!("PassGroup with only a present!")
            }
        }
    }
}
