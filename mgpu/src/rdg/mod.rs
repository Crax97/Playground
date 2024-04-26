use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    hal::{
        AttachmentType, QueueType, Resource, ResourceAccessMode, ResourceInfo, ResourceTransition,
    },
    util::check,
    AttachmentStoreOp, Buffer, FilterMode, Image, ImageRegion, MgpuResult, RenderPassInfo,
    SwapchainImage,
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
    CopyBufferToImage {
        source: Buffer,
        dest: Image,
        source_offset: usize,
        dest_region: ImageRegion,
    },
    Blit {
        source: Image,
        source_region: ImageRegion,
        dest: Image,
        dest_region: ImageRegion,
        filter: FilterMode,
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

    // This node's predecessor on the same queue, if None this is the first node on the queue
    pub(crate) predecessor: Option<usize>,
}

#[derive(Default, Debug)]
pub struct Rdg {
    nodes: Vec<RdgNode>,
    last_on_graphics: Option<usize>,
    last_on_compute: Option<usize>,
    last_on_transfer: Option<usize>,
    ownerships: HashMap<Resource, QueueOwnership>,
}

#[derive(Debug)]
struct QueueOwnership {
    queue: QueueType,
    node: usize,
    access_mode: ResourceAccessMode,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct OwnershipTransfer {
    pub source: QueueType,
    pub source_usage: ResourceAccessMode,
    pub destination: QueueType,
    pub dest_usage: ResourceAccessMode,
    pub resource: ResourceInfo,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Pass {
    pub prequisites: Vec<ResourceTransition>,
    pub node_id: usize,
}

#[derive(Default, Debug)]
pub struct PassGroup {
    pub graphics_passes: Vec<Pass>,
    pub compute_passes: Vec<Pass>,
    pub transfer_passes: Vec<Pass>,
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
    pub topological_sorting: Vec<usize>,
}

impl Rdg {
    pub fn add_graphics_pass(&mut self, node: Node) {
        Self::add_on_queue(
            QueueType::Graphics,
            &mut self.nodes,
            node,
            &mut self.last_on_graphics,
        );
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
        Self::add_on_queue(
            QueueType::AsyncCompute,
            &mut self.nodes,
            node,
            &mut self.last_on_compute,
        );
    }

    pub fn add_async_transfer_node(&mut self, node: Node) {
        check!(
            !matches!(node, Node::RenderPass { .. }),
            "Cannot add a render pass node on a transfer queue!"
        );
        Self::add_on_queue(
            QueueType::AsyncTransfer,
            &mut self.nodes,
            node,
            &mut self.last_on_transfer,
        );
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
                Node::CopyBufferToImage { .. } => {}
                Node::Blit { .. } => {}
            }
            if stop {
                return;
            };
        }
    }

    pub fn clear(&mut self) {
        self.last_on_compute = None;
        self.last_on_graphics = None;
        self.last_on_transfer = None;
        self.nodes.clear();
    }

    fn add_on_queue(
        queue: QueueType,
        nodes: &mut Vec<RdgNode>,
        pass: Node,
        last_on_queue: &mut Option<usize>,
    ) {
        let toi = nodes.len();
        let node = pass.into_rdg(toi, queue, *last_on_queue);

        *last_on_queue = Some(toi);

        nodes.push(node);
    }

    pub fn compile(&mut self) -> MgpuResult<RdgCompiledGraph> {
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
            if let Some(predecessor) = node.predecessor {
                adjacency_list[predecessor].insert(node.global_index);
            }
        }

        adjacency_list
    }

    fn do_topological_sorting(&self, adjacency_list: &[HashSet<usize>]) -> MgpuResult<Vec<usize>> {
        let mut visited = Vec::with_capacity(adjacency_list.len());
        visited.resize(adjacency_list.len(), false);

        let mut node_depths = Vec::with_capacity(adjacency_list.len());
        node_depths.resize(adjacency_list.len(), usize::MAX);

        let mut sorted = Vec::with_capacity(adjacency_list.len());

        let mut to_visit = VecDeque::new();
        to_visit.push_front((0usize, 0usize));
        while let Some((node, depth)) = to_visit.pop_front() {
            if node_depths[node] != usize::MAX && depth > node_depths[node] {
                panic!("Loop!");
            }
            node_depths[node] = depth;
            if visited[node] {
                continue;
            }
            let children_to_visit = adjacency_list[node]
                .iter()
                .filter(|&&node| !visited[node])
                .collect::<Vec<_>>();
            if !children_to_visit.is_empty() {
                to_visit.push_front((node, depth));
                for &child in children_to_visit {
                    to_visit.push_front((child, depth + 1));
                }
            } else {
                visited[node] = true;
                sorted.push(node);
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
        let ownerships = &mut self.ownerships;
        let mut steps = Vec::<Step>::new();

        let mut barrier_info = Vec::<OwnershipTransfer>::default();
        let mut current_group = PassGroup::default();
        for &node in topological_sorting {
            let node_info = &self.nodes[node];
            let mut prequisites = vec![];
            for &read_resource in &node_info.reads {
                let ownership =
                    ownerships
                        .entry(read_resource.resource)
                        .or_insert(QueueOwnership {
                            queue: node_info.queue,
                            node: node_info.global_index,
                            access_mode: ResourceAccessMode::Undefined,
                        });

                ownership.node = node_info.global_index;
                if ownership.queue != node_info.queue {
                    barrier_info.push(OwnershipTransfer {
                        source: ownership.queue,
                        source_usage: ownership.access_mode,
                        destination: node_info.queue,
                        dest_usage: read_resource.access_mode,
                        resource: read_resource,
                    });
                    ownership.queue = node_info.queue;
                    ownership.access_mode = read_resource.access_mode;
                } else if ownership.access_mode != read_resource.access_mode {
                    prequisites.push(ResourceTransition {
                        resource: read_resource.resource,
                        old_usage: ownership.access_mode,
                        new_usage: read_resource.access_mode,
                    });
                    ownership.access_mode = read_resource.access_mode;
                }
            }

            for &written_resource in &node_info.writes {
                let ownership =
                    ownerships
                        .entry(written_resource.resource)
                        .or_insert(QueueOwnership {
                            queue: node_info.queue,
                            node: node_info.global_index,
                            access_mode: ResourceAccessMode::Undefined,
                        });

                ownership.node = node_info.global_index;
                if ownership.queue != node_info.queue {
                    barrier_info.push(OwnershipTransfer {
                        source: ownership.queue,
                        source_usage: ownership.access_mode,
                        destination: node_info.queue,
                        dest_usage: written_resource.access_mode,
                        resource: written_resource,
                    });

                    ownership.queue = node_info.queue;
                } else if ownership.access_mode != written_resource.access_mode {
                    prequisites.push(ResourceTransition {
                        resource: written_resource.resource,
                        old_usage: ownership.access_mode,
                        new_usage: written_resource.access_mode,
                    });
                }

                ownership.access_mode = written_resource.access_mode;
            }

            if !barrier_info.is_empty() {
                steps.push(Step::ExecutePasses(std::mem::take(&mut current_group)));
                steps.push(Step::OwnershipTransfer {
                    transfers: std::mem::take(&mut barrier_info),
                });
            }
            match node_info.queue {
                QueueType::Graphics => current_group.graphics_passes.push(Pass {
                    prequisites,
                    node_id: node_info.global_index,
                }),
                QueueType::AsyncCompute => current_group.compute_passes.push(Pass {
                    prequisites,
                    node_id: node_info.global_index,
                }),
                QueueType::AsyncTransfer => current_group.transfer_passes.push(Pass {
                    prequisites,
                    node_id: node_info.global_index,
                }),
            }
        }

        if !current_group.compute_passes.is_empty()
            || !current_group.graphics_passes.is_empty()
            || !current_group.transfer_passes.is_empty()
        {
            steps.push(Step::ExecutePasses(std::mem::take(&mut current_group)));
        }

        Ok(RdgCompiledGraph {
            sequence: steps,
            nodes: std::mem::take(&mut self.nodes),
            adjacency_list,
            topological_sorting: topological_sorting.to_vec(),
        })
    }
}

impl Node {
    fn into_rdg(self, toi: usize, queue: QueueType, predecessor: Option<usize>) -> RdgNode {
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

                let all_resource_bindings_read = info.steps.iter().flat_map(|step| {
                    step.commands.iter().flat_map(|cmd| {
                        cmd.binding_sets.iter().flat_map(|set| {
                            set.bindings.iter().filter_map(|b| match b.ty {
                                crate::BindingType::Sampler(_) => None,
                                crate::BindingType::UniformBuffer {
                                    buffer,
                                    offset,
                                    range,
                                } => Some(ResourceInfo {
                                    access_mode: ResourceAccessMode::ShaderRead,
                                    resource: Resource::Buffer {
                                        buffer,
                                        offset,
                                        size: range,
                                    },
                                }),
                                crate::BindingType::SampledImage { view, .. } => {
                                    Some(ResourceInfo {
                                        access_mode: ResourceAccessMode::ShaderRead,
                                        resource: Resource::Image {
                                            image: view.owner,
                                            subresource: view.subresource,
                                        },
                                    })
                                }
                            })
                        })
                    })
                });

                let attachments_read = info
                    .framebuffer
                    .render_targets
                    .iter()
                    .filter_map(|rt| match rt.store_op {
                        AttachmentStoreOp::DontCare => Some(ResourceInfo {
                            resource: Resource::Image {
                                image: rt.view.owner,
                                subresource: rt.view.subresource,
                            },
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
                                    resource: Resource::Image {
                                        image: dt.view.owner,
                                        subresource: dt.view.subresource,
                                    },
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
                                resource: Resource::Image {
                                    image: rt.view.owner,
                                    subresource: rt.view.subresource,
                                },
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
                                        resource: Resource::Image {
                                            image: dt.view.owner,
                                            subresource: dt.view.subresource,
                                        },
                                        access_mode: ResourceAccessMode::AttachmentWrite(
                                            AttachmentType::DepthStencil,
                                        ),
                                    })
                                }
                            }),
                    );
                (
                    attachments_read
                        .chain(all_buffers_read)
                        .chain(all_resource_bindings_read)
                        .collect(),
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
            Node::CopyBufferToImage {
                source,
                dest,
                source_offset,
                dest_region,
            } => (
                [ResourceInfo {
                    access_mode: ResourceAccessMode::TransferSrc,
                    resource: Resource::Buffer {
                        buffer: *source,
                        offset: *source_offset,
                        size: dest_region.extents.area() as usize * dest.format.byte_size(),
                    },
                }]
                .into(),
                [ResourceInfo {
                    access_mode: ResourceAccessMode::TransferDst,
                    resource: Resource::Image {
                        image: *dest,
                        subresource: dest_region.to_image_subresource(),
                    },
                }]
                .into(),
            ),
            Node::Blit {
                source,
                source_region,
                dest,
                dest_region,
                ..
            } => (
                [ResourceInfo {
                    resource: Resource::Image {
                        image: *source,
                        subresource: source_region.to_image_subresource(),
                    },
                    access_mode: ResourceAccessMode::TransferSrc,
                }]
                .into(),
                [ResourceInfo {
                    resource: Resource::Image {
                        image: *dest,
                        subresource: dest_region.to_image_subresource(),
                    },
                    access_mode: ResourceAccessMode::TransferDst,
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
            predecessor,
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
                    "Copy {:?}:{:?} -> {:?}:{:?} {} bytes",
                    source.id, source_offset, dest.id, dest_offset, size
                ),
                Node::CopyBufferToImage {
                    source,
                    dest,
                    source_offset,
                    dest_region,
                } => format!(
                    "Copy {} texels {}:{} -> {}",
                    dest_region.extents.area(),
                    source.id,
                    source_offset,
                    dest.id,
                ),
                Node::Blit {
                    source,
                    source_region,
                    dest,
                    dest_region,
                    filter,
                } => {
                    format!(
                        "{}Blit {} l{}m{} -> {} l{}m{}",
                        match filter {
                            FilterMode::Nearest => "Near",
                            FilterMode::Linear => "Lin",
                        },
                        source.id,
                        source_region.base_array_layer,
                        source_region.mip,
                        dest.id,
                        dest_region.base_array_layer,
                        dest_region.mip
                    )
                }
            };
            let label = format!("\t{} [label = \"{}\"];\n", node, node_label);
            nodes += label.as_str();

            let mut node_connections: String = Default::default();
            for child in children {
                node_connections += &format!("\t{} -> {};\n", node, child)
            }
            edges += node_connections.as_str();
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

    use crate::{rdg::Node, Buffer, DrawCommand, GraphicsPipeline, Image, ImageView, RenderStep};

    use super::Rdg;

    #[test]
    fn present() {
        let mut rdg = Rdg::default();

        let image_0 = Image {
            id: 0,
            usage_flags: Default::default(),
            extents: Default::default(),
            dimension: crate::ImageDimension::D1,
            num_mips: 1.try_into().unwrap(),
            array_layers: 1.try_into().unwrap(),
            samples: crate::SampleCount::One,
            format: crate::ImageFormat::Rgba8,
        };
        let view_0 = ImageView {
            owner: image_0,
            subresource: image_0.whole_subresource(),
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
    }
    #[test]
    fn ttg() {
        let mut rdg = Rdg::default();

        let buffer_src = Buffer {
            id: 0,
            usage_flags: Default::default(),
            size: 1000,
            memory_domain: crate::MemoryDomain::DeviceLocal,
        };
        let buffer_0 = Buffer {
            id: 1,
            usage_flags: Default::default(),
            size: 10,
            memory_domain: crate::MemoryDomain::DeviceLocal,
        };

        let buffer_1 = Buffer {
            id: 2,
            usage_flags: Default::default(),
            size: 10,
            memory_domain: crate::MemoryDomain::DeviceLocal,
        };

        let gp = GraphicsPipeline { id: 0 };

        rdg.add_async_transfer_node(Node::CopyBufferToBuffer {
            source: buffer_src,
            dest: buffer_0,
            source_offset: 0,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_transfer_node(Node::CopyBufferToBuffer {
            source: buffer_src,
            dest: buffer_1,
            source_offset: 10,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_graphics_pass(Node::RenderPass {
            info: crate::RenderPassInfo {
                label: None,
                framebuffer: Default::default(),
                render_area: Default::default(),
                steps: vec![RenderStep {
                    color_attachments: vec![],
                    depth_stencil_attachment: None,
                    commands: vec![DrawCommand {
                        pipeline: gp,
                        vertex_buffers: vec![buffer_0, buffer_1],
                        index_buffer: None,
                        binding_sets: vec![],
                        draw_type: crate::DrawType::Draw {
                            vertices: 1,
                            instances: 1,
                            first_vertex: 1,
                            first_instance: 1,
                        },
                    }],
                }],
            },
        });

        let compiled = rdg.compile().unwrap();
        println!("Topo {:?}", compiled.topological_sorting);
        println!("Adj {:?}", compiled.adjacency_list);
        println!("Dot\n{}", compiled.dump_dot());
        assert_eq!(compiled.sequence.len(), 3);
        let step_0 = &compiled.sequence[0];
        match step_0 {
            crate::rdg::Step::OwnershipTransfer { .. } => panic!("Barrier with only a present"),
            crate::rdg::Step::ExecutePasses(passes) => {
                assert_eq!(passes.transfer_passes.len(), 2);
                assert_eq!(passes.graphics_passes.len(), 0);
                assert_eq!(passes.compute_passes.len(), 0);
            }
        }
    }
}
