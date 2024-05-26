use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    hal::{
        AttachmentType, QueueType, Resource, ResourceAccessMode, ResourceInfo, ResourceTransition,
    },
    util::check,
    AttachmentStoreOp, Buffer, ComputePassInfo, FilterMode, Image, ImageRegion, ImageView,
    MgpuResult, RenderPassInfo, StorageAccessMode,
};

#[derive(Debug)]
pub enum Node {
    RenderPass {
        info: RenderPassInfo,
    },
    ComputePass {
        info: ComputePassInfo,
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
    Clear {
        target: ImageView,
        // If the image is a depth image, only the first value is used
        // IF the image is a stencil image, the value is transformed to an u8
        color: [f32; 4],
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
    pub async_compute_passes: Vec<Pass>,
    pub async_copy_passes: Vec<Pass>,
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
            "Cannot add a render pass node on a compute queue!"
        );
        check!(
            !matches!(node, Node::CopyBufferToBuffer { .. }),
            "Cannot add a copy pass node on a compute queue!"
        );
        Self::add_on_queue(
            QueueType::AsyncCompute,
            &mut self.nodes,
            node,
            &mut self.last_on_compute,
        );
    }

    pub fn add_async_copy_node(&mut self, node: Node) {
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

        let mut parent_list = Vec::<HashSet<_>>::with_capacity(self.nodes.len());
        parent_list.resize(self.nodes.len(), HashSet::<usize>::default());
        // Helper map to track which node currently owns a resource

        let mut ownerships = HashMap::<Resource, usize>::new();

        for node in &self.nodes {
            let mut inner_ownerships = HashMap::<Resource, usize>::new();
            for written_resource in &node.writes {
                ownerships.insert(written_resource.resource, node.global_index);
                inner_ownerships.insert(written_resource.resource, node.global_index);

                for other in &self.nodes {
                    if other.global_index == node.global_index {
                        continue;
                    }
                    let has_dependency_on_other_node = other.reads.iter().any(|r| {
                        if r.resource == written_resource.resource {
                            true
                        } else {
                            // if the other node reads a whole resource, and this node wrote a part of it
                            // there's a dependency
                            match (&r.resource, &written_resource.resource) {
                                (
                                    Resource::Image {
                                        image: ri,
                                        subresource: sr,
                                    },
                                    Resource::Image { image: wi, .. },
                                ) => ri == wi && *sr == ri.whole_subresource(),
                                (
                                    Resource::Buffer {
                                        buffer: sb,
                                        offset: so,
                                        size: ss,
                                    },
                                    Resource::Buffer { buffer: wb, .. },
                                ) => sb == wb && *so == 0 && *ss == sb.size,

                                _ => false,
                            }
                        }
                    });
                    if has_dependency_on_other_node
                        // Avoid incorrectly adding a dependency when a resource changes in ownership
                        && inner_ownerships
                            .get(&written_resource.resource)
                            .is_some_and(|o| *o == node.global_index)
                    {
                        adjacency_list[node.global_index].insert(other.global_index);
                        parent_list[other.global_index].insert(node.global_index);
                    }

                    for other_written in &other.writes {
                        inner_ownerships.insert(other_written.resource, other.global_index);
                    }
                }
            }
        }

        self.simplify_adjacency_list(&mut adjacency_list, parent_list);

        // Make sure each node depends on its parent on the queue
        for node in 0..self.nodes.len() {
            if let Some(pred) = self.nodes[node].predecessor {
                adjacency_list[pred].insert(node);
            }
        }

        adjacency_list
    }

    fn simplify_adjacency_list(
        &self,
        adjacency_list: &mut Vec<HashSet<usize>>,
        parent_list: Vec<HashSet<usize>>,
    ) {
        *adjacency_list = Vec::with_capacity(self.nodes.len());
        adjacency_list.resize(self.nodes.len(), HashSet::default());
        #[allow(clippy::needless_range_loop)]
        for node in 0..self.nodes.len() {
            let parents: &HashSet<usize> = &parent_list[node];

            let mut nearest_gfx: Option<usize> = None;
            let mut nearest_copy: Option<usize> = None;
            let mut nearest_compute: Option<usize> = None;

            for &parent in parents {
                let parent_node = &self.nodes[parent];
                let nearest = match parent_node.queue {
                    QueueType::Graphics => &mut nearest_gfx,
                    QueueType::AsyncCompute => &mut nearest_copy,
                    QueueType::AsyncTransfer => &mut nearest_compute,
                };

                nearest.replace(nearest.unwrap_or_default().max(parent));
            }

            let mut nearest_parents = HashSet::new();
            if let Some(n) = nearest_gfx {
                nearest_parents.insert(n);
            }
            if let Some(n) = nearest_compute {
                nearest_parents.insert(n);
            }
            if let Some(n) = nearest_copy {
                nearest_parents.insert(n);
            }

            for parent in nearest_parents {
                adjacency_list[parent].insert(node);
            }
        }
    }

    fn do_topological_sorting(&self, adjacency_list: &[HashSet<usize>]) -> MgpuResult<Vec<usize>> {
        #[derive(Clone, Copy, Debug, Default)]
        struct NodeToVisit {
            node_index: usize,
            first_encouter_depth: usize,
        }

        let mut visited = Vec::with_capacity(adjacency_list.len());
        visited.resize(adjacency_list.len(), false);

        let mut node_depths = Vec::with_capacity(adjacency_list.len());
        node_depths.resize(adjacency_list.len(), usize::MAX);

        let mut sorted = Vec::with_capacity(adjacency_list.len());

        let mut to_visit = VecDeque::new();
        to_visit.push_front(NodeToVisit {
            node_index: 0,
            first_encouter_depth: 0,
        });
        while let Some(node_to_visit) = to_visit.pop_front() {
            let NodeToVisit {
                node_index: node,
                first_encouter_depth: depth,
            } = node_to_visit;
            if node_depths[node] != usize::MAX && depth > node_depths[node] {
                panic!(
                    "Encountered a loop while sorting the nodes for execution
                    On node
                    {:#?}
                    first encountered at depth {} and then at depth {}",
                    self.nodes[node], node_depths[node], depth
                );
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
                to_visit.push_front(node_to_visit);
                for &child in children_to_visit {
                    to_visit.push_front(NodeToVisit {
                        node_index: child,
                        first_encouter_depth: depth + 1,
                    });
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
                QueueType::AsyncCompute => current_group.async_compute_passes.push(Pass {
                    prequisites,
                    node_id: node_info.global_index,
                }),
                QueueType::AsyncTransfer => current_group.async_copy_passes.push(Pass {
                    prequisites,
                    node_id: node_info.global_index,
                }),
            }
        }

        if !current_group.async_compute_passes.is_empty()
            || !current_group.graphics_passes.is_empty()
            || !current_group.async_copy_passes.is_empty()
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

                let all_resource_bindings_written = info.steps.iter().flat_map(|step| {
                    step.commands.iter().flat_map(|cmd| {
                        cmd.binding_sets.iter().flat_map(|set| {
                            set.bindings.iter().filter_map(|b| match b.ty {
                                crate::BindingType::Sampler(_) => None,
                                crate::BindingType::UniformBuffer { .. } => None,
                                crate::BindingType::StorageBuffer {
                                    buffer,
                                    offset,
                                    range,
                                    access_mode,
                                } => {
                                    if access_mode != StorageAccessMode::Read {
                                        Some(ResourceInfo {
                                            resource: Resource::Buffer {
                                                buffer,
                                                offset,
                                                size: range,
                                            },
                                            access_mode: ResourceAccessMode::ShaderWrite(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
                                }
                                crate::BindingType::SampledImage { .. } => None,
                                crate::BindingType::StorageImage { view, access_mode } => {
                                    if access_mode != StorageAccessMode::Read {
                                        Some(ResourceInfo {
                                            resource: Resource::Image {
                                                image: view.owner,
                                                subresource: view.subresource,
                                            },
                                            access_mode: ResourceAccessMode::ShaderWrite(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
                                }
                            })
                        })
                    })
                });
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
                                    access_mode: ResourceAccessMode::ShaderRead(b.visibility),
                                    resource: Resource::Buffer {
                                        buffer,
                                        offset,
                                        size: range,
                                    },
                                }),

                                crate::BindingType::StorageBuffer {
                                    buffer,
                                    offset,
                                    range,
                                    access_mode,
                                } => {
                                    if access_mode == StorageAccessMode::Read {
                                        Some(ResourceInfo {
                                            resource: Resource::Buffer {
                                                buffer,
                                                offset,
                                                size: range,
                                            },
                                            access_mode: ResourceAccessMode::ShaderRead(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
                                }
                                crate::BindingType::SampledImage { view, .. } => {
                                    Some(ResourceInfo {
                                        access_mode: ResourceAccessMode::ShaderRead(b.visibility),
                                        resource: Resource::Image {
                                            image: view.owner,
                                            subresource: view.subresource,
                                        },
                                    })
                                }
                                crate::BindingType::StorageImage { view, access_mode } => {
                                    if access_mode == StorageAccessMode::Read {
                                        Some(ResourceInfo {
                                            resource: Resource::Image {
                                                image: view.owner,
                                                subresource: view.subresource,
                                            },
                                            access_mode: ResourceAccessMode::ShaderRead(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
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
                        .chain(all_resource_bindings_written)
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
            Node::Clear { target, .. } => (
                [].into(),
                [ResourceInfo {
                    resource: Resource::Image {
                        image: target.owner,
                        subresource: target.subresource,
                    },
                    access_mode: ResourceAccessMode::TransferDst,
                }]
                .into(),
            ),
            Node::ComputePass { info } => {
                let all_resource_bindings_written = info.steps.iter().flat_map(|step| {
                    step.commands.iter().flat_map(|cmd| {
                        cmd.binding_sets.iter().flat_map(|set| {
                            set.bindings.iter().filter_map(|b| match b.ty {
                                crate::BindingType::Sampler(_) => None,
                                crate::BindingType::UniformBuffer { .. } => None,
                                crate::BindingType::StorageBuffer {
                                    buffer,
                                    offset,
                                    range,
                                    access_mode,
                                } => {
                                    if access_mode != StorageAccessMode::Read {
                                        Some(ResourceInfo {
                                            resource: Resource::Buffer {
                                                buffer,
                                                offset,
                                                size: range,
                                            },
                                            access_mode: ResourceAccessMode::ShaderWrite(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
                                }
                                crate::BindingType::SampledImage { .. } => None,
                                crate::BindingType::StorageImage { view, access_mode } => {
                                    if access_mode != StorageAccessMode::Read {
                                        Some(ResourceInfo {
                                            resource: Resource::Image {
                                                image: view.owner,
                                                subresource: view.subresource,
                                            },
                                            access_mode: ResourceAccessMode::ShaderWrite(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
                                }
                            })
                        })
                    })
                });
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
                                    access_mode: ResourceAccessMode::ShaderRead(b.visibility),
                                    resource: Resource::Buffer {
                                        buffer,
                                        offset,
                                        size: range,
                                    },
                                }),
                                crate::BindingType::StorageBuffer {
                                    buffer,
                                    offset,
                                    range,
                                    access_mode,
                                } => {
                                    if access_mode == StorageAccessMode::Read {
                                        Some(ResourceInfo {
                                            resource: Resource::Buffer {
                                                buffer,
                                                offset,
                                                size: range,
                                            },
                                            access_mode: ResourceAccessMode::ShaderRead(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
                                }

                                crate::BindingType::SampledImage { view, .. } => {
                                    Some(ResourceInfo {
                                        access_mode: ResourceAccessMode::ShaderRead(b.visibility),
                                        resource: Resource::Image {
                                            image: view.owner,
                                            subresource: view.subresource,
                                        },
                                    })
                                }
                                crate::BindingType::StorageImage { view, access_mode } => {
                                    if access_mode != StorageAccessMode::Write {
                                        Some(ResourceInfo {
                                            resource: Resource::Image {
                                                image: view.owner,
                                                subresource: view.subresource,
                                            },
                                            access_mode: ResourceAccessMode::ShaderRead(
                                                b.visibility,
                                            ),
                                        })
                                    } else {
                                        None
                                    }
                                }
                            })
                        })
                    })
                });

                (
                    all_resource_bindings_read.collect(),
                    all_resource_bindings_written.collect(),
                )
            }
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
    fn node_name(info: &RdgNode) -> String {
        match &info.ty {
            Node::RenderPass { info } => {
                format!(
                    "RenderPass '{}'",
                    info.label.as_deref().unwrap_or("Unknown")
                )
            }
            Node::ComputePass { info } => format!(
                "ComputePass '{}'",
                info.label.as_deref().unwrap_or("Unknown")
            ),
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
            Node::Clear { target, color } => {
                format!("Clear {} color {:?}", target.id, color)
            }
        }
    }

    pub fn dump_dot(&self) -> String {
        let mut content = "digraph rdg {\n".into();
        let mut edges = "".into();

        let mut subgraph_id = 0;
        for step in &self.sequence {
            match step {
                Step::OwnershipTransfer { .. } => {}
                Step::ExecutePasses(pass) => {
                    let mut subgraph = format!("\tsubgraph clusterStep{} {{\n", subgraph_id);
                    subgraph_id += 1;
                    self.extract_pass_info(
                        &pass.graphics_passes,
                        subgraph_id,
                        "Graphics",
                        &mut subgraph,
                        &mut edges,
                    );
                    self.extract_pass_info(
                        &pass.async_compute_passes,
                        subgraph_id,
                        "Compute",
                        &mut subgraph,
                        &mut edges,
                    );
                    self.extract_pass_info(
                        &pass.async_copy_passes,
                        subgraph_id,
                        "Copy",
                        &mut subgraph,
                        &mut edges,
                    );
                    subgraph += "\t};\n";

                    content += subgraph.as_str();
                }
            }
        }

        content += edges.as_str();
        content += "}";
        content
    }

    #[cfg(feature = "rdg_to_svg")]
    pub fn save_to_svg(&self, path: &str) {
        use layout::backends::svg::SVGWriter;
        use layout::gv::*;
        use log::{error, info};
        let content = self.dump_dot();
        let mut svg = SVGWriter::new();

        let mut parser = DotParser::new(&content);

        let mut graph = if let Ok(graph) = parser.process() {
            let mut gb = GraphBuilder::new();
            gb.visit_graph(&graph);
            gb.get()
        } else {
            error!(
                "Failed to parse dot graph, source
            {content}
            "
            );
            return;
        };

        if graph.dag.is_empty() {
            info!("save_to_svg: the graph was empty");
            return;
        }
        graph.do_it(false, false, false, &mut svg);
        let content = svg.finalize();

        if let Err(e) = std::fs::write(path, &content) {
            error!(
                "Failed to write svg file to {path}, error {e:?}, content
            {}",
                content
            );
        }
    }

    fn extract_pass_info(
        &self,
        passes: &[Pass],
        pass_id: usize,
        prefix: &str,
        nodes: &mut String,
        edges: &mut String,
    ) {
        if !passes.is_empty() {
            let mut node = format!(
                "\t\tsubgraph cluster{}{} {{\n\t\t\tlabel=\"{}{}\";\n",
                prefix, pass_id, prefix, pass_id
            );
            for gfx in passes {
                let info = &self.nodes[gfx.node_id];
                let name = Self::node_name(info);
                node += &format!("\t\t\t{} [label=\"{}\"];\n", gfx.node_id, name);

                for child in self.adjacency_list[gfx.node_id].iter() {
                    *edges += &format!("\t{} -> {};\n", gfx.node_id, child);
                }
            }
            node += "\t\t};\n";
            *nodes += node.as_str();
        }
    }
}

#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicU64;

    use crate::{
        rdg::Node, Binding, BindingSet, Buffer, ComputeStep, DispatchCommand, DrawCommand,
        Extents2D, Extents3D, GraphicsPipeline, Image, ImageCreationFlags, ImageFormat,
        ImageSubresource, ImageUsageFlags, ImageView, Rect2D, RenderAttachmentReference,
        RenderStep, RenderTarget,
    };

    use super::Rdg;
    static BUFFER_ID: AtomicU64 = AtomicU64::new(0);
    static IMAGE_ID: AtomicU64 = AtomicU64::new(0);
    static IMAGE_VIEW_ID: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy)]
    struct BufferBuilder {
        buffer: Buffer,
    }

    impl BufferBuilder {
        fn new() -> Self {
            Self {
                buffer: Buffer {
                    id: BUFFER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    usage_flags: Default::default(),
                    size: 100,
                    memory_domain: crate::MemoryDomain::Gpu,
                },
            }
        }
        fn build(self) -> Buffer {
            self.buffer
        }
    }

    #[derive(Clone, Copy)]
    struct ImageBuilder {
        image: Image,
    }
    impl ImageBuilder {
        fn new() -> Self {
            Self {
                image: Image {
                    id: IMAGE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    usage_flags: ImageUsageFlags::default(),
                    extents: Extents3D {
                        width: 512,
                        height: 512,
                        depth: 1,
                    },
                    dimension: crate::ImageDimension::D2,
                    num_mips: 1.try_into().unwrap(),
                    array_layers: 1.try_into().unwrap(),
                    samples: crate::SampleCount::One,
                    format: crate::ImageFormat::Rgba8,
                    creation_flags: ImageCreationFlags::default(),
                },
            }
        }

        #[allow(dead_code)]
        fn width_height(mut self, width: u32, height: u32) -> Self {
            self.image.extents.width = width;
            self.image.extents.height = height;
            self
        }

        #[allow(dead_code)]
        fn format(mut self, format: ImageFormat) -> Self {
            self.image.format = format;
            self
        }

        fn build(self) -> Image {
            self.image
        }
    }
    #[derive(Clone, Copy)]
    struct ImageViewBuilder {
        view: ImageView,
    }
    impl ImageViewBuilder {
        fn new(owner: Image) -> Self {
            Self {
                view: ImageView {
                    owner,
                    subresource: ImageSubresource {
                        mip: 0,
                        num_mips: owner.num_mips,
                        base_array_layer: 0,
                        num_layers: owner.array_layers,
                    },
                    id: IMAGE_VIEW_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                },
            }
        }

        fn build(self) -> ImageView {
            self.view
        }
    }

    #[test]
    fn ttg() {
        let mut rdg = Rdg::default();

        let buffer_src = Buffer {
            id: 0,
            usage_flags: Default::default(),
            size: 1000,
            memory_domain: crate::MemoryDomain::Gpu,
        };
        let buffer_0 = Buffer {
            id: 1,
            usage_flags: Default::default(),
            size: 10,
            memory_domain: crate::MemoryDomain::Gpu,
        };

        let buffer_1 = Buffer {
            id: 2,
            usage_flags: Default::default(),
            size: 10,
            memory_domain: crate::MemoryDomain::Gpu,
        };

        let gp = GraphicsPipeline { id: 0 };

        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_src,
            dest: buffer_0,
            source_offset: 0,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_src,
            dest: buffer_1,
            source_offset: 10,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_graphics_pass(Node::RenderPass {
            info: crate::RenderPassInfo {
                label: None,
                flags: Default::default(),
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
                        push_constants: None,
                        label: None,

                        scissor_rect: None,
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
                assert_eq!(passes.async_copy_passes.len(), 2);
                assert_eq!(passes.graphics_passes.len(), 0);
                assert_eq!(passes.async_compute_passes.len(), 0);
            }
        }
    }

    #[test]
    fn compute_1() {
        /*
        0. Copy buffer S -> buffer A
        1. Copy buffer S -> buffer B
        2. Copy buffer S -> buffer C
        3. Copy buffer S -> image D
        4. Compute uses C to write image F
        5. Render pass uses A, B, F, D to write P
        Topological sorting should impose that 5 is executed after (4, 3, 1, 0)
        and that 4 is executed before 2
        */
        let buffer_s = BufferBuilder::new().build();
        let buffer_a = BufferBuilder::new().build();
        let buffer_b = BufferBuilder::new().build();
        let buffer_c = BufferBuilder::new().build();
        let image_d = ImageBuilder::new().build();
        let image_f = ImageBuilder::new().build();
        let image_p = ImageBuilder::new().build();
        let view_d = ImageViewBuilder::new(image_d).build();
        let view_f = ImageViewBuilder::new(image_f).build();
        let view_p = ImageViewBuilder::new(image_p).build();

        let mut rdg = Rdg::default();
        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_s,
            dest: buffer_a,
            source_offset: 0,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_s,
            dest: buffer_b,
            source_offset: 10,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_s,
            dest: buffer_c,
            source_offset: 20,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_copy_node(Node::CopyBufferToImage {
            source: buffer_s,
            dest: image_d,
            source_offset: 30,
            dest_region: image_d.whole_region(),
        });
        rdg.add_async_compute_pass(Node::ComputePass {
            info: crate::ComputePassInfo {
                label: None,
                steps: vec![ComputeStep {
                    commands: vec![DispatchCommand {
                        pipeline: crate::ComputePipeline { id: 0 },
                        binding_sets: vec![BindingSet {
                            id: 1,
                            bindings: vec![
                                Binding {
                                    binding: 0,
                                    ty: crate::BindingType::UniformBuffer {
                                        buffer: buffer_c,
                                        offset: 0,
                                        range: 10,
                                    },
                                    visibility: Default::default(),
                                },
                                Binding {
                                    binding: 1,
                                    ty: crate::BindingType::StorageImage {
                                        view: view_f,
                                        access_mode: crate::StorageAccessMode::Write,
                                    },
                                    visibility: Default::default(),
                                },
                            ],
                        }],
                        push_constants: None,
                        dispatch_type: crate::DispatchType::Dispatch(1, 1, 1),

                        label: None,
                    }],
                }],
            },
        });

        rdg.add_graphics_pass(Node::RenderPass {
            info: crate::RenderPassInfo {
                label: None,
                flags: Default::default(),
                framebuffer: crate::Framebuffer {
                    render_targets: vec![RenderTarget {
                        view: view_p,
                        sample_count: crate::SampleCount::One,
                        load_op: crate::RenderTargetLoadOp::DontCare,
                        store_op: crate::AttachmentStoreOp::Store,
                    }],
                    depth_stencil_target: None,
                    extents: Extents2D::default(),
                },
                render_area: Rect2D::default(),
                steps: vec![RenderStep {
                    color_attachments: vec![RenderAttachmentReference {
                        index: 0,
                        access_mode: crate::AttachmentAccessMode::Write,
                    }],
                    depth_stencil_attachment: None,
                    commands: vec![DrawCommand {
                        pipeline: GraphicsPipeline { id: 0 },
                        vertex_buffers: vec![buffer_a],
                        index_buffer: Some(buffer_b),
                        binding_sets: vec![BindingSet {
                            id: 1,
                            bindings: vec![
                                Binding {
                                    binding: 0,
                                    ty: crate::BindingType::SampledImage { view: view_f },
                                    visibility: Default::default(),
                                },
                                Binding {
                                    binding: 1,
                                    ty: crate::BindingType::SampledImage { view: view_d },
                                    visibility: Default::default(),
                                },
                            ],
                        }],
                        push_constants: None,
                        draw_type: crate::DrawType::DrawIndexed {
                            indices: 10,
                            instances: 1,
                            first_index: 0,
                            vertex_offset: 0,
                            first_instance: 0,
                        },

                        label: None,

                        scissor_rect: None,
                    }],
                }],
            },
        });

        let compiled = rdg.compile().unwrap();
        /*
        Topological sorting should impose that 5 is executed after (4, 3, 1, 0)
        and that 4 is executed before 2
        */
        assert!(*compiled.topological_sorting.last().unwrap() == 5);
        assert!(
            compiled
                .topological_sorting
                .iter()
                .position(|&i| i == 4)
                .unwrap()
                > compiled
                    .topological_sorting
                    .iter()
                    .position(|&i| i == 2)
                    .unwrap()
        );
    }

    #[test]
    fn post_process() {
        /*
        0. Copy buffer S -> buffer A
        1. Copy buffer S -> buffer B
        2. Copy buffer S -> buffer C
        3. Copy buffer S -> image D
        4. Compute uses C to write image F
        5. Render pass uses A, B, F, D to write P1
        6. Render pass uses P1 to write P2
        7. Render pass uses P2 to write P1 again
        Topological sorting should impose that 5 is executed after (4, 3, 1, 0)
        and that 4 is executed before 2
        */
        let buffer_s = BufferBuilder::new().build();
        let buffer_a = BufferBuilder::new().build();
        let buffer_b = BufferBuilder::new().build();
        let buffer_c = BufferBuilder::new().build();
        let image_d = ImageBuilder::new().build();
        let image_f = ImageBuilder::new().build();
        let image_p1 = ImageBuilder::new().build();
        let image_p2 = ImageBuilder::new().build();
        let view_d = ImageViewBuilder::new(image_d).build();
        let view_f = ImageViewBuilder::new(image_f).build();
        let view_p1 = ImageViewBuilder::new(image_p1).build();
        let view_p2 = ImageViewBuilder::new(image_p2).build();

        let mut rdg = Rdg::default();
        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_s,
            dest: buffer_a,
            source_offset: 0,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_s,
            dest: buffer_b,
            source_offset: 10,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_copy_node(Node::CopyBufferToBuffer {
            source: buffer_s,
            dest: buffer_c,
            source_offset: 20,
            dest_offset: 0,
            size: 10,
        });
        rdg.add_async_copy_node(Node::CopyBufferToImage {
            source: buffer_s,
            dest: image_d,
            source_offset: 30,
            dest_region: image_d.whole_region(),
        });
        rdg.add_async_compute_pass(Node::ComputePass {
            info: crate::ComputePassInfo {
                label: None,
                steps: vec![ComputeStep {
                    commands: vec![DispatchCommand {
                        pipeline: crate::ComputePipeline { id: 0 },
                        binding_sets: vec![BindingSet {
                            id: 1,
                            bindings: vec![
                                Binding {
                                    binding: 0,
                                    ty: crate::BindingType::UniformBuffer {
                                        buffer: buffer_c,
                                        offset: 0,
                                        range: 10,
                                    },
                                    visibility: Default::default(),
                                },
                                Binding {
                                    binding: 1,
                                    ty: crate::BindingType::StorageImage {
                                        view: view_f,
                                        access_mode: crate::StorageAccessMode::Write,
                                    },
                                    visibility: Default::default(),
                                },
                            ],
                        }],
                        push_constants: None,
                        dispatch_type: crate::DispatchType::Dispatch(1, 1, 1),
                        label: None,
                    }],
                }],
            },
        });

        rdg.add_graphics_pass(Node::RenderPass {
            info: crate::RenderPassInfo {
                label: None,
                flags: Default::default(),
                framebuffer: crate::Framebuffer {
                    render_targets: vec![RenderTarget {
                        view: view_p1,
                        sample_count: crate::SampleCount::One,
                        load_op: crate::RenderTargetLoadOp::DontCare,
                        store_op: crate::AttachmentStoreOp::Store,
                    }],
                    depth_stencil_target: None,
                    extents: Extents2D::default(),
                },
                render_area: Rect2D::default(),
                steps: vec![RenderStep {
                    color_attachments: vec![RenderAttachmentReference {
                        index: 0,
                        access_mode: crate::AttachmentAccessMode::Write,
                    }],
                    depth_stencil_attachment: None,
                    commands: vec![DrawCommand {
                        pipeline: GraphicsPipeline { id: 0 },
                        vertex_buffers: vec![buffer_a],
                        index_buffer: Some(buffer_b),
                        binding_sets: vec![BindingSet {
                            id: 1,
                            bindings: vec![
                                Binding {
                                    binding: 0,
                                    ty: crate::BindingType::SampledImage { view: view_f },
                                    visibility: Default::default(),
                                },
                                Binding {
                                    binding: 1,
                                    ty: crate::BindingType::SampledImage { view: view_d },
                                    visibility: Default::default(),
                                },
                            ],
                        }],
                        push_constants: None,
                        draw_type: crate::DrawType::DrawIndexed {
                            indices: 10,
                            instances: 1,
                            first_index: 0,
                            vertex_offset: 0,
                            first_instance: 0,
                        },
                        label: None,

                        scissor_rect: None,
                    }],
                }],
            },
        });
        rdg.add_graphics_pass(Node::RenderPass {
            info: crate::RenderPassInfo {
                label: None,
                flags: Default::default(),
                framebuffer: crate::Framebuffer {
                    render_targets: vec![RenderTarget {
                        view: view_p2,
                        sample_count: crate::SampleCount::One,
                        load_op: crate::RenderTargetLoadOp::DontCare,
                        store_op: crate::AttachmentStoreOp::Store,
                    }],
                    depth_stencil_target: None,
                    extents: Extents2D::default(),
                },
                render_area: Rect2D::default(),
                steps: vec![RenderStep {
                    color_attachments: vec![RenderAttachmentReference {
                        index: 0,
                        access_mode: crate::AttachmentAccessMode::Write,
                    }],
                    depth_stencil_attachment: None,
                    commands: vec![DrawCommand {
                        pipeline: GraphicsPipeline { id: 0 },
                        vertex_buffers: vec![],
                        index_buffer: None,
                        binding_sets: vec![BindingSet {
                            id: 1,
                            bindings: vec![Binding {
                                binding: 0,
                                ty: crate::BindingType::SampledImage { view: view_p1 },
                                visibility: Default::default(),
                            }],
                        }],
                        push_constants: None,
                        draw_type: crate::DrawType::DrawIndexed {
                            indices: 10,
                            instances: 1,
                            first_index: 0,
                            vertex_offset: 0,
                            first_instance: 0,
                        },
                        label: None,
                        scissor_rect: None,,
                    }],
                }],
            },
        });
        rdg.add_graphics_pass(Node::RenderPass {
            info: crate::RenderPassInfo {
                label: None,
                flags: Default::default(),
                framebuffer: crate::Framebuffer {
                    render_targets: vec![RenderTarget {
                        view: view_p1,
                        sample_count: crate::SampleCount::One,
                        load_op: crate::RenderTargetLoadOp::DontCare,
                        store_op: crate::AttachmentStoreOp::Store,
                    }],
                    depth_stencil_target: None,
                    extents: Extents2D::default(),
                },
                render_area: Rect2D::default(),
                steps: vec![RenderStep {
                    color_attachments: vec![RenderAttachmentReference {
                        index: 0,
                        access_mode: crate::AttachmentAccessMode::Write,
                    }],
                    depth_stencil_attachment: None,
                    commands: vec![DrawCommand {
                        pipeline: GraphicsPipeline { id: 0 },
                        vertex_buffers: vec![],
                        index_buffer: None,
                        binding_sets: vec![BindingSet {
                            id: 1,
                            bindings: vec![Binding {
                                binding: 0,
                                ty: crate::BindingType::SampledImage { view: view_p2 },
                                visibility: Default::default(),
                            }],
                        }],
                        push_constants: None,
                        draw_type: crate::DrawType::DrawIndexed {
                            indices: 10,
                            instances: 1,
                            first_index: 0,
                            vertex_offset: 0,
                            first_instance: 0,
                        },
                        label: None,
                        scissor_rect: None,
                    }],
                }],
            },
        });
        let compiled = rdg.compile().unwrap();
        let index_of_pass = |id| {
            compiled
                .topological_sorting
                .iter()
                .position(|&i| i == id)
                .unwrap()
        };
        /*
        Topological sorting should impose that 5 is executed after (4, 3, 1, 0),
        7 after 6 and 6 after 5 and that 4 is executed before 2
        */
        assert!(*compiled.topological_sorting.last().unwrap() == 7);
        assert!(index_of_pass(7) > index_of_pass(6));
        assert!(index_of_pass(7) - index_of_pass(6) == 1);
        assert!(index_of_pass(6) > index_of_pass(5));
        assert!(index_of_pass(6) - index_of_pass(5) == 1);
        assert!(index_of_pass(5) > index_of_pass(4));
        assert!(index_of_pass(4) > index_of_pass(2));
    }
}
