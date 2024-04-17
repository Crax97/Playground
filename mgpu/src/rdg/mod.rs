use std::collections::{HashMap, HashSet, VecDeque};

use crate::{AttachmentStoreOp, Image, ImageView, MgpuResult, RenderPassInfo, SwapchainImage};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Resource {
    Image { image: Image },
    ImageView { view: ImageView },
}

pub enum Node {
    RenderPass { info: RenderPassInfo },
}

pub struct RdgNode {
    // The resources read by this node
    pub(crate) reads: HashSet<Resource>,

    // The resources written by this node
    pub(crate) writes: HashSet<Resource>,

    // The operation executed by this node
    pub(crate) ty: Node,

    // A monotonically increasing index that identifies this pass
    pub(crate) global_index: usize,

    // The queue where this node should be placed on
    pub(crate) queue: QueueType,
}

#[derive(Default)]
pub struct Rdg {
    nodes: Vec<RdgNode>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum QueueType {
    // This queue can execute both graphics commands (such as drawing) and sync compute commands
    Graphics,

    // This queue can execute only compute commands, and it runs asynchronously from the Graphics queue
    AsyncCompute,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct OwnershipTransfer {
    pub before: QueueType,
    pub after: QueueType,
    pub resource: Resource,
}

#[derive(Default)]
pub struct PassGroup {
    pub graphics_nodes: Vec<usize>,
    pub compute_nodes: Vec<usize>,
}

pub enum Step {
    Barrier { transfers: Vec<OwnershipTransfer> },
    PassGroup(PassGroup),
}

#[derive(Default)]
pub struct RdgCompiledGraph {
    pub sequence: Vec<Step>,
    pub nodes: Vec<RdgNode>,
}

impl Rdg {
    pub fn add_graphics_pass(&mut self, node: Node) {
        Self::add_on_queue(QueueType::Graphics, &mut self.nodes, node);
    }
    pub fn add_async_compute_pass(&mut self, node: Node) {
        Self::add_on_queue(QueueType::AsyncCompute, &mut self.nodes, node);
    }

    pub fn inform_present(&mut self, swapchain_image: SwapchainImage, swapchain_id: u64) {
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
            }
            if stop {
                return;
            };
        }
    }

    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    pub fn clear(&mut self) {
        *self = Default::default()
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
        self.create_compiled_graph(&topological_sorting)
    }

    fn create_adjacency_list(&self) -> Vec<HashSet<usize>> {
        let mut adjacency_list = Vec::with_capacity(self.nodes.len());
        adjacency_list.resize(self.nodes.len(), HashSet::default());

        // Helper map to track which node currently owns a resource
        let mut ownerships = HashMap::<Resource, usize>::new();

        for node in &self.nodes {
            let mut inner_ownerships = HashMap::<Resource, usize>::new();
            for written_resource in &node.writes {
                ownerships.insert(*written_resource, node.global_index);
                inner_ownerships.insert(*written_resource, node.global_index);

                for other in &self.nodes {
                    if other.reads.contains(written_resource)
                        // Avoid incorrectly adding a dependency when a resource changes in ownership
                        && inner_ownerships
                            .get(written_resource)
                            .is_some_and(|o| *o == node.global_index)
                    {
                        adjacency_list[node.global_index].insert(other.global_index);
                    }

                    for other_written in &other.writes {
                        inner_ownerships.insert(*other_written, other.global_index);
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
                let ownership = ownerships.entry(read_resource).or_insert(QueueOwnership {
                    queue: node_info.queue,
                    node: node_info.global_index,
                });

                if ownership.queue != node_info.queue {
                    barrier_info.push(OwnershipTransfer {
                        before: ownership.queue,
                        after: node_info.queue,
                        resource: read_resource,
                    });
                    ownership.node = node_info.global_index;
                    ownership.queue = node_info.queue;
                }
            }

            for &written_resource in &node_info.writes {
                let ownership = ownerships
                    .entry(written_resource)
                    .or_insert(QueueOwnership {
                        queue: node_info.queue,
                        node: node_info.global_index,
                    });

                ownership.node = node_info.global_index;
                ownership.queue = node_info.queue;
            }

            if !barrier_info.is_empty() {
                steps.push(Step::PassGroup(std::mem::take(&mut current_group)));
                steps.push(Step::Barrier {
                    transfers: std::mem::take(&mut barrier_info),
                });
            } else {
                match node_info.queue {
                    QueueType::Graphics => {
                        current_group.graphics_nodes.push(node_info.global_index)
                    }
                    QueueType::AsyncCompute => {
                        current_group.compute_nodes.push(node_info.global_index)
                    }
                }
            }
        }

        if !current_group.compute_nodes.is_empty() || !current_group.graphics_nodes.is_empty() {
            steps.push(Step::PassGroup(std::mem::take(&mut current_group)));
        }

        Ok(RdgCompiledGraph {
            sequence: steps,
            nodes: std::mem::take(&mut self.nodes),
        })
    }

    fn patch_latest_render_pass_writing_to_swapchain(
        &mut self,
        image: SwapchainImage,
        topological_sorting: &[usize],
    ) {
        let swapchain_image = image;
        let last_writing_node = topological_sorting.iter().find(|node_idx| {
            let node_info = &self.nodes[**node_idx];
            node_info.writes.iter().any(|res| match res {
                Resource::ImageView { view } => *view == swapchain_image.view,
                _ => false,
            })
        });
        if let Some(&last_writing_node) = last_writing_node {
            let node_info = &mut self.nodes[last_writing_node];
            match &mut node_info.ty {
                Node::RenderPass { info } => info.steps.iter_mut().for_each(|step| {
                    step.color_attachments.iter_mut().for_each(|rta| {
                        if info.framebuffer.render_targets[rta.index].view == swapchain_image.view {
                            info.framebuffer.render_targets[rta.index].store_op =
                                AttachmentStoreOp::Present;
                        }
                    })
                }),
            }
        }
    }
}

impl Node {
    fn into_rdg(self, toi: usize, queue: QueueType) -> RdgNode {
        let (read, write) = match &self {
            Node::RenderPass { info } => (
                info.framebuffer
                    .render_targets
                    .iter()
                    .filter_map(|rt| match rt.store_op {
                        AttachmentStoreOp::DontCare => Some(Resource::ImageView { view: rt.view }),
                        AttachmentStoreOp::Store | AttachmentStoreOp::Present => None,
                    })
                    .collect(),
                info.framebuffer
                    .render_targets
                    .iter()
                    .filter_map(|rt| match rt.store_op {
                        AttachmentStoreOp::DontCare => None,
                        AttachmentStoreOp::Store | AttachmentStoreOp::Present => {
                            Some(Resource::ImageView { view: rt.view })
                        }
                    })
                    .collect(),
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
            crate::rdg::Step::Barrier { .. } => panic!("Barrier with only a present"),
            crate::rdg::Step::PassGroup(..) => {
                panic!("PassGroup with only a present!")
            }
        }
    }
}
