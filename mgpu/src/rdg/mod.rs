use std::collections::{HashMap, HashSet, VecDeque};

use crate::{ImageView, MgpuResult, SwapchainImage};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Resource {
    ImageView { view: ImageView },
}

pub enum Node {
    Present { id: u64, image: SwapchainImage },
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
    Graphics,
    Compute,
    Transfer,
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
    pub transfer_nodes: Vec<usize>,
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
    pub fn add_compute_pass(&mut self, node: Node) {
        Self::add_on_queue(QueueType::Compute, &mut self.nodes, node);
    }
    pub fn add_transfer_pass(&mut self, node: Node) {
        Self::add_on_queue(QueueType::Transfer, &mut self.nodes, node);
    }

    pub fn add_present_pass(&mut self, image: SwapchainImage, id: u64) {
        self.add_graphics_pass(Node::Present { image, id })
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
                    QueueType::Compute => current_group.compute_nodes.push(node_info.global_index),
                    QueueType::Transfer => {
                        current_group.transfer_nodes.push(node_info.global_index)
                    }
                }
            }
        }

        if !current_group.compute_nodes.is_empty()
            || !current_group.graphics_nodes.is_empty()
            || !current_group.transfer_nodes.is_empty()
        {
            steps.push(Step::PassGroup(std::mem::take(&mut current_group)));
        }

        Ok(RdgCompiledGraph {
            sequence: steps,
            nodes: std::mem::take(&mut self.nodes),
        })
    }
}

impl Node {
    fn into_rdg(self, toi: usize, queue: QueueType) -> RdgNode {
        match &self {
            Node::Present { image, .. } => RdgNode {
                reads: [Resource::ImageView { view: image.view }].into(),
                writes: [].into(),
                ty: self,
                global_index: toi,
                queue,
            },
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{rdg::Resource, Image, ImageView};

    use super::Rdg;

    #[test]
    fn present() {
        let mut rdg = Rdg::default();

        let image_0 = Image { id: 0 };
        let view_0 = ImageView {
            owner: image_0,
            id: 0,
        };
        rdg.add_present_pass(
            crate::SwapchainImage {
                image: image_0,
                view: view_0,
            },
            0,
        );
        let compiled = rdg.compile().unwrap();
        assert_eq!(compiled.sequence.len(), 1);
        let step_0 = &compiled.sequence[0];
        match step_0 {
            crate::rdg::Step::Barrier { .. } => panic!("Barrier with only a present"),
            crate::rdg::Step::PassGroup(passes) => {
                assert_eq!(passes.compute_nodes.len(), 0);
                assert_eq!(passes.transfer_nodes.len(), 0);
                assert_eq!(passes.graphics_nodes.len(), 1);
                let pass = passes.graphics_nodes[0];

                let pass = &compiled.nodes[pass];
                assert_eq!(
                    *pass.reads.iter().next().unwrap(),
                    Resource::ImageView { view: view_0 }
                );
            }
        }
    }
}
