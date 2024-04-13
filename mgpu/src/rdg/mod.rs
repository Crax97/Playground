use crate::{ImageView, SwapchainImage};

pub enum Resource {
    ImageView { view: ImageView },
}

pub enum Node {
    Present { id: u64, image: SwapchainImage },
}

struct RdgNode {
    // The resources read by this node
    reads: Vec<Resource>,

    // The resources written by this node
    writes: Vec<Resource>,

    // The operation executed by this node
    ty: Node,

    // This is the toi of this pass
    toi: u64,
}

#[derive(Default)]
pub struct Queues {
    graphics_nodes: Vec<RdgNode>,
    compute_nodes: Vec<RdgNode>,
    transfer_nodes: Vec<RdgNode>,
}

#[derive(Default)]
pub struct Rdg {
    queues: Queues,
    last_node_inserted: u64,
}

impl Rdg {
    pub fn add_graphics_pass(&mut self, node: Node) {
        Self::add_on_queue(
            &mut self.queues.graphics_nodes,
            node,
            &mut self.last_node_inserted,
        );
    }
    pub fn add_compute_pass(&mut self, node: Node) {
        Self::add_on_queue(
            &mut self.queues.compute_nodes,
            node,
            &mut self.last_node_inserted,
        );
    }
    pub fn add_transfer_pass(&mut self, node: Node) {
        Self::add_on_queue(
            &mut self.queues.transfer_nodes,
            node,
            &mut self.last_node_inserted,
        );
    }

    pub fn add_present_pass(&mut self, image: SwapchainImage, id: u64) {
        self.add_graphics_pass(Node::Present { image, id })
    }

    pub fn clear(&mut self) {
        *self = Default::default()
    }
    fn add_on_queue(queue: &mut Vec<RdgNode>, pass: Node, toi: &mut u64) {
        let node = pass.into_rdg(*toi);
        *toi += 1;

        queue.push(node)
    }
}

impl Node {
    fn into_rdg(self, toi: u64) -> RdgNode {
        match &self {
            Node::Present { image, .. } => RdgNode {
                reads: vec![Resource::ImageView { view: image.view }],
                writes: vec![],
                ty: self,
                toi,
            },
        }
    }
}
