use nalgebra::{vector, Vector3};

enum BvhNodeKind<T> {
    Leaf(T),
    Inner {
        left_child: Option<BvhNodeId>,
        right_child: Option<BvhNodeId>,
    },
}

impl<T: std::fmt::Debug> std::fmt::Debug for BvhNodeKind<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leaf(arg0) => f.debug_tuple("Leaf").field(arg0).finish(),
            Self::Inner {
                left_child,
                right_child,
            } => f
                .debug_struct("Inner")
                .field("left_child", left_child)
                .field("right_child", right_child)
                .finish(),
        }
    }
}

pub struct BvhNode<T> {
    generation: usize,
    parent: Option<BvhNodeId>,
    aabb_min: Vector3<f32>,
    aabb_max: Vector3<f32>,
    kind: BvhNodeKind<T>,
}

impl<T> BvhNode<T> {
    fn contains_bounds(&self, aabb_min: Vector3<f32>, aabb_max: Vector3<f32>) -> bool {
        self.aabb_min.x <= aabb_min.x
            && self.aabb_max.x >= aabb_max.x
            && self.aabb_min.y <= aabb_min.y
            && self.aabb_max.y >= aabb_max.y
            && self.aabb_min.z <= aabb_min.z
            && self.aabb_max.z >= aabb_max.z
    }
    fn extend_bounds(
        &self,
        aabb_min: Vector3<f32>,
        aabb_max: Vector3<f32>,
    ) -> (Vector3<f32>, Vector3<f32>) {
        (
            vector![
                aabb_min.x.min(self.aabb_min.x),
                aabb_min.y.min(self.aabb_min.y),
                aabb_min.z.min(self.aabb_min.z),
            ],
            vector![
                aabb_max.x.max(self.aabb_max.x),
                aabb_max.y.max(self.aabb_max.y),
                aabb_max.z.max(self.aabb_max.z),
            ],
        )
    }
}

impl<T: Default> Default for BvhNode<T> {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            parent: Default::default(),
            aabb_min: Default::default(),
            aabb_max: Default::default(),
            kind: BvhNodeKind::Leaf(T::default()),
        }
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for BvhNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BvhNode")
            .field("generation", &self.generation)
            .field("parent", &self.parent)
            .field("aabb_min", &self.aabb_min)
            .field("aabb_max", &self.aabb_max)
            .field("kind", &self.kind)
            .finish()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct BvhNodeId {
    id: usize,
    generation: usize,
}

#[derive(Debug)]
pub struct Bvh<T> {
    nodes: Vec<BvhNode<T>>,
    unused_nodes_id: Vec<BvhNodeId>,
    root_node_idx: Option<BvhNodeId>,
}

impl<T> Default for Bvh<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Bvh<T> {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            unused_nodes_id: vec![],
            root_node_idx: None,
        }
    }

    pub fn add(&mut self, payload: T, aabb_min: Vector3<f32>, aabb_max: Vector3<f32>) -> BvhNodeId {
        let node = BvhNode {
            parent: None,
            aabb_max,
            aabb_min,
            kind: BvhNodeKind::Leaf(payload),
            generation: 0,
        };

        let new_node_idx = self.insert_node(node);
        if self.root_node_idx.is_none() {
            self.root_node_idx = Some(new_node_idx);
            return new_node_idx;
        }

        let mut visit_queue = vec![self.root_node_idx.unwrap()];
        while let Some(current_node_idx) = visit_queue.pop() {
            let current_node = self.get_node_mut(current_node_idx);
            if current_node.contains_bounds(aabb_min, aabb_max) {
                match &mut current_node.kind {
                    BvhNodeKind::Leaf(_) => {
                        // create a new subtree, with the current node and the new node
                        let (aabb_min, aabb_max) = current_node.extend_bounds(aabb_min, aabb_max);
                        self.add_subtree(aabb_min, aabb_max, new_node_idx, current_node_idx);
                        break;
                    }
                    BvhNodeKind::Inner {
                        left_child,
                        right_child,
                    } => {
                        // Check if there's space in this subtree
                        // Otherwise visit the subtree
                        if let Some(child) = left_child {
                            visit_queue.push(*child);
                        } else {
                            *left_child = Some(new_node_idx);
                            self.get_node_mut(new_node_idx).parent = Some(current_node_idx);
                            break;
                        }
                        if let Some(child) = right_child {
                            visit_queue.push(*child);
                        } else {
                            *right_child = Some(new_node_idx);
                            self.get_node_mut(new_node_idx).parent = Some(current_node_idx);
                            break;
                        }
                    }
                }
            } else {
                let (aabb_min, aabb_max) = current_node.extend_bounds(aabb_min, aabb_max);
                self.add_subtree(aabb_min, aabb_max, new_node_idx, current_node_idx);
                break;
            }
        }

        new_node_idx
    }

    fn add_subtree(
        &mut self,
        aabb_min: Vector3<f32>,
        aabb_max: Vector3<f32>,
        new_node_idx: BvhNodeId,
        root_subtree_idx: BvhNodeId,
    ) {
        let current_root_parent = self.get_node_mut(root_subtree_idx).parent;
        let new_subtree_root_node = BvhNode {
            generation: 0,
            parent: current_root_parent,
            aabb_min,
            aabb_max,
            kind: BvhNodeKind::Inner {
                left_child: Some(new_node_idx),
                right_child: Some(root_subtree_idx),
            },
        };
        let new_root_index = self.insert_node(new_subtree_root_node);

        // The old subtree root's parent is now the new node just created
        self.get_node_mut(root_subtree_idx).parent = Some(new_root_index);
        self.get_node_mut(new_node_idx).parent = Some(new_root_index);

        // Ensure that the old subtree root's parent points to new root
        if let Some(parent) = current_root_parent {
            match &mut self.get_node_mut(parent).kind {
                BvhNodeKind::Leaf(_) => unreachable!(),
                BvhNodeKind::Inner {
                    left_child,
                    right_child,
                } => {
                    if left_child.is_some_and(|c| c == root_subtree_idx) {
                        *left_child = Some(new_root_index);
                        self.get_node_mut(new_root_index).parent = Some(parent);
                    } else if right_child.is_some_and(|c| c == root_subtree_idx) {
                        *right_child = Some(new_root_index);
                        self.get_node_mut(new_root_index).parent = Some(parent);
                    }
                }
            }
        } else {
            // The old subtree root was the bvh's root node
            assert!(self.root_node_idx.is_some_and(|r| r == root_subtree_idx));
            self.root_node_idx = Some(new_root_index);
        }
    }

    pub fn get(&self, node_id: BvhNodeId) -> &T {
        match &self.nodes[node_id.id].kind {
            BvhNodeKind::Leaf(payload) => payload,
            _ => unreachable!(),
        }
    }

    fn get_node_mut(&mut self, node_id: BvhNodeId) -> &mut BvhNode<T> {
        let node = &mut self.nodes[node_id.id];
        assert!(node.generation == node_id.generation);
        node
    }

    fn get_node(&self, node_id: BvhNodeId) -> &BvhNode<T> {
        let node = &self.nodes[node_id.id];
        assert!(node.generation == node_id.generation);
        node
    }

    fn insert_node(&mut self, mut node: BvhNode<T>) -> BvhNodeId {
        if let Some(reusable_node) = self.unused_nodes_id.pop() {
            node.generation = reusable_node.generation;
            self.nodes[reusable_node.id] = node;
            reusable_node
        } else {
            let new_node_idx = self.nodes.len();
            self.nodes.push(node);
            BvhNodeId {
                id: new_node_idx,
                generation: 0,
            }
        }
    }

    pub fn visit<F: FnMut(&T)>(&self, mut f: F) {
        if self.nodes.is_empty() {
            return;
        }
        let mut visit_queue = vec![self.root_node_idx.unwrap()];
        while let Some(node) = visit_queue.pop() {
            let node = self.get_node(node);
            match node.kind {
                BvhNodeKind::Leaf(ref payload) => {
                    f(payload);
                }
                BvhNodeKind::Inner {
                    left_child,
                    right_child,
                } => {
                    if let Some(child) = left_child {
                        visit_queue.push(child);
                    }
                    if let Some(child) = right_child {
                        visit_queue.push(child);
                    }
                }
            }
        }
    }
}

impl<T: Copy> Bvh<T> {
    pub fn remove(&mut self, node_id: BvhNodeId) -> T {
        let node_parent = self.get_node_mut(node_id).parent;

        // Disconnect the subtree by unreferencing the node from the node's parent
        if let Some(parent) = node_parent {
            let parent = self.get_node_mut(parent);
            match &mut parent.kind {
                BvhNodeKind::Leaf(_) => unreachable!(),
                BvhNodeKind::Inner {
                    left_child,
                    right_child,
                } => {
                    if left_child.is_some_and(|c| c == node_id) {
                        *left_child = None;
                    }
                    if right_child.is_some_and(|c| c == node_id) {
                        *right_child = None;
                    }
                }
            }
        } else {
            // The node being removed is the bvh's root
            assert!(self.root_node_idx.is_some_and(|r| r == node_id));
            self.root_node_idx = None;
        }

        // Reclaim the node by incremeting it's generation
        self.unused_nodes_id.push(BvhNodeId {
            id: node_id.id,
            generation: node_id.generation + 1,
        });
        let node = &self.nodes[node_id.id].kind;
        match node {
            BvhNodeKind::Leaf(payload) => *payload,
            BvhNodeKind::Inner { .. } => unreachable!(),
        }
    }
}
#[cfg(test)]
mod tests {
    use nalgebra::{vector, Vector3};

    use super::{Bvh, BvhNodeId, BvhNodeKind};

    fn random_vec3() -> Vector3<f32> {
        vector![rand::random(), rand::random(), rand::random()]
    }

    fn ensure_bvh_is_valid<T: Default>(
        bvh: &Bvh<T>,
        parent: Option<BvhNodeId>,
        current: BvhNodeId,
        current_iter: usize,
        max_iter: usize,
    ) {
        assert!(current_iter < max_iter);
        let current_node = bvh.get_node(current);
        assert!(current_node.parent == parent);
        if let Some(parent) = parent {
            let parent_node = bvh.get_node(parent);
            assert!(parent_node.contains_bounds(current_node.aabb_min, current_node.aabb_max));
        } else {
            // The only node allowed to NOT have a parent is the root node
            assert!(current == bvh.root_node_idx.unwrap());
        }

        if let BvhNodeKind::Inner {
            left_child,
            right_child,
        } = current_node.kind
        {
            if let Some(child) = left_child {
                ensure_bvh_is_valid(bvh, Some(current), child, current_iter + 1, max_iter);
            }
            if let Some(child) = right_child {
                ensure_bvh_is_valid(bvh, Some(current), child, current_iter + 1, max_iter);
            }
        }
    }

    #[test]
    fn test_random_operations() {
        for test_id in 0..100 {
            let n: usize = rand::random::<usize>() % 1000;
            let n = n + 10; // Insert at least ten node

            let mut counter: u32 = 0;
            let mut inc = || {
                let i = counter;
                counter += 1;
                i
            };
            let mut node_ids = vec![];
            let mut bvh = Bvh::new();

            for _ in 0..n {
                let payload = inc();
                let node_id = bvh.add(payload, random_vec3(), random_vec3());
                node_ids.push((node_id, payload));
            }
            let payload = inc();
            let node_id = bvh.add(payload, random_vec3(), random_vec3());
            node_ids.push((node_id, payload));

            ensure_bvh_is_valid(&bvh, None, bvh.root_node_idx.unwrap(), 0, 1000);

            let mut leaves = 0;
            bvh.visit(|_| {
                leaves += 1;
            });
            assert_eq!(node_ids.len(), leaves, "Addition is broken");

            for _ in 0..(rand::random::<usize>() % n) {
                let random_node_id = rand::random::<usize>() % node_ids.len();
                let (random_node_id, _) = node_ids.remove(random_node_id);
                if bvh.root_node_idx.is_some_and(|n| n != random_node_id) {
                    bvh.remove(random_node_id);
                }
            }

            if let Some(root_node) = bvh.root_node_idx {
                ensure_bvh_is_valid(&bvh, None, root_node, 0, 100000);
                let mut leaves = 0;
                bvh.visit(|_| {
                    leaves += 1;
                });
                assert_eq!(node_ids.len(), leaves, "Removal is broken");
            } else {
                println!("Test {test_id}): bvh was emptied, {bvh:?}");
            }

            for (node, payload) in node_ids {
                assert!(*bvh.get(node) == payload);
            }
        }
    }
}
