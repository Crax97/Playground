use nalgebra::{vector, Vector3};
use thunderdome::{Arena, Index};

use super::AccelerationStructure;

enum BvhNodeKind<T> {
    Leaf(T),
    Inner {
        left_child: Option<usize>,
        right_child: Option<usize>,
    },
}

pub struct BvhNode<T> {
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

pub struct BvhNodeId(usize);

#[derive(Default)]
pub struct Bvh<T: Default> {
    nodes: Vec<BvhNode<T>>,
}

impl<T: Default> Bvh<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, payload: T, aabb_min: Vector3<f32>, aabb_max: Vector3<f32>) -> BvhNodeId {
        let node = BvhNode {
            aabb_max,
            aabb_min,
            kind: BvhNodeKind::Leaf(payload),
        };

        let new_node_idx = self.nodes.len();
        self.nodes.push(node);

        if self.nodes.len() == 1 {
            return BvhNodeId(new_node_idx);
        }

        let mut visit_queue = vec![0];
        while let Some(current_node_idx) = visit_queue.pop() {
            let current_node = &mut self.nodes[current_node_idx];
            if current_node.contains_bounds(aabb_min, aabb_max) {
                match &mut current_node.kind {
                    BvhNodeKind::Leaf(_) => {
                        self.add_subtree(aabb_min, aabb_max, new_node_idx, current_node_idx);
                        break;
                    }
                    BvhNodeKind::Inner {
                        left_child,
                        right_child,
                    } => {
                        if let Some(child) = left_child {
                            visit_queue.push(*child);
                        } else {
                            *left_child = Some(new_node_idx);
                            break;
                        }
                        if let Some(child) = right_child {
                            visit_queue.push(*child);
                        } else {
                            *right_child = Some(new_node_idx);
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

        BvhNodeId(new_node_idx)
    }

    pub fn remove(&mut self, node_id: BvhNodeId) -> T {
        todo!()
    }

    pub fn get(&self, node_id: BvhNodeId) -> &T {
        match &self.nodes[node_id.0].kind {
            BvhNodeKind::Leaf(payload) => payload,
            _ => unreachable!(),
        }
    }

    fn add_subtree(
        &mut self,
        aabb_min: Vector3<f32>,
        aabb_max: Vector3<f32>,
        new_node_idx: usize,
        root_subtree_idx: usize,
    ) {
        let new_root_index = self.nodes.len();
        let new_subtree_root_node = BvhNode {
            aabb_min,
            aabb_max,
            kind: BvhNodeKind::Inner {
                left_child: Some(new_node_idx),
                right_child: Some(new_root_index),
            },
        };
        self.nodes.push(new_subtree_root_node);
        self.nodes.swap(new_root_index, root_subtree_idx);
    }

    pub fn visit<F: FnMut(&T)>(&self, mut f: F) {
        if self.nodes.is_empty() {
            return;
        }
        let mut visit_queue = vec![0];
        while let Some(node) = visit_queue.pop() {
            let node = &self.nodes[node];
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

#[cfg(test)]
mod tests {
    use nalgebra::{vector, Vector3};

    use super::{Bvh, BvhNodeKind};

    fn random_vec3() -> Vector3<f32> {
        vector![rand::random(), rand::random(), rand::random()]
    }

    fn ensure_bvh_is_valid<T: Default>(
        bvh: &Bvh<T>,
        parent: Option<usize>,
        current: usize,
        current_iter: usize,
        max_iter: usize,
    ) {
        assert!(current_iter < max_iter);
        let current_node = &bvh.nodes[current];
        if let Some(parent) = parent {
            let parent_node = &bvh.nodes[parent];
            assert!(parent_node.contains_bounds(current_node.aabb_min, current_node.aabb_max));
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
    fn test_random_add_n() {
        let n: usize = rand::random::<usize>() % 100;
        let n = n + 1; // Insert at least one node
        let mut counter: u32 = 0;
        let mut inc = || {
            let i = counter;
            counter += 1;
            i
        };
        let mut bvh = Bvh::new();
        for _ in 0..n {
            bvh.add(inc(), random_vec3(), random_vec3());
        }

        ensure_bvh_is_valid(&bvh, None, 0, 0, 1000);

        let mut leaves = 0;
        bvh.visit(|_| {
            leaves += 1;
            assert!(leaves < n + 1);
        });
        assert_eq!(leaves, n);
    }
}
