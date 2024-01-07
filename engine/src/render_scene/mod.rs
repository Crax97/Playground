use nalgebra::Point3;

mod bvh;
pub use bvh::*;

use crate::{components::Transform, math::shape::BoundingShape, Frustum, Mesh, ResourceHandle};
pub enum PrimitiveKind {
    Mesh(ResourceHandle<Mesh>),
}

pub struct ScenePrimitive {
    pub kind: PrimitiveKind,
    pub transform: Transform,
}

pub trait AccelerationStructure<N>: Default {
    type NodeId;
    fn add(&mut self, node: N, bounds: BoundingShape, position: Point3<f32>) -> Self::NodeId;
    fn remove(&mut self, node_id: Self::NodeId);
    fn intersect_frustum(&self, frustum: &Frustum) -> Vec<Self::NodeId>;
    fn get(&self, target: Self::NodeId) -> &N;
}

#[derive(Default)]
pub struct RenderScene<A: AccelerationStructure<ScenePrimitive>> {
    accelleration_structure: A,
}

impl<A: AccelerationStructure<ScenePrimitive>> RenderScene<A> {
    fn new() -> Self {
        Default::default()
    }
    fn tick(&mut self, delta_frame: f32) {}

    fn add(&mut self, primitive: ScenePrimitive, bounds: BoundingShape) -> A::NodeId {
        let pos = primitive.transform.position;
        self.accelleration_structure.add(primitive, bounds, pos)
    }

    fn remove(&mut self, id: A::NodeId) {
        self.accelleration_structure.remove(id);
    }

    fn get_visible_nodes(&self, frustum: &Frustum) -> Vec<A::NodeId> {
        self.accelleration_structure.intersect_frustum(frustum)
    }
}
