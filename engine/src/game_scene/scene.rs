use std::marker::PhantomData;

use egui::ahash::HashSet;
use nalgebra::{vector, Matrix, Matrix4, Point3, UnitQuaternion, Vector3};
use thunderdome::{Arena, Index};

use crate::{components::Transform, math::shape::BoundingShape, LightHandle, RenderScene};

#[derive(Default)]
pub struct Scene<T: 'static> {
    render_scene: RenderScene,
    nodes: Arena<SceneNode<T>>,
}

#[derive(Default, Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct SceneNodeId(Option<Index>);

#[derive(Default, Clone)]
pub enum SceneNodeKind {
    #[default]
    Empty,
    Mesh,
    Light(LightHandle),
}

pub struct SceneNode<T: 'static> {
    pub children: HashSet<SceneNodeId>,
    pub parent: Option<SceneNodeId>,
    pub world_transform: Transform,
    pub local_transform: Transform,
    pub payload: T,
    pub collision_shape: Option<BoundingShape>,
    pub tags: Vec<String>,
    pub local_to_world: Matrix4<f32>,
    pub world_to_local: Matrix4<f32>,

    _marker: PhantomData<()>,
}

impl<T: Default> Default for SceneNode<T> {
    fn default() -> Self {
        Self {
            children: Default::default(),
            parent: Default::default(),
            world_transform: Default::default(),
            local_transform: Default::default(),
            payload: Default::default(),
            collision_shape: Default::default(),
            tags: Default::default(),
            local_to_world: Matrix4::identity(),
            world_to_local: Matrix4::identity(),
            _marker: Default::default(),
        }
    }
}

impl<T: Clone> Clone for SceneNode<T> {
    fn clone(&self) -> Self {
        Self {
            payload: self.payload.clone(),
            children: self.children.clone(),
            parent: self.parent,
            world_transform: self.world_transform,
            local_transform: self.local_transform,
            collision_shape: self.collision_shape,
            tags: self.tags.clone(),
            local_to_world: self.local_to_world,
            world_to_local: self.world_to_local,
            _marker: self._marker,
        }
    }
}

pub struct SceneNodeBuilder<'s, T: 'static> {
    scene: &'s mut Scene<T>,
    new_node: SceneNode<T>,
}

#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Copy, Clone)]
pub enum TransformSpace {
    Local,
    World,
}

impl<'s, T: Clone + Default> SceneNodeBuilder<'s, T> {
    fn new(scene: &'s mut Scene<T>) -> SceneNodeBuilder<'_, T> {
        Self {
            scene,
            new_node: Default::default(),
        }
    }

    pub fn with_tags<I: IntoIterator>(&mut self, tags: I) -> &mut Self
    where
        I::Item: Into<String>,
    {
        self.new_node
            .tags
            .extend(tags.into_iter().map(|s| s.into()));
        self
    }

    pub fn build(&mut self) -> SceneNodeId {
        let new_node = self.scene.nodes.insert(self.new_node.clone());
        SceneNodeId(Some(new_node))
    }
}

impl<T: 'static + Default + Clone> Scene<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self) -> SceneNodeBuilder<T> {
        SceneNodeBuilder::new(self)
    }
    pub fn get_node(&self, node_id: SceneNodeId) -> Option<&SceneNode<T>> {
        node_id.0.and_then(|id| self.nodes.get(id))
    }
    pub fn remove_node(&mut self, node_id: SceneNodeId) -> Option<SceneNode<T>> {
        node_id.0.and_then(|n| self.nodes.remove(n))
    }
    pub fn set_position(
        &mut self,
        node_id: SceneNodeId,
        position: Point3<f32>,
        transform_space: TransformSpace,
    ) {
        let id = if let Some(index) = node_id.0 {
            index
        } else {
            return;
        };
        if let Some(node) = self.nodes.get_mut(id) {
            let transform = pick_transform_mut(transform_space, node);
            transform.position = position;

            if transform_space == TransformSpace::Local {
                self.recompute_world_from_relative(node_id);
            } else {
                self.recompute_relative_from_world(node_id);
            }
        }
    }
    pub fn set_rotation(
        &mut self,
        node_id: SceneNodeId,
        rotation: UnitQuaternion<f32>,
        transform_space: TransformSpace,
    ) {
        let id = if let Some(index) = node_id.0 {
            index
        } else {
            return;
        };
        if let Some(node) = self.nodes.get_mut(id) {
            let transform = pick_transform_mut(transform_space, node);
            transform.rotation = rotation;
            if transform_space == TransformSpace::Local {
                self.recompute_world_from_relative(node_id);
            } else {
                self.recompute_relative_from_world(node_id);
            }
        }
    }
    pub fn set_scale(
        &mut self,
        node_id: SceneNodeId,
        scale: Vector3<f32>,
        transform_space: TransformSpace,
    ) {
        let id = if let Some(index) = node_id.0 {
            index
        } else {
            return;
        };
        if let Some(node) = self.nodes.get_mut(id) {
            let transform = pick_transform_mut(transform_space, node);
            transform.scale = scale;
            if transform_space == TransformSpace::Local {
                self.recompute_world_from_relative(node_id);
            } else {
                self.recompute_relative_from_world(node_id);
            }
        }
    }

    pub fn set_transform(
        &mut self,
        node_id: SceneNodeId,
        transform: Transform,
        transform_space: TransformSpace,
    ) {
        let id = if let Some(index) = node_id.0 {
            index
        } else {
            return;
        };
        if let Some(node) = self.nodes.get_mut(id) {
            let old_local_to_world = node.local_to_world;
            let children = node.children.clone();
            let old_transform = pick_transform_mut(transform_space, node);
            *old_transform = transform;

            if transform_space == TransformSpace::Local {
                self.recompute_world_from_relative(node_id);
            } else {
                self.recompute_relative_from_world(node_id);
            }
            let node = self.nodes.get_mut(id).unwrap();
            let new_local_to_world = node.local_to_world;

            let transform_delta = new_local_to_world * old_local_to_world.try_inverse().unwrap();

            for child in children {
                self.apply_world_delta(child, transform_delta);
            }
        }
    }

    pub fn get_position(
        &self,
        node_id: SceneNodeId,
        transform_space: TransformSpace,
    ) -> Option<Point3<f32>> {
        let id = node_id.0?;
        if let Some(node) = self.nodes.get(id) {
            let transform = pick_transform(transform_space, node);
            Some(transform.position)
        } else {
            None
        }
    }
    pub fn get_rotation(
        &self,
        node_id: SceneNodeId,
        transform_space: TransformSpace,
    ) -> Option<UnitQuaternion<f32>> {
        let id = node_id.0?;
        if let Some(node) = self.nodes.get(id) {
            let transform = pick_transform(transform_space, node);
            Some(transform.rotation)
        } else {
            None
        }
    }
    pub fn get_scale(
        &self,
        node_id: SceneNodeId,
        transform_space: TransformSpace,
    ) -> Option<Vector3<f32>> {
        let id = node_id.0?;
        if let Some(node) = self.nodes.get(id) {
            let transform = pick_transform(transform_space, node);
            Some(transform.scale)
        } else {
            None
        }
    }

    pub fn get_transform(
        &self,
        node_id: SceneNodeId,
        transform_space: TransformSpace,
    ) -> Option<Transform> {
        let id = node_id.0?;
        if let Some(node) = self.nodes.get(id) {
            let transform = pick_transform(transform_space, node);
            Some(*transform)
        } else {
            None
        }
    }

    pub fn set_parent(&mut self, node_id: SceneNodeId, new_parent: Option<SceneNodeId>) {
        assert!(
            new_parent.is_none()
                || new_parent.is_some_and(|n| n != node_id && !self.is_child_of(n, node_id)),
            "The new parent is either a child of the current node, or the node itself"
        );
        node_id
            .0
            .map(|node_id| self.nodes.get_mut(node_id).map(|n| n.parent = new_parent));
        new_parent
            .and_then(|p| p.0)
            .and_then(|n| self.nodes.get_mut(n))
            .map(|node| node.children.insert(node_id));
    }
    pub fn get_parent(&self, node_id: SceneNodeId) -> Option<SceneNodeId> {
        node_id
            .0
            .and_then(|n| self.nodes.get(n).map(|n| n.parent))
            .flatten()
    }

    pub fn add_child(&mut self, node_id: SceneNodeId, new_child: SceneNodeId) {
        assert!(
            new_child
                .0
                .is_some_and(|_| !self.is_parent_of(new_child, node_id)),
            "Either the new child is null, or it's a parent of the current node"
        );

        node_id.0.map(|n| {
            self.nodes
                .get_mut(n)
                .map(|node| node.children.insert(new_child))
        });
        let old_parent = new_child
            .0
            .and_then(|n| self.nodes.get_mut(n).unwrap().parent.replace(node_id));
        if let Some(old_parent) = old_parent {
            let parent = self.nodes.get_mut(old_parent.0.unwrap()).unwrap();
            parent.children.remove(&new_child);
        }
    }
    pub fn remove_child(&mut self, node_id: SceneNodeId, child: SceneNodeId) {
        let removed = node_id
            .0
            .and_then(|n| {
                self.nodes
                    .get_mut(n)
                    .map(|node| node.children.remove(&child))
            })
            .unwrap_or_default();

        if removed {
            self.nodes.get_mut(child.0.unwrap()).unwrap().parent = None;
        }
    }
    pub fn is_child_of(&self, who: SceneNodeId, maybe_child_of: SceneNodeId) -> bool {
        if let (Some(_), Some(maybe_child_of)) = (who.0, maybe_child_of.0) {
            if let Some(node) = self.nodes.get(maybe_child_of) {
                for &child in &node.children {
                    if child == who {
                        return true;
                    }
                    if self.is_child_of(who, child) {
                        return true;
                    }
                }
                return false;
            }
        }
        false
    }

    pub fn is_parent_of(&self, who: SceneNodeId, maybe_parent_of: SceneNodeId) -> bool {
        let mut current = maybe_parent_of;
        while let Some(index) = current.0 {
            if let Some(node) = self.nodes.get(index) {
                if node.parent.is_some_and(|n| n == who) {
                    return true;
                }

                current = node.parent.unwrap_or_default();
            }
        }
        false
    }
    pub fn get_children(&self, node_id: SceneNodeId) -> Option<impl Iterator<Item = &SceneNodeId>> {
        dbg!(node_id
            .0
            .and_then(|id| self.nodes.get(id).map(|n| &n.children))
            .map(|chi| chi.iter()))
    }

    pub fn find_by_tag(&self, tag: impl AsRef<str>) -> Option<SceneNodeId> {
        self.nodes
            .iter()
            .find(|(_, node)| node.tags.iter().any(|s| s.as_str() == tag.as_ref()))
            .map(|(i, _)| SceneNodeId(Some(i)))
    }

    fn recompute_world_from_relative(&mut self, node_id: SceneNodeId) {
        let (node_parent, model_matrix) = self
            .nodes
            .get(node_id.0.unwrap())
            .map(|node| (node.parent, node.local_transform.matrix()))
            .unwrap();
        if node_parent.is_some() {
            let local_to_world = self.get_node_local_to_world(node_parent) * model_matrix;
            let world_to_local = local_to_world.try_inverse().unwrap();
            if let Some(node) = self.nodes.get_mut(node_id.0.unwrap()) {
                node.local_to_world = local_to_world;
                node.world_to_local = world_to_local;

                let world_translation = Point3::from(local_to_world.column(3).xyz());
                println!("New world translation is {world_translation}");
                let scale = vector![
                    local_to_world.column(0).magnitude(),
                    local_to_world.column(1).magnitude(),
                    local_to_world.column(2).magnitude()
                ];
                let mut rotation = local_to_world.remove_column(3).remove_row(3);
                rotation.set_column(0, &rotation.column(0).map(|c| c / scale.x));
                rotation.set_column(1, &rotation.column(1).map(|c| c / scale.y));
                rotation.set_column(2, &rotation.column(2).map(|c| c / scale.z));
                let rotation = UnitQuaternion::from_matrix(&rotation);
                node.world_transform.position = world_translation;
                node.world_transform.scale = scale;
                node.world_transform.rotation = rotation;
            }
        } else if let Some(node) = self.nodes.get_mut(node_id.0.unwrap()) {
            node.world_transform = node.local_transform;
        }
    }
    fn recompute_relative_from_world(&mut self, node_id: SceneNodeId) {
        let (node_parent, model_matrix) = self
            .nodes
            .get(node_id.0.unwrap())
            .map(|node| (node.parent, node.world_transform.matrix()))
            .unwrap();
        let local_to_world = self.get_node_local_to_world(node_parent) * model_matrix;
        let world_to_local = local_to_world.try_inverse().unwrap();
        if let Some(node) = self.nodes.get_mut(node_id.0.unwrap()) {
            node.local_to_world = local_to_world;
            node.world_to_local = world_to_local;

            if let Some(parent) = node_parent {
                let world_translation = Point3::from(local_to_world.column(3).xyz());
                let scale = vector![
                    local_to_world.column(0).magnitude(),
                    local_to_world.column(1).magnitude(),
                    local_to_world.column(2).magnitude()
                ];
                let mut rotation = local_to_world.remove_column(3).remove_row(3);
                rotation.set_column(0, &rotation.column(0).map(|c| c / scale.x));
                rotation.set_column(1, &rotation.column(1).map(|c| c / scale.y));
                rotation.set_column(2, &rotation.column(2).map(|c| c / scale.z));
                let rotation = UnitQuaternion::from_matrix(&rotation);
                node.world_transform.position = world_translation;
                node.world_transform.scale = scale;
                node.world_transform.rotation = rotation;
            } else {
                node.local_transform = node.world_transform;
            }
        }
    }
    fn get_node_local_to_world(&self, node_parent: Option<SceneNodeId>) -> Matrix4<f32> {
        node_parent
            .map(|p| self.nodes.get(p.0.unwrap()).unwrap().local_to_world)
            .unwrap_or(Matrix4::identity())
    }

    fn apply_world_delta(
        &mut self,
        child: SceneNodeId,
        transform_delta: nalgebra::Matrix<
            f32,
            nalgebra::Const<4>,
            nalgebra::Const<4>,
            nalgebra::ArrayStorage<f32, 4, 4>,
        >,
    ) {
        if let Some(node) = self.nodes.get_mut(child.0.unwrap()) {
            node.local_to_world *= transform_delta;
            let children = node.children.clone();
            self.recompute_relative_from_world(child);
            for child in children {
                self.apply_world_delta(child, transform_delta);
            }
        }
    }
}

fn pick_transform_mut<T>(
    transform_space: TransformSpace,
    node: &mut SceneNode<T>,
) -> &mut Transform {
    match transform_space {
        TransformSpace::Local => &mut node.local_transform,
        TransformSpace::World => &mut node.world_transform,
    }
}

fn pick_transform<T>(transform_space: TransformSpace, node: &SceneNode<T>) -> &Transform {
    match transform_space {
        TransformSpace::Local => &node.local_transform,
        TransformSpace::World => &node.world_transform,
    }
}
