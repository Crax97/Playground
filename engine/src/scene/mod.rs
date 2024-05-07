use std::{collections::HashSet, default};

use glam::Vec3;

use crate::{
    arena::{Arena, Index},
    asset_map::AssetHandle,
    assets::{material::Material, mesh::Mesh},
    math::Transform,
};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneNodeId(Index);

pub struct SceneMesh {
    pub handle: AssetHandle<Mesh>,
    pub material: AssetHandle<Material>,
}

#[derive(Default)]
pub enum ScenePrimitive {
    #[default]
    Group,
    Mesh(SceneMesh),
}

#[derive(Default)]
pub struct SceneNode {
    pub label: Option<String>,
    pub enabled: bool,
    pub transform: Transform,
    pub primitive_type: ScenePrimitive,
}

pub struct Scene {
    pub nodes: Arena<SceneNode>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            nodes: Arena::default(),
        }
    }

    pub fn add_node(&mut self, node: SceneNode) -> SceneNodeId {
        let index = self.nodes.add(node);
        SceneNodeId(index)
    }

    pub fn remove_node(&mut self, node: SceneNodeId) {
        self.nodes.remove(node.0);
    }
    pub fn set_node_world_transform(&mut self, node: SceneNodeId, transform: Transform) {
        if let Some(node) = self.nodes.get_mut(node.0) {
            node.transform = transform;
        }
    }

    pub fn get_node_world_transform(&mut self, node: SceneNodeId) -> Option<Transform> {
        self.nodes.get_mut(node.0).map(|node| node.transform)
    }

    pub fn add_node_world_offset(&mut self, node: SceneNodeId, offset: Vec3) {
        if let Some(mut node_transform) = self.get_node_world_transform(node) {
            node_transform.location += offset;
            self.set_node_world_transform(node, node_transform);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SceneNode> {
        self.nodes.iter()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneNode {
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn primitive(mut self, primitive: ScenePrimitive) -> Self {
        self.primitive_type = primitive;
        self
    }

    pub fn transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }
}
