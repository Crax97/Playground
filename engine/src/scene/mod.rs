pub mod serializable_scene;

use std::collections::{HashMap, HashSet};

use glam::Vec3;
use log::warn;
use serde::{Deserialize, Serialize};

use crate::{
    arena::{Arena, Index},
    asset_map::{AssetHandle, AssetMap},
    assets::{material::Material, mesh::Mesh},
    math::Transform,
};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneNodeId(Index);

#[derive(Serialize, Deserialize, Clone)]
pub struct SceneMesh {
    pub handle: AssetHandle<Mesh>,
    pub material: AssetHandle<Material>,
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub enum ScenePrimitive {
    #[default]
    Group,
    Mesh(SceneMesh),
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct SceneNode {
    pub label: Option<String>,
    pub tags: Vec<String>,
    pub enabled: bool,
    pub transform: Transform,
    pub primitive_type: ScenePrimitive,
}

pub struct Scene {
    pub nodes: Arena<SceneNode>,
    pub children: HashMap<SceneNodeId, HashSet<SceneNodeId>>,
    pub parents: HashMap<SceneNodeId, SceneNodeId>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            nodes: Arena::default(),
            children: Default::default(),
            parents: Default::default(),
        }
    }

    pub fn preload(&self, asset_map: &mut AssetMap) -> anyhow::Result<()> {
        for node in self.nodes.iter() {
            match &node.primitive_type {
                ScenePrimitive::Group => {}
                ScenePrimitive::Mesh(SceneMesh { handle, material }) => {
                    asset_map.increment_reference(handle)?;
                    asset_map.increment_reference(material)?;

                    let material = asset_map.get(material).unwrap();
                    let textures = material.get_used_textures();
                    for tex in textures {
                        asset_map.increment_reference(&tex)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn release_assets(&self, asset_map: &mut AssetMap) {
        for node in self.nodes.iter() {
            match &node.primitive_type {
                ScenePrimitive::Group => {}
                ScenePrimitive::Mesh(SceneMesh {
                    handle,
                    material: material_handle,
                }) => {
                    let Some(material) = asset_map.get(material_handle) else {
                        warn!(
                            "Material {:?} was already released",
                            material_handle.identifier()
                        );
                        continue;
                    };
                    let textures = material.get_used_textures();
                    for tex in textures {
                        asset_map.decrement_reference(tex);
                    }

                    asset_map.decrement_reference(material_handle.clone());
                    asset_map.decrement_reference(handle.clone());
                }
            }
        }
    }

    pub fn add_node(&mut self, node: SceneNode) -> SceneNodeId {
        let index = self.nodes.add(node);
        SceneNodeId(index)
    }

    pub fn remove_node(&mut self, node: SceneNodeId) {
        self.nodes.remove(node.0);
    }

    pub fn add_child(&mut self, node: SceneNodeId, child: SceneNodeId) {
        if self.nodes.get(node.0).is_none() || self.nodes.get(child.0).is_none() {
            return;
        }

        self.children.entry(node).or_default().insert(child);
        self.set_parent(child, Some(node));
    }

    pub fn set_parent(&mut self, node: SceneNodeId, new_parent: Option<SceneNodeId>) {
        if self.nodes.get(node.0).is_none() {
            return;
        }

        if let Some(new_parent) = new_parent {
            let old_parent = self.parents.insert(node, new_parent);

            if let Some(old_parent) = old_parent {
                self.children.get_mut(&old_parent).unwrap().remove(&node);
            }

            let children = self.children.entry(new_parent).or_default();
            children.insert(node);
        } else {
            self.parents.remove(&node);
        }
    }

    pub fn set_node_world_transform(&mut self, node: SceneNodeId, transform: Transform) {
        let mut transform_queue = vec![(node, transform)];
        while let Some((node, transform)) = transform_queue.pop() {
            let old_transform = if let Some(scene_node) = self.nodes.get_mut(node.0) {
                let old_transform = scene_node.transform;
                scene_node.transform = transform;
                old_transform
            } else {
                continue;
            };

            if let Some(children) = self.children.get(&node) {
                let delta_transform = transform.difference(&old_transform);
                for child in children {
                    if let Some(child_transform) = self.nodes.get(child.0) {
                        transform_queue
                            .push((*child, child_transform.transform.compose(&delta_transform)))
                    }
                }
            }
        }
    }

    pub fn set_node_world_location(&mut self, node: SceneNodeId, location: Vec3) {
        let Some(transf) = self.get_node_world_transform(node) else {
            return;
        };

        self.set_node_world_transform(node, Transform { location, ..transf })
    }

    pub fn get_node_world_transform(&mut self, node: SceneNodeId) -> Option<Transform> {
        self.nodes.get_mut(node.0).map(|node| node.transform)
    }
    pub fn add_node_world_offset(&mut self, node: SceneNodeId, offset: Transform) {
        if let Some(scene_node) = self.nodes.get_mut(node.0) {
            let transform = scene_node.transform;
            let composed = transform.compose(&offset);
            self.set_node_world_transform(node, composed);
        }
    }

    pub fn add_node_world_offset_location(&mut self, node: SceneNodeId, offset: Vec3) {
        if let Some(mut node_transform) = self.get_node_world_transform(node) {
            node_transform.location += offset;
            self.set_node_world_transform(node, node_transform);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SceneNode> {
        self.nodes.iter()
    }

    pub fn children_of(
        &self,
        scene_node_id: SceneNodeId,
    ) -> Option<impl Iterator<Item = SceneNodeId> + '_> {
        self.children.get(&scene_node_id).map(|c| c.iter().copied())
    }

    pub fn find_with_tag(&self, tag: impl AsRef<str>) -> Option<SceneNodeId> {
        self.nodes
            .iter_with_index()
            .find(|(_, node)| node.tags.iter().any(|s| s.as_str() == tag.as_ref()))
            .map(|(i, _)| SceneNodeId(i))
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

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}
