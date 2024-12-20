pub mod serializable_scene;

use std::collections::{HashMap, HashSet};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::{
    arena::{Arena, Index},
    asset_map::{AssetHandle, AssetMap},
    assets::{material::Material, mesh::Mesh},
    math::Transform,
};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneNodeId(Index);

#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct SceneMesh {
    pub handle: AssetHandle<Mesh>,
    pub material: AssetHandle<Material>,
}

#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
pub enum LightType {
    #[default]
    Directional,
    Point {
        radius: f32,
    },
    Spot {
        radius: f32,
        inner_angle: f32,
        outer_angle: f32,
    }
}

#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct LightInfo {
    pub ty: LightType,
    pub color: Vec3,
    pub strength: f32,
}

#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
pub enum ScenePrimitive {
    #[default]
    Group,
    Mesh(SceneMesh),
    Light(LightInfo)
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
                ScenePrimitive::Group | ScenePrimitive::Light(_) => {}
                ScenePrimitive::Mesh(SceneMesh { handle, material }) => {
                    if !handle.is_null() {
                        asset_map.increment_reference(handle)?;
                    }

                    if !material.is_null() {
                        asset_map.increment_reference(material)?;
                    }

                    let Some(material) = asset_map.get(material) else {
                        continue;
                    };
                    let textures = material.get_used_textures();
                    for tex in textures {
                        if tex.is_null() {
                            asset_map.increment_reference(&tex)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn dispose(&self, asset_map: &mut AssetMap) {
        for node in self.nodes.iter() {
            match &node.primitive_type {
                ScenePrimitive::Group | ScenePrimitive::Light(_) => {}
                ScenePrimitive::Mesh(SceneMesh {
                    handle,
                    material: material_handle,
                }) => {
                    let Some(material) = asset_map.get(material_handle) else {
                        continue;
                    };
                    let textures = material.get_used_textures();
                    for tex in textures {
                        if tex.is_null() {
                            asset_map.increment_reference(&tex).unwrap();
                        }
                    }
                    if !handle.is_null() {
                        asset_map.increment_reference(handle).unwrap();
                    }

                    if !material_handle.is_null() {
                        asset_map.increment_reference(material_handle).unwrap();
                    }
                }
            }
        }
    }

    pub fn add_node(&mut self, node: SceneNode) -> SceneNodeId {
        let index = self.nodes.add(node);
        self.children.insert(SceneNodeId(index), Default::default());
        SceneNodeId(index)
    }

    pub fn remove_node(&mut self, node: SceneNodeId) {
        self.nodes.remove(node.0);
        self.parents.remove(&node);
        self.children.remove(&node).unwrap();
    }

    pub fn add_child(&mut self, node: SceneNodeId, child: SceneNodeId) {
        assert!(node != child);
        if self.nodes.get(node.0).is_none() || self.nodes.get(child.0).is_none() {
            return;
        }

        self.children.get_mut(&node).unwrap().insert(child);
        self.set_parent(child, Some(node));
    }

    pub fn set_parent(&mut self, node: SceneNodeId, new_parent: Option<SceneNodeId>) {
        if self.nodes.get(node.0).is_none() {
            return;
        }

        if let Some(new_parent) = new_parent {
            assert!(new_parent != node);
            let old_parent = self.parents.insert(node, new_parent);

            if let Some(old_parent) = old_parent {
                self.children.get_mut(&old_parent).unwrap().remove(&node);
            }

            let children = self.children.get_mut(&new_parent).unwrap();
            children.insert(node);
        } else {
            self.parents.remove(&node);
        }
    }

    pub fn set_node_world_transform(&mut self, node: SceneNodeId, transform: Transform) {
        let mut transform_queue = vec![(node, transform)];
        if self.nodes.get(node.0).is_none() {
            return;
        }
        while let Some((node, transform)) = transform_queue.pop() {
            let scene_node = self.nodes.get_mut(node.0).unwrap();
            let old_transform = std::mem::replace(&mut scene_node.transform, transform);
            let children = self.children.get(&node).unwrap();
            if children.is_empty() {
                continue;
            }

            let delta_transform = transform.difference(&old_transform);
            for child in children {
                let child_transform = self.nodes.get(child.0).unwrap();
                transform_queue.push((*child, child_transform.transform.compose(&delta_transform)))
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

    pub fn iter_with_ids(&self) -> impl Iterator<Item = (SceneNodeId, &SceneNode)> {
        self.nodes
            .iter_with_index()
            .map(|(i, n)| (SceneNodeId(i), n))
    }

    pub fn parent_of(&self, node: SceneNodeId) -> Option<SceneNodeId> {
        self.parents.get(&node).copied()
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

    pub fn get_node(&self, node: SceneNodeId) -> Option<&SceneNode> {
        self.nodes.get(node.0)
    }

    pub fn get_node_mut(&mut self, node: SceneNodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(node.0)
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
