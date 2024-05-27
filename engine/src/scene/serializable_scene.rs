use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{Scene, SceneNode, SceneNodeId};

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SerializableScene {
    pub nodes: Vec<SceneNode>,
    pub adjacency_list: Vec<Vec<usize>>,
}

impl SerializableScene {
    pub fn to_scene(self) -> anyhow::Result<Scene> {
        let mut scene = Scene::new();
        let mut usize_to_scene_node_id = HashMap::new();
        for (i, node) in self.nodes.into_iter().enumerate() {
            usize_to_scene_node_id.insert(i, scene.add_node(node));
        }

        for (i, children) in self.adjacency_list.into_iter().enumerate() {
            for child in children {
                scene.add_child(usize_to_scene_node_id[&i], usize_to_scene_node_id[&child]);
            }
        }

        Ok(scene)
    }
}

impl From<&Scene> for SerializableScene {
    fn from(value: &Scene) -> Self {
        let mut nodes = vec![];
        let mut adjacency_list = vec![];
        let mut scene_node_id_to_usize = HashMap::new();

        for (index, node) in value.nodes.iter_with_index() {
            scene_node_id_to_usize.insert(SceneNodeId(index), nodes.len());
            nodes.push(node.clone());
            adjacency_list.push(vec![]);
        }

        for (i, (index, _)) in value.nodes.iter_with_index().enumerate() {
            let scene_node_id = SceneNodeId(index);
            let Some(children) = value.children_of(scene_node_id) else {
                continue;
            };
            for child in children {
                adjacency_list[i].push(scene_node_id_to_usize[&child]);
            }
        }

        SerializableScene {
            nodes,
            adjacency_list,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        asset_map::AssetHandle,
        scene::{Scene, SceneMesh, SceneNode, SceneNodeId},
    };

    use super::SerializableScene;

    fn make_node(scene: &mut Scene, name: &str) -> SceneNodeId {
        scene.add_node(
            SceneNode::default()
                .label(name)
                .primitive(crate::scene::ScenePrimitive::Mesh(SceneMesh {
                    handle: AssetHandle::new("test"),
                    material: AssetHandle::new("material"),
                }))
                .with_tag(name),
        )
    }

    #[test]
    fn serialize_scene_1() {
        let mut scene = Scene::new();
        let a = make_node(&mut scene, "a");
        let b = make_node(&mut scene, "b");
        let c = make_node(&mut scene, "c");
        let _ = make_node(&mut scene, "f");

        scene.add_child(a, b);
        scene.add_child(a, c);

        let serializable = SerializableScene::from(&scene);
        let to_json = serde_json::to_string_pretty(&serializable).unwrap();
        let serializable = serde_json::from_str::<SerializableScene>(&to_json).unwrap();
        let scene = serializable.to_scene().unwrap();

        let a = scene.find_with_tag("a").unwrap();
        let b = scene.find_with_tag("b").unwrap();
        let c = scene.find_with_tag("c").unwrap();
        let f = scene.find_with_tag("f").unwrap();

        assert!(scene.children_of(a).unwrap().any(|n| n == b));
        assert!(scene.children_of(a).unwrap().any(|n| n == c));
        assert!(scene.children_of(f).unwrap().next().is_none())
    }

    #[test]
    fn serialize_scene_2() {
        let mut scene = Scene::new();
        let a = make_node(&mut scene, "a");
        for _ in 0..1000 {
            let c = make_node(&mut scene, "c");
            scene.add_child(a, c);
        }

        let serializable = SerializableScene::from(&scene);
        let to_json = serde_json::to_string_pretty(&serializable).unwrap();
        let serializable = serde_json::from_str::<SerializableScene>(&to_json).unwrap();
        let scene = serializable.to_scene().unwrap();

        let a = scene.find_with_tag("a").unwrap();

        assert!(scene.children_of(a).unwrap().count() == 1000);
    }
}
