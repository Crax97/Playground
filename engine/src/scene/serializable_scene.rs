use serde::{Deserialize, Serialize};

use super::{Scene, SceneNode};

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SerializableScene {
    pub nodes: Vec<SceneNode>,
}

impl SerializableScene {
    pub fn to_scene(self) -> anyhow::Result<Scene> {
        let mut scene = Scene::new();
        for node in self.nodes {
            scene.add_node(node);
        }
        Ok(scene)
    }
}

impl From<&Scene> for SerializableScene {
    fn from(value: &Scene) -> Self {
        let mut nodes = vec![];

        for node in value.nodes.iter() {
            nodes.push(node.clone());
        }

        SerializableScene { nodes }
    }
}
