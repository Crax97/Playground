use egui_mgpu::egui::Ui;
use egui_mgpu::egui::{self, Sense};
use log::error;

use crate::asset_map::AssetMap;
use crate::scene::serializable_scene::SerializableScene;
use crate::scene::{Scene, SceneNodeId};

use super::asset_picker::AssetPicker;
use super::{edit_transform, edit_vec};

#[derive(Default)]
pub struct SceneEditor {
    selected_node: Option<SceneNodeId>,
    asset_picker: AssetPicker,
}

impl SceneEditor {
    pub fn new() -> Self {
        Self {
            selected_node: None,
            asset_picker: AssetPicker::default(),
        }
    }
    pub fn show(&mut self, ui: &mut Ui, scene: &mut Scene, asset_map: &mut AssetMap) {
        let root_nodes = scene
            .iter_with_ids()
            .filter(|(i, _)| scene.parent_of(*i).is_none())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Save scene").clicked() {
                    if let Some(file) = rfd::FileDialog::new()
                        .add_filter("Scene file", &["json"])
                        .save_file()
                    {
                        let scene_serializable = SerializableScene::from(&*scene);
                        let scene = serde_json::to_string_pretty(&scene_serializable).unwrap();
                        let _ = std::fs::write(&file, scene)
                            .inspect_err(|e| error!("Failed to write scene to {file:?}: {e:?}"));
                    }
                }

                if ui.button("Load scene").clicked() {
                    if let Some(file) = rfd::FileDialog::new()
                        .add_filter("Scene file", &["json"])
                        .pick_file()
                    {
                        let Ok(content) = std::fs::read_to_string(file) else {
                            return;
                        };
                        let Ok(scene_serialized): Result<SerializableScene, _> =
                            serde_json::from_str(&content)
                        else {
                            return;
                        };
                        let Ok(new_scene) = scene_serialized.to_scene() else {
                            return;
                        };
                        new_scene.preload(asset_map).unwrap();
                        let old_scene = std::mem::replace(scene, new_scene);
                        old_scene.dispose(asset_map)
                    }
                }
            })
        });

        egui::CollapsingHeader::new("Scene outliner").show(ui, |ui| {
            for root_node in root_nodes {
                self.scene_node_outliner(root_node, ui, scene);
            }
        });

        egui::CollapsingHeader::new("Node Editor").show(ui, |ui| {
            if let Some(selected_node) = self.selected_node {
                self.node_ui(selected_node, ui, scene, asset_map);
            } else {
                ui.label("Select a node");
            }
        });
    }

    fn scene_node_outliner(&mut self, node: SceneNodeId, ui: &mut Ui, scene: &mut Scene) {
        let children = scene.children_of(node).unwrap().collect::<Vec<_>>();
        let node_label = scene
            .get_node(node)
            .unwrap()
            .label
            .clone()
            .unwrap_or(format!("Scene Node {:?}", node));
        let highlight = self.selected_node.is_some_and(|n| n == node);
        if children.is_empty() {
            let mut label =
                ui.add(egui::Label::new(&node_label).sense(Sense::click().union(Sense::hover())));

            if highlight {
                label = label.highlight();
            }
            if label.clicked() {
                self.selected_node = Some(node);
            }
        } else {
            let mut response = egui::CollapsingHeader::new(&node_label)
                .show(ui, |ui| {
                    for child in children {
                        self.scene_node_outliner(child, ui, scene);
                    }
                })
                .header_response;

            if highlight {
                response = response.highlight();
            }

            if response.clicked() {
                self.selected_node = Some(node);
            }
        }
    }

    fn node_ui(
        &mut self,
        node_id: SceneNodeId,
        ui: &mut Ui,
        scene: &mut Scene,
        asset_map: &mut AssetMap,
    ) {
        let node = scene.get_node_mut(node_id).unwrap();
        let mut label = node.label.clone().unwrap_or_default();
        egui::Grid::new(node_id).show(ui, |ui| {
            ui.checkbox(&mut node.enabled, "Enabled?");
            ui.end_row();
            ui.label("Node label");
            ui.text_edit_singleline(&mut label);
            if !label.is_empty() {
                node.label = Some(label);
            } else {
                node.label = None;
            }
            ui.end_row();

            match &mut node.primitive_type {
                crate::scene::ScenePrimitive::Group => {
                    ui.label("Group node");
                }
                crate::scene::ScenePrimitive::Mesh(info) => {
                    ui.group(|ui| {
                        self.asset_picker.draw(ui, &mut info.handle, asset_map);
                    });
                }
            }
            ui.end_row();

            ui.heading("Transform");
            ui.end_row();

            let transform = &mut node.transform;
            edit_transform(ui, transform);
            ui.end_row();

            ui.heading("Tags");
            ui.end_row();
            edit_vec(ui, &mut node.tags, |ui, elt| {
                ui.text_edit_singleline(elt);
            });
            ui.end_row();
        });
    }
}
