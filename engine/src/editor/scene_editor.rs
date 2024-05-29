use egui_mgpu::egui::Ui;
use egui_mgpu::egui::{self, Sense};
use log::error;
use mgpu::Device;

use crate::asset_map::{AssetHandle, AssetMap};
use crate::scene::serializable_scene::SerializableScene;
use crate::scene::{Scene, SceneMesh, SceneNode, SceneNodeId, ScenePrimitive};

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
    pub fn show(
        &mut self,
        device: &Device,
        ui: &mut Ui,
        scene: &mut Scene,
        asset_map: &mut AssetMap,
    ) {
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
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    if ui.button("Add node").clicked() {
                        let id = scene.add_node(
                            SceneNode::default().label(format!("SceneNode {}", scene.nodes.len())),
                        );
                        self.selected_node = Some(id);
                    }
                    if ui
                        .add_enabled(self.selected_node.is_some(), egui::Button::new("Remove"))
                        .clicked()
                    {
                        scene.remove_node(self.selected_node.unwrap());
                        self.selected_node = None;
                    }
                })
            });
        });

        egui::CollapsingHeader::new("Node Editor").show(ui, |ui| {
            if let Some(selected_node) = self.selected_node {
                self.node_ui(device, selected_node, ui, scene, asset_map);
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
        device: &Device,
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

            let stringify_prim = |prim_ty: &ScenePrimitive| match prim_ty {
                ScenePrimitive::Group => "Group",
                ScenePrimitive::Mesh(_) => "Mesh",
            };

            egui::containers::ComboBox::new(
                ui.next_auto_id(),
                stringify_prim(&node.primitive_type),
            )
            .selected_text(stringify_prim(&node.primitive_type))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut node.primitive_type, ScenePrimitive::Group, "Group");
                ui.selectable_value(
                    &mut node.primitive_type,
                    ScenePrimitive::Mesh(SceneMesh {
                        handle: AssetHandle::null(),
                        material: AssetHandle::null(),
                    }),
                    "Mesh",
                );
            });
            ui.end_row();

            match &mut node.primitive_type {
                crate::scene::ScenePrimitive::Group => {
                    ui.label("Group node");
                }
                crate::scene::ScenePrimitive::Mesh(info) => {
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            self.asset_picker.modify(ui, &mut info.handle, asset_map);
                        });
                        ui.group(|ui| {
                            self.asset_picker.modify(ui, &mut info.material, asset_map);
                        });
                    });
                    ui.end_row();

                    ui.collapsing("Material info", |ui| {
                        ui.group(|ui| {
                            let tex_asset_map =
                                unsafe { (asset_map as *mut AssetMap).as_mut().unwrap() };
                            if info.material.is_null() {
                                return;
                            }
                            let Some(material) = asset_map.get_mut(&info.material) else {
                                return;
                            };
                            let mut changed_textures = true;
                            egui::Grid::new(ui.next_auto_id()).show(ui, |ui| {
                                for texture in &mut material.parameters.textures {
                                    ui.horizontal(|ui| {
                                        ui.label(&texture.name);
                                        if self.asset_picker.modify(
                                            ui,
                                            &mut texture.texture,
                                            tex_asset_map,
                                        ) {
                                            changed_textures = true;
                                        }
                                    });
                                    ui.end_row();
                                }
                            });

                            if changed_textures {
                                material
                                    .recreate_binding_set_layout(device, tex_asset_map)
                                    .unwrap();
                            }
                        });
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
