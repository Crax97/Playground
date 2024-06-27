use egui_mgpu::egui::{self, Sense, Stroke};
use egui_mgpu::egui::{Context, Ui};
use glam::{vec3, Quat};
use log::error;
use mgpu::Device;
use transform_gizmo_egui::math::Transform as GizmoTransform;
use transform_gizmo_egui::{enum_set, Gizmo, GizmoConfig, GizmoExt, GizmoMode, GizmoOrientation};

use crate::asset_map::{AssetHandle, AssetMap};
use crate::math::Transform;
use crate::scene::serializable_scene::SerializableScene;
use crate::scene::{Scene, SceneMesh, SceneNode, SceneNodeId, ScenePrimitive};
use crate::scene_renderer::PointOfView;

use super::asset_picker::AssetPicker;
use super::{edit_transform, edit_vec};

pub struct SceneEditor {
    selected_node: Option<SceneNodeId>,
    asset_picker: AssetPicker,
    gizmo: Gizmo,
}

impl SceneEditor {
    pub fn new() -> Self {
        Self {
            selected_node: None,
            asset_picker: AssetPicker::default(),
            gizmo: transform_gizmo_egui::Gizmo::new(GizmoConfig {
                modes: enum_set!(GizmoMode::Rotate | GizmoMode::Translate | GizmoMode::Scale),
                orientation: GizmoOrientation::Local,
                ..Default::default()
            }),
        }
    }
    pub fn show(
        &mut self,
        device: &Device,
        context: &Context,
        scene: &mut Scene,
        asset_map: &mut AssetMap,
    ) {
        let root_nodes = scene
            .iter_with_ids()
            .filter(|(i, _)| scene.parent_of(*i).is_none())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: egui::Color32::TRANSPARENT,
                stroke: Stroke::NONE,
                ..Default::default()
            })
            .show(context, |ui| {
                self.draw_gizmo(ui, scene);
            });

        egui::panel::TopBottomPanel::top("EditorTools").show(context, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Save scene").clicked() {
                        if let Some(file) = rfd::FileDialog::new()
                            .add_filter("Scene file", &["json"])
                            .save_file()
                        {
                            let scene_serializable = SerializableScene::from(&*scene);
                            let scene = serde_json::to_string_pretty(&scene_serializable).unwrap();
                            let _ = std::fs::write(&file, scene).inspect_err(|e| {
                                error!("Failed to write scene to {file:?}: {e:?}")
                            });
                        }
                    }

                    if ui.button("Load scene").clicked() {
                        self.selected_node = None;
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
        });

        egui::SidePanel::right("SceneStuff").show(context, |ui| {
            egui::CollapsingHeader::new("Scene outliner").show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for root_node in root_nodes {
                        self.scene_node_outliner(root_node, ui, scene);
                    }
                });
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Add node").clicked() {
                            let id = scene.add_node(
                                SceneNode::default()
                                    .label(format!("SceneNode {}", scene.nodes.len())),
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

                        if ui
                            .add_enabled(self.selected_node.is_some(), egui::Button::new("Duplicate"))
                            .clicked() && self.selected_node.is_some()
                        {
                            let id = scene.add_node(scene.get_node(self.selected_node.unwrap()).unwrap().clone());
                            self.selected_node = Some(id);
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
        });
    }

    fn draw_gizmo(&mut self, ui: &mut Ui, scene: &mut Scene) {
        let viewport = ui.clip_rect();
        let cfg = *self.gizmo.config();
        self.gizmo.update_config(GizmoConfig { viewport, ..cfg });
        if let Some(node) = self.selected_node.and_then(|id| scene.get_node_mut(id)) {
            let transform = node.transform;
            if let Some((_, transforms)) = self.gizmo.interact(
                ui,
                &[GizmoTransform::from_scale_rotation_translation(
                    transform.scale.as_dvec3(),
                    transform.rotation.as_dquat().normalize(),
                    transform.location.as_dvec3(),
                )],
            ) {
                let GizmoTransform {
                    translation,
                    rotation,
                    scale,
                } = transforms[0];
                let transform = Transform {
                    location: vec3(
                        translation.x as f32,
                        translation.y as f32,
                        translation.z as f32,
                    ),
                    rotation: Quat::from_xyzw(
                        rotation.v.x as _,
                        rotation.v.y as _,
                        rotation.v.z as _,
                        rotation.s as _,
                    )
                    .normalize(),
                    scale: vec3(scale.x as _, scale.y as _, scale.z as _),
                };
                node.transform = transform;
            }
        }
    }

    pub fn update_pov(&mut self, pov: &PointOfView, pixels_per_point: f32) {
        self.gizmo.update_config(GizmoConfig {
            view_matrix: pov.view_matrix().as_dmat4().into(),
            projection_matrix: pov.projection_matrix().as_dmat4().into(),
            modes: enum_set!(GizmoMode::Rotate | GizmoMode::Translate | GizmoMode::Scale),
            orientation: GizmoOrientation::Global,
            pixels_per_point,
            ..Default::default()
        })
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
                            let mut changed_textures = false;
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
                                material.recreate_binding_set_layout(tex_asset_map).unwrap();
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

impl Default for SceneEditor {
    fn default() -> Self {
        Self::new()
    }
}
