use egui::{Id, Sense, Separator, Ui};
use gpu::Gpu;
use kecs::{ComponentId, Entity, World};
use log::warn;
use nalgebra::UnitQuaternion;
use std::collections::HashMap;

use crate::{
    app::{app_state::AppState, egui_support::EguiSupport},
    fps_camera::FpsCamera,
    immutable_string::ImmutableString,
    input::InputState,
    kecs_app::{Plugin, SharedAssetMap, SimulationState, SimulationStep},
    AssetHandle, AssetMap, GameScene, PrimitiveHandle, SceneLightInfo, ScenePrimitive,
    ScenePrimitiveType, Time,
};
use winit::{
    dpi::{PhysicalPosition, Position},
    event::MouseButton,
    window::Window,
};

use super::ui_extension::UiExtension;

const PLAY: &str = "\u{23F5}";
const PAUSE: &str = "\u{23F8}";
const STOP: &str = "\u{23F9}";

pub trait TypeEditor {
    type EditedType: Default;

    fn show_ui(&mut self, value: &mut Self::EditedType, ui: &mut Ui);
}

trait ErasedTypeEditor {
    fn show_ui(&mut self, ui: &mut Ui, entity: Entity, world: &mut World);
    fn add_component(&self, entity: Entity, world: &mut World);
}

impl<T: TypeEditor + 'static> ErasedTypeEditor for T {
    fn show_ui(&mut self, ui: &mut Ui, entity: Entity, world: &mut World) {
        let component = world
            .get_component_mut::<T::EditedType>(entity)
            .expect("No component");

        self.show_ui(component, ui)
    }

    fn add_component(&self, entity: Entity, world: &mut World) {
        world.add_component(entity, T::EditedType::default());
    }
}

pub struct EguiSceneEditor {
    egui_support: EguiSupport,
    current_node: Option<PrimitiveHandle>,
    registered_types: HashMap<ComponentId, Box<dyn ErasedTypeEditor>>,
    input: InputState,
    camera: FpsCamera,
    asset_picker: AssetPicker,

    wants_to_add_scene_node: bool,
    new_scene_node: ScenePrimitive,
}

impl EguiSceneEditor {
    pub fn new(window: &Window, gpu: &dyn Gpu) -> anyhow::Result<Self> {
        Ok(Self {
            egui_support: EguiSupport::new(window, gpu)?,
            current_node: None,
            registered_types: HashMap::new(),
            input: InputState::default(),
            camera: FpsCamera::default(),
            asset_picker: Default::default(),
            wants_to_add_scene_node: false,
            new_scene_node: ScenePrimitive::default(),
        })
    }

    pub fn register_type<E: TypeEditor + 'static>(&mut self, world: &mut World, editor: E) {
        let registration = world.get_type_registration::<E::EditedType>();
        let registration = self.registered_types.insert(registration, Box::new(editor));
        if registration.is_some() {
            warn!(
                "A registration for type {} already exists",
                std::any::type_name::<E::EditedType>()
            );
        }
    }

    fn draw_scene_node(
        ui: &mut Ui,
        node: &mut ScenePrimitive,
        asset_picker: &mut AssetPicker,
        asset_map: &mut AssetMap,
    ) {
        egui::Grid::new("edit_node").show(ui, |ui| {
            let node_transform = &mut node.transform;
            let mut changed_transform = false;

            let (r, p, y) = node_transform.rotation.euler_angles();
            let mut rotation_euler = [r.to_degrees(), p.to_degrees(), y.to_degrees()];

            ui.label("Node Name");
            ui.text_edit_singleline(&mut node.label);
            ui.end_row();

            egui::ComboBox::new(&node.label, "Node type")
                .selected_text(match node.ty {
                    ScenePrimitiveType::Empty => "Empty",
                    ScenePrimitiveType::Mesh(_) => "Mesh",
                    ScenePrimitiveType::Light(_) => "Light",
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(matches!(node.ty, ScenePrimitiveType::Empty), "Empty")
                        .clicked()
                    {
                        node.ty = ScenePrimitiveType::Empty;
                    }

                    if ui
                        .selectable_label(matches!(node.ty, ScenePrimitiveType::Mesh(_)), "Mesh")
                        .clicked()
                    {
                        node.ty = ScenePrimitiveType::Mesh(crate::SceneMesh::default());
                    }

                    if ui
                        .selectable_label(matches!(node.ty, ScenePrimitiveType::Light(_)), "Light")
                        .clicked()
                    {
                        node.ty = ScenePrimitiveType::Light(crate::SceneLightInfo::default());
                    }
                });
            ui.end_row();

            match &mut node.ty {
                ScenePrimitiveType::Empty => {}
                ScenePrimitiveType::Mesh(m) => Self::scene_mesh_ui(ui, m, asset_picker, asset_map),
                ScenePrimitiveType::Light(l) => Self::scene_light_ui(ui, l),
            }

            changed_transform |=
                ui.input_floats("Position", node_transform.position.coords.as_mut_slice());
            ui.end_row();

            changed_transform |=
                ui.input_floats("Rotation (roll, pitch, yaw)", &mut rotation_euler);
            ui.end_row();

            changed_transform |= ui.input_floats("Scale", node_transform.scale.as_mut_slice());
            ui.end_row();

            ui.collapsing("Tags", |ui| {
                vec_ui(ui, &mut node.tags, |ui, tag| {
                    ui.text_edit_singleline(tag);
                })
            });
            ui.end_row();

            if node.label.is_empty() {
                node.label = "(None)".to_owned();
            }

            if changed_transform {
                node_transform.rotation = UnitQuaternion::from_euler_angles(
                    rotation_euler[0].to_radians(),
                    rotation_euler[1].to_radians(),
                    rotation_euler[2].to_radians(),
                );
            }
        });
    }

    fn editor_ui(&mut self, world: &mut kecs::KecsWorld) {
        let context = self.egui_support.create_context();
        egui::TopBottomPanel::top("toolbar").show(&context, |ui| {
            // let _ = egui::menu::bar(ui, |ui| {});
            egui::menu::bar(ui, |ui| {
                egui::menu::menu_button(ui, "test", |_| {});

                ui.add(Separator::default().vertical());
                play_pause_button(world, ui);
            });
        });
        if self.wants_to_add_scene_node {
            egui::Window::new("Configure new Scene Node")
                .open(&mut self.wants_to_add_scene_node)
                .show(&context, |ui| {
                    let asset_map = world.get_resource::<SharedAssetMap>().unwrap().clone();
                    let asset_map = &mut asset_map.write();
                    Self::draw_scene_node(
                        ui,
                        &mut self.new_scene_node,
                        &mut self.asset_picker,
                        asset_map,
                    );
                    ui.separator();

                    if ui.button("Confirm").clicked() {
                        let node = std::mem::take(&mut self.new_scene_node);
                        let scene = world.get_resource_mut::<GameScene>().expect("No scene");
                        scene.add_primitive(node);
                    }
                });
        } else {
            egui::SidePanel::new(egui::panel::Side::Right, "scene_entity").show(&context, |ui| {
                self.scene_editor(world, ui);
            });
        }
        // let current_selected_entity = None;

        // egui::CollapsingHeader::new("Current Entity Components").show(ui, |ui| {
        //     if let Some(entity) = current_selected_entity {
        //         let mut selected_component = None;
        //         egui::ComboBox::new("edit_entity", "")
        //             .selected_text("Add new component")
        //             .show_ui(ui, |ui| {
        //                 for component in self.registered_types.keys() {
        //                     if ui.selectable_label(false, component.name()).clicked() {
        //                         selected_component = Some(*component);
        //                     }
        //                 }
        //             });

        //         if let (Some(id), Some(entity)) = (selected_component, current_selected_entity)
        //         {
        //             let editor = self.registered_types.get(&id).expect("No editor");
        //             editor.add_component(entity, world);
        //         }
        //         let entity_info = world.get_entity_info(entity);
        //         if let Some(info) = entity_info {
        //             for (component, _) in info.components.iter() {
        //                 if let Some(editor) = self.registered_types.get_mut(&component) {
        //                     egui::CollapsingHeader::new(component.name())
        //                         .show(ui, |ui| editor.show_ui(ui, entity, world));
        //                 }
        //             }
        //         }
        //     }
        // });
    }

    fn scene_editor(&mut self, world: &mut kecs::KecsWorld, ui: &mut Ui) {
        let asset_map = world.get_resource::<SharedAssetMap>().unwrap().clone();
        let asset_map = &mut asset_map.write();
        ui.horizontal(|ui| {
            ui.heading("Scene Editor");

            if ui.button("Add New Scene Node").clicked() {
                self.wants_to_add_scene_node = true;
                self.new_scene_node = ScenePrimitive::default();
            }
        });
        let mut scene = world.get_resource_mut::<GameScene>();
        let height = ui.available_height() * 0.4;

        ui.group(|ui| {
            egui::ScrollArea::vertical()
                .max_height(height)
                .show(ui, |ui| {
                    if let Some(scene) = &mut scene {
                        scene.all_primitives().for_each(|(handle, _)| {
                            Self::draw_entity_outliner(ui, scene, handle, &mut self.current_node);
                        });
                    }
                });
        });
        ui.separator();
        ui.group(|ui| {
            if let (Some(current_node), Some(scene)) = (self.current_node, scene) {
                let current_node = scene.get_mut(current_node);
                Self::draw_scene_node(ui, current_node, &mut self.asset_picker, asset_map);
            } else {
                ui.label("No entity selected");
            }
        });
    }

    fn scene_mesh_ui(
        ui: &mut Ui,
        m: &mut crate::SceneMesh,
        asset_picker: &mut AssetPicker,
        asset_map: &mut AssetMap,
    ) {
        egui::Grid::new("mesh").show(ui, |ui| {
            ui.label("Current mesh");
            asset_picker.show(&mut m.mesh, asset_map, ui);
            ui.end_row();

            ui.label("Materials");
            ui.end_row();
            vec_ui(ui, &mut m.materials, |ui, _mat_inst| {
                ui.label("todo...");
            });

            ui.end_row();
        });
    }

    fn scene_light_ui(ui: &mut Ui, light: &mut SceneLightInfo) {
        egui::Grid::new("light edit").show(ui, |ui| {
            ui.checkbox(&mut light.enabled, "Enabled");
            ui.end_row();

            ui.color_edit3(
                "Light color",
                light.color.data.as_mut_slice().try_into().unwrap(),
            );
            ui.end_row();

            ui.label("Intensity");
            ui.add(
                egui::DragValue::new(&mut light.intensity)
                    .clamp_range(0.0..=f32::MAX)
                    .speed(0.01),
            );
            ui.end_row();

            ui.label("Radius");
            ui.add(
                egui::DragValue::new(&mut light.radius)
                    .clamp_range(0.0..=f32::MAX)
                    .speed(0.01),
            );
            ui.end_row();

            if let Some(shadow_config) = &mut light.shadow_configuration {
                ui.label("Depth bias");
                ui.add(egui::DragValue::new(&mut shadow_config.depth_bias).speed(0.01));
                ui.end_row();

                ui.label("Depth slope");
                ui.add(egui::DragValue::new(&mut shadow_config.depth_slope).speed(0.01));
                ui.end_row();

                ui.label("Shadow map width/height");
                ui.add(
                    egui::DragValue::new(&mut shadow_config.shadow_map_width).clamp_range(0..=4096),
                );
                ui.add(
                    egui::DragValue::new(&mut shadow_config.shadow_map_width).clamp_range(0..=4096),
                );
                ui.end_row();
            }
        });
    }
}

impl Plugin for EguiSceneEditor {
    fn on_start(&mut self, world: &mut kecs::World) {
        let status = world.get_resource_mut::<SimulationState>().unwrap();
        status.step = SimulationStep::Idle;
    }

    fn on_event(
        &mut self,
        app_state: &crate::app::app_state::AppState,
        _world: &mut kecs::World,
        event: &winit::event::Event<()>,
    ) {
        self.input.update(event);
        if let winit::event::Event::WindowEvent { event, .. } = event {
            self.egui_support
                .handle_window_event(&app_state.window, event);
        }
    }

    fn on_resize(
        &mut self,
        _world: &mut kecs::World,
        _app_state: &crate::app::app_state::AppState,
        _new_size: winit::dpi::PhysicalSize<u32>,
    ) {
    }

    fn pre_update(&mut self, _world: &mut kecs::World) {}

    fn update(&mut self, world: &mut kecs::World, state: &mut AppState) {
        if self.input.is_mouse_button_just_pressed(MouseButton::Right) {
            state
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .unwrap();

            state.window.set_cursor_visible(false);
        }

        if self.input.is_mouse_button_just_released(MouseButton::Right) {
            state
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            state.window.set_cursor_visible(true);
        }
        if self
            .input
            .is_mouse_button_pressed(winit::event::MouseButton::Right)
        {
            let time = world.get_resource::<Time>().unwrap();
            self.camera.update(&self.input, time.delta_frame());
            let window_size = state.window.inner_size();

            state
                .window
                .set_cursor_position(Position::Physical(PhysicalPosition {
                    x: window_size.width as i32 / 2,
                    y: window_size.height as i32 / 2,
                }))
                .expect("Failed to set cursor pos");
        }

        world.add_resource(self.camera.camera());
    }

    fn post_update(&mut self, _world: &mut kecs::World) {
        self.input.end_frame();
    }

    fn draw(
        &mut self,
        world: &mut kecs::World,
        app_state: &mut crate::app::app_state::AppState,
        backbuffer: &crate::Backbuffer,
        command_buffer: &mut gpu::CommandBuffer,
    ) -> anyhow::Result<()> {
        let time = world.get_resource::<Time>().unwrap();

        self.egui_support.begin_frame(&app_state.window, time);

        self.editor_ui(world);

        let output = self.egui_support.end_frame(&app_state.window);
        self.egui_support.paint_frame(
            app_state.gpu(),
            command_buffer,
            backbuffer,
            output.textures_delta,
            output.shapes,
        )?;
        self.egui_support
            .handle_platform_output(&app_state.window, output.platform_output);

        Ok(())
    }
    fn shutdown(&mut self, gpu: &dyn Gpu) {
        self.egui_support.destroy(gpu);
    }
}

fn play_pause_button(world: &mut kecs::KecsWorld, ui: &mut Ui) {
    let status = world.get_resource_mut::<SimulationState>().unwrap();

    if status.is_stopped() {
        if ui.button(PLAY).clicked() {
            status.play();
        }
    } else {
        let play_pause = if status.is_paused() { PLAY } else { PAUSE };
        if ui.button(play_pause).clicked() {
            status.toggle_play_pause()
        }
        if ui.button(STOP).clicked() {
            status.stop();
        }
    }
}

impl EguiSceneEditor {
    fn draw_entity_outliner(
        ui: &mut Ui,
        scene: &GameScene,
        current_node: PrimitiveHandle,
        selected_node: &mut Option<PrimitiveHandle>,
    ) {
        ui.fill_width();
        let selected = selected_node.is_some_and(|n| n == current_node);
        let node = scene.get(current_node);
        // let children = scene
        //     .get_children(current_node)
        //     .unwrap()
        //     .cloned()
        //     .collect::<Vec<_>>();

        let clicked = ui
            .add(egui::SelectableLabel::new(selected, &node.label))
            .clicked();
        if clicked {
            *selected_node = Some(current_node);
        }
    }
}

pub fn vec_ui<T: Default>(
    ui: &mut Ui,
    elements: &mut Vec<T>,
    mut elem_ui: impl FnMut(&mut Ui, &mut T),
) {
    ui.collapsing("Values", |ui| {
        ui.horizontal(|ui| {
            if ui.button("+").clicked() {
                elements.push(T::default());
            }
            if ui.button("-").clicked() {
                elements.pop();
            }
            if ui.button("Clear").clicked() {
                elements.clear();
            }
        });

        egui::Grid::new(ui.next_auto_id()).show(ui, |ui| {
            for (i, el) in elements.iter_mut().enumerate() {
                ui.label(format!("#{}", i));
                elem_ui(ui, el);
                ui.end_row();
            }
        });
    });
}

#[derive(Default)]
pub struct AssetPicker {
    is_shown: bool,
    selected_id: Option<ImmutableString>,
    picker_id: Option<Id>,
}

impl AssetPicker {
    pub fn show<T: crate::asset_map::Asset>(
        &mut self,
        handle: &mut AssetHandle<T>,
        asset_map: &mut AssetMap,
        ui: &mut Ui,
    ) {
        let button_label = handle.to_string();

        let mut open = self.is_shown;
        if open && self.picker_id.is_some_and(|id| id == ui.id()) {
            egui::Window::new("Pick an asset")
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(400.0)
                        .max_width(300.0)
                        .show(ui, |ui| {
                            ui.fill_width();
                            asset_map.iter_ids::<T>(|id| {
                                let highlight = self.selected_id.as_ref().is_some_and(|i| i == &id);
                                if ui.selectable_label(highlight, id.to_string()).clicked() {
                                    self.selected_id = Some(id);
                                }
                            });
                        });

                    if ui
                        .add(
                            egui::Button::new("Confirm").sense(if self.selected_id.is_some() {
                                Sense::click()
                            } else {
                                Sense::focusable_noninteractive()
                            }),
                        )
                        .clicked()
                    {
                        if let Some(id) = self.selected_id.take() {
                            *handle = asset_map.upcast_index(id);
                            self.is_shown = false;
                            self.picker_id = None;
                        }
                    }

                    if ui.button("Load new asset").clicked() {
                        let path = rfd::FileDialog::new()
                            .set_title("Pick an asset")
                            .pick_file();
                        if let Some(path) = path {
                            match asset_map.load::<T>(path.to_str().unwrap()) {
                                Ok(_) => {}
                                Err(e) => {
                                    log::error!("While loading asset: {e:?}");
                                }
                            }
                        }
                    }
                });
        }

        if self.is_shown && !open {
            self.is_shown = false;
            self.selected_id = None;
            self.picker_id = None;
        }

        let button_sense = if self.is_shown || self.picker_id.is_some() {
            Sense::focusable_noninteractive()
        } else {
            Sense::click()
        };

        if ui
            .add(egui::Button::new(button_label).sense(button_sense))
            .clicked()
        {
            self.is_shown = true;
            self.picker_id = Some(ui.id());
        }
    }
}
