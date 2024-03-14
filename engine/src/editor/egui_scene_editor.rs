use egui::Ui;
use gpu::Gpu;
use kecs::{ComponentId, Entity, Resource, World};
use log::warn;
use nalgebra::UnitQuaternion;
use std::collections::HashMap;

use crate::{
    app::egui_support::EguiSupport, game_scene::SceneNodeId, kecs_app::Plugin, GameScene, Time,
};
use winit::window::Window;

use super::ui_extension::UiExtension;

pub trait TypeEditor {
    type EditedType: Default;

    fn show_ui(&self, value: &mut Self::EditedType, ui: &mut Ui);
}

trait ErasedTypeEditor {
    fn show_ui(&self, ui: &mut Ui, entity: Entity, world: &mut World);
    fn add_component(&self, entity: Entity, world: &mut World);
}

impl<T: TypeEditor + 'static> ErasedTypeEditor for T {
    fn show_ui(&self, ui: &mut Ui, entity: Entity, world: &mut World) {
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
    current_node: Option<SceneNodeId>,
    registered_types: HashMap<ComponentId, Box<dyn ErasedTypeEditor>>,
}

#[derive(Default)]
struct CurrentEditorScene(GameScene);

impl Resource for CurrentEditorScene {}

impl EguiSceneEditor {
    pub fn new(window: &Window, gpu: &dyn Gpu) -> anyhow::Result<Self> {
        Ok(Self {
            egui_support: EguiSupport::new(window, gpu)?,
            current_node: None,
            registered_types: HashMap::new(),
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

    fn draw_scene_node(&self, ui: &mut Ui, scene: &mut GameScene, current_node: SceneNodeId) {
        let node = scene.get_node_mut(current_node).unwrap();

        ui.text_edit_singleline(&mut node.label);
        ui.group(|ui| {
            let mut node_transform = scene
                .get_transform(current_node, crate::game_scene::TransformSpace::World)
                .unwrap();
            let mut changed_transform = false;
            changed_transform |=
                ui.input_floats("Position", node_transform.position.coords.as_mut_slice());
            let (r, p, y) = node_transform.rotation.euler_angles();
            let mut rotation_euler = [r.to_degrees(), p.to_degrees(), y.to_degrees()];
            changed_transform |=
                ui.input_floats("Rotation (roll, pitch, yaw)", &mut rotation_euler);
            changed_transform |= ui.input_floats("Scale", node_transform.scale.as_mut_slice());

            if changed_transform {
                node_transform.rotation = UnitQuaternion::from_euler_angles(
                    rotation_euler[0].to_radians(),
                    rotation_euler[1].to_radians(),
                    rotation_euler[2].to_radians(),
                );
                scene.set_transform(
                    current_node,
                    node_transform,
                    crate::game_scene::TransformSpace::World,
                );
            }
        });
    }
}

impl Plugin for EguiSceneEditor {
    fn on_start(&mut self, world: &mut kecs::World) {
        world.add_resource(CurrentEditorScene::default());
    }

    fn on_event(
        &mut self,
        app_state: &crate::app::app_state::AppState,
        _world: &mut kecs::World,
        event: &winit::event::Event<()>,
    ) {
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

    fn update(&mut self, _world: &mut kecs::World) {}

    fn post_update(&mut self, _world: &mut kecs::World) {}

    fn draw(
        &mut self,
        world: &mut kecs::World,
        app_state: &mut crate::app::app_state::AppState,
        backbuffer: &crate::Backbuffer,
        command_buffer: &mut gpu::CommandBuffer,
    ) -> anyhow::Result<()> {
        let time = world.get_resource::<Time>().unwrap();

        self.egui_support.begin_frame(&app_state.window, time);

        let mut current_selected_entity = None;
        let mut wants_to_add_entity = false;

        let context = self.egui_support.create_context();
        let _ = egui::Window::new("Entities").show(&context, |ui| {
            let scene = world.get_resource_mut::<CurrentEditorScene>();
            if let Some(scene) = scene {
                let roots = scene.0.root_nodes().cloned().collect::<Vec<_>>();
                roots.into_iter().for_each(|root_node| {
                    Self::draw_entity_outliner(ui, &mut scene.0, root_node, &mut self.current_node);
                });

                if ui.button("Add New Entity").clicked() {
                    wants_to_add_entity = true;
                }

                ui.collapsing("Current Scene Node", |ui| {
                    if let Some(current_node) = self.current_node {
                        current_selected_entity =
                            Some(scene.0.get_node(current_node).unwrap().payload);
                        self.draw_scene_node(ui, &mut scene.0, current_node);
                    }
                });
                if wants_to_add_entity {
                    let entity = world.new_entity();

                    let scene = world.get_resource_mut::<CurrentEditorScene>().unwrap();
                    scene
                        .0
                        .add_node(entity)
                        .label(format!("{:?}", entity))
                        .build();
                }
            }

            egui::CollapsingHeader::new("Current Entity Components").show(ui, |ui| {
                if let Some(entity) = current_selected_entity {
                    let mut selected_component = None;
                    egui::ComboBox::new("edit_entity", "")
                        .selected_text("Add new component")
                        .show_ui(ui, |ui| {
                            for component in self.registered_types.keys() {
                                if ui.selectable_label(false, component.name()).clicked() {
                                    selected_component = Some(*component);
                                }
                            }
                        });

                    if let (Some(id), Some(entity)) = (selected_component, current_selected_entity)
                    {
                        let editor = self.registered_types.get(&id).expect("No editor");
                        editor.add_component(entity, world);
                    }
                    let entity_info = world.get_entity_info(entity);
                    if let Some(info) = entity_info {
                        for (component, _) in info.components.iter() {
                            egui::CollapsingHeader::new(component.name()).show(ui, |ui| {
                                if let Some(editor) = self.registered_types.get(&component) {
                                    editor.show_ui(ui, entity, world)
                                }
                            });
                        }
                    }
                }
            });
        });

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
}

impl EguiSceneEditor {
    fn draw_entity_outliner(
        ui: &mut Ui,
        scene: &mut GameScene,
        current_node: SceneNodeId,
        selected_node: &mut Option<SceneNodeId>,
    ) {
        let selected = selected_node.is_some_and(|n| n == current_node);
        let node = scene.get_node_mut(current_node).unwrap();
        let label = node.label.clone();
        let children = scene
            .get_children(current_node)
            .unwrap()
            .cloned()
            .collect::<Vec<_>>();

        let clicked = egui::CollapsingHeader::new(label)
            .show_background(selected)
            .show(ui, |ui| {
                for child in children {
                    Self::draw_entity_outliner(ui, scene, child, selected_node);
                }
            })
            .header_response
            .clicked();
        if clicked {
            *selected_node = Some(current_node);
        }
    }
}
