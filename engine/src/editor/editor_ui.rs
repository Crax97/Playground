use bevy_ecs::world::World;
use egui::Ui;

use crate::{
    components::Transform2D,
    physics::{Collider2DHandle, PhysicsContext2D, RigidBody2DHandle},
};

use super::ui_extension::UiExtension;

pub trait EditorUi: 'static {
    fn name(&self) -> &'static str;
    fn ui(&mut self, world: &mut World, ui: &mut Ui);
}

impl EditorUi for Transform2D {
    fn name(&self) -> &'static str {
        "Transform 2D"
    }

    fn ui(&mut self, _world: &mut World, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Location");
            ui.edit_numbers(self.position.coords.data.as_mut_slice());
        });
        ui.horizontal(|ui| {
            ui.label("Rotation");
            ui.edit_number(&mut self.rotation);
        });
        ui.horizontal_with_label("Layer", |ui| ui.edit_number(&mut self.layer));
        ui.horizontal(|ui| {
            ui.label("Scale");
            ui.edit_numbers(self.scale.data.as_mut_slice());
        });
    }
}

impl EditorUi for RigidBody2DHandle {
    fn name(&self) -> &'static str {
        "Rigidbody 2D"
    }

    fn ui(&mut self, world: &mut World, ui: &mut Ui) {
        let mut ctx = world.get_resource_mut::<PhysicsContext2D>().unwrap();
        let rigidbody_2d = ctx
            .rigid_body_set
            .get_mut(self.0)
            .expect("Failed to find Rigidbody 2D");
        ui.label("Todo Rigidbody 2D");
    }
}

impl EditorUi for Collider2DHandle {
    fn name(&self) -> &'static str {
        "Collider 2D"
    }

    fn ui(&mut self, world: &mut World, ui: &mut Ui) {
        let mut ctx = world.get_resource_mut::<PhysicsContext2D>().unwrap();
        let collider_2d = ctx
            .collider_set
            .get_mut(self.0)
            .expect("Failed to find Collider 2D");
        ui.label("Todo Collider 2D");
    }
}
