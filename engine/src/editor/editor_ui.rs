use bevy_ecs::world::World;
use egui::Ui;
use nalgebra::Unit;
use rapier2d::dynamics::RigidBody;

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
        egui::Grid::new("Transform 2D").show(ui, |ui| {
            ui.label("Location");
            ui.edit_numbers(self.position.coords.data.as_mut_slice());
            ui.end_row();

            ui.label("Rotation");
            ui.edit_number(&mut self.rotation);
            ui.end_row();

            ui.label("Layer");
            ui.edit_number(&mut self.layer);
            ui.end_row();

            ui.label("Scale");
            ui.edit_numbers(self.scale.data.as_mut_slice());
            ui.end_row();
        });
    }
}

macro_rules! set_get {
    ($target:expr, $get:path, $set:path, $ui:expr, $($set_params:expr,)* ) => {{
        let mut temp = $get($target);
        if $ui.edit_number(&mut temp).changed {
            $set($target, temp, $($set_params,)*);
        }
    }};
}

macro_rules! set_get_bool {
    ($target:expr, $get:path, $set:path, $ui:expr, $($set_params:expr,)* ) => {{
        let mut temp = $get($target);
        if $ui.checkbox(&mut temp, "").changed {
            $set($target, temp, $($set_params,)*);
        }
    }};
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
        egui::Grid::new("Rigidbody2D").show(ui, |ui| {
            ui.label("Enabled");
            set_get_bool!(
                rigidbody_2d,
                RigidBody::is_enabled,
                RigidBody::set_enabled,
                ui,
            );
            ui.end_row();

            ui.label("Gravity Scale");
            set_get!(
                rigidbody_2d,
                RigidBody::mass,
                RigidBody::set_additional_mass,
                ui,
                false,
            );
            ui.end_row();

            ui.label("Mass");
            set_get!(
                rigidbody_2d,
                RigidBody::gravity_scale,
                RigidBody::set_gravity_scale,
                ui,
                false,
            );
            ui.end_row();

            ui.label("CCD Enabled");
            set_get_bool!(
                rigidbody_2d,
                RigidBody::is_ccd_enabled,
                RigidBody::enable_ccd,
                ui,
            );
            ui.end_row();
        });
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

        let mut position = collider_2d.position().clone();
        let mut changed_pos = false;
        egui::Grid::new("Collider2D").show(ui, |ui| {
            ui.label("Position");
            if ui.edit_numbers(position.translation.vector.data.as_mut_slice()) {
                changed_pos = true;
            }
            ui.end_row();

            ui.label("Rotation");
            let mut rotation = position.rotation.angle();
            if ui.edit_number(&mut rotation).changed() {
                position.rotation = Unit::from_angle(rotation);
                changed_pos = true;
            }

            ui.end_row();
        });

        if changed_pos {
            collider_2d.set_position(position);
        }
    }
}
