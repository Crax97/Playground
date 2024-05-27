mod scene_editor;
use glam::Quat;
pub use scene_editor::*;

use crate::math::Transform;
use egui_mgpu::egui::{self, Ui};

pub fn edit_transform(ui: &mut Ui, transform: &mut Transform) {
    egui::Grid::new(ui.next_auto_id()).show(ui, |ui| {
        ui.label("Location");
        transform.location.as_mut().iter_mut().for_each(|v| {
            ui.add(egui::DragValue::new(v).speed(0.01));
        });
        ui.end_row();

        ui.label("Scale");
        transform.scale.as_mut().iter_mut().for_each(|v| {
            ui.add(egui::DragValue::new(v).speed(0.01));
        });
        ui.end_row();

        ui.label("Rotation");
        let rotation_euler = transform.rotation.to_euler(glam::EulerRot::XYZ);
        let mut rotation_euler = [rotation_euler.0, rotation_euler.1, rotation_euler.2];

        let changed = rotation_euler.iter_mut().fold(false, |b, v| {
            ui.add(egui::DragValue::new(v).speed(0.01)).changed() | b
        });

        if changed {
            transform.rotation = Quat::from_euler(
                glam::EulerRot::XYZ,
                rotation_euler[0],
                rotation_euler[1],
                rotation_euler[2],
            );
        }
    });
}

pub fn edit_vec<T: Default>(
    ui: &mut Ui,
    elements: &mut Vec<T>,
    edit_element: impl Fn(&mut Ui, &mut T),
) {
    egui::Grid::new(ui.next_auto_id()).show(ui, |ui| {
        if ui.button("+").clicked() {
            elements.push(T::default());
        };
        if ui.button("-").clicked() {
            elements.pop();
        }
        if ui.button("clear").clicked() {
            elements.clear();
        }

        ui.end_row();

        let mut remove = None;
        for (i, elt) in elements.iter_mut().enumerate() {
            edit_element(ui, elt);
            if ui.button("X").clicked() {
                remove = Some(i);
            }

            ui.end_row();
        }

        if let Some(elt) = remove {
            elements.remove(elt);
        }
    });
}
