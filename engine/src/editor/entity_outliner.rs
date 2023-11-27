use std::any::TypeId;

use bevy_ecs::{
    component::Component,
    entity::Entity,
    reflect::ReflectComponent,
    world::{EntityWorldMut, World},
};
use bevy_reflect::{Reflect, TypeRegistry};
use egui::Ui;

use crate::{
    components::Transform2D,
    physics::{Collider2DHandle, RigidBody2DHandle},
};
use std::collections::HashMap;

use super::{ui_extension::UiExtension, EditorUi};

#[derive(Default)]
pub(super) struct EntityOutliner {
    selected_entity: Option<Entity>,
}

impl EntityOutliner {
    pub(super) fn draw(
        &mut self,
        components_with_custom_ui: &mut HashMap<
            TypeId,
            Box<dyn FnMut(&mut EntityWorldMut, &mut Ui)>,
        >,
        world: &mut World,
        ui: &mut Ui,
        registry: &TypeRegistry,
    ) {
        ui.label("Entity outliner");
        ui.group(|ui| {
            for entity in world.iter_entities().map(|e| e.id()) {
                if ui
                    .selectable_label(
                        self.selected_entity.is_some_and(|e| e == entity),
                        format!("{:?}", entity),
                    )
                    .clicked()
                {
                    self.selected_entity = Some(entity);
                }
            }
        });

        if let Some(entity) = self.selected_entity {
            try_edit_type::<Transform2D>(entity, world, ui);
            try_edit_type::<RigidBody2DHandle>(entity, world, ui);
            try_edit_type::<Collider2DHandle>(entity, world, ui);
            components_ui(components_with_custom_ui, entity, world, ui, registry);
        }
    }
}

fn try_edit_type<T: EditorUi + Component>(entity: Entity, world: &mut World, ui: &mut Ui) {
    let world_2 = world as *mut World;
    let mut entity_mut = world.entity_mut(entity);
    if let Some(mut component) = entity_mut.get_mut::<T>() {
        let world_2 = unsafe { world_2.as_mut().unwrap() };
        ui.heading(component.name());
        component.ui(world_2, ui);
    }
}

fn components_ui(
    components_with_custom_ui: &mut HashMap<TypeId, Box<dyn FnMut(&mut EntityWorldMut, &mut Ui)>>,
    entity: Entity,
    world: &mut World,
    ui: &mut Ui,
    registry: &TypeRegistry,
) {
    let mut entity_mut = world.entity_mut(entity);
    for ty in registry.iter() {
        if let Some(component_func) = components_with_custom_ui.get_mut(&ty.type_id()) {
            component_func(&mut entity_mut, ui);
        } else if let Some(reflect_component) = ty.data::<ReflectComponent>() {
            if let Some(mut component) = reflect_component.reflect_mut(&mut entity_mut) {
                let component_name = component
                    .get_represented_type_info()
                    .unwrap()
                    .type_path_table()
                    .ident()
                    .unwrap();
                ui.heading(component_name);
                reflect_type_recursive(component.as_reflect_mut(), ui);
            }
        }
    }
}

fn reflect_type_recursive(field: &mut dyn Reflect, ui: &mut Ui) {
    match field.reflect_mut() {
        bevy_reflect::ReflectMut::Struct(stru) => {
            egui::Grid::new(stru.reflect_type_path()).show(ui, |ui| {
                for i in 0..stru.field_len() {
                    ui.label(stru.name_at(i).unwrap());
                    let field = stru.field_at_mut(i).unwrap();
                    reflect_type_recursive(field, ui);
                    ui.end_row();
                }
            });
        }
        bevy_reflect::ReflectMut::TupleStruct(_) => todo!(),
        bevy_reflect::ReflectMut::Tuple(_) => todo!(),
        bevy_reflect::ReflectMut::List(_) => todo!(),
        bevy_reflect::ReflectMut::Array(_) => todo!(),
        bevy_reflect::ReflectMut::Map(_) => todo!(),
        bevy_reflect::ReflectMut::Enum(_) => todo!(),
        bevy_reflect::ReflectMut::Value(value) => {
            if value.is::<f32>() {
                ui.edit_number(value.as_any_mut().downcast_mut::<f32>().unwrap());
            } else if value.is::<u32>() {
                ui.edit_number(value.as_any_mut().downcast_mut::<u32>().unwrap());
            } else if value.is::<String>() {
                let str_ref = value.as_any_mut().downcast_mut::<String>().unwrap();
                ui.text_edit_singleline(str_ref);
            }
        }
    }
}
