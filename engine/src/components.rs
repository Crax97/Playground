use std::ops::{Deref, DerefMut};

use crate::editor::{AssetPicker, TypeEditor};
use crate::kecs_app::SharedAssetMap;
use crate::material_v2::Material2;
use crate::math::shape::BoundingShape;
use crate::{asset_map::AssetHandle, LightType, ShadowConfiguration, Texture};
use bevy_ecs::reflect::ReflectComponent;
use bevy_ecs::{
    component::Component,
    system::{Resource},
};
use bevy_reflect::Reflect;



use nalgebra::{
    vector, Matrix4, Point3, UnitQuaternion, Vector2, Vector3,
};
use serde::{Deserialize, Serialize};
use winit::window::Window;

use crate::{MasterMaterial, Mesh};

pub struct EngineWindow(pub(crate) Window);

impl kecs::Resource for EngineWindow {}

impl Deref for EngineWindow {
    type Target = Window;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for EngineWindow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct DebugName(pub String);

#[derive(Serialize, Deserialize)]
pub struct Transform {
    pub position: Point3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Default::default(),
            rotation: Default::default(),
            scale: vector![1.0, 1.0, 1.0],
        }
    }
}

impl Transform {
    pub fn new_translation(position: Point3<f32>) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }
    pub fn matrix(&self) -> Matrix4<f32> {
        Matrix4::new_translation(&self.position.to_homogeneous().xyz())
            * Matrix4::new_nonuniform_scaling(&self.scale)
            * self.rotation.to_homogeneous()
    }

    pub fn forward(&self) -> Vector3<f32> {
        let matrix = self.rotation.to_rotation_matrix().to_homogeneous();
        matrix.column(2).xyz()
    }
}

#[derive(Component)]
pub struct MeshComponent {
    pub mesh: AssetHandle<Mesh>,
    pub materials: Vec<AssetHandle<Material2>>,
    pub bounding_shape: BoundingShape,
}

impl Default for MeshComponent {
    fn default() -> Self {
        Self {
            mesh: AssetHandle::null(),
            materials: Default::default(),
            bounding_shape: BoundingShape::Sphere {
                radius: 0.0,
                origin: Default::default(),
            },
        }
    }
}

impl MeshComponent {
    fn bounds(&self) -> BoundingShape {
        self.bounding_shape
    }
}

pub struct SpriteComponentDescription {
    pub texture: AssetHandle<Texture>,
    pub material: AssetHandle<MasterMaterial>,
    pub atlas_offset: Vector2<u32>,
    pub atlas_size: Vector2<u32>,
    pub sprite_size: Vector2<f32>,
    pub z_layer: u32,
}

#[derive(Component)]
pub struct LightComponent {
    pub ty: LightType,
    pub radius: f32,
    pub intensity: f32,
    pub color: Vector3<f32>,

    pub enabled: bool,
    pub shadow_setup: Option<ShadowConfiguration>,
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct TestComponent {
    pub num: u32,
    pub flo: f32,
    pub stri: String,
}

#[derive(Resource)]
pub struct SceneSetup {
    pub skybox_material: Option<AssetHandle<Material2>>,
    pub skybox_texture: Option<AssetHandle<Texture>>,
}

pub struct MeshComponentEditor {
    asset_map: SharedAssetMap,
    mesh_picker: AssetPicker,
    master_picker: AssetPicker,
    texture_picker: AssetPicker,
}

impl MeshComponentEditor {
    pub fn new(asset_map: SharedAssetMap) -> Self {
        Self {
            asset_map,
            mesh_picker: AssetPicker::default(),
            master_picker: AssetPicker::default(),
            texture_picker: AssetPicker::default(),
        }
    }
}

impl TypeEditor for MeshComponentEditor {
    type EditedType = MeshComponent;

    fn show_ui(&mut self, value: &mut Self::EditedType, ui: &mut egui::Ui) {
        let mut asset_map = self.asset_map.write();
        egui::Grid::new("mesh_editor").show(ui, |ui| {
            ui.label("Current mesh");
            self.mesh_picker.show(&mut value.mesh, &mut asset_map, ui);
            ui.end_row();

            ui.label("Materials");
            if ui.button("Add").clicked() {
                value.materials.push(AssetHandle::null());
            }

            ui.end_row();
            egui::Grid::new("mats").show(ui, |ui| {
                for _material in &mut value.materials {
                    egui::Grid::new("parameters").show(ui, |ui| {
                        ui.label("todo...");
                        ui.end_row();
                    });

                    ui.end_row();
                }
            });
        });
    }
}
