use std::ops::{Deref, DerefMut};

use crate::editor::TypeEditor;
use crate::kecs_app::SharedAssetMap;
use crate::math::shape::BoundingShape;
use crate::{
    asset_map::AssetHandle, bevy_ecs_app::CommonResources, LightType, RenderScene,
    ShadowConfiguration, Texture,
};
use bevy_ecs::reflect::ReflectComponent;
use bevy_ecs::{
    component::Component,
    schedule::Schedule,
    system::{Commands, Query, Res, Resource},
    world::World,
};
use bevy_reflect::Reflect;
use bytemuck::{Pod, Zeroable};
use egui::Ui;
use gpu::{BufferCreateInfo, BufferHandle, BufferUsageFlags, Gpu, MemoryDomain};
use nalgebra::{
    point, vector, Matrix4, Point2, Point3, UnitQuaternion, UnitVector3, Vector2, Vector3, Vector4,
};
use winit::window::Window;

use crate::{
    AssetMap, EntityToSceneNode, GameScene, GpuDevice, MasterMaterial, MaterialInstance, Mesh,
};

#[derive(Resource)]
pub struct EngineWindow(pub(crate) Window);

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

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct DebugName(pub String);

#[derive(Component, Copy, Clone)]
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

#[derive(Component, Debug, Clone, Copy)]
pub struct Transform2D {
    pub position: Point2<f32>,
    pub layer: u32,
    pub rotation: f32,
    pub scale: Vector2<f32>,
}

impl Default for Transform2D {
    fn default() -> Self {
        Self {
            position: Default::default(),
            layer: 0,
            rotation: 0.0,
            scale: vector![1.0, 1.0],
        }
    }
}

impl Transform2D {
    pub fn matrix(&self) -> Matrix4<f32> {
        Matrix4::new_translation(&vector![
            self.position.x,
            self.position.y,
            self.layer as f32 + 510.0
        ]) * Matrix4::new_nonuniform_scaling(&vector![self.scale.x, self.scale.y, 1.0])
            * UnitQuaternion::from_axis_angle(
                &UnitVector3::new_normalize(vector![0.0, 0.0, 1.0]),
                self.rotation.to_radians(),
            )
            .to_homogeneous()
    }

    fn to_3d(self) -> Transform {
        Transform {
            position: point![self.position.x, self.position.y, self.layer as f32],
            rotation: UnitQuaternion::from_euler_angles(self.rotation.to_radians(), 0.0, 0.0),
            scale: vector![self.scale.x, self.scale.y, 1.0],
        }
    }
}
#[derive(Component)]
pub struct MeshComponent {
    pub mesh: AssetHandle<Mesh>,
    pub materials: Vec<MaterialInstance>,
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

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SpriteGpuData {
    // in pixel coords
    offset_size: Vector4<u32>,
}

unsafe impl Pod for SpriteGpuData {}
unsafe impl Zeroable for SpriteGpuData {}

pub struct SpriteComponentDescription {
    pub texture: AssetHandle<Texture>,
    pub material: AssetHandle<MasterMaterial>,
    pub atlas_offset: Vector2<u32>,
    pub atlas_size: Vector2<u32>,
    pub sprite_size: Vector2<f32>,
    pub z_layer: u32,
}

#[derive(Component)]
pub struct SpriteComponent {
    pub texture: AssetHandle<Texture>,
    pub material: AssetHandle<MasterMaterial>,
    pub sprite_size: Vector2<f32>,
    pub z_layer: u32,

    sprite_gpu_data: SpriteGpuData,
    parameter_buffer: BufferHandle,
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
    pub skybox_material: Option<MaterialInstance>,
    pub skybox_texture: Option<AssetHandle<Texture>>,
}

impl SpriteComponent {
    pub fn new(gpu: &GpuDevice, description: SpriteComponentDescription) -> Self {
        let sprite_gpu_data = SpriteGpuData {
            offset_size: vector![
                description.atlas_offset.x,
                description.atlas_offset.y,
                description.atlas_size.x,
                description.atlas_size.y
            ],
        };
        let parameter_buffer = gpu
            .make_buffer(
                &BufferCreateInfo {
                    label: Some("Sprite parameters"),
                    size: std::mem::size_of::<SpriteGpuData>(),
                    usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
                },
                MemoryDomain::HostVisible,
            )
            .unwrap();
        gpu.write_buffer(
            &parameter_buffer,
            0,
            bytemuck::cast_slice(&[sprite_gpu_data]),
        )
        .expect("Failed to write buffer");
        Self {
            texture: description.texture,
            z_layer: description.z_layer,
            material: description.material,
            sprite_size: description.sprite_size,
            parameter_buffer,
            sprite_gpu_data,
        }
    }

    pub fn update(&self, gpu: &dyn Gpu) {
        gpu.write_buffer(
            &self.parameter_buffer,
            0,
            bytemuck::cast_slice(&[self.sprite_gpu_data]),
        )
        .expect("Failed to write buffer");
    }
}

#[deprecated]
pub fn rendering_system(
    meshes: Query<(&MeshComponent, &Transform)>,
    lights: Query<(&LightComponent, &Transform)>,
    world: &World,
    mut commands: Commands,
) {
    let mut scene = RenderScene::new();
    if let Some(setup) = world.get_resource::<SceneSetup>() {
        scene.set_skybox_material(setup.skybox_material.clone());
        scene.set_skybox_texture(setup.skybox_texture.clone());
    }
    for (mesh_component, transform) in meshes.iter() {
        let bounds = mesh_component.bounds().transformed(transform.matrix());
        scene.add_mesh(
            crate::SceneMesh {
                mesh: mesh_component.mesh.clone(),
                materials: mesh_component.materials.clone(),
                bounds,
            },
            *transform,
            None,
        );
    }
    for (light, transform) in lights.iter() {
        scene.add_light(
            crate::SceneLightInfo {
                ty: light.ty,
                radius: light.radius,
                color: light.color,
                intensity: light.intensity,
                enabled: light.enabled,
                shadow_configuration: light.shadow_setup,
            },
            *transform,
            None,
        );
    }

    commands.insert_resource(scene)
}

#[deprecated]
pub fn rendering_system_kecs(
    meshes: kecs::Query<(&MeshComponent, &EntityToSceneNode)>,
    lights: kecs::Query<(&LightComponent, &Transform)>,
    mut commands: kecs::Commands,
    game_scene: kecs::ResMut<GameScene>,
) {
    let mut scene = RenderScene::new();
    for (mesh_component, node_id) in meshes.iter() {
        let transform = game_scene
            .get_transform(node_id.node_id, crate::game_scene::TransformSpace::World)
            .expect("No transform");
        let bounds = mesh_component.bounds().transformed(transform.matrix());
        scene.add_mesh(
            crate::SceneMesh {
                mesh: mesh_component.mesh.clone(),
                materials: mesh_component.materials.clone(),
                bounds,
            },
            transform,
            None,
        );
    }
    for (light, transform) in lights.iter() {
        scene.add_light(
            crate::SceneLightInfo {
                ty: light.ty,
                radius: light.radius,
                color: light.color,
                intensity: light.intensity,
                enabled: light.enabled,
                shadow_configuration: light.shadow_setup,
            },
            *transform,
            None,
        );
    }

    commands.add_resource(scene)
}

#[deprecated]
pub fn rendering_system_2d(
    common_resources: Res<CommonResources>,
    sprites: Query<(&SpriteComponent, &Transform2D)>,
    world: &World,
    mut commands: Commands,
) {
    let mut scene = RenderScene::new();
    if let Some(setup) = world.get_resource::<SceneSetup>() {
        scene.set_skybox_material(setup.skybox_material.clone());
        scene.set_skybox_texture(setup.skybox_texture.clone());
    }
    for (sprite_component, transform) in sprites.iter() {
        let material_instance = MaterialInstance {
            owner: sprite_component.material.clone(),
            parameter_buffers: vec![sprite_component.parameter_buffer],
            textures: vec![sprite_component.texture.clone()],
        };
        let correct_scale = [
            transform.scale.x * sprite_component.sprite_size.x,
            transform.scale.y * sprite_component.sprite_size.y,
        ]
        .into();
        let transform = Transform2D {
            position: transform.position,
            layer: transform.layer,
            rotation: transform.rotation,
            scale: correct_scale,
        };
        let bounds = BoundingShape::BoundingBox {
            min: point![
                -sprite_component.sprite_size.x,
                -sprite_component.sprite_size.y,
                0.0
            ],
            max: point![
                sprite_component.sprite_size.x,
                sprite_component.sprite_size.y,
                0.0
            ],
        };
        let bounds = bounds.transformed(transform.matrix());
        scene.add_mesh(
            crate::SceneMesh {
                mesh: common_resources.quad_mesh.clone(),
                materials: vec![material_instance],
                bounds,
            },
            transform.to_3d(),
            None,
        );
    }

    commands.insert_resource(scene)
}

pub fn init(schedule: &mut Schedule) {
    schedule.add_systems(rendering_system);
}

pub struct MeshComponentEditor {
    asset_map: SharedAssetMap,
    mesh_picker: ResourceHandleEditor,
    master_picker: ResourceHandleEditor,
    texture_picker: ResourceHandleEditor,
}

impl MeshComponentEditor {
    pub fn new(asset_map: SharedAssetMap) -> Self {
        Self {
            asset_map,
            mesh_picker: ResourceHandleEditor::default(),
            master_picker: ResourceHandleEditor::default(),
            texture_picker: ResourceHandleEditor::default(),
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
                value.materials.push(MaterialInstance::default());
            }

            ui.end_row();
            egui::Grid::new("mats").show(ui, |ui| {
                for material in &mut value.materials {
                    self.master_picker
                        .show(&mut material.owner, &mut asset_map, ui);
                    egui::Grid::new("textures").show(ui, |ui| {
                        if ui.button("Add Texture").clicked() {
                            material.textures.push(Default::default());
                        }
                        for tex in &mut material.textures {
                            self.texture_picker.show(tex, &mut asset_map, ui);
                        }
                        ui.end_row();
                    });

                    ui.end_row();
                }
            });
        });
    }
}

#[derive(Default)]
pub struct ResourceHandleEditor {
    is_shown: bool,
}

impl ResourceHandleEditor {
    pub fn show<T: crate::asset_map::Asset>(
        &mut self,
        handle: &mut AssetHandle<T>,
        asset_map: &mut AssetMap,
        ui: &mut Ui,
    ) {
        let button_label = if handle.is_null() {
            "None".to_owned()
        } else {
            let metadata = asset_map.asset_metadata(handle);
            metadata.name
        };

        if self.is_shown {
            let mut selected_id = None;
            egui::Window::new("Pick an asset")
                .open(&mut self.is_shown)
                .show(ui.ctx(), |ui| {
                    egui::ScrollArea::new([true, false]).show(ui, |ui| {
                        asset_map.iter_ids::<T>(|id, meta| {
                            if ui.selectable_label(false, &meta.name).double_clicked() {
                                selected_id = Some(id);
                            }
                        });
                    });

                    if ui.button("Load new asset").clicked() {
                        let path = rfd::FileDialog::new()
                            .set_title("Pick an asset")
                            .pick_file();
                        if let Some(path) = path {
                            match asset_map.load::<T>(path) {
                                Ok(_) => {}
                                Err(e) => {
                                    log::error!("While loading asset: {e:?}");
                                }
                            }
                        }
                    }
                });

            if let Some(id) = selected_id {
                *handle = asset_map.upcast_index(id);
                self.is_shown = false;
            }
        }
        if ui.button(button_label).clicked() {
            self.is_shown = true;
        }
    }
}
