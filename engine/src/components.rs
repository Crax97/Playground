use std::ops::{Deref, DerefMut};

use crate::{
    bevy_ecs_app::CommonResources, resource_map::ResourceHandle, LightType, Scene, ShadowSetup,
    Texture,
};
use bevy_ecs::reflect::ReflectComponent;
use bevy_ecs::{
    component::Component,
    schedule::Schedule,
    system::{Commands, Query, Res, Resource},
    world::World,
};
use bevy_reflect::Reflect;
use nalgebra::{vector, Matrix4, Point2, Point3, UnitQuaternion, UnitVector3, Vector2, Vector3};
use winit::window::Window;

use crate::{MaterialInstance, Mesh};

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

#[derive(Component)]
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
    pub fn matrix(&self) -> Matrix4<f32> {
        Matrix4::new_translation(&self.position.to_homogeneous().xyz())
            * Matrix4::new_nonuniform_scaling(&self.scale)
            * self.rotation.to_homogeneous()
    }
}

#[derive(Component, Debug)]
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
}
#[derive(Component)]
pub struct MeshComponent {
    pub mesh: ResourceHandle<Mesh>,
    pub materials: Vec<ResourceHandle<MaterialInstance>>,
}

#[derive(Component)]
pub struct SpriteComponent {
    pub material: ResourceHandle<MaterialInstance>,
    pub z_layer: u32,
}

#[derive(Component)]
pub struct LightComponent {
    pub ty: LightType,
    pub radius: f32,
    pub intensity: f32,
    pub color: Vector3<f32>,

    pub enabled: bool,
    pub shadow_setup: Option<ShadowSetup>,
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
    pub skybox_material: Option<ResourceHandle<MaterialInstance>>,
    pub skybox_texture: Option<ResourceHandle<Texture>>,
}

pub fn rendering_system(
    meshes: Query<(&MeshComponent, &Transform)>,
    lights: Query<(&LightComponent, &Transform)>,
    world: &World,
    mut commands: Commands,
) {
    let mut scene = Scene::new();
    if let Some(setup) = world.get_resource::<SceneSetup>() {
        scene.set_skybox_material(setup.skybox_material.clone());
        scene.set_skybox_texture(setup.skybox_texture.clone());
    }
    for (mesh_component, transform) in meshes.iter() {
        scene.add(crate::ScenePrimitive {
            mesh: mesh_component.mesh.clone(),
            materials: mesh_component.materials.clone(),
            transform: transform.matrix(),
        });
    }
    for (light, transform) in lights.iter() {
        scene.add_light(crate::Light {
            ty: light.ty,
            position: transform.position,
            radius: light.radius,
            color: light.color,
            intensity: light.intensity,
            enabled: light.enabled,
            shadow_setup: light.shadow_setup,
        });
    }

    commands.insert_resource(scene)
}

pub fn rendering_system_2d(
    common_resources: Res<CommonResources>,
    sprites: Query<(&SpriteComponent, &Transform2D)>,
    world: &World,
    mut commands: Commands,
) {
    let mut scene = Scene::new();
    if let Some(setup) = world.get_resource::<SceneSetup>() {
        scene.set_skybox_material(setup.skybox_material.clone());
        scene.set_skybox_texture(setup.skybox_texture.clone());
    }
    for (sprite_component, transform) in sprites.iter() {
        scene.add(crate::ScenePrimitive {
            mesh: common_resources.quad_mesh.clone(),
            materials: vec![sprite_component.material.clone()],
            transform: transform.matrix(),
        });
    }

    commands.insert_resource(scene)
}

pub fn init(schedule: &mut Schedule) {
    schedule.add_systems(rendering_system);
}
