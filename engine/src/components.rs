use crate::{resource_map::ResourceHandle, LightType, Scene, ShadowSetup, Texture};
use bevy_ecs::{
    component::Component,
    schedule::Schedule,
    system::{Commands, Query, Resource},
    world::World,
};
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};

use crate::{MaterialInstance, Mesh};

#[derive(Component)]
pub struct Transform {
    pub position: Point3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Transform {
    pub fn matrix(&self) -> Matrix4<f32> {
        Matrix4::new_translation(&self.position.to_homogeneous().xyz())
            * Matrix4::new_nonuniform_scaling(&self.scale)
            * self.rotation.to_homogeneous()
    }
}

#[derive(Component)]
pub struct MeshComponent {
    pub mesh: ResourceHandle<Mesh>,
    pub materials: Vec<ResourceHandle<MaterialInstance>>,
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

pub fn init(schedule: &mut Schedule) {
    schedule.add_systems(rendering_system);
}
