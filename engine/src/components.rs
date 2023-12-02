use std::ops::{Deref, DerefMut};

use crate::app::app_state::app_state;
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
use bytemuck::{Pod, Zeroable};
use gpu::{BufferCreateInfo, BufferHandle, BufferUsageFlags, Gpu, MemoryDomain};
use nalgebra::{
    point, vector, Matrix4, Point2, Point3, UnitQuaternion, UnitVector3, Vector2, Vector3, Vector4,
};
use winit::window::Window;

use crate::{MasterMaterial, MaterialInstance, Mesh};

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
}
#[derive(Component)]
pub struct MeshComponent {
    pub mesh: ResourceHandle<Mesh>,
    pub materials: Vec<MaterialInstance>,
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
    pub texture: ResourceHandle<Texture>,
    pub material: ResourceHandle<MasterMaterial>,
    pub sprite_offset: Vector2<u32>,
    pub sprite_size: Vector2<u32>,
    pub z_layer: u32,
}

#[derive(Component)]
pub struct SpriteComponent {
    pub texture: ResourceHandle<Texture>,
    pub material: ResourceHandle<MasterMaterial>,
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
    pub skybox_material: Option<MaterialInstance>,
    pub skybox_texture: Option<ResourceHandle<Texture>>,
}

impl SpriteComponent {
    pub fn new(description: SpriteComponentDescription) -> Self {
        let sprite_gpu_data = SpriteGpuData {
            offset_size: vector![
                description.sprite_offset.x,
                description.sprite_offset.y,
                description.sprite_size.x,
                description.sprite_size.y
            ],
        };
        let parameter_buffer = app_state()
            .gpu
            .make_buffer(
                &BufferCreateInfo {
                    label: Some("Sprite parameters"),
                    size: std::mem::size_of::<SpriteGpuData>(),
                    usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
                },
                MemoryDomain::HostVisible,
            )
            .unwrap();
        app_state()
            .gpu
            .write_buffer(
                &parameter_buffer,
                0,
                bytemuck::cast_slice(&[sprite_gpu_data]),
            )
            .expect("Failed to write buffer");
        Self {
            texture: description.texture,
            z_layer: description.z_layer,
            material: description.material,
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
        let material_instance = MaterialInstance {
            owner: sprite_component.material.clone(),
            parameter_buffers: vec![sprite_component.parameter_buffer.clone()],
            textures: vec![sprite_component.texture.clone()],
        };
        let correct_scale = [
            transform.scale.x * sprite_component.sprite_gpu_data.offset_size.z as f32,
            transform.scale.y * sprite_component.sprite_gpu_data.offset_size.w as f32,
        ]
        .into();
        let transform = Transform2D {
            position: transform.position,
            layer: transform.layer,
            rotation: transform.rotation,
            scale: correct_scale,
        };
        scene.add(crate::ScenePrimitive {
            mesh: common_resources.quad_mesh.clone(),
            materials: vec![material_instance],
            transform: transform.matrix(),
        });
    }

    commands.insert_resource(scene)
}

pub fn init(schedule: &mut Schedule) {
    schedule.add_systems(rendering_system);
}
