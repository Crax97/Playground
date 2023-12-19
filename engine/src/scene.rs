use crate::{
    resource_map::{ResourceHandle, ResourceMap},
    CvarManager,
};
use bevy_ecs::system::Resource;
use gpu::{
    CommandBuffer, Extent2D, ImageFormat, ImageHandle, ImageViewHandle, VkCommandBuffer, VkGpu,
};
use nalgebra::{vector, Matrix4, Point3, Vector2, Vector3};
use std::num::NonZeroU32;

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameData {
    view: nalgebra::Matrix4<f32>,
    projection: nalgebra::Matrix4<f32>,
}

use crate::{mesh::Mesh, Camera, MasterMaterial, MaterialDescription, MaterialInstance, Texture};

#[derive(Clone)]
pub struct ScenePrimitive {
    pub mesh: ResourceHandle<Mesh>,
    pub materials: Vec<MaterialInstance>,
    pub transform: Matrix4<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Point,
    Directional {
        direction: Vector3<f32>,
        size: Vector2<f32>,
    },
    Spotlight {
        direction: Vector3<f32>,
        inner_cone_degrees: f32,
        outer_cone_degrees: f32,
    },
    Rect {
        direction: Vector3<f32>,
        width: f32,
        height: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowSetup {
    pub importance: NonZeroU32,
}

impl Default for ShadowSetup {
    fn default() -> Self {
        Self {
            importance: NonZeroU32::new(1).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub ty: LightType,
    pub position: Point3<f32>,
    pub radius: f32,
    pub color: Vector3<f32>,
    pub intensity: f32,

    pub enabled: bool,
    pub shadow_setup: Option<ShadowSetup>,
}
impl Light {
    pub fn set_direction(&mut self, forward: Vector3<f32>) {
        match &mut self.ty {
            LightType::Point => {}
            LightType::Directional { direction, .. }
            | LightType::Spotlight { direction, .. }
            | LightType::Rect { direction, .. } => *direction = forward,
        }
    }

    pub fn direction(&self) -> Vector3<f32> {
        match self.ty {
            LightType::Point => vector![1.0, 0.0, 0.0],
            LightType::Directional { direction, .. }
            | LightType::Spotlight { direction, .. }
            | LightType::Rect { direction, .. } => direction,
        }
    }

    pub(crate) fn shadow_view_matrices(&self) -> Vec<Matrix4<f32>> {
        match self.ty {
            LightType::Point => vec![
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![1.0, 0.0, 0.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![-1.0, 0.0, 0.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, 1.0, 0.0]),
                    &vector![0.0, 0.0, 1.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, -1.0, 0.0]),
                    &vector![0.0, 0.0, 1.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, 0.0, 1.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, 0.0, -1.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
            ],
            LightType::Directional { direction, .. } => vec![Matrix4::look_at_rh(
                &self.position,
                &(self.position + direction),
                &vector![0.0, 1.0, 0.0],
            )],
            _ => vec![Matrix4::look_at_rh(
                &self.position,
                &(self.position + self.direction()),
                &vector![0.0, 1.0, 0.0],
            )],
        }
    }

    pub(crate) fn projection_matrix(&self) -> Matrix4<f32> {
        const ZNEAR: f32 = 0.001;
        match self.ty {
            LightType::Point => Matrix4::new_perspective(
                1.0,
                90.0f32.to_radians(),
                ZNEAR,
                // Use max to avoid superimposing far plane onto near plane
                self.radius.max(ZNEAR + 0.01),
            ),
            LightType::Directional { size, .. } => Matrix4::new_orthographic(
                -size.x * 0.5,
                size.x * 0.5,
                -size.y * 0.5,
                size.y * 0.5,
                -self.radius * 0.5,
                (self.radius * 0.5).max(ZNEAR),
            ),
            LightType::Spotlight {
                outer_cone_degrees, ..
            } => Matrix4::new_perspective(
                1.0,
                // the outer cone spans from the light center to the light side: |-/
                //                                                               |/
                // the fov spans from side to side: \-|-/
                //                                   \|/
                2.0 * outer_cone_degrees
                    .to_radians()
                    .min(std::f32::consts::FRAC_PI_2)
                    .max(0.01),
                ZNEAR,
                self.radius.max(ZNEAR + 0.1),
            ),
            LightType::Rect { width, height, .. } => {
                Matrix4::new_perspective(width / height, 90.0, ZNEAR, self.radius.max(ZNEAR + 0.1))
            }
        }
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq)]
pub struct LightHandle(pub usize);

#[derive(Resource, Default)]
pub struct Scene {
    pub primitives: Vec<ScenePrimitive>,
    pub lights: Vec<Light>,

    skybox_material: Option<MaterialInstance>,
    skybox_texture: Option<ResourceHandle<Texture>>,
    current_lights_iteration: u64,
}

impl Scene {
    fn increment_light_counter(&mut self) {
        self.current_lights_iteration = self.current_lights_iteration.wrapping_add(1);
    }

    pub fn get_skybox_texture_handle(&self) -> &Option<ResourceHandle<Texture>> {
        &self.skybox_texture
    }

    pub fn get_skybox_material(&self) -> &Option<MaterialInstance> {
        &self.skybox_material
    }
}

impl Scene {
    pub fn new() -> Self {
        Self {
            primitives: vec![],
            lights: vec![],
            current_lights_iteration: 0,

            skybox_texture: None,
            skybox_material: None,
        }
    }

    pub fn add(&mut self, primitive: ScenePrimitive) {
        self.primitives.push(primitive);
    }

    pub fn add_light(&mut self, light: Light) -> LightHandle {
        let idx = self.lights.len();
        self.increment_light_counter();
        self.lights.push(light);
        LightHandle(idx)
    }

    pub fn edit_light(&mut self, handle: &LightHandle) -> &mut Light {
        self.increment_light_counter();
        &mut self.lights[handle.0]
    }

    pub fn all_primitives(&self) -> &[ScenePrimitive] {
        &self.primitives
    }
    pub fn all_lights(&self) -> &[Light] {
        &self.lights
    }

    pub fn all_lights_mut(&mut self) -> &mut [Light] {
        &mut self.lights
    }

    pub fn all_enabled_lights(&self) -> impl Iterator<Item = &Light> {
        self.lights.iter().filter(|l| l.enabled)
    }

    pub fn lights_iteration(&self) -> u64 {
        /*
         * When a light is added/removed, the current iteration counter is incremented to
         * notify that the lights in the scene have changed.
         * */
        self.current_lights_iteration
    }

    pub fn set_skybox_texture(&mut self, new_skybox_texture: Option<ResourceHandle<Texture>>) {
        self.skybox_texture = new_skybox_texture;
    }

    pub fn set_skybox_material(&mut self, new_skybox_material: Option<MaterialInstance>) {
        self.skybox_material = new_skybox_material;
    }
}

pub struct Backbuffer {
    pub size: Extent2D,
    pub format: ImageFormat,
    pub image: ImageHandle,
    pub image_view: ImageViewHandle,
}

pub trait RenderingPipeline {
    fn render<'a>(
        &'a mut self,
        gpu: &'a VkGpu,
        pov: &Camera,
        scene: &Scene,
        backbuffer: &Backbuffer,
        resource_map: &ResourceMap,
        cvar_manager: &CvarManager,
    ) -> anyhow::Result<CommandBuffer>;

    fn create_material(
        &mut self,
        gpu: &VkGpu,
        material_description: MaterialDescription,
    ) -> anyhow::Result<MasterMaterial>;
}
