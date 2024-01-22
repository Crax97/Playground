use crate::{
    math::shape::BoundingShape,
    render_scene::{BinaryBvh, Bvh},
    resource_map::{ResourceHandle, ResourceMap},
    CvarManager, Frustum,
};
use bevy_ecs::system::Resource;
use gpu::{Extent2D, Gpu, ImageFormat, ImageHandle, ImageViewHandle};
use nalgebra::{vector, Matrix4, Point3, Vector2, Vector3};

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
    pub bounds: BoundingShape,
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
pub struct ShadowConfiguration {
    pub shadow_map_width: u32,
    pub shadow_map_height: u32,
    pub depth_bias: f32,
    pub depth_slope: f32,
}

impl Default for ShadowConfiguration {
    fn default() -> Self {
        Self {
            shadow_map_width: 512,
            shadow_map_height: 512,
            depth_bias: 0.0,
            depth_slope: 0.0,
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
    pub shadow_configuration: Option<ShadowConfiguration>,
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

    pub fn light_cameras(&self) -> Vec<Camera> {
        const ZNEAR: f32 = 0.001;
        let mut povs = vec![];
        match self.ty {
            LightType::Point => {
                let mut camera =
                    Camera::new_perspective(90.0, 1.0, 1.0, ZNEAR, self.radius.max(ZNEAR + 0.1));
                camera.location = self.position;
                let directions = [
                    vector![1.0, 0.0, 0.0],
                    vector![-1.0, 0.0, 0.0],
                    vector![0.0, 1.0, 0.0],
                    vector![0.0, -1.0, 0.0],
                    vector![0.0, 0.0, 1.0],
                    vector![0.0, 0.0, -1.0],
                ];

                for direction in directions {
                    let mut camera = camera;
                    camera.forward = direction;
                    povs.push(camera);
                }
            }
            LightType::Directional { direction, size } => {
                let mut camera =
                    Camera::new_orthographic(size.x, size.y, -self.radius * 0.5, self.radius * 0.5);
                camera.location = self.position;
                camera.forward = direction;
                povs.push(camera);
            }
            LightType::Spotlight {
                direction,
                outer_cone_degrees,
                ..
            } => {
                let mut camera = Camera::new_perspective(
                    2.0 * outer_cone_degrees.max(0.01),
                    1.0,
                    1.0,
                    ZNEAR,
                    self.radius.max(ZNEAR + 0.01),
                );
                camera.location = self.position;
                camera.forward = direction;
                povs.push(camera);
            }
            LightType::Rect { .. } => todo!(),
        }

        povs
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq)]
pub struct LightHandle(pub usize);

#[derive(Resource, Default)]
pub struct RenderScene {
    pub bvh: BinaryBvh<usize>,
    pub use_frustum_culling: bool,
    pub use_bvh: bool,
    pub primitives: Vec<ScenePrimitive>,
    pub lights: Vec<Light>,

    skybox_material: Option<MaterialInstance>,
    skybox_texture: Option<ResourceHandle<Texture>>,
    current_lights_iteration: u64,
}

impl RenderScene {
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

impl RenderScene {
    pub fn new() -> Self {
        Self {
            bvh: Bvh::new(),
            use_frustum_culling: true,
            use_bvh: true,
            primitives: vec![],
            lights: vec![],
            current_lights_iteration: 0,

            skybox_texture: None,
            skybox_material: None,
        }
    }

    pub fn add(&mut self, primitive: ScenePrimitive) {
        let prim_index = self.primitives.len();
        let (aabb_min, aabb_max) = primitive.bounds.box_extremes();
        self.primitives.push(primitive);
        self.bvh.add(prim_index, aabb_min, aabb_max);
    }

    pub fn intersect_frustum(&self, frustum: &Frustum) -> Vec<&ScenePrimitive> {
        if !self.use_frustum_culling {
            return self.primitives.iter().collect();
        } else if self.use_bvh {
            let indices = self.bvh.intersect_frustum_copy(frustum);
            indices.iter().map(|id| &self.primitives[*id]).collect()
        } else {
            self.primitives
                .iter()
                .filter(|prim| frustum.contains_shape(&prim.bounds))
                .collect()
        }
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
    fn render(
        &mut self,
        gpu: &dyn Gpu,
        pov: &Camera,
        scene: &RenderScene,
        resource_map: &ResourceMap,
        cvar_manager: &CvarManager,
    ) -> anyhow::Result<ImageViewHandle>;

    fn on_resolution_changed(&mut self, new_resolution: Extent2D);

    fn create_material(
        &mut self,
        gpu: &dyn Gpu,
        material_description: MaterialDescription,
    ) -> anyhow::Result<MasterMaterial>;
}
