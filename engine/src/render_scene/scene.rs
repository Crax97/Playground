use std::marker::PhantomData;

use crate::{
    asset_map::{AssetHandle, AssetMap},
    components::Transform,
    material::Material,
    math::shape::BoundingShape,
    render_scene::BinaryBvh,
    CvarManager, Frustum,
};
use bevy_ecs::system::Resource;
use gpu::{CommandBuffer, Extent2D, Gpu, ImageFormat, ImageHandle, ImageViewHandle, Rect2D};
use nalgebra::{vector, Vector2, Vector3};
use serde::{de::Visitor, Deserialize, Serialize};
use thunderdome::{Arena, Index};

#[derive(Serialize, Deserialize, Resource, Default)]
pub struct GameScene {
    pub content: SceneContent,

    skybox_material: Option<AssetHandle<Material>>,
    skybox_texture: Option<AssetHandle<Texture>>,

    #[serde(skip)]
    current_lights_iteration: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameData {
    view: nalgebra::Matrix4<f32>,
    projection: nalgebra::Matrix4<f32>,
}

use crate::{mesh::Mesh, Camera, Texture};

#[derive(Clone, Serialize, Deserialize)]
pub struct SceneMesh {
    pub mesh: AssetHandle<Mesh>,
    pub materials: Vec<AssetHandle<Material>>,
    pub bounds: BoundingShape,
}

impl Default for SceneMesh {
    fn default() -> Self {
        Self {
            mesh: Default::default(),
            materials: Default::default(),
            bounds: BoundingShape::Sphere {
                radius: 0.0,
                origin: Default::default(),
            },
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct ScenePrimitive {
    pub ty: ScenePrimitiveType,
    pub transform: Transform,
    pub label: String,
    pub tags: Vec<String>,
}

#[derive(Default, Serialize, Deserialize)]
pub enum ScenePrimitiveType {
    #[default]
    Empty,
    Mesh(SceneMesh),
    Light(SceneLightInfo),
}
impl ScenePrimitiveType {
    pub fn as_light(&self) -> &SceneLightInfo {
        match self {
            ScenePrimitiveType::Light(l) => l,
            _ => panic!("Primitive is not a light"),
        }
    }

    pub fn as_light_mut(&mut self) -> &mut SceneLightInfo {
        match self {
            ScenePrimitiveType::Light(l) => l,
            _ => panic!("Primitive is not a light"),
        }
    }

    pub fn as_mesh(&self) -> &SceneMesh {
        match self {
            ScenePrimitiveType::Mesh(l) => l,
            _ => panic!("Primitive is not a mesh"),
        }
    }

    pub fn as_mesh_mut(&mut self) -> &mut SceneMesh {
        match self {
            ScenePrimitiveType::Mesh(l) => l,
            _ => panic!("Primitive is not a mesh"),
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LightType {
    #[default]
    Point,
    Directional {
        size: Vector2<f32>,
    },
    Spotlight {
        inner_cone_degrees: f32,
        outer_cone_degrees: f32,
    },
    Rect {
        width: f32,
        height: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SceneLightInfo {
    pub enabled: bool,
    pub ty: LightType,
    pub radius: f32,
    pub color: Vector3<f32>,
    pub intensity: f32,
    pub shadow_configuration: Option<ShadowConfiguration>,
}
impl SceneLightInfo {
    pub fn light_cameras(&self, transform: &Transform) -> Vec<Camera> {
        let position = &transform.position;
        let direction = transform.forward();
        const ZNEAR: f32 = 0.001;
        let mut povs = vec![];
        match self.ty {
            LightType::Point => {
                let mut camera =
                    Camera::new_perspective(90.0, 1.0, 1.0, ZNEAR, self.radius.max(ZNEAR + 0.1));
                camera.location = *position;
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
            LightType::Directional { size } => {
                let mut camera =
                    Camera::new_orthographic(size.x, size.y, -self.radius * 0.5, self.radius * 0.5);
                camera.location = *position;
                camera.forward = direction;
                povs.push(camera);
            }
            LightType::Spotlight {
                outer_cone_degrees, ..
            } => {
                let mut camera = Camera::new_perspective(
                    2.0 * outer_cone_degrees.max(0.01),
                    1.0,
                    1.0,
                    ZNEAR,
                    self.radius.max(ZNEAR + 0.01),
                );
                camera.location = *position;
                camera.forward = direction;
                povs.push(camera);
            }
            LightType::Rect { .. } => todo!(),
        }

        povs
    }
}

#[derive(Debug, Clone, Copy, Eq, Ord, PartialOrd, PartialEq)]
pub struct PrimitiveHandle(pub Index);

#[derive(Default)]
pub struct SceneContent {
    pub bvh: BinaryBvh<thunderdome::Index>,
    pub primitives: Arena<ScenePrimitive>,
}

#[derive(Default, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum IntersectionMode {
    #[default]
    Bvh,
    Frustum,
    None,
}

impl kecs::Resource for GameScene {}

impl GameScene {
    fn increment_light_counter(&mut self) {
        self.current_lights_iteration = self.current_lights_iteration.wrapping_add(1);
    }

    pub fn get_skybox_texture_handle(&self) -> &Option<AssetHandle<Texture>> {
        &self.skybox_texture
    }

    pub fn get_skybox_material(&self) -> &Option<AssetHandle<Material>> {
        &self.skybox_material
    }

    pub fn get(&self, handle: PrimitiveHandle) -> &ScenePrimitive {
        &self.content.primitives[handle.0]
    }

    pub fn get_mut(&mut self, handle: PrimitiveHandle) -> &mut ScenePrimitive {
        &mut self.content.primitives[handle.0]
    }

    pub fn all_lights(&self) -> impl Iterator<Item = (PrimitiveHandle, &ScenePrimitive)> {
        self.all_primitives().filter_map(|(i, p)| match p.ty {
            ScenePrimitiveType::Light(_) => Some((i, p)),
            _ => None,
        })
    }

    pub fn add_primitive(&mut self, node: ScenePrimitive) -> PrimitiveHandle {
        let extremes = match &node.ty {
            ScenePrimitiveType::Mesh(mesh) => Some(mesh.bounds.box_extremes()),
            _ => None,
        };

        let index = self.content.primitives.insert(node);

        if let Some((aabb_min, aabb_max)) = extremes {
            self.content.bvh.add(index, aabb_min, aabb_max);
        }

        PrimitiveHandle(index)
    }

    pub fn clear(&mut self) {
        self.content = Default::default();
    }
}

impl GameScene {
    pub fn new() -> Self {
        Self {
            content: Default::default(),
            current_lights_iteration: 0,

            skybox_texture: None,
            skybox_material: None,
        }
    }

    pub fn add_mesh(
        &mut self,
        primitive: SceneMesh,
        transform: Transform,
        label: Option<String>,
    ) -> PrimitiveHandle {
        let (aabb_min, aabb_max) = primitive.bounds.box_extremes();
        let prim_index = self.content.primitives.insert(ScenePrimitive {
            ty: ScenePrimitiveType::Mesh(primitive),
            transform,
            label: label.unwrap_or("Mesh".to_owned()),
            tags: Default::default(),
        });
        self.content.bvh.add(prim_index, aabb_min, aabb_max);

        PrimitiveHandle(prim_index)
    }

    pub fn intersect_frustum(
        &self,
        frustum: &Frustum,
        intersection_mode: IntersectionMode,
    ) -> Vec<&ScenePrimitive> {
        if intersection_mode == IntersectionMode::None {
            return self
                .content
                .primitives
                .iter()
                .filter_map(|(_, p)| match &p.ty {
                    ScenePrimitiveType::Mesh(_) => Some(p),
                    _ => None,
                })
                .collect();
        } else if intersection_mode == IntersectionMode::Bvh {
            let indices = self.content.bvh.intersect_frustum_copy(frustum);
            indices
                .iter()
                .map(|id| &self.content.primitives[*id])
                .map(|p| match &p.ty {
                    ScenePrimitiveType::Mesh(_) => p,
                    ScenePrimitiveType::Light(_) | ScenePrimitiveType::Empty => {
                        panic!("BVH returned a light")
                    }
                })
                .collect()
        } else {
            self.content
                .primitives
                .iter()
                .filter_map(|(_, prim)| match &prim.ty {
                    ScenePrimitiveType::Mesh(m) if frustum.contains_shape(&m.bounds) => Some(prim),
                    _ => None,
                })
                .collect()
        }
    }

    pub fn add_light(
        &mut self,
        light: SceneLightInfo,
        transform: Transform,
        label: Option<String>,
    ) -> PrimitiveHandle {
        self.increment_light_counter();
        let idx = self.content.primitives.insert(ScenePrimitive {
            ty: ScenePrimitiveType::Light(light),
            transform,
            label: label.unwrap_or_else(|| {
                match light.ty {
                    LightType::Point => "Point Light",
                    LightType::Directional { .. } => "Directional Light",
                    LightType::Spotlight { .. } => "Spot Light",
                    LightType::Rect { .. } => "Rect Light",
                }
                .to_string()
            }),
            tags: Default::default(),
        });
        PrimitiveHandle(idx)
    }

    pub fn all_primitives(&self) -> impl Iterator<Item = (PrimitiveHandle, &ScenePrimitive)> {
        self.content
            .primitives
            .iter()
            .map(|(i, p)| (PrimitiveHandle(i), p))
    }

    pub fn all_primitives_mut(
        &mut self,
    ) -> impl Iterator<Item = (PrimitiveHandle, &mut ScenePrimitive)> {
        self.content
            .primitives
            .iter_mut()
            .map(|(i, p)| (PrimitiveHandle(i), p))
    }

    pub fn all_enabled_lights(&self) -> impl Iterator<Item = (PrimitiveHandle, &ScenePrimitive)> {
        self.all_primitives().filter_map(|(i, p)| match p.ty {
            ScenePrimitiveType::Light(l) if l.enabled => Some((i, p)),
            _ => None,
        })
    }

    pub fn lights_iteration(&self) -> u64 {
        /*
         * When a light is added/removed, the current iteration counter is incremented to
         * notify that the lights in the scene have changed.
         * */
        self.current_lights_iteration
    }

    pub fn set_skybox_texture(&mut self, new_skybox_texture: Option<AssetHandle<Texture>>) {
        self.skybox_texture = new_skybox_texture;
    }

    pub fn set_skybox_material(&mut self, new_skybox_material: Option<AssetHandle<Material>>) {
        self.skybox_material = new_skybox_material;
    }

    pub fn clean_resources(&mut self, _gpu: &dyn Gpu) {}
}

pub struct Backbuffer {
    pub size: Extent2D,
    pub format: ImageFormat,
    pub image: ImageHandle,
    pub image_view: ImageViewHandle,
}

impl Backbuffer {
    pub fn whole_area(&self) -> Rect2D {
        Rect2D {
            offset: Default::default(),
            extent: self.size,
        }
    }
}

pub trait RenderingPipeline {
    fn render(
        &mut self,
        gpu: &dyn Gpu,
        graphics_command_buffer: &mut CommandBuffer,
        pov: &Camera,
        scene: &GameScene,
        resource_map: &AssetMap,
        cvar_manager: &CvarManager,
    ) -> anyhow::Result<ImageViewHandle>;

    fn on_resolution_changed(&mut self, new_resolution: Extent2D);

    fn destroy(&mut self, gpu: &dyn Gpu);
}

impl Serialize for SceneContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let num_prims = self.primitives.len();
        let mut state = serializer.serialize_seq(Some(num_prims))?;
        for (_, prim) in self.primitives.iter() {
            state.serialize_element(prim)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for SceneContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::SeqAccess;
        use std::fmt;

        #[derive(Default)]
        struct ArenaVisitor<T> {
            marker: PhantomData<T>,
        }

        impl<'de, T> Visitor<'de> for ArenaVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = Arena<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let capacity = seq.size_hint().unwrap_or_default();
                let mut values = Arena::<T>::with_capacity(capacity);

                while let Some(value) = seq.next_element()? {
                    values.insert(value);
                }

                Ok(values)
            }
        }
        let primitives = deserializer.deserialize_seq(ArenaVisitor::<ScenePrimitive>::default())?;
        let mut bvh = BinaryBvh::new();
        for (index, prim) in &primitives {
            if let ScenePrimitiveType::Mesh(mesh) = &prim.ty {
                let (aabb_min, aabb_max) = mesh.bounds.box_extremes();
                bvh.add(index, aabb_min, aabb_max);
            }
        }
        Ok(SceneContent { primitives, bvh })
    }
}
