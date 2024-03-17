use bytemuck::{Pod, Zeroable};
use gpu::{BufferHandle, Gpu};
use nalgebra::{vector, Matrix4, Point3, Point4, Vector4};
use rapier2d::math::Rotation;

use crate::{components::Transform, LightType, SceneLightInfo};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ObjectDrawInfo {
    pub model: Matrix4<f32>,
    pub camera_index: u32,
}

unsafe impl Pod for ObjectDrawInfo {}
unsafe impl Zeroable for ObjectDrawInfo {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PointOfViewData {
    pub eye: Point4<f32>,
    pub eye_forward: Vector4<f32>,
    pub view: nalgebra::Matrix4<f32>,
    pub projection: nalgebra::Matrix4<f32>,
}

unsafe impl Pod for PointOfViewData {}
unsafe impl Zeroable for PointOfViewData {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuLightInfo {
    pub position_radius: Vector4<f32>,
    pub direction: Vector4<f32>,
    pub color: Vector4<f32>,
    pub extras: Vector4<f32>,
    // x: type, y: shadow map index, z: csm split base index
    pub ty_shadow_map_idx_csm_split: [i32; 4],
}
unsafe impl Pod for GpuLightInfo {}
unsafe impl Zeroable for GpuLightInfo {}

impl SceneLightInfo {
    pub(crate) fn to_gpu_data(&self, transform: &Transform) -> GpuLightInfo {
        let position = transform.position;
        let direction = transform.forward();
        let (direction, extras, ty) = match self.ty {
            LightType::Point => (Default::default(), Default::default(), 0),
            LightType::Directional { .. } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                Default::default(),
                1,
            ),
            LightType::Spotlight {
                inner_cone_degrees,
                outer_cone_degrees,
            } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                vector![
                    inner_cone_degrees.to_radians().cos(),
                    outer_cone_degrees.to_radians().cos(),
                    0.0,
                    0.0
                ],
                2,
            ),
            LightType::Rect { width, height } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                vector![width, height, 0.0, 0.0],
                3,
            ),
        };
        GpuLightInfo {
            position_radius: vector![position.x, position.y, position.z, self.radius],
            color: vector![self.color.x, self.color.y, self.color.z, self.intensity],
            direction,
            extras,
            ty_shadow_map_idx_csm_split: [ty, -1, -1, 0],
        }
    }
}

pub(crate) struct FrameBuffers {
    pub(crate) camera_buffer: BufferHandle,
    pub(crate) light_buffer: BufferHandle,
}
