use bytemuck::{Pod, Zeroable};
use gpu::BufferHandle;
use nalgebra::{vector, Matrix4, Point4, Vector4};

use crate::{Light, LightType};

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

impl From<&Light> for GpuLightInfo {
    fn from(light: &Light) -> Self {
        let (direction, extras, ty) = match light.ty {
            LightType::Point => (Default::default(), Default::default(), 0),
            LightType::Directional { direction, .. } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                Default::default(),
                1,
            ),
            LightType::Spotlight {
                direction,
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
            LightType::Rect {
                direction,
                width,
                height,
            } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                vector![width, height, 0.0, 0.0],
                3,
            ),
        };
        Self {
            position_radius: vector![
                light.position.x,
                light.position.y,
                light.position.z,
                light.radius
            ],
            color: vector![light.color.x, light.color.y, light.color.z, light.intensity],
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
