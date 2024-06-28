use crate::math::Transform;
use crate::scene::{LightInfo, LightType};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GPULight {
    pub light_ty: [u32; 4],
    pub color: [f32; 4],
    pub position_strength: [f32; 4],
    pub direction_radius: [f32; 4],
    // in radians
    pub inner_outer_angles: [f32; 4],
}

impl GPULight {
    pub fn from_light(value: &LightInfo, transform: &Transform) -> Self {

        const LIGHT_DIRECTIONAL: u32 = 0;
        const LIGHT_POINT: u32 = 1;
        const LIGHT_SPOT: u32 = 2;

        let light_ty = match &value.ty {
            LightType::Directional => LIGHT_DIRECTIONAL,
            LightType::Point { .. } => LIGHT_POINT,
            LightType::Spot { .. } => LIGHT_SPOT
        };

        let light_radius = match &value.ty {
            LightType::Directional => 0.0,
            LightType::Point { radius, .. } |
            LightType::Spot { radius, .. } => *radius
        };
        let light_strength = value.strength;

        let (light_inner_angle, light_outer_angle) = match &value.ty {
            LightType::Directional | LightType::Point {..} => (0.0, 0.0),
            LightType::Spot { inner_angle, outer_angle, .. } => (inner_angle.to_radians(), outer_angle.to_radians())
        };

        Self {
            light_ty: [light_ty, 0, 0, 0],
            color: value.color.extend(0.0).to_array(),
            position_strength: transform.location.extend(light_strength).to_array(),
            direction_radius: transform.forward().extend(light_radius).to_array(),
            inner_outer_angles: [light_inner_angle, light_outer_angle, 0.0, 0.0],
        }
    }
}