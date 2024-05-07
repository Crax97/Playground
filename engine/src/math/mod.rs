pub mod color;

use glam::{vec3, Mat4, Quat, Vec3, Vec4Swizzles};
use serde::{Deserialize, Serialize};

pub mod constants {
    use glam::{vec3, Vec3};

    pub const UP: Vec3 = vec3(0.0, 1.0, 0.0);
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub struct Transform {
    pub location: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.location)
    }
    pub fn from_matrix(matrix: Mat4) -> Self {
        let (scale, rotation, location) = matrix.to_scale_rotation_translation();
        Self {
            location,
            rotation,
            scale,
        }
    }

    pub fn difference_matrix(&self, other: &Self) -> Mat4 {
        let inv_other = other.matrix().inverse();
        self.matrix() * inv_other
    }

    pub fn difference(&self, other: &Self) -> Self {
        Self::from_matrix(self.difference_matrix(other))
    }

    pub fn compose(&self, transform: &Self) -> Self {
        let matrix = self.matrix() * transform.matrix();
        Self::from_matrix(matrix)
    }

    pub fn inverse(&self) -> Self {
        let matrix = self.matrix().inverse();
        Self::from_matrix(matrix)
    }

    pub fn forward(&self) -> Vec3 {
        self.matrix().col(2).xyz().normalize()
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            location: Default::default(),
            rotation: Default::default(),
            scale: vec3(1.0, 1.0, 1.0),
        }
    }
}
