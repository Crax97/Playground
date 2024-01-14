use bevy_ecs::system::Resource;
use nalgebra::{vector, Matrix4, Point3, Vector3};

use crate::{
    math::{plane::Plane, shape::BoundingShape},
    SMALL_NUMBER,
};

#[derive(Clone, Copy, Debug)]
pub enum FrustumTestResult {
    Inside,
    Outside,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Frustum {
    pub(crate) top: Plane,
    pub(crate) bottom: Plane,
    pub(crate) right: Plane,
    pub(crate) left: Plane,
    pub(crate) far: Plane,
    pub(crate) near: Plane,
}

impl Frustum {
    pub fn contains_shape(&self, shape: &BoundingShape) -> bool {
        self.top.is_shape_behind_plane(shape)
            && self.bottom.is_shape_behind_plane(shape)
            && self.left.is_shape_behind_plane(shape)
            && self.right.is_shape_behind_plane(shape)
            && self.near.is_shape_behind_plane(shape)
            && self.far.is_shape_behind_plane(shape)
    }
}

#[derive(Clone, Copy)]
pub enum CameraMode {
    Perspective { fov_degrees: f32 },
    Orthographic,
}

#[derive(Resource, Clone, Copy)]
pub struct Camera {
    pub location: Point3<f32>,
    pub forward: Vector3<f32>,
    pub width: f32,
    pub height: f32,
    pub near: f32,
    pub far: f32,

    pub mode: CameraMode,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            location: Default::default(),
            forward: vector![0.0, 0.0, 1.0],
            width: 1240.0,
            height: 720.0,
            near: 0.1,
            far: 100.0,
            mode: CameraMode::Perspective { fov_degrees: 45.0 },
        }
    }
}

impl Camera {
    pub fn new_perspective(fov_degrees: f32, width: f32, height: f32, near: f32, far: f32) -> Self {
        Self {
            mode: CameraMode::Perspective { fov_degrees },
            width,
            height,
            near,
            far,
            ..Default::default()
        }
    }

    pub fn new_orthographic(width: f32, height: f32, near: f32, far: f32) -> Self {
        Self {
            mode: CameraMode::Orthographic,
            width,
            height,
            near,
            far,
            ..Default::default()
        }
    }

    pub fn view(&self) -> Matrix4<f32> {
        let up = if self.forward.y >= 1.0 - SMALL_NUMBER {
            vector![0.0, 0.0, 1.0]
        } else if self.forward.y <= -1.0 + SMALL_NUMBER {
            vector![0.0, 0.0, -1.0]
        } else {
            vector![0.0, 1.0, 0.0]
        };
        Matrix4::look_at_rh(&self.location, &(self.location + self.forward), &up)
    }

    pub fn view_projection(&self) -> Matrix4<f32> {
        self.projection() * self.view()
    }
    pub fn projection(&self) -> Matrix4<f32> {
        match self.mode {
            CameraMode::Perspective { fov_degrees } => Matrix4::new_perspective(
                self.width / self.height,
                fov_degrees.to_radians(),
                self.near,
                self.far,
            ),
            CameraMode::Orthographic => Matrix4::new_orthographic(
                -self.width * 0.5,
                self.width * 0.5,
                -self.height * 0.5,
                self.height * 0.5,
                self.near,
                self.far,
            ),
        }
    }

    pub fn frustum(&self) -> Frustum {
        let view_projection_matrix = self.projection() * self.view();
        let left = -(view_projection_matrix.row(3) + view_projection_matrix.row(0));
        let right = -(view_projection_matrix.row(3) - view_projection_matrix.row(0));
        let bottom = -(view_projection_matrix.row(3) + view_projection_matrix.row(1));
        let top = -(view_projection_matrix.row(3) - view_projection_matrix.row(1));
        let near = -(view_projection_matrix.row(3) + view_projection_matrix.row(2));
        let far = -(view_projection_matrix.row(3) - view_projection_matrix.row(2));
        Frustum {
            left: Plane::from_slice(left.as_slice()),
            right: Plane::from_slice(right.as_slice()),
            bottom: Plane::from_slice(bottom.as_slice()),
            top: Plane::from_slice(top.as_slice()),
            near: Plane::from_slice(near.as_slice()),
            far: Plane::from_slice(far.as_slice()),
        }
    }

    pub fn split_into_slices(&self, num_slices: u8) -> Vec<Camera> {
        let mut splits = Vec::with_capacity(num_slices as usize);
        let ratio = self.far / self.near;
        let range = self.far - self.near;
        let mut z_near = self.near;

        const LAMBDA: f32 = 1.0;

        for i in 1..=num_slices {
            let p = i as f32 / num_slices as f32;
            let z_far =
                LAMBDA * self.near * ratio.powf(p) + (1.0 - LAMBDA) * (self.near + p * range);
            let mut camera = match self.mode {
                CameraMode::Perspective { fov_degrees } => {
                    Camera::new_perspective(fov_degrees, self.width, self.height, z_near, z_far)
                }
                CameraMode::Orthographic => {
                    Camera::new_orthographic(self.width, self.height, z_near, z_far)
                }
            };
            camera.location = self.location;
            camera.forward = self.forward;
            z_near = z_far;
            splits.push(camera);
        }
        splits
    }
}

#[cfg(test)]
mod tests {
    use crate::Camera;

    #[test]
    fn frustum() {
        let camera = Camera::new_perspective(90.0, 1.0, 1.0, 0.01, 1000.0);

        let frustum = camera.frustum();

        println!("{frustum:?}");
    }
}
