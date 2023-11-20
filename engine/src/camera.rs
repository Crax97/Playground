use bevy_ecs::system::Resource;
use nalgebra::{vector, Matrix4, Point3, Vector3};

/*
view: nalgebra::Matrix4::look_at_rh(
    &point![2.0, 2.0, 2.0],
    &point![0.0, 0.0, 0.0],
    &vector![0.0, 0.0, -1.0],
),
projection: nalgebra::Matrix4::new_perspective(
    1240.0 / 720.0,
    45.0,
    0.1,
    10.0,
),
*/

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
        let up = vector![0.0, 1.0, 0.0];
        Matrix4::look_at_rh(&self.location, &(self.location + self.forward), &up)
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
}
