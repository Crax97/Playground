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
pub struct Camera {
    pub location: Point3<f32>,
    pub forward: Vector3<f32>,
    pub fov: f32,
    pub width: f32,
    pub height: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            location: Default::default(),
            forward: vector![0.0, 1.0, 0.0],
            fov: 45.0,
            width: 1240.0,
            height: 720.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    pub fn view(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(
            &self.location,
            &(self.location + self.forward),
            &vector![0.0, 1.0, 0.0],
        )
    }
    pub fn projection(&self) -> Matrix4<f32> {
        Matrix4::new_perspective(self.width / self.height, self.fov, self.near, self.far)
    }
}
