use nalgebra::{Point3, Vector3};

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere { radius: f32, origin: Point3<f32> },
}
impl Shape {
    pub fn translated(&self, offset: Vector3<f32>) -> Shape {
        match self {
            Shape::Sphere { radius, origin } => Shape::Sphere {
                radius: *radius,
                origin: origin + offset,
            },
        }
    }
}
