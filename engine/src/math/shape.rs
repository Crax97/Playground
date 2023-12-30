use nalgebra::{Matrix4, Point3};

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere { radius: f32, origin: Point3<f32> },
}
impl Shape {
    pub fn transformed(&self, transform: Matrix4<f32>) -> Shape {
        match self {
            Shape::Sphere { radius, origin } => {
                let origin = origin.to_homogeneous();
                let origin = transform * origin;
                let radius = transform.m11.max(transform.m22).max(transform.m33) * radius;
                Shape::Sphere {
                    radius,
                    origin: Point3::from_homogeneous(origin).unwrap_or_default(),
                }
            }
        }
    }
}
