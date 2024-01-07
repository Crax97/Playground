use nalgebra::{point, Matrix4, Point3};
#[derive(Clone, Copy, Debug)]
pub enum BoundingShape {
    Sphere { radius: f32, origin: Point3<f32> },
    BoundingBox { min: Point3<f32>, max: Point3<f32> },
}
impl BoundingShape {
    pub fn transformed(&self, transform: Matrix4<f32>) -> BoundingShape {
        match self {
            BoundingShape::Sphere { radius, origin } => {
                let origin = origin.to_homogeneous();
                let origin = transform * origin;
                let radius = transform.m11.max(transform.m22).max(transform.m33) * radius;
                BoundingShape::Sphere {
                    radius,
                    origin: Point3::from_homogeneous(origin).unwrap_or_default(),
                }
            }
            BoundingShape::BoundingBox { min, max } => {
                let points = Self::bounding_box_points(min, max);

                let points = points
                    .iter()
                    .map(|pt| transform * pt.to_homogeneous())
                    .map(|pt| point![pt.x, pt.y, pt.z]);
                Self::bounding_box_from_points(points)
            }
        }
    }

    pub fn bounding_box_points(min: &Point3<f32>, max: &Point3<f32>) -> [Point3<f32>; 8] {
        [
            *min,
            point![min.x, min.y, max.z],
            point![max.x, min.y, max.z],
            point![max.x, min.y, min.z],
            *max,
            point![min.x, max.y, max.z],
            point![max.x, max.y, max.z],
            point![max.x, max.y, min.z],
        ]
    }

    pub fn bounding_box_from_points<I: IntoIterator<Item = Point3<f32>>>(i: I) -> BoundingShape {
        let (mut min, mut max): (Point3<f32>, Point3<f32>) =
            (Default::default(), Default::default());

        for pt in i.into_iter() {
            min.x = min.x.min(pt.x);
            min.y = min.y.min(pt.y);
            min.z = min.z.min(pt.z);
            max.x = max.x.max(pt.x);
            max.y = max.y.max(pt.y);
            max.z = max.z.max(pt.z);
        }

        BoundingShape::BoundingBox { min, max }
    }

    pub fn box_extremes(&self) -> (Point3<f32>, Point3<f32>) {
        match self {
            BoundingShape::Sphere { radius, origin } => (
                point![-radius + origin.x, -radius + origin.y, -radius + origin.z],
                point![*radius + origin.x, *radius + origin.y, *radius + origin.z],
            ),
            BoundingShape::BoundingBox { min, max } => (*min, *max),
        }
    }
}
