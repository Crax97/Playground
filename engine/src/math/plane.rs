use nalgebra::{vector, Point3, UnitVector3, Vector3};

use super::shape::Shape;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationResult {
    Inside,
    OnFrustum,
    Outside,
}

#[derive(Clone, Copy, Debug)]
// ax + by + cz + d = 0
pub struct Plane {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

impl Default for Plane {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: Default::default(),
            c: Default::default(),
            d: Default::default(),
        }
    }
}

impl Plane {
    pub fn from_slice(slice: &[f32]) -> Plane {
        assert!(slice.len() >= 4);
        let norm = vector![slice[0], slice[1], slice[2]].norm();
        Plane {
            a: slice[0] / norm,
            b: slice[1] / norm,
            c: slice[2] / norm,
            d: slice[3] / norm,
        }
    }

    pub fn from_normal_origin(normal: UnitVector3<f32>, origin: Point3<f32>) -> Self {
        let d = origin.to_homogeneous().magnitude();
        Self {
            a: normal.x,
            b: normal.y,
            c: normal.z,
            d,
        }
    }

    pub fn from_normal_d(normal: UnitVector3<f32>, d: f32) -> Self {
        Self {
            a: normal.x,
            b: normal.y,
            c: normal.z,
            d,
        }
    }

    // Distance from the origin
    // Since the normal is stored as a normalized vector, we can return d directly
    pub fn distance_from_origin(&self) -> f32 {
        self.d
    }

    pub fn normal(&self) -> Vector3<f32> {
        vector![self.a, self.b, self.c]
    }

    pub fn closest_point(&self, p: Point3<f32>) -> Point3<f32> {
        p - self.normal() * self.signed_distance_to_point(&p)
    }

    pub fn signed_distance_to_point(&self, p: &Point3<f32>) -> f32 {
        let p_vec = vector![p.x, p.y, p.z];
        self.normal().dot(&p_vec) + self.d
    }

    pub fn classify_shape(&self, shape: &Shape) -> ClassificationResult {
        match shape {
            Shape::Sphere { radius, origin } => {
                let distance = self.signed_distance_to_point(origin);
                let signed_distance = distance - *radius;
                if signed_distance > 0.0 {
                    ClassificationResult::Outside
                } else if signed_distance < 0.0 {
                    ClassificationResult::Inside
                } else {
                    ClassificationResult::OnFrustum
                }
            }
        }
    }

    pub fn is_shape_behind_plane(&self, shape: &Shape) -> bool {
        self.classify_shape(shape) != ClassificationResult::Outside
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{point, vector, UnitVector3};

    use super::Plane;

    #[test]
    fn planes() {
        let origin = point![10.0, 0.0, 0.0];
        let plane =
            Plane::from_normal_origin(UnitVector3::new_normalize(vector![1.0, 0.0, 0.0]), origin);

        assert!((plane.distance_from_origin() - 10.0).abs() <= 0.5);
        let p0 = origin + plane.normal() * 10.0;
        let closest = plane.closest_point(p0);
        assert!((origin - closest).magnitude() <= 0.5);
    }
}
