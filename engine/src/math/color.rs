use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Clone, Copy, Serialize, Deserialize, Pod, Zeroable)]
pub struct LinearColor {
    pub data: [f32; 4],
}

impl Default for LinearColor {
    fn default() -> Self {
        LinearColor { data: [0.0; 4] }
    }
}

impl LinearColor {
    pub const VIOLET: LinearColor = LinearColor::new(0.498, 0.0, 1.0, 1.0);
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { data: [r, g, b, a] }
    }

    pub fn r(self) -> f32 {
        self.data[0]
    }

    pub fn g(self) -> f32 {
        self.data[1]
    }

    pub fn b(self) -> f32 {
        self.data[2]
    }

    pub fn a(self) -> f32 {
        self.data[3]
    }

    pub fn to_u8_array(self) -> [u8; 4] {
        self.data.map(|v| (v * 255.0).floor() as u8)
    }

    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice()
    }

    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        self.data.as_mut()
    }
}

impl std::ops::Mul<f32> for LinearColor {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            data: self.data.map(|v| v * rhs).map(|v| v.clamp(0.0, 1.0)),
        }
    }
}
