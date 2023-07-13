use engine::Camera;
use nalgebra::{vector, Point3, Rotation3};

use crate::input::{key::Key, InputState};

pub struct FpsCamera {
    location: Point3<f32>,
    rotation: Rotation3<f32>,
    speed: f32,
    rotation_speed: f32,

    roll: f32,
    pitch: f32,
}

impl Default for FpsCamera {
    fn default() -> Self {
        Self {
            location: Default::default(),
            rotation: Default::default(),
            speed: 10.0,
            rotation_speed: 45.0,
            roll: Default::default(),
            pitch: Default::default(),
        }
    }
}

impl FpsCamera {
    pub fn update(&mut self, input_state: &InputState, delta_time: f32) {
        let transform = self.rotation.to_homogeneous();
        let forward = transform.column(2).xyz();
        let right = transform.column(0).xyz();

        let mouse_delta = input_state
            .normalized_mouse_position()
            .map(|v| v * 10.0)
            .map(|v| f32::from(v.abs() > 0.1) * v);
        self.roll += mouse_delta.y * self.rotation_speed * delta_time;
        self.roll = self.roll.clamp(-89.0, 89.0);
        self.pitch += mouse_delta.x * 5.0 * self.rotation_speed * delta_time;
        let rotation =
            Rotation3::from_euler_angles(self.roll.to_radians(), self.pitch.to_radians(), 0.0);
        self.rotation = rotation;

        if input_state.is_key_pressed(Key::W) {
            self.location += forward * self.speed * delta_time;
        }
        if input_state.is_key_pressed(Key::S) {
            self.location -= forward * self.speed * delta_time;
        }
        if input_state.is_key_pressed(Key::A) {
            self.location -= right * self.speed * delta_time;
        }
        if input_state.is_key_pressed(Key::D) {
            self.location += right * self.speed * delta_time;
        }
    }
    pub fn camera(&self) -> Camera {
        let forward = self.rotation.transform_vector(&vector![0.0, 0.0, 1.0]);
        Camera {
            location: self.location,
            forward,
            fov: 45.0,
            width: 1920.0,
            height: 1080.0,
            near: 0.001,
            far: 1000.0,
        }
    }
}
