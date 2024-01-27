use std::time::*;

use bevy_ecs::system::Resource;

#[derive(Resource)]
pub struct Time {
    app_start: Instant,
    last_frame: Instant,
    delta: f32,
    since_app_start: f32,
    frame_counter: u64,
}

impl Time {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            app_start: now,
            last_frame: now,
            delta: 0.0,
            since_app_start: 0.0,
            frame_counter: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        let now = Instant::now();
        let delta = now - self.last_frame;
        let delta = delta.as_millis() as f32 / 1000.0;

        self.delta = delta;
        self.last_frame = now;

        let delta = now - self.app_start;
        let delta = delta.as_millis() as f32 / 1000.0;

        self.since_app_start = delta;
    }

    pub fn end_frame(&mut self) {
        self.frame_counter += 1;
    }

    pub fn since_app_start(&self) -> f32 {
        self.since_app_start
    }

    pub fn delta_frame(&self) -> f32 {
        self.delta
    }

    pub fn frames_since_start(&self) -> u64 {
        self.frame_counter
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}
