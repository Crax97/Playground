use std::time::*;

pub struct Time {
    app_start: Instant,
    last_frame: Instant,
    delta: f32,
    since_app_start: f32,
}

impl Time {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            app_start: now.clone(),
            last_frame: now,
            delta: 0.0,
            since_app_start: 0.0,
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let delta = now - self.last_frame;
        let delta = delta.as_millis() as f32 / 1000.0;

        self.delta = delta;
        self.last_frame = now;

        let delta = now - self.app_start;
        let delta = delta.as_millis() as f32 / 1000.0;

        self.since_app_start = delta;
    }

    pub fn since_app_start(&self) -> f32 {
        self.since_app_start
    }

    pub fn delta_frame(&self) -> f32 {
        self.delta
    }
}
