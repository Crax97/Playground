use std::time::Duration;

pub struct FpsLimiter {
    target_ms: f64,
}

impl FpsLimiter {
    pub fn new(target_fps: u64) -> Self {
        let target_ms = 1.0 / target_fps as f64;

        Self { target_ms }
    }

    pub fn update(&self, frame_time: f64) {
        let delta = self.target_ms - frame_time;
        if delta > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(delta))
        }
    }
}
