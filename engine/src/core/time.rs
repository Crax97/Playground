use std::time::SystemTime;

pub struct Time {
    delta_seconds: f64,
    total_time: f64,
    total_ticks: u64,
    now: SystemTime,
}

impl Time {
    pub fn new() -> Self {
        Self {
            delta_seconds: 0.0,
            total_time: 0.0,
            total_ticks: 0,
            now: SystemTime::now(),
        }
    }

    pub fn begin_frame(&mut self) {
        self.now = SystemTime::now();
    }

    pub fn end_frame(&mut self) {
        self.delta_seconds = self.delta_from_frame_begin();
        self.total_time += self.delta_seconds;
        self.total_ticks += 1;
    }

    pub fn delta_seconds(&self) -> f64 {
        self.delta_seconds
    }

    pub fn delta_from_frame_begin(&self) -> f64 {
        let difference = self.now.elapsed().unwrap();
        difference.as_secs_f64()
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}
