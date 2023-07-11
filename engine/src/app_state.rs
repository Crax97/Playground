use ash::prelude::VkResult;
use gpu::Gpu;

use crate::Time;

pub struct AppState {
    pub gpu: Gpu,
    pub time: Time,
}
impl AppState {
    pub fn new(gpu: Gpu) -> Self {
        Self {
            gpu,
            time: Time::new(),
        }
    }

    pub fn begin_frame(&mut self) -> VkResult<()> {
        self.time.begin_frame();
        Ok(())
    }

    pub fn end_frame(&mut self) -> VkResult<()> {
        self.gpu.present()?;
        self.time.end_frame();
        Ok(())
    }

    pub fn time(&self) -> &Time {
        &self.time
    }
}
