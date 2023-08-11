use gpu::{Gpu, Swapchain};
use winit::window::Window;

use crate::Time;

pub struct AppState {
    pub gpu: Gpu,
    pub time: Time,
    pub swapchain: Swapchain,
    pub window: Window,
}

impl AppState {
    pub fn new(gpu: Gpu, window: Window) -> Self {
        let swapchain = Swapchain::new(&gpu, &window).expect("Failed to create swapchain!");

        Self {
            gpu,
            time: Time::new(),
            swapchain,
            window,
        }
    }

    pub fn begin_frame(&mut self) -> anyhow::Result<()> {
        self.time.begin_frame();
        Ok(())
    }

    pub fn end_frame(&mut self) -> anyhow::Result<()> {
        self.swapchain.present()?;
        self.time.end_frame();
        Ok(())
    }

    pub fn time(&self) -> &Time {
        &self.time
    }
    pub fn swapchain(&self) -> &Swapchain {
        &self.swapchain
    }

    pub fn swapchain_mut(&mut self) -> &mut Swapchain {
        &mut self.swapchain
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}
