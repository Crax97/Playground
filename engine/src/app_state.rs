use gpu::{VkSwapchain, VkGpu};
use winit::{dpi::PhysicalSize, window::Window};

use crate::Time;

pub struct AppState {
    pub gpu: VkGpu,
    pub time: Time,
    pub swapchain: VkSwapchain,
    pub window: Window,
    pub new_size: Option<PhysicalSize<u32>>,
}

impl AppState {
    pub fn new(gpu: VkGpu, window: Window) -> Self {
        let swapchain = VkSwapchain::new(&gpu, &window).expect("Failed to create swapchain!");

        Self {
            gpu,
            time: Time::new(),
            swapchain,
            window,
            new_size: None,
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
    pub fn swapchain(&self) -> &VkSwapchain {
        &self.swapchain
    }

    pub fn swapchain_mut(&mut self) -> &mut VkSwapchain {
        &mut self.swapchain
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}
