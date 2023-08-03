use ash::prelude::VkResult;
use ash::vk::CommandPoolResetFlags;
use winit::window::Window;
use gpu::{Gpu, Swapchain};

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
            window
        }
    }

    pub fn begin_frame(&mut self) -> VkResult<()> {
        self.time.begin_frame();
        unsafe {
            self.gpu.vk_logical_device().reset_command_pool(
                self.gpu.thread_local_state.graphics_command_pool,
                CommandPoolResetFlags::empty(),
            )?;
        }
        Ok(())
    }

    pub fn end_frame(&mut self) -> VkResult<()> {
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
