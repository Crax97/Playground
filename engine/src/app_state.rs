use gpu::{Gpu, VkGpu, VkSwapchain};
use crate::resource_map::ResourceMap;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    cvar_manager::{CvarFlags, CvarManager},
    input::InputState,
    Time,
};

pub struct AppState {
    pub gpu: VkGpu,
    pub time: Time,
    pub swapchain: VkSwapchain,
    pub window: Window,
    pub new_size: Option<PhysicalSize<u32>>,
    pub input: InputState,
    pub cvar_manager: CvarManager,
    pub resource_map: ResourceMap,
}

impl AppState {
    pub fn new(gpu: VkGpu, window: Window) -> Self {
        let swapchain = VkSwapchain::new(&gpu, &window).expect("Failed to create swapchain!");
        let mut cvar_manager = CvarManager::new();
        cvar_manager.register_cvar("g_test", 10, CvarFlags::empty());

        let resource_manager = ResourceMap::new();

        Self {
            gpu,
            time: Time::new(),
            swapchain,
            window,
            new_size: None,
            cvar_manager,
            input: InputState::new(),
            resource_map: resource_manager,
        }
    }

    pub fn begin_frame(&mut self) -> anyhow::Result<()> {
        self.time.begin_frame();
        self.gpu.update();
        Ok(())
    }

    pub fn end_frame(&mut self) -> anyhow::Result<()> {
        self.swapchain.present()?;
        self.time.end_frame();
        self.input.end_frame();
        self.resource_map.update();
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
