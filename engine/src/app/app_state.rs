use std::sync::Arc;

use gpu::{Gpu, GpuConfiguration, Swapchain};
use winit::{dpi::PhysicalSize, window::Window};

pub struct AppState {
    pub gpu: Arc<dyn Gpu>,
    pub swapchain: Swapchain,
    pub needs_new_swapchain: bool,
    pub current_window_size: PhysicalSize<u32>,
}

impl AppState {
    pub fn new(gpu: Arc<dyn Gpu>, window: &Window) -> Self {
        let swapchain = gpu
            .create_swapchain(window)
            .expect("Failed to create swapchain");
        Self {
            gpu,
            swapchain,
            needs_new_swapchain: false,
            current_window_size: window.inner_size(),
        }
    }

    pub fn begin_frame(&mut self) -> anyhow::Result<()> {
        self.gpu.update();
        Ok(())
    }

    pub fn end_frame(&mut self) -> anyhow::Result<()> {
        self.swapchain.present()?;
        Ok(())
    }

    pub fn swapchain(&self) -> &Swapchain {
        &self.swapchain
    }

    pub fn swapchain_mut(&mut self) -> &mut Swapchain {
        &mut self.swapchain
    }

    pub fn gpu(&self) -> &dyn Gpu {
        self.gpu.as_ref()
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        self.swapchain.destroy(self.gpu.as_ref());
        self.gpu.destroy();
    }
}

/*
    Creates a global AppState, which is going to belong to a single thread.
    The AppState can be only accessed by the thread that ran engine::init()
*/
pub fn init(app_name: &str, window: &winit::window::Window) -> anyhow::Result<AppState> {
    let enable_debug_utilities = if cfg!(debug_assertions) {
        true
    } else {
        std::env::var("ENABLE_DEBUG_UTILITIES").is_ok()
    };

    let gpu = gpu::make_gpu(GpuConfiguration {
        app_name,
        enable_debug_utilities,
        pipeline_cache_path: Some("pipeline_cache.pso"),
        window: Some(window),
    })?;

    let app_state = AppState::new(gpu, window);
    Ok(app_state)
}
