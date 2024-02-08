use std::sync::Arc;

use gpu::{Gpu, GpuConfiguration, Swapchain};
use winit::{dpi::PhysicalSize, window::Window};

pub struct AppState {
    pub gpu: Arc<dyn Gpu>,
    pub swapchain: Swapchain,
    pub window: Window,
    pub(crate) needs_new_swapchain: bool,
    pub(crate) current_window_size: PhysicalSize<u32>,
}

impl AppState {
    pub fn begin_frame(&mut self) -> anyhow::Result<()> {
        self.gpu.update();
        Ok(())
    }

    pub fn end_frame(&mut self) -> anyhow::Result<()> {
        self.gpu.submit_work();
        self.swapchain.present()?;
        self.gpu.end_frame();
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
pub fn init(app_name: &str, window: winit::window::Window) -> anyhow::Result<AppState> {
    let enable_debug_utilities = if cfg!(debug_assertions) {
        true
    } else {
        std::env::var("ENABLE_DEBUG_UTILITIES").is_ok()
    };

    let gpu = gpu::make_gpu(GpuConfiguration {
        app_name,
        enable_debug_utilities,
        pipeline_cache_path: Some("pipeline_cache.pso"),

        window: Some(&window),
    })?;
    let swapchain = gpu
        .create_swapchain(&window)
        .expect("Failed to create swapchain");
    let size = window.inner_size();
    let app_state = AppState {
        window,
        gpu,
        swapchain,
        current_window_size: size,
        needs_new_swapchain: false,
    };
    Ok(app_state)
}
