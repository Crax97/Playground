use std::thread::ThreadId;

use gpu::{Gpu, GpuConfiguration, VkGpu, VkSwapchain};
use once_cell::unsync::OnceCell;
use winit::{dpi::PhysicalSize, window::Window};

pub struct AppState {
    pub gpu: VkGpu,
    pub swapchain: VkSwapchain,
    pub needs_new_swapchain: bool,
    pub current_window_size: PhysicalSize<u32>,
}

impl AppState {
    pub fn new(gpu: VkGpu, window: &Window) -> Self {
        let swapchain = VkSwapchain::new(&gpu, window).expect("Failed to create swapchain!");
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

    pub fn swapchain(&self) -> &VkSwapchain {
        &self.swapchain
    }

    pub fn swapchain_mut(&mut self) -> &mut VkSwapchain {
        &mut self.swapchain
    }
}

struct GlobalState {
    app: *mut AppState,
    creator_id: Option<ThreadId>,
}

impl GlobalState {
    const UNINIT: GlobalState = GlobalState {
        app: std::ptr::null_mut(),
        creator_id: None,
    };
}

static mut STATE: GlobalState = GlobalState::UNINIT;

/*
    Creates a global AppState, which is going to belong to a single thread.
    The AppState can be only accessed by the thread that ran engine::init()
*/
pub fn init(app_name: &str, window: &winit::window::Window) -> anyhow::Result<()> {
    unsafe {
        assert!(
            STATE.app.is_null(),
            "Application can only be initialized once!"
        );

        static mut DATA: OnceCell<AppState> = once_cell::unsync::OnceCell::new();

        let enable_debug_utilities = if cfg!(debug_assertions) {
            true
        } else {
            std::env::var("ENABLE_DEBUG_UTILITIES").is_ok()
        };

        let gpu = VkGpu::new(GpuConfiguration {
            app_name,
            enable_debug_utilities,
            pipeline_cache_path: Some("pipeline_cache.pso"),
            window: Some(&window),
        })?;

        let app_state = AppState::new(gpu, &window);
        assert!(DATA.set(app_state).is_ok());

        STATE = GlobalState {
            app: DATA.get_mut().unwrap() as *mut AppState,
            creator_id: Some(std::thread::current().id()),
        }
    }
    Ok(())
}
pub fn app_state() -> &'static AppState {
    unsafe {
        assert!(
            !STATE.app.is_null(),
            "Application has not been initialized!"
        );
        assert_eq!(
            std::thread::current().id(),
            STATE.creator_id.unwrap(),
            "Tried to access app state from a thread that is not the main thread!"
        );
        &*STATE.app
    }
}

pub fn app_state_mut() -> &'static mut AppState {
    unsafe {
        assert!(
            !STATE.app.is_null(),
            "Application has not been initialized!"
        );
        assert_eq!(
            std::thread::current().id(),
            STATE.creator_id.unwrap(),
            "Tried to access app state from a thread that is not the main thread!"
        );
        &mut *STATE.app
    }
}
