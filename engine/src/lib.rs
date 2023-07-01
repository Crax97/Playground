mod app_state;
mod camera;
mod gpu_pipeline;
mod material;
mod mesh;
mod render_graph;
mod scene;
mod static_deferred_renderer;
mod texture;
mod time;
mod utils;

use std::thread::ThreadId;

use gpu::{Gpu, GpuConfiguration};
use once_cell::unsync::OnceCell;

pub use app_state::*;
pub use camera::*;
pub use gpu_pipeline::*;
pub use material::*;
pub use mesh::*;
pub use render_graph::*;
pub use scene::*;
pub use static_deferred_renderer::*;
pub use texture::*;
pub use time::*;
pub use utils::constants::*;

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
pub fn init(app_name: &str, window: winit::window::Window) -> anyhow::Result<()> {
    unsafe {
        assert!(
            STATE.app == std::ptr::null_mut(),
            "Application can only be initialized once!"
        );

        static mut DATA: OnceCell<AppState> = once_cell::unsync::OnceCell::new();

        let enable_debug_utilities = std::env::var("ENABLE_DEBUG_UTILITIES").is_ok();

        let gpu = Gpu::new(GpuConfiguration {
            app_name,
            engine_name: "Hello Engine",
            enable_debug_utilities,
            window: &window,
        })?;

        let swapchain = gpu::Swapchain::new(&gpu, window)?;

        let app_state = AppState::new(gpu, swapchain);
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
            STATE.app != std::ptr::null_mut(),
            "Application has not been initialized!"
        );
        assert!(
            std::thread::current().id() == STATE.creator_id.unwrap(),
            "Tried to access app state from a thread that is not the main thread!"
        );
        &*STATE.app
    }
}

pub fn app_state_mut() -> &'static mut AppState {
    unsafe {
        assert!(
            STATE.app != std::ptr::null_mut(),
            "Application has not been initialized!"
        );
        assert!(
            std::thread::current().id() == STATE.creator_id.unwrap(),
            "Tried to access app state from a thread that is not the main thread!"
        );
        &mut *STATE.app
    }
}
