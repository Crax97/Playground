use engine::app::{bootstrap, App, AppContext, AppDescription};

struct BlankApp;

impl App for BlankApp {
    fn create(_context: &engine::app::AppContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self)
    }

    fn handle_window_event(
        &mut self,
        _event: &winit::event::WindowEvent,
        _context: &AppContext,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    fn handle_device_event(
        &mut self,
        _event: &winit::event::DeviceEvent,
        _context: &AppContext,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, _context: &engine::app::AppContext) -> anyhow::Result<()> {
        Ok(())
    }

    fn render(
        &mut self,
        _context: &engine::app::AppContext,
        _render_context: engine::app::RenderContext,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn resized(
        &mut self,
        _context: &engine::app::AppContext,
        _new_extents: mgpu::Extents2D,
    ) -> mgpu::MgpuResult<()> {
        Ok(())
    }

    fn shutdown(&mut self, _context: &engine::app::AppContext) -> anyhow::Result<()> {
        Ok(())
    }
}

fn main() {
    bootstrap::<BlankApp>(AppDescription::default()).unwrap()
}
