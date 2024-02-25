use crate::{
    app::App,
    game_framework::world::{World, WorldBuilder},
    ResourceMap,
};

pub struct EngineApp {
    world: World,
}

impl EngineApp {
    pub fn new(world: World) -> Self {
        Self { world }
    }
}

impl App for EngineApp {
    fn window_name(&self, app_state: &crate::app::app_state::AppState) -> std::borrow::Cow<str> {
        todo!()
    }

    fn create(
        app_state: &mut crate::app::app_state::AppState,
        event_loop: &winit::event_loop::EventLoop<()>,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn on_shutdown(&mut self, app_state: &mut crate::app::app_state::AppState) {
        self.world.shutdown();
        if let Some(mut resource_map) = self.world.resources().try_get_mut::<ResourceMap>() {
            resource_map.update();
        }
    }

    fn update(&mut self, app_state: &mut crate::app::app_state::AppState) -> anyhow::Result<()> {
        self.world.update(1.0 / 60.0);
        if let Some(mut resource_map) = self.world.resources().try_get_mut::<ResourceMap>() {
            resource_map.update();
        }
        Ok(())
    }

    fn on_event(
        &mut self,
        event: &winit::event::Event<()>,
        _app_state: &crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
        self.world.on_os_event(event);
        Ok(())
    }

    fn draw<'a>(
        &'a mut self,
        app_state: &'a mut crate::app::app_state::AppState,
        backbuffer: &crate::Backbuffer,
    ) -> anyhow::Result<gpu::CommandBuffer> {
        let mut command_buffer = app_state
            .gpu
            .start_command_buffer(gpu::QueueType::Graphics)?;

        self.world.draw(&mut command_buffer, backbuffer);

        Ok(command_buffer)
    }
}
