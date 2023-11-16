use bevy_ecs::{
    schedule::{Schedule, ScheduleLabel},
    world::World,
};

use crate::{
    app::{App, ImguiConsole},
    input::InputState,
    utils, Camera, CvarManager, DeferredRenderingPipeline, RenderingPipeline, ResourceMap, Scene,
};

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct StartupSchedule;

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct UpdateSchedule;

pub struct BevyEcsApp {
    world: World,
    startup_schedule: Schedule,
    update_schedule: Schedule,
    imgui_console: ImguiConsole,
    renderer: DeferredRenderingPipeline,
}

impl BevyEcsApp {
    fn world(&mut self) -> &mut World {
        &mut self.world
    }

    fn startup_schedule(&mut self) -> &mut Schedule {
        &mut self.startup_schedule
    }

    fn update_schedule(&mut self) -> &mut Schedule {
        &mut self.update_schedule
    }
}

impl App for BevyEcsApp {
    fn window_name(&self, _app_state: &crate::app::app_state::AppState) -> String {
        "Bevy ECS app".to_owned()
    }

    fn create(
        app_state: &mut crate::app::app_state::AppState,
        _event_loop: &winit::event_loop::EventLoop<()>,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut world = World::new();
        let mut resource_map = ResourceMap::new();
        let mut cvar_manager = CvarManager::new();
        let imgui_console = ImguiConsole::new();
        let input = InputState::new();
        let cube_mesh = utils::load_cube_to_resource_map(&app_state.gpu, &mut resource_map)?;
        let renderer = DeferredRenderingPipeline::new(
            &app_state.gpu,
            &mut resource_map,
            cube_mesh,
            &mut cvar_manager,
        )?;
        let startup_schedule = Schedule::new(StartupSchedule);
        let update_schedule = Schedule::new(UpdateSchedule);

        world.insert_resource(resource_map);
        world.insert_resource(cvar_manager);
        world.insert_resource(input);

        Ok(Self {
            world,
            startup_schedule,
            update_schedule,
            imgui_console,
            renderer,
        })
    }

    fn on_startup(&mut self, _app_state: &mut crate::app::app_state::AppState) {
        self.startup_schedule.run(&mut self.world);
    }

    fn input(
        &mut self,
        app_state: &crate::app::app_state::AppState,
        event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_event(
        &mut self,
        event: &winit::event::Event<()>,
        _app_state: &crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
        self.world
            .get_resource_mut::<InputState>()
            .unwrap()
            .update(event);
        self.world
            .get_resource_mut::<ResourceMap>()
            .unwrap()
            .update();
        Ok(())
    }

    fn update(
        &mut self,
        _app_state: &mut crate::app::app_state::AppState,
        ui: &mut imgui::Ui,
    ) -> anyhow::Result<()> {
        self.imgui_console
            .update(self.world.get_resource::<InputState>().unwrap());
        self.imgui_console.imgui_update(
            ui,
            &mut self.world.get_resource_mut::<CvarManager>().unwrap(),
        );
        self.update_schedule.run(&mut self.world);
        Ok(())
    }

    fn end_frame(&mut self, _app_state: &crate::app::app_state::AppState) {
        self.world
            .get_resource_mut::<InputState>()
            .unwrap()
            .end_frame();
    }
    fn draw<'a>(
        &'a mut self,
        app_state: &'a crate::app::app_state::AppState,
        backbuffer: &crate::Backbuffer,
    ) -> anyhow::Result<gpu::VkCommandBuffer> {
        let scene = self.world.get_resource::<Scene>().unwrap();
        let resource_map = self.world.get_resource::<ResourceMap>().unwrap();
        let cvar_manager = self.world.get_resource::<CvarManager>().unwrap();
        let pov = if let Some(pov) = self.world.get_resource::<Camera>() {
            *pov
        } else {
            Camera::default()
        };
        self.renderer.render(
            &app_state.gpu,
            &pov,
            scene,
            backbuffer,
            resource_map,
            cvar_manager,
        )
    }
}
