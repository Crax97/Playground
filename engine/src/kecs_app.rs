use std::{borrow::Cow, ops::Deref, sync::Arc};

use gpu::{CommandBuffer, Gpu};
use kecs::{Label, Resource, World};
use winit::{dpi::PhysicalSize, event::Event, event_loop::EventLoop};

use crate::{
    app::{app_state::AppState, App},
    Backbuffer, DeferredRenderingPipeline, Time,
};

#[derive(Clone)]
pub struct GpuDevice(Arc<dyn Gpu>);

impl Deref for GpuDevice {
    type Target = dyn Gpu;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Resource for GpuDevice {}

pub trait Plugin: 'static {
    fn on_start(&mut self, _world: &mut World) {}
    fn on_event(&mut self, _app_state: &AppState, _world: &mut World, _event: &Event<()>) {}
    fn on_resize(
        &mut self,
        _world: &mut World,
        _app_state: &AppState,
        _new_size: PhysicalSize<u32>,
    ) {
    }

    fn pre_update(&mut self, _world: &mut World) {}
    fn update(&mut self, _world: &mut World) {}
    fn post_update(&mut self, _world: &mut World) {}
    fn draw(
        &mut self,
        _world: &mut World,
        _app_state: &mut AppState,
        _backbuffer: &Backbuffer,
        _command_buffer: &mut CommandBuffer,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

pub struct KecsApp {
    pub world: World,
    plugins: Vec<Box<dyn Plugin>>,
}

impl KecsApp {
    pub const START: Label = Label::new(0);
    pub const PRE_UPDATE: Label = Label::new(1);
    pub const UPDATE: Label = Label::new(2);
    pub const POST_UPDATE: Label = Label::new(3);
    pub const END: Label = Label::new(4);
    pub fn create() -> anyhow::Result<(KecsApp, EventLoop<()>, AppState)> {
        let (app, evt_loop, state) = crate::app::create_app::<Self>()?;

        Ok((app, evt_loop, state))
    }

    pub fn add_plugin<P: Plugin>(&mut self, plugin: P) {
        let plugin = Box::new(plugin);
        self.plugins.push(plugin);
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

impl App for KecsApp {
    fn window_name(&self, _app_state: &crate::app::app_state::AppState) -> Cow<str> {
        Cow::Borrowed("Kecs ECS app")
    }

    fn create(
        app_state: &mut crate::app::app_state::AppState,
        _event_loop: &winit::event_loop::EventLoop<()>,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut world = World::new();
        world.add_resource(Time::default());
        Ok(Self {
            world,
            plugins: vec![],
        })
    }

    fn on_startup(
        &mut self,
        _app_state: &mut crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
        self.world.update(Self::START);
        for plugin in &mut self.plugins {
            plugin.on_start(&mut self.world);
        }
        Ok(())
    }

    fn on_event(
        &mut self,
        event: &winit::event::Event<()>,
        app_state: &crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
        // self.world
        //     .get_resource_mut::<InputState>()
        //     .unwrap()
        //     .update(event);
        // self.world
        //     .get_resource_mut::<ResourceMap>()
        //     .unwrap()
        //     .update();

        for plugin in &mut self.plugins {
            plugin.on_event(app_state, &mut self.world, event);
        }
        Ok(())
    }

    fn on_resized(&mut self, app_state: &AppState, size: PhysicalSize<u32>) {
        for plugin in &mut self.plugins {
            plugin.on_resize(&mut self.world, app_state, size);
        }
    }

    fn begin_frame(
        &mut self,
        _app_state: &mut crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
        let time = self.world.get_resource_mut::<Time>().unwrap();
        time.begin_frame();
        Ok(())
    }

    fn update(&mut self, _app_state: &mut crate::app::app_state::AppState) -> anyhow::Result<()> {
        // self.imgui_console
        //     .update(self.world.get_resource::<InputState>().unwrap());
        // self.imgui_console.imgui_update(
        //     ui,
        //     &mut self.world.get_resource_mut::<CvarManager>().unwrap(),
        // );
        self.world.update(Self::PRE_UPDATE);
        for plugin in &mut self.plugins {
            plugin.pre_update(&mut self.world);
        }

        self.world.update(Self::UPDATE);
        for plugin in &mut self.plugins {
            plugin.update(&mut self.world);
        }

        self.world.update(Self::POST_UPDATE);
        for plugin in &mut self.plugins {
            plugin.post_update(&mut self.world);
        }
        Ok(())
    }

    fn end_frame(&mut self, _app_state: &crate::app::app_state::AppState) {
        let time = self.world.get_resource_mut::<Time>().unwrap();
        time.end_frame();
    }
    fn draw<'a>(
        &'a mut self,
        app_state: &'a mut crate::app::app_state::AppState,
        backbuffer: &crate::Backbuffer,
    ) -> anyhow::Result<gpu::CommandBuffer> {
        let mut command_buffer = app_state
            .gpu
            .start_command_buffer(gpu::QueueType::Graphics)?;

        for plugin in &mut self.plugins {
            plugin.draw(&mut self.world, app_state, backbuffer, &mut command_buffer)?;
        }

        Ok(command_buffer)
    }

    fn on_shutdown(&mut self, _app_state: &mut AppState) {
        self.world.update(Self::END);
        // for plugin in &mut self.plugins {
        //     plugin.post_update(&mut self.world);
        // }
    }
}
