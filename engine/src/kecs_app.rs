use std::{
    borrow::Cow,
    ops::Deref,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use gpu::{CommandBuffer, Gpu, Offset2D, Rect2D};
use kecs::{Label, Resource, World};
use winit::{dpi::PhysicalSize, event::Event, event_loop::EventLoop};

use crate::{
    app::{app_state::AppState, App},
    AssetMap, Backbuffer, Camera, CvarManager, DeferredRenderingPipeline, GameScene, RenderScene,
    RenderingPipeline, Time,
};

#[derive(Clone)]
pub struct GpuDevice(Arc<dyn Gpu>);

#[derive(Clone)]
pub struct SharedAssetMap(Arc<RwLock<AssetMap>>);

impl SharedAssetMap {
    pub fn new(gpu: Arc<dyn Gpu>) -> Self {
        Self(Arc::new(RwLock::new(AssetMap::new(gpu))))
    }
    pub fn read(&self) -> RwLockReadGuard<'_, AssetMap> {
        self.0.read().expect("Failed to lock asset map")
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, AssetMap> {
        self.0.write().expect("Failed to lock asset map")
    }
}

impl Deref for GpuDevice {
    type Target = dyn Gpu;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Resource for GpuDevice {}
impl Resource for SharedAssetMap {}

#[derive(Clone, Copy, Default, Eq, PartialEq, PartialOrd, Ord)]
pub enum SimulationStep {
    #[default]
    FirstTick,
    Running,
    Stopped,
    Paused,

    Idle,
}

#[derive(Clone, Copy, Default)]
pub struct SimulationState {
    pub(crate) step: SimulationStep,
}

impl SimulationState {
    pub fn play(&mut self) {
        if self.step == SimulationStep::Paused {
            self.step = SimulationStep::Running
        } else {
            self.step = SimulationStep::FirstTick;
        }
    }

    pub fn stop(&mut self) {
        self.step = SimulationStep::Stopped;
    }

    pub fn pause(&mut self) {
        self.step = SimulationStep::Paused;
    }

    pub fn step(&self) -> SimulationStep {
        self.step
    }

    pub fn is_paused(&self) -> bool {
        self.step == SimulationStep::Paused
    }

    pub fn is_stopped(&self) -> bool {
        self.step == SimulationStep::Stopped || self.step == SimulationStep::Idle
    }

    pub fn toggle_play_pause(&mut self) {
        if self.is_paused() {
            self.play()
        } else {
            self.pause()
        }
    }
}

impl Resource for SimulationState {}

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

    fn shutdown(&mut self, _gpu: &dyn Gpu) {}
}

pub struct KecsApp {
    pub world: World,
    plugins: Vec<Box<dyn Plugin>>,
    pub renderer: DeferredRenderingPipeline,
}

impl KecsApp {
    pub const START: Label = Label::new(0);
    pub const PRE_UPDATE: Label = Label::new(1);
    pub const UPDATE: Label = Label::new(2);
    pub const POST_UPDATE: Label = Label::new(3);
    pub const END: Label = Label::new(4);
    pub const DRAW: Label = Label::new(5);
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
        world.add_resource(SharedAssetMap::new(app_state.gpu.clone()));
        world.add_resource(CvarManager::new());
        world.add_resource(RenderScene::new());
        world.add_resource(SimulationState::default());
        world.add_resource(GameScene::default());
        let combine_shader = DeferredRenderingPipeline::make_3d_combine_shader(app_state.gpu())?;
        let renderer = DeferredRenderingPipeline::new(app_state.gpu(), combine_shader)?;
        Ok(Self {
            world,
            plugins: vec![],
            renderer,
        })
    }

    fn on_startup(
        &mut self,
        _app_state: &mut crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
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
        let mut simulation = self
            .world
            .get_resource::<SimulationState>()
            .cloned()
            .unwrap();

        for plugin in &mut self.plugins {
            plugin.pre_update(&mut self.world);
        }

        for plugin in &mut self.plugins {
            plugin.update(&mut self.world);
        }

        for plugin in &mut self.plugins {
            plugin.post_update(&mut self.world);
        }

        let new_state = match simulation.step {
            SimulationStep::FirstTick => {
                self.world.update(Self::START);
                SimulationStep::Running
            }
            SimulationStep::Running => {
                self.world.update(Self::PRE_UPDATE);

                self.world.update(Self::UPDATE);

                self.world.update(Self::POST_UPDATE);
                SimulationStep::Running
            }
            SimulationStep::Stopped => {
                self.world.update(Self::END);
                SimulationStep::Idle
            }
            SimulationStep::Paused => SimulationStep::Paused,
            SimulationStep::Idle => SimulationStep::Idle,
        };

        simulation.step = new_state;
        self.world.add_resource(simulation);
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
        self.world.update(Self::DRAW);

        let mut command_buffer = app_state
            .gpu
            .start_command_buffer(gpu::QueueType::Graphics)?;

        let scene = self.world.get_resource::<RenderScene>();
        let camera = self
            .world
            .get_resource::<Camera>()
            .cloned()
            .unwrap_or_default();
        let resource_map = self.world.get_resource::<SharedAssetMap>().unwrap();
        let resource_map = resource_map.read();
        let cvar_manager = self.world.get_resource::<CvarManager>().unwrap();
        if let Some(scene) = scene {
            let final_render = self.renderer.render(
                app_state.gpu(),
                &mut command_buffer,
                &camera,
                scene,
                &resource_map,
                cvar_manager,
            )?;
            self.renderer.draw_textured_quad(
                &mut command_buffer,
                &backbuffer.image_view,
                &final_render,
                Rect2D {
                    offset: Offset2D::default(),
                    extent: backbuffer.size,
                },
                true,
                None,
            )?;
        }
        drop(resource_map);

        for plugin in &mut self.plugins {
            plugin.draw(&mut self.world, app_state, backbuffer, &mut command_buffer)?;
        }

        Ok(command_buffer)
    }

    fn on_shutdown(&mut self, app_state: &mut AppState) {
        for plugin in &mut self.plugins {
            plugin.shutdown(app_state.gpu());
        }
        self.renderer.destroy(app_state.gpu());
    }
}
