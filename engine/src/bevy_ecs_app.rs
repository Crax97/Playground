use std::ops::{Deref, DerefMut};

use bevy_ecs::{
    schedule::{Schedule, ScheduleLabel},
    system::Resource,
    world::World,
};
use nalgebra::vector;
use winit::event_loop::EventLoop;

use crate::{
    app::{App, ImguiConsole},
    input::InputState,
    resource_map, utils, Camera, CvarManager, DeferredRenderingPipeline, Mesh, MeshCreateInfo,
    MeshPrimitiveCreateInfo, RenderingPipeline, ResourceHandle, ResourceMap, Scene, Texture,
};

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct StartupSchedule;

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct UpdateSchedule;

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct PostUpdateSchedule;

pub struct BevyEcsApp {
    world: World,
    startup_schedule: Schedule,
    update_schedule: Schedule,
    post_update_schedule: Schedule,
    imgui_console: ImguiConsole,
    renderer: DeferredRenderingPipeline,
}

pub struct BevyEcsAppWithLoop {
    app: BevyEcsApp,
    evt_loop: EventLoop<()>,
}

#[derive(Resource)]
pub struct CommonResources {
    pub cube_mesh: ResourceHandle<Mesh>,
    pub quad_mesh: ResourceHandle<Mesh>,

    pub white_texture: ResourceHandle<Texture>,
    pub black_texture: ResourceHandle<Texture>,
}

impl BevyEcsApp {
    pub fn new() -> anyhow::Result<BevyEcsAppWithLoop> {
        let (app, evt_loop) = crate::app::create_app::<Self>()?;

        Ok(BevyEcsAppWithLoop { app, evt_loop })
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn startup_schedule(&mut self) -> &mut Schedule {
        &mut self.startup_schedule
    }

    pub fn update_schedule(&mut self) -> &mut Schedule {
        &mut self.update_schedule
    }

    pub fn post_update_schedule(&mut self) -> &mut Schedule {
        &mut self.post_update_schedule
    }

    fn create_common_resources(
        gpu: &gpu::VkGpu,
        resource_map: &mut ResourceMap,
    ) -> anyhow::Result<CommonResources> {
        let quad_mesh = {
            let mesh_data = MeshCreateInfo {
                label: Some("Quad mesh"),
                primitives: &[MeshPrimitiveCreateInfo {
                    indices: vec![0, 1, 2, 2, 3, 0],
                    positions: vec![
                        vector![-0.5, -0.5, 0.0],
                        vector![0.5, -0.5, 0.0],
                        vector![0.5, 0.5, 0.0],
                        vector![-0.5, 0.5, 0.0],
                    ],
                    colors: vec![
                        vector![1.0, 0.0, 0.0],
                        vector![0.0, 1.0, 0.0],
                        vector![0.0, 0.0, 1.0],
                        vector![1.0, 1.0, 1.0],
                    ],
                    normals: vec![
                        vector![0.0, 1.0, 0.0],
                        vector![0.0, 1.0, 0.0],
                        vector![0.0, 1.0, 0.0],
                        vector![0.0, 1.0, 0.0],
                    ],
                    tangents: vec![
                        vector![0.0, 0.0, 1.0],
                        vector![0.0, 0.0, 1.0],
                        vector![0.0, 0.0, 1.0],
                        vector![0.0, 0.0, 1.0],
                    ],
                    uvs: vec![
                        vector![1.0, 0.0],
                        vector![0.0, 0.0],
                        vector![0.0, 1.0],
                        vector![1.0, 1.0],
                    ],
                }],
            };

            let mesh = Mesh::new(gpu, &mesh_data)?;
            resource_map.add(mesh)
        };

        let cube_mesh = utils::load_cube_to_resource_map(gpu, resource_map)?;
        let white_texture = Texture::new_with_data(
            gpu,
            1,
            1,
            &[0, 0, 0, 255],
            Some("White texture"),
            gpu::ImageFormat::Rgba8,
            gpu::ImageViewType::Type2D,
        )?;
        let black_texture = Texture::new_with_data(
            gpu,
            1,
            1,
            &[255, 255, 255, 255],
            Some("Black texture"),
            gpu::ImageFormat::Rgba8,
            gpu::ImageViewType::Type2D,
        )?;
        let white_texture = resource_map.add(white_texture);
        let black_texture = resource_map.add(black_texture);

        Ok(CommonResources {
            cube_mesh,
            quad_mesh,
            white_texture,
            black_texture,
        })
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
        let (imgui_console, console_writer) = ImguiConsole::new_with_writer();
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
        let post_update_schedule = Schedule::new(PostUpdateSchedule);

        let default_resources = Self::create_common_resources(&app_state.gpu, &mut resource_map)?;

        world.insert_resource(resource_map);
        world.insert_resource(cvar_manager);
        world.insert_resource(input);
        world.insert_resource(console_writer);
        world.insert_resource(default_resources);

        Ok(Self {
            world,
            startup_schedule,
            update_schedule,
            post_update_schedule,
            imgui_console,
            renderer,
        })
    }

    fn on_startup(
        &mut self,
        _app_state: &mut crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
        self.startup_schedule.run(&mut self.world);
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
        self.post_update_schedule.run(&mut self.world);
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
        let empty_scene = Scene::default();
        let scene = self.world.get_resource::<Scene>().unwrap_or(&empty_scene);
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

impl BevyEcsAppWithLoop {
    pub fn run(self) -> anyhow::Result<()> {
        crate::app::run(self.app, self.evt_loop)
    }

    pub fn event_loop(&self) -> &EventLoop<()> {
        &self.evt_loop
    }

    pub fn event_loop_mut(&mut self) -> &mut EventLoop<()> {
        &mut self.evt_loop
    }
}

impl Deref for BevyEcsAppWithLoop {
    type Target = BevyEcsApp;

    fn deref(&self) -> &Self::Target {
        &self.app
    }
}

impl DerefMut for BevyEcsAppWithLoop {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.app
    }
}
