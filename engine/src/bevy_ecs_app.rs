use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use bevy_ecs::{
    schedule::{IntoSystemConfigs, Schedule, ScheduleLabel},
    system::Resource,
    world::{Mut, World},
};
use bevy_reflect::{GetTypeRegistration, TypeRegistry};
use egui::mutex::RwLock;
use engine_macros::glsl;
use gpu::{CommandBuffer, Gpu, Offset2D, Rect2D, ShaderModuleCreateInfo, ShaderStage};
use nalgebra::vector;
use winit::{dpi::PhysicalSize, event::Event, event_loop::EventLoop, window::Window};

use crate::{
    app::{app_state::AppState, App},
    components::EngineWindow,
    input::InputState,
    loaders::FileSystemTextureLoader,
    physics::PhysicsContext2D,
    render_scene::camera::Camera,
    utils, CvarManager, DeferredRenderingPipeline, MasterMaterial, Mesh, MeshCreateInfo,
    MeshPrimitiveCreateInfo, RenderScene, RenderingPipeline, ResourceHandle, ResourceMap, Texture,
    TextureInput, Time,
};

const DEFAULT_DEFERRED_FS: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/default_fragment_deferred.frag",
    entry_point = "main"
);

const DEFAULT_DEFERRED_TRANSPARENCY_FS: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/default_fragment_deferred_discard.frag",
    entry_point = "main"
);

const DEFAULT_FRAGMENT_SPRITE: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/default_fragment_sprite.frag",
    entry_point = "main"
);

const DEFAULT_DEFERRED_VS: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/default_vertex_deferred.vert",
    entry_point = "main"
);

#[derive(Clone, Resource)]
pub struct GpuDevice(Arc<dyn Gpu>);

impl Deref for GpuDevice {
    type Target = dyn Gpu;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

#[derive(Default, Clone)]
pub struct AppTypeRegistry(Arc<RwLock<TypeRegistry>>);

impl Deref for AppTypeRegistry {
    type Target = RwLock<TypeRegistry>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait Plugin: 'static {
    fn on_start(&mut self, _world: &mut World) {}
    fn on_event(&mut self, _world: &mut World, _event: &Event<()>) {}
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
        _command_buffer: &mut CommandBuffer,
    ) {
    }
}

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct StartupSchedule;

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct PreUpdateSchedule;

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct UpdateSchedule;

#[derive(ScheduleLabel, Debug, Hash, Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct PostUpdateSchedule;

pub struct BevyEcsApp {
    pub world: World,
    startup_schedule: Schedule,
    pre_update_schedule: Schedule,
    update_schedule: Schedule,
    post_update_schedule: Schedule,
    renderer: DeferredRenderingPipeline,
    plugins: Vec<Box<dyn Plugin>>,
    type_registry: AppTypeRegistry,
}

pub struct BevyEcsAppWithLoop {
    pub app: BevyEcsApp,
    evt_loop: EventLoop<()>,
    pub state: AppState,
}

#[derive(Resource)]
pub struct CommonResources {
    pub cube_mesh: ResourceHandle<Mesh>,
    pub quad_mesh: ResourceHandle<Mesh>,

    pub white_texture: ResourceHandle<Texture>,
    pub black_texture: ResourceHandle<Texture>,

    pub default_material: ResourceHandle<MasterMaterial>,
    pub default_material_transparency: ResourceHandle<MasterMaterial>,
    pub default_sprite_material: ResourceHandle<MasterMaterial>,
}

impl BevyEcsApp {
    pub fn new() -> anyhow::Result<BevyEcsAppWithLoop> {
        let (app, evt_loop, state) = crate::app::create_app::<Self>()?;

        Ok(BevyEcsAppWithLoop {
            app,
            evt_loop,
            state,
        })
    }

    pub fn add_plugin<P: Plugin>(&mut self, plugin: P) {
        let plugin = Box::new(plugin);
        self.plugins.push(plugin);
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn renderer(&mut self) -> &mut DeferredRenderingPipeline {
        &mut self.renderer
    }

    pub fn resource_map(&mut self) -> Mut<'_, ResourceMap> {
        self.world.get_resource_mut::<ResourceMap>().unwrap()
    }

    pub fn startup_schedule(&mut self) -> &mut Schedule {
        &mut self.startup_schedule
    }

    pub fn pre_update_schedule(&mut self) -> &mut Schedule {
        &mut self.post_update_schedule
    }

    pub fn update_schedule(&mut self) -> &mut Schedule {
        &mut self.update_schedule
    }

    pub fn post_update_schedule(&mut self) -> &mut Schedule {
        &mut self.post_update_schedule
    }

    pub fn type_registry(&self) -> &AppTypeRegistry {
        &self.type_registry
    }

    pub fn register_type<T: GetTypeRegistration>(&mut self) -> &mut Self {
        self.type_registry
            .0
            .write()
            .add_registration(T::get_type_registration());
        self
    }

    pub fn setup_2d(&mut self) {
        self.world().insert_resource(PhysicsContext2D::new());
        self.post_update_schedule()
            .add_systems(crate::physics::update_positions_before_physics_system)
            .add_systems(crate::physics::update_positions_after_physics_system)
            .add_systems(
                crate::physics::update_physics_2d_context
                    .before(crate::physics::update_positions_after_physics_system)
                    .after(crate::physics::update_positions_before_physics_system),
            )
            .add_systems(
                crate::components::rendering_system_2d
                    .after(crate::physics::update_positions_after_physics_system),
            );
    }

    fn create_common_resources(
        gpu: &dyn Gpu,
        resource_map: &mut ResourceMap,
    ) -> anyhow::Result<CommonResources> {
        let quad_mesh = {
            let mesh_data = MeshCreateInfo {
                label: Some("Quad mesh"),
                primitives: &[MeshPrimitiveCreateInfo {
                    indices: vec![2, 1, 0, 0, 3, 2],
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
                        vector![1.0, 1.0],
                        vector![0.0, 1.0],
                        vector![0.0, 0.0],
                        vector![1.0, 0.0],
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

        let vertex_module = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Default vertex shader"),
            code: bytemuck::cast_slice(DEFAULT_DEFERRED_VS),
        })?;
        let fragment_module = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Default fragment shader"),
            code: bytemuck::cast_slice(DEFAULT_DEFERRED_FS),
        })?;
        let fragment_module_transparency = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Default transparency fragment shader"),
            code: bytemuck::cast_slice(DEFAULT_DEFERRED_TRANSPARENCY_FS),
        })?;
        let fragment_sprite = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Default sprite shader"),
            code: bytemuck::cast_slice(DEFAULT_FRAGMENT_SPRITE),
        })?;
        let default_material = MasterMaterial::new(&crate::MasterMaterialDescription {
            name: "Default master deferred",
            domain: crate::MaterialDomain::Surface,
            material_parameters: Default::default(),
            parameters_visibility: ShaderStage::FRAGMENT,
            vertex_info: &gpu::VertexStageInfo {
                entry_point: "main",
                module: vertex_module.clone(),
            },
            fragment_info: &gpu::FragmentStageInfo {
                entry_point: "main",
                module: fragment_module,
            },

            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::CounterClockWise,

            texture_inputs: &[TextureInput {
                name: "texSampler".to_owned(),
                format: gpu::ImageFormat::Rgba8,
                shader_stage: ShaderStage::FRAGMENT,
            }],
        })?;
        let default_material_transparency =
            MasterMaterial::new(&crate::MasterMaterialDescription {
                name: "Default master deferred - transparency",
                domain: crate::MaterialDomain::Surface,
                material_parameters: Default::default(),
                parameters_visibility: ShaderStage::FRAGMENT,
                vertex_info: &gpu::VertexStageInfo {
                    entry_point: "main",
                    module: vertex_module.clone(),
                },
                fragment_info: &gpu::FragmentStageInfo {
                    entry_point: "main",
                    module: fragment_module_transparency,
                },

                cull_mode: gpu::CullMode::Back,
                front_face: gpu::FrontFace::CounterClockWise,

                texture_inputs: &[TextureInput {
                    name: "texSampler".to_owned(),
                    format: gpu::ImageFormat::Rgba8,
                    shader_stage: ShaderStage::FRAGMENT,
                }],
            })?;
        let default_sprite_material = MasterMaterial::new(&crate::MasterMaterialDescription {
            name: "Default sprite material",
            domain: crate::MaterialDomain::Surface,
            material_parameters: Default::default(),
            parameters_visibility: ShaderStage::FRAGMENT,
            vertex_info: &gpu::VertexStageInfo {
                entry_point: "main",
                module: vertex_module,
            },
            fragment_info: &gpu::FragmentStageInfo {
                entry_point: "main",
                module: fragment_sprite,
            },

            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::CounterClockWise,

            texture_inputs: &[TextureInput {
                name: "texSampler".to_owned(),
                format: gpu::ImageFormat::Rgba8,
                shader_stage: ShaderStage::FRAGMENT,
            }],
        })?;
        let default_material = resource_map.add(default_material);
        let default_material_transparency = resource_map.add(default_material_transparency);
        let default_sprite_material = resource_map.add(default_sprite_material);

        Ok(CommonResources {
            cube_mesh,
            quad_mesh,
            white_texture,
            black_texture,

            default_material,
            default_material_transparency,
            default_sprite_material,
        })
    }

    fn setup_resource_map(resource_map: &mut ResourceMap, gpu: Arc<dyn Gpu>) {
        resource_map.install_resource_loader(FileSystemTextureLoader { gpu: gpu.clone() });
    }
}

impl App for BevyEcsApp {
    fn window_name(&self, _app_state: &crate::app::app_state::AppState) -> Cow<str> {
        Cow::Borrowed("Bevy ECS app")
    }

    fn create(
        app_state: &mut crate::app::app_state::AppState,
        _event_loop: &winit::event_loop::EventLoop<()>,
        window: Window,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let startup_schedule = Schedule::new(StartupSchedule);
        let pre_update_schedule = Schedule::new(StartupSchedule);
        let update_schedule = Schedule::new(UpdateSchedule);
        let post_update_schedule = Schedule::new(PostUpdateSchedule);

        let mut world = World::new();
        let mut resource_map = ResourceMap::new(app_state.gpu.clone());

        Self::setup_resource_map(&mut resource_map, app_state.gpu.clone());

        let cvar_manager = CvarManager::new();
        let cube_mesh = utils::load_cube_to_resource_map(app_state.gpu(), &mut resource_map)?;
        let renderer = DeferredRenderingPipeline::new(
            app_state.gpu(),
            &mut resource_map,
            cube_mesh,
            DeferredRenderingPipeline::make_3d_combine_shader(app_state.gpu.as_ref())?,
        )?;

        let default_resources = Self::create_common_resources(app_state.gpu(), &mut resource_map)?;

        setup_world_resources(
            &mut world,
            resource_map,
            cvar_manager,
            default_resources,
            window,
            app_state,
        );

        Ok(Self {
            world,
            startup_schedule,
            pre_update_schedule,
            update_schedule,
            post_update_schedule,
            renderer,
            plugins: vec![],
            type_registry: AppTypeRegistry::default(),
        })
    }

    fn on_startup(
        &mut self,
        _app_state: &mut crate::app::app_state::AppState,
    ) -> anyhow::Result<()> {
        self.startup_schedule.run(&mut self.world);
        for plugin in &mut self.plugins {
            plugin.on_start(&mut self.world);
        }
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

        for plugin in &mut self.plugins {
            plugin.on_event(&mut self.world, event);
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
        self.world.get_resource_mut::<Time>().unwrap().begin_frame();

        Ok(())
    }

    fn update(&mut self, _app_state: &mut crate::app::app_state::AppState) -> anyhow::Result<()> {
        // self.imgui_console
        //     .update(self.world.get_resource::<InputState>().unwrap());
        // self.imgui_console.imgui_update(
        //     ui,
        //     &mut self.world.get_resource_mut::<CvarManager>().unwrap(),
        // );
        self.pre_update_schedule.run(&mut self.world);
        for plugin in &mut self.plugins {
            plugin.pre_update(&mut self.world);
        }

        self.update_schedule.run(&mut self.world);
        for plugin in &mut self.plugins {
            plugin.update(&mut self.world);
        }

        self.post_update_schedule.run(&mut self.world);
        for plugin in &mut self.plugins {
            plugin.post_update(&mut self.world);
        }
        Ok(())
    }

    fn end_frame(&mut self, _app_state: &crate::app::app_state::AppState) {
        self.world.get_resource_mut::<Time>().unwrap().end_frame();
        self.world
            .get_resource_mut::<InputState>()
            .unwrap()
            .end_frame();
    }
    fn draw<'a>(
        &'a mut self,
        app_state: &'a mut crate::app::app_state::AppState,
        backbuffer: &crate::Backbuffer,
    ) -> anyhow::Result<gpu::CommandBuffer> {
        let mut command_buffer = app_state
            .gpu
            .start_command_buffer(gpu::QueueType::Graphics)?;
        let empty_scene = RenderScene::default();
        let scene = self
            .world
            .get_resource::<RenderScene>()
            .unwrap_or(&empty_scene);
        let resource_map = self.world.get_resource::<ResourceMap>().unwrap();
        let cvar_manager = self.world.get_resource::<CvarManager>().unwrap();
        let pov = if let Some(pov) = self.world.get_resource::<Camera>() {
            *pov
        } else {
            Camera::default()
        };
        let render_final_color = self.renderer.render(
            app_state.gpu.as_ref(),
            &mut command_buffer,
            &pov,
            scene,
            resource_map,
            cvar_manager,
        )?;

        self.renderer.draw_textured_quad(
            &mut command_buffer,
            &backbuffer.image_view,
            &render_final_color,
            Rect2D {
                offset: Offset2D::default(),
                extent: backbuffer.size,
            },
            true,
            None,
        )?;

        for plugin in &mut self.plugins {
            plugin.draw(&mut self.world, app_state, &mut command_buffer)
        }

        Ok(command_buffer)
    }

    fn on_shutdown(&mut self, app_state: &mut AppState) {
        todo!()
    }
}

fn setup_world_resources(
    world: &mut World,
    resource_map: ResourceMap,
    cvar_manager: CvarManager,
    default_resources: CommonResources,
    window: Window,
    app_state: &mut AppState,
) {
    world.insert_resource(resource_map);
    world.insert_resource(cvar_manager);
    world.insert_resource(default_resources);
    world.insert_resource(InputState::new());
    world.insert_resource(EngineWindow(window));
    world.insert_resource(Time::new());
    world.insert_resource(GpuDevice(app_state.gpu.clone()));
}

impl BevyEcsAppWithLoop {
    pub fn run(self) -> anyhow::Result<()> {
        crate::app::run(self.app, self.evt_loop, self.state)
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
