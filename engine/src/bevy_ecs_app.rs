use std::ops::{Deref, DerefMut};

use bevy_ecs::{
    schedule::{Schedule, ScheduleLabel},
    system::Resource,
    world::World,
};
use engine_macros::glsl;
use gpu::{Gpu, ShaderModuleCreateInfo, ShaderStage};
use nalgebra::vector;
use winit::event_loop::EventLoop;

use crate::{
    app::{App, ImguiConsole},
    input::InputState,
    loaders::FileSystemTextureLoader,
    utils, Camera, CvarManager, DeferredRenderingPipeline, MasterMaterial, Mesh, MeshCreateInfo,
    MeshPrimitiveCreateInfo, RenderingPipeline, ResourceHandle, ResourceMap, Scene, Texture,
    TextureInput,
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

const DEFAULT_DEFERRED_VS: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/default_vertex_deferred.vert",
    entry_point = "main"
);

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

    pub default_material: ResourceHandle<MasterMaterial>,
    pub default_material_transparency: ResourceHandle<MasterMaterial>,
}

impl BevyEcsApp {
    pub fn new() -> anyhow::Result<BevyEcsAppWithLoop> {
        let (app, evt_loop) = crate::app::create_app::<Self>()?;

        Ok(BevyEcsAppWithLoop { app, evt_loop })
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn renderer(&mut self) -> &mut DeferredRenderingPipeline {
        &mut self.renderer
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
            code: bytemuck::cast_slice(DEFAULT_DEFERRED_VS),
        })?;
        let fragment_module = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(DEFAULT_DEFERRED_FS),
        })?;
        let fragment_module_transparency = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(DEFAULT_DEFERRED_TRANSPARENCY_FS),
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
                    module: vertex_module,
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

        let default_material = resource_map.add(default_material);
        let default_material_transparency = resource_map.add(default_material_transparency);

        Ok(CommonResources {
            cube_mesh,
            quad_mesh,
            white_texture,
            black_texture,

            default_material,
            default_material_transparency,
        })
    }

    fn setup_resource_map(resource_map: &mut ResourceMap) {
        resource_map.install_resource_loader(FileSystemTextureLoader);
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

        Self::setup_resource_map(&mut resource_map);

        let cvar_manager = CvarManager::new();
        let (imgui_console, console_writer) = ImguiConsole::new_with_writer();
        let input = InputState::new();
        let cube_mesh = utils::load_cube_to_resource_map(&app_state.gpu, &mut resource_map)?;
        let renderer = DeferredRenderingPipeline::new(
            &app_state.gpu,
            &mut resource_map,
            cube_mesh,
            DeferredRenderingPipeline::make_3d_combine_shader(&app_state.gpu)?,
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
