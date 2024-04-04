mod fps_camera;
mod gltf_loader;
mod utils;

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use engine::app::egui_support::EguiSupport;
use engine::app::{app_state::*, bootstrap, App, Console};
use engine::components::Transform;
use engine::editor::ui_extension::UiExtension;
use engine::material_v2::{MaterialBuilder, Shader};
use engine::{
    egui, GameScene, LightType, SceneLightInfo, ScenePrimitiveType, ShadowConfiguration, Tick, Time,
};

use engine::input::InputState;
use engine::post_process_pass::TonemapPass;
use engine_macros::glsl;
use fps_camera::FpsCamera;
use gpu::{
    CommandBuffer, Extent2D, Offset2D, PresentMode, Rect2D, ShaderModuleCreateInfo,
    ShaderModuleHandle,
};
use winit::dpi::{PhysicalPosition, Position};

use crate::gltf_loader::{GltfLoadOptions, GltfLoader};
use engine::input::key::Key;
use engine::{
    post_process_pass::FxaaPass, AssetMap, Backbuffer, CvarManager, DeferredRenderingPipeline,
    PrimitiveHandle, RenderingPipeline,
};
use nalgebra::*;
use winit::event::MouseButton;
use winit::event_loop::EventLoop;

use clap::Parser;

const DEPTH_DRAW: &[u32] = glsl!(
    path = "src/shaders/depth_draw.frag",
    kind = fragment,
    entry_point = "main"
);

#[derive(Parser)]
pub struct GltfViewerArgs {
    #[arg(value_name = "FILE")]
    gltf_file: String,

    #[arg(long, default_value_t = false)]
    no_use_bvh: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthDrawConstants {
    near: f32,
    far: f32,
}

pub struct GLTFViewer {
    camera: FpsCamera,
    scene_renderer: DeferredRenderingPipeline,
    scene: GameScene,
    selected_primitive: Option<PrimitiveHandle>,

    input: InputState,
    console: Console,
    cvar_manager: CvarManager,
    resource_map: AssetMap,
    time: Time,
    egui_support: EguiSupport,

    depth_draw: ShaderModuleHandle,
}

impl GLTFViewer {
    pub(crate) fn print_lights(&self) {
        for (light, _) in self.scene.all_lights() {
            println!("{light:?}");
        }
    }
}

impl GLTFViewer {
    fn lights_ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Scene Setup", |ui| {
            ui.separator();

            ui.color_edit3(
                "Ambient light color",
                &mut self.scene_renderer.ambient_color.data.0[0],
            );
            ui.input_float(
                "Ambient light intensity",
                &mut self.scene_renderer.ambient_intensity,
            );

            self.scene.all_primitives().for_each(|(handle, prim)| {
                if ui
                    .selectable_label(
                        self.selected_primitive.is_some_and(|l| l == handle),
                        &prim.label,
                    )
                    .clicked()
                {
                    self.selected_primitive = Some(handle);
                }
            });

            if let Some(cl) = self.selected_primitive {
                let l = self.scene.get_mut(cl);
                let transform = &mut l.transform;
                let degrees_rot = transform.rotation.euler_angles();
                let mut degrees_rot =
                    [degrees_rot.0, degrees_rot.1, degrees_rot.2].map(|v| v.to_degrees());
                ui.collapsing("Transform", |ui| {
                    ui.input_floats("Position", transform.position.coords.as_mut_slice());

                    if ui.input_floats("Rotation", &mut degrees_rot) {
                        let rad_rot = degrees_rot.map(|v| v.to_radians());
                        let rot = nalgebra::UnitQuaternion::from_euler_angles(
                            rad_rot[0], rad_rot[1], rad_rot[2],
                        );
                        transform.rotation = rot;
                    }

                    ui.input_floats("Scale", transform.scale.as_mut_slice());
                });
                if let ScenePrimitiveType::Light(l) = &mut l.ty {
                    ui.indent(
                        "cameralight
                ",
                        |ui| {
                            ui.checkbox(&mut l.enabled, "Enabled");

                            ui.color_edit3("Color", &mut l.color.data.0[0]);
                            ui.slider("Intensity", 0.0, 1000.0, &mut l.intensity);
                            ui.slider("Radius", 0.0, 1000.0, &mut l.radius);

                            if let Some(setup) = l.shadow_configuration.as_mut() {
                                ui.separator();
                                ui.slider("Depth Bias", -10.0, 10.0, &mut setup.depth_bias);
                                ui.slider("Depth Slope", -10.0, 10.0, &mut setup.depth_slope);
                            }
                            match &mut l.ty {
                                LightType::Point => {}
                                LightType::Directional { size } => {
                                    ui.input_floats("Shadow size", &mut size.data.0[0]);
                                }
                                LightType::Spotlight {
                                    inner_cone_degrees,
                                    outer_cone_degrees,
                                } => {
                                    ui.slider(
                                        "Outer cone",
                                        *inner_cone_degrees,
                                        45.0,
                                        outer_cone_degrees,
                                    );
                                    ui.slider(
                                        "Inner cone",
                                        0.0,
                                        *outer_cone_degrees,
                                        inner_cone_degrees,
                                    );
                                }
                                LightType::Rect { width, height } => {
                                    ui.slider("Width", 0.0, 100000.0, width);
                                    ui.slider("Height", 0.0, 100000.0, height);
                                }
                            }
                        },
                    );
                }
            }
            ui.separator();

            if ui.button("Add new spotlight").clicked() {
                self.scene.add_light(
                    SceneLightInfo {
                        ty: LightType::Spotlight {
                            inner_cone_degrees: 15.0,
                            outer_cone_degrees: 35.0,
                        },
                        radius: 100.0,
                        color: vector![1.0, 1.0, 1.0],
                        intensity: 10.0,
                        enabled: true,
                        shadow_configuration: Some(ShadowConfiguration {
                            shadow_map_width: 512,
                            shadow_map_height: 512,
                            ..Default::default()
                        }),
                    },
                    Transform::default(),
                    None,
                );
            }
            if ui.button("Add new directional light").clicked() {
                self.scene.add_light(
                    SceneLightInfo {
                        ty: LightType::Directional {
                            size: vector![10.0, 10.0],
                        },
                        radius: 100.0,
                        color: vector![1.0, 1.0, 1.0],
                        intensity: 10.0,
                        enabled: true,
                        shadow_configuration: Some(ShadowConfiguration {
                            shadow_map_width: 2048,
                            shadow_map_height: 2048,
                            depth_bias: 0.05,
                            depth_slope: 1.0,
                        }),
                    },
                    Transform::default(),
                    None,
                );
            }
            if ui.button("Add new point light").clicked() {
                self.scene.add_light(
                    SceneLightInfo {
                        ty: LightType::Point,
                        radius: 100.0,
                        color: vector![1.0, 1.0, 1.0],
                        intensity: 10.0,
                        enabled: true,
                        shadow_configuration: Some(ShadowConfiguration {
                            shadow_map_width: 512,
                            shadow_map_height: 512,
                            ..Default::default()
                        }),
                    },
                    Transform::default(),
                    None,
                );
            }
        });
    }
}

impl App for GLTFViewer {
    fn window_name(&self, _app_state: &AppState) -> Cow<str> {
        Cow::Owned(format!(
            "GLTF Viewer : {} FPS",
            1.0 / self.time.delta_frame()
        ))
    }

    fn create(app_state: &mut AppState, _event_loop: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let input = InputState::new();
        let console = Console::new();
        let mut cvar_manager = CvarManager::new();
        let args = GltfViewerArgs::parse();
        let time = Time::new();

        let mut resource_map = AssetMap::new(app_state.gpu.clone(), true);
        let cube_mesh = utils::load_cube_to_resource_map(app_state.gpu(), &mut resource_map)?;

        // TODO: avoid duplicating this module creation
        let vertex_module =
            utils::read_file_to_vk_module(app_state.gpu(), "./shaders/vertex_deferred.spirv")?;
        let skybox_fragment =
            utils::read_file_to_vk_module(app_state.gpu(), "./shaders/skybox_master.spirv")?;

        let david_texture = utils::load_hdr_to_cubemap(
            app_state.gpu.as_ref(),
            Extent2D {
                width: 1024,
                height: 1024,
            },
            cube_mesh.clone(),
            &mut resource_map,
            "images/skybox/hdr/evening_road.hdr",
        )?;
        let irradiance_map = utils::generate_irradiance_map(
            app_state.gpu(),
            &david_texture,
            &mut resource_map,
            &cube_mesh,
        )?;
        let david_texture = resource_map.add(david_texture, Some("David texture"));
        let irradiance_map = resource_map.add(irradiance_map, Some("Irradiance map"));

        let mut scene_renderer = DeferredRenderingPipeline::new(
            app_state.gpu(),
            DeferredRenderingPipeline::make_3d_combine_shader(app_state.gpu.as_ref())?,
        )?;

        scene_renderer.add_post_process_pass(TonemapPass::new(app_state.gpu.as_ref())?);
        scene_renderer
            .add_post_process_pass(FxaaPass::new(app_state.gpu.as_ref(), &mut cvar_manager)?);

        scene_renderer.set_irradiance_texture(Some(irradiance_map));
        let vertex_shader = resource_map.add(
            Shader {
                name: "Skybox VS".into(),
                handle: vertex_module,
            },
            Some("Skybox VS"),
        );
        let fragment_shader = resource_map.add(
            Shader {
                name: "Skybox FS".into(),
                handle: skybox_fragment,
            },
            Some("Skybox FS"),
        );
        let skybox_instance = MaterialBuilder::new(
            vertex_shader,
            fragment_shader,
            engine::MaterialDomain::Surface,
        )
        .parameter(
            "baseColorSampler",
            engine::material_v2::MaterialParameter::Texture(david_texture),
        )
        .build();
        let skybox_instance = resource_map.add(skybox_instance, Some("Skybox Material"));

        let mut scene = GltfLoader::load(
            &args.gltf_file,
            app_state.gpu(),
            &mut scene_renderer,
            &mut resource_map,
            GltfLoadOptions {
                use_bvh: !args.no_use_bvh,
            },
        )?;

        scene.set_skybox_material(Some(skybox_instance));

        app_state
            .swapchain_mut()
            .select_present_mode(PresentMode::Immediate)?;

        let egui_support = EguiSupport::new(&app_state.window, app_state.gpu())?;

        scene.add_light(
            SceneLightInfo {
                ty: LightType::Directional {
                    size: vector![50.0, 50.0],
                },
                radius: 1.0,
                color: vector![1.0, 1.0, 1.0],

                intensity: 10.0,
                enabled: true,
                shadow_configuration: Some(ShadowConfiguration {
                    shadow_map_width: 2048,
                    shadow_map_height: 2048,
                    depth_bias: 0.05,
                    depth_slope: 1.0,
                }),
            },
            Default::default(),
            None,
        );

        let depth_draw = app_state.gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Depth draw"),
            code: bytemuck::cast_slice(DEPTH_DRAW),
        })?;
        let mut camera = FpsCamera::default();
        camera.location = point![-292.7, 136.0, -216.7];
        camera.roll = 19.0;
        camera.pitch = 61.0;

        Ok(Self {
            scene_renderer,
            scene,
            camera,
            selected_primitive: None,
            console,
            input,
            cvar_manager,
            resource_map,
            time,
            egui_support,
            depth_draw,
        })
    }

    fn input(
        &mut self,
        _app_state: &AppState,
        _event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_event(
        &mut self,
        event: &winit::event::Event<()>,
        app_state: &AppState,
    ) -> anyhow::Result<()> {
        self.input.update(event);
        if let winit::event::Event::WindowEvent { event, .. } = event {
            let _ = self
                .egui_support
                .handle_window_event(&app_state.window, event);
        }
        Ok(())
    }

    fn on_resized(&mut self, app_state: &AppState, size: winit::dpi::PhysicalSize<u32>) {
        self.egui_support.swapchain_updated(&app_state.swapchain);
        self.scene_renderer.on_resolution_changed(Extent2D {
            width: size.width,
            height: size.height,
        })
    }

    fn begin_frame(&mut self, app_state: &mut AppState) -> anyhow::Result<()> {
        self.time.begin_frame();
        self.egui_support.begin_frame(&app_state.window, &self.time);
        Ok(())
    }

    fn update(&mut self, app_state: &mut AppState) -> anyhow::Result<()> {
        let fps = 1.0 / self.time.delta_frame();
        let win_name = format!("GLTF Viewer : {} FPS", fps);
        app_state.window.set_title(&win_name);
        self.console.update(&self.input);
        self.egui_support
            .paint_console(&mut self.console, &mut self.cvar_manager);

        let context = self.egui_support.create_context();
        let mut early_return = false;
        egui::Window::new("GLTF Viewer").show(&context, |ui| {
            egui::ScrollArea::new([false, true]).show(ui, |ui| {
                ui.group(|ui| {
                    ui.heading("Stats");
                    ui.label(format!("FPS {}", fps));
                    ui.label(format!("Delta time {}", self.time.delta_frame()));
                    ui.label(format!(
                        "Primitives drawn last frame: {}",
                        self.scene_renderer.drawcalls_last_frame,
                    ));
                });

                ui.group(|ui| {
                    ui.heading("Camera");
                    ui.input_floats(
                        "Camera location",
                        self.camera.location.coords.as_mut_slice(),
                    );
                    let (rx, ry, rz) = self.camera.rotation.euler_angles();
                    let mut rotation = vec![rx.to_degrees(), ry.to_degrees(), rz.to_degrees()];
                    if ui.input_floats("Camera rotation", &mut rotation) {
                        let (roll, pitch, yaw) = (rotation[0], rotation[1], rotation[2]);
                        let rotation = Rotation3::from_euler_angles(
                            roll.to_radians(),
                            pitch.to_radians(),
                            yaw.to_radians(),
                        );
                        self.camera.rotation = rotation;
                    }
                    let mut fwd = self.camera.forward();
                    ui.input_floats("Camera forward", fwd.data.as_mut_slice());

                    ui.input_float("Camera speed", &mut self.camera.speed);
                    ui.input_float("Camera rotation speed", &mut self.camera.rotation_speed);
                    ui.input_float("Camera FOV", &mut self.camera.fov_degrees);

                    ui.checkbox(&mut self.scene.use_frustum_culling, "Use frustum culling");

                    ui.checkbox(&mut self.scene.use_bvh, "Use BVH for frustum culling");

                    if ui.button("Reset camera").clicked() {
                        self.camera.location = Default::default();
                        self.camera.rotation = Default::default();
                    }

                    ui.checkbox(&mut self.scene_renderer.update_frustum, "Update frustum");
                });

                ui.separator();

                ui.slider(
                    "FXAA iterations",
                    0,
                    12,
                    self.cvar_manager
                        .get_named_ref_mut::<i32>(FxaaPass::FXAA_ITERATIONS_CVAR_NAME)
                        .unwrap(),
                );
                ui.slider(
                    "FXAA subpix",
                    0.0,
                    1.0,
                    self.cvar_manager
                        .get_named_ref_mut::<f32>(FxaaPass::FXAA_SUBPIX_CVAR_NAME)
                        .unwrap(),
                );
                ui.slider(
                    "FXAA Edge Threshold",
                    0.0,
                    1.0,
                    self.cvar_manager
                        .get_named_ref_mut::<f32>(FxaaPass::FXAA_EDGE_THRESHOLD_CVAR_NAME)
                        .unwrap(),
                );
                ui.slider(
                    "FXAA Edge Threshold min",
                    0.0,
                    1.0,
                    self.cvar_manager
                        .get_named_ref_mut::<f32>(FxaaPass::FXAA_EDGE_THRESHOLD_MIN_CVAR_NAME)
                        .unwrap(),
                );

                ui.separator();

                ui.collapsing("Global shadow settings", |ui| {
                    ui.slider(
                        "CSM Splits",
                        0,
                        4,
                        &mut self.scene_renderer.cascaded_shadow_map.num_cascades,
                    );
                    ui.slider(
                        "CSM Split lambda",
                        0.0,
                        1.0,
                        &mut self.scene_renderer.cascaded_shadow_map.csm_split_lambda,
                    );
                    ui.slider(
                        "CSM Z Multiplier",
                        0.1,
                        20.0,
                        &mut self.scene_renderer.cascaded_shadow_map.z_mult,
                    );
                    ui.checkbox(
                        &mut self.scene_renderer.cascaded_shadow_map.stabilize_cascades,
                        "Stabilize cascades",
                    );
                    ui.checkbox(
                        &mut self.scene_renderer.cascaded_shadow_map.debug_csm_splits,
                        "Debug CSM Splits",
                    );
                    ui.checkbox(
                        &mut self.scene_renderer.cascaded_shadow_map.is_pcf_enabled,
                        "Enable Percentage Close Filtering",
                    );
                });

                self.lights_ui(ui);
                if ui.ui_contains_pointer() {
                    early_return = true;
                }
            });
        });

        if early_return {
            return Ok(());
        }
        if self.input.is_key_just_pressed(Key::P) {
            self.print_lights();
        }

        if self.input.is_mouse_button_just_pressed(MouseButton::Right) {
            app_state
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)?;
            app_state.window.set_cursor_visible(false);
        }
        if self.input.is_mouse_button_just_released(MouseButton::Right) {
            app_state
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::None)?;
            app_state.window.set_cursor_visible(true);
        }

        if self
            .input
            .is_mouse_button_pressed(winit::event::MouseButton::Right)
        {
            self.camera.update(&self.input, self.time.delta_frame());
            let window_size = app_state.window.inner_size();

            app_state
                .window
                .set_cursor_position(Position::Physical(PhysicalPosition {
                    x: window_size.width as i32 / 2,
                    y: window_size.height as i32 / 2,
                }))?;
        }

        if self.input.is_mouse_button_pressed(MouseButton::Left) {
            if let Some(camera_light) = self.selected_primitive {
                self.scene.get_mut(camera_light).transform.position = self.camera.location;
            }
        }

        Ok(())
    }

    fn end_frame(&mut self, _app_state: &AppState) {
        self.resource_map.update();
        self.input.end_frame();
        self.time.end_frame();
    }

    fn draw<'a>(
        &'a mut self,
        app_state: &'a mut AppState,
        backbuffer: &Backbuffer,
    ) -> anyhow::Result<CommandBuffer> {
        let mut command_buffer = app_state
            .gpu
            .start_command_buffer(gpu::QueueType::Graphics)?;
        let final_render = self.scene_renderer.render(
            app_state.gpu(),
            &mut command_buffer,
            &self.camera.camera(),
            &self.scene,
            &self.resource_map,
            &self.cvar_manager,
        )?;
        // write_image_to_swapchain(backbuffer);
        let output = self.egui_support.end_frame(&app_state.window);
        self.scene_renderer.draw_textured_quad(
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

        self.egui_support.paint_frame(
            app_state.gpu(),
            &mut command_buffer,
            backbuffer,
            output.textures_delta,
            output.shapes,
        )?;
        self.egui_support
            .handle_platform_output(&app_state.window, output.platform_output);

        Ok(command_buffer)
    }

    fn on_shutdown(&mut self, app_state: &mut AppState) {
        app_state.gpu().destroy_shader_module(self.depth_draw);
        self.scene.clean_resources(app_state.gpu());
        self.scene_renderer.destroy(app_state.gpu());
        self.egui_support.destroy(app_state.gpu());
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
