mod fps_camera;
mod gltf_loader;
mod utils;

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroU32;

use engine::app::egui_support::EguiSupport;
use engine::app::{app_state::*, bootstrap, App, Console};
use engine::editor::ui_extension::UiExtension;
use engine::{egui, Light, LightType, ShadowConfiguration, Time};

use engine::input::InputState;
use engine::post_process_pass::TonemapPass;
use engine_macros::glsl;
use fps_camera::FpsCamera;
use gpu::{
    CommandBuffer, Extent2D, ImageFormat, Offset2D, PipelineBarrierInfo, PipelineStageFlags,
    PresentMode, Rect2D, ShaderModuleCreateInfo, ShaderModuleHandle, ShaderStage,
};
use winit::{
    dpi::{PhysicalPosition, Position},
    window::Window,
};

use crate::gltf_loader::{GltfLoadOptions, GltfLoader};
use engine::input::key::Key;
use engine::{
    post_process_pass::FxaaPass, Backbuffer, CvarManager, DeferredRenderingPipeline, LightHandle,
    MaterialInstance, RenderingPipeline, ResourceMap, TextureInput,
};
use nalgebra::*;
use winit::event::MouseButton;
use winit::event_loop::EventLoop;

use clap::Parser;

const SHADOW_DRAW: &[u32] = glsl!(
    path = "src/shaders/depth_draw.frag",
    entry_point = "main",
    kind = fragment
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

pub struct GLTFViewer {
    camera: FpsCamera,
    scene_renderer: DeferredRenderingPipeline,
    gltf_loader: GltfLoader,
    camera_light: Option<LightHandle>,

    input: InputState,
    console: Console,
    cvar_manager: CvarManager,
    resource_map: ResourceMap,
    time: Time,
    window: Window,
    egui_support: EguiSupport,
    depth_draw: ShaderModuleHandle,
}

impl GLTFViewer {
    pub(crate) fn print_lights(&self) {
        for light in self.gltf_loader.scene().all_lights() {
            println!("{light:?}");
        }
    }
}

impl GLTFViewer {
    fn lights_ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Lighting settings", |ui| {
            ui.separator();

            ui.color_edit3(
                "Ambient light color",
                &mut self.scene_renderer.ambient_color.data.0[0],
            );
            ui.input_float(
                "Ambient light intensity",
                &mut self.scene_renderer.ambient_intensity,
            );

            self.gltf_loader
                .scene()
                .all_lights()
                .iter()
                .enumerate()
                .for_each(|(i, l)| {
                    let light_string = match l.ty {
                        LightType::Point => "Point light",
                        LightType::Directional { .. } => "Directional light",
                        LightType::Spotlight { .. } => "Spot light",
                        LightType::Rect { .. } => "Rect light",
                    };

                    let handle = LightHandle(i);

                    if ui
                        .selectable_label(
                            self.camera_light.is_some_and(|l| l == handle),
                            format!("{light_string} #{}", i),
                        )
                        .clicked()
                    {
                        self.camera_light = Some(handle);
                    }
                });

            if let Some(cl) = self.camera_light {
                let l = self.gltf_loader.scene_mut().edit_light(&cl);
                ui.indent(
                    "cameralight
                ",
                    |ui| {
                        ui.checkbox(&mut l.enabled, "Enabled");
                        ui.input_floats("Position", &mut l.position.coords.data.0[0]);
                        let mut degrees_rot = l.direction().map(|v| v.to_degrees());
                        if ui.input_floats("Rotation", &mut degrees_rot.data.0[0]) {
                            let rad_rot = degrees_rot.map(|v| v.to_radians());
                            l.set_direction(rad_rot);
                        }

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
                            LightType::Directional { direction, size } => {
                                ui.input_floats("Direction", &mut direction.data.0[0]);
                                ui.input_floats("Shadow size", &mut size.data.0[0]);
                            }
                            LightType::Spotlight {
                                direction,
                                inner_cone_degrees,
                                outer_cone_degrees,
                            } => {
                                ui.input_floats("Direction", &mut direction.data.0[0]);
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
                            LightType::Rect {
                                direction,
                                width,
                                height,
                            } => {
                                ui.input_floats("Direction", &mut direction.data.0[0]);
                                ui.slider("Width", 0.0, 100000.0, width);
                                ui.slider("Height", 0.0, 100000.0, height);
                            }
                        }
                    },
                );
            }
            ui.separator();

            if ui.button("Add new spotlight").clicked() {
                self.gltf_loader.scene_mut().add_light(Light {
                    ty: LightType::Spotlight {
                        direction: vector![0.454, -0.324, -0.830],
                        inner_cone_degrees: 15.0,
                        outer_cone_degrees: 35.0,
                    },
                    position: point![9.766, -0.215, 2.078],
                    radius: 100.0,
                    color: vector![1.0, 1.0, 1.0],
                    intensity: 10.0,
                    enabled: true,
                    shadow_configuration: Some(ShadowConfiguration {
                        importance: NonZeroU32::new(5).unwrap(),
                        ..Default::default()
                    }),
                });
            }
            if ui.button("Add new directional light").clicked() {
                self.gltf_loader.scene_mut().add_light(Light {
                    ty: LightType::Directional {
                        direction: vector![0.454, -0.324, -0.830],
                        size: vector![10.0, 10.0],
                    },
                    position: point![9.766, -0.215, 2.078],
                    radius: 100.0,
                    color: vector![1.0, 1.0, 1.0],
                    intensity: 10.0,
                    enabled: true,
                    shadow_configuration: Some(ShadowConfiguration {
                        importance: NonZeroU32::new(20).unwrap(),
                        depth_bias: 0.05,
                        depth_slope: 1.0,
                    }),
                });
            }
            if ui.button("Add new point light").clicked() {
                self.gltf_loader.scene_mut().add_light(Light {
                    ty: LightType::Point,
                    position: point![9.766, -0.215, 2.078],
                    radius: 100.0,
                    color: vector![1.0, 1.0, 1.0],
                    intensity: 10.0,
                    enabled: true,
                    shadow_configuration: Some(ShadowConfiguration {
                        importance: NonZeroU32::new(2).unwrap(),
                        ..Default::default()
                    }),
                });
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

    fn create(
        app_state: &mut AppState,
        _event_loop: &EventLoop<()>,
        window: Window,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let input = InputState::new();
        let console = Console::new();
        let mut cvar_manager = CvarManager::new();
        let args = GltfViewerArgs::parse();
        let time = Time::new();

        let mut resource_map = ResourceMap::new();
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
        let david_texture = resource_map.add(david_texture);
        let irradiance_map = resource_map.add(irradiance_map);

        let mut scene_renderer = DeferredRenderingPipeline::new(
            app_state.gpu(),
            &mut resource_map,
            cube_mesh,
            DeferredRenderingPipeline::make_3d_combine_shader(app_state.gpu.as_ref())?,
        )?;

        scene_renderer.add_post_process_pass(TonemapPass::new(app_state.gpu.as_ref())?);
        scene_renderer
            .add_post_process_pass(FxaaPass::new(app_state.gpu.as_ref(), &mut cvar_manager)?);

        scene_renderer.set_irradiance_texture(Some(irradiance_map));

        let skybox_material = scene_renderer.create_material(
            app_state.gpu(),
            engine::MaterialDescription {
                name: "skybox material",
                domain: engine::MaterialDomain::Surface,
                texture_inputs: &[TextureInput {
                    name: "Cubemap".to_owned(),
                    format: ImageFormat::Rgba8,
                    shader_stage: ShaderStage::ALL_GRAPHICS,
                }],
                material_parameters: HashMap::new(),
                fragment_module: skybox_fragment,
                vertex_module,
                parameter_shader_visibility: ShaderStage::ALL_GRAPHICS,
                cull_mode: gpu::CullMode::None,
            },
        )?;
        let skybox_master = resource_map.add(skybox_material);

        let skybox_textures = vec![david_texture];

        let skybox_instance = MaterialInstance::create_instance(
            skybox_master,
            &engine::MaterialInstanceDescription {
                name: "david skybox",
                textures: skybox_textures,
                ..Default::default()
            },
        )?;

        let mut gltf_loader = GltfLoader::load(
            &args.gltf_file,
            app_state.gpu(),
            &mut scene_renderer,
            &mut resource_map,
            GltfLoadOptions {
                use_bvh: !args.no_use_bvh,
            },
        )?;

        gltf_loader
            .scene_mut()
            .set_skybox_material(Some(skybox_instance));

        app_state_mut()
            .swapchain_mut()
            .select_present_mode(PresentMode::Immediate)?;

        let egui_support = EguiSupport::new(&window, &app_state.gpu, &app_state.swapchain)?;

        let depth_draw = app_state.gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(SHADOW_DRAW),
        })?;

        gltf_loader.scene_mut().add_light(Light {
            ty: LightType::Directional {
                direction: vector![0.45, -0.45, 0.0],
                size: vector![50.0, 50.0],
            },
            position: Default::default(),
            radius: 1000.0,
            color: vector![1.0, 1.0, 1.0],

            intensity: 10.0,
            enabled: true,
            shadow_configuration: Some(ShadowConfiguration {
                importance: NonZeroU32::new(10).unwrap(),
                depth_bias: 0.05,
                depth_slope: 1.0,
            }),
        });

        Ok(Self {
            scene_renderer,
            gltf_loader,
            camera: FpsCamera::default(),
            camera_light: None,
            console,
            input,
            cvar_manager,
            resource_map,
            time,
            window,
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
        _app_state: &AppState,
    ) -> anyhow::Result<()> {
        self.input.update(event);
        if let winit::event::Event::WindowEvent { event, .. } = event {
            let _ = self.egui_support.handle_event(event);
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

    fn begin_frame(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        self.time.begin_frame();
        self.egui_support.begin_frame(&self.window);
        Ok(())
    }

    fn update(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        let fps = 1.0 / self.time.delta_frame();
        let win_name = format!("GLTF Viewer : {} FPS", fps);
        self.window.set_title(&win_name);
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

                    ui.checkbox(
                        &mut self.gltf_loader.scene_mut().use_frustum_culling,
                        "Use frustum culling",
                    );

                    ui.checkbox(
                        &mut self.gltf_loader.scene_mut().use_bvh,
                        "Use BVH for frustum culling",
                    );

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
                    ui.slider("CSM Splits", 0, 6, &mut self.scene_renderer.csm_slices);
                    ui.slider(
                        "CSM Split lambda",
                        0.0,
                        1.0,
                        &mut self.scene_renderer.csm_split_lambda,
                    );
                    ui.slider(
                        "CSM Z Multiplier",
                        0.1,
                        20.0,
                        &mut self.scene_renderer.z_mult,
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
            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)?;
            self.window.set_cursor_visible(false);
        }
        if self.input.is_mouse_button_just_released(MouseButton::Right) {
            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::None)?;
            self.window.set_cursor_visible(true);
        }

        if self
            .input
            .is_mouse_button_pressed(winit::event::MouseButton::Right)
        {
            self.camera.update(&self.input, self.time.delta_frame());
            let window_size = self.window.inner_size();

            self.window
                .set_cursor_position(Position::Physical(PhysicalPosition {
                    x: window_size.width as i32 / 2,
                    y: window_size.height as i32 / 2,
                }))?;
        }

        if self.input.is_mouse_button_pressed(MouseButton::Left) {
            if let Some(camera_light) = self.camera_light {
                self.gltf_loader
                    .scene_mut()
                    .edit_light(&camera_light)
                    .position = self.camera.location;
                self.gltf_loader
                    .scene_mut()
                    .edit_light(&camera_light)
                    .set_direction(self.camera.forward());
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
        app_state: &'a AppState,
        backbuffer: &Backbuffer,
    ) -> anyhow::Result<CommandBuffer> {
        let final_render = self.scene_renderer.render(
            app_state.gpu(),
            &self.camera.camera(),
            self.gltf_loader.scene(),
            &self.resource_map,
            &self.cvar_manager,
        )?;
        // write_image_to_swapchain(backbuffer);
        let output = self.egui_support.end_frame(&self.window);
        let mut command_buffer = app_state
            .gpu
            .start_command_buffer(gpu::QueueType::Graphics)?;

        command_buffer.pipeline_barrier(&PipelineBarrierInfo {
            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
            ..Default::default()
        });

        let shadow_texture = self.scene_renderer.get_shadow_texture(app_state.gpu());

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

        self.scene_renderer.draw_textured_quad(
            &mut command_buffer,
            &backbuffer.image_view,
            &shadow_texture,
            Rect2D {
                offset: Offset2D::default(),
                extent: Extent2D {
                    width: backbuffer.size.width / 4,
                    height: backbuffer.size.height / 4,
                },
            },
            true,
            None,
        )?;
        self.egui_support
            .paint_frame(output, &app_state.swapchain, &command_buffer);

        Ok(command_buffer)
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
