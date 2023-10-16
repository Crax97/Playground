mod app;
mod fps_camera;
mod gltf_loader;
mod input;
mod utils;

use std::collections::HashMap;

use std::num::NonZeroU32;

use app::{bootstrap, App};

use fps_camera::FpsCamera;
use gpu::{ImageFormat, PresentMode, VkCommandBuffer};
use imgui::{TreeNodeFlags, Ui};
use input::InputState;
use winit::dpi::{PhysicalPosition, Position};

use crate::gltf_loader::{GltfLoadOptions, GltfLoader};
use crate::input::key::Key;
use engine::{
    AppState, Backbuffer, DeferredRenderingPipeline, Light, LightHandle, LightType,
    MaterialInstance, RenderingPipeline, ShadowSetup, TextureInput,
};
use nalgebra::*;
use resource_map::ResourceMap;
use winit::event::MouseButton;
use winit::event_loop::EventLoop;

use clap::Parser;

#[derive(Parser)]
pub struct GltfViewerArgs {
    #[arg(value_name = "FILE")]
    gltf_file: String,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}

pub struct GLTFViewer {
    resource_map: ResourceMap,
    camera: FpsCamera,
    scene_renderer: DeferredRenderingPipeline,
    gltf_loader: GltfLoader,
    input: InputState,
    camera_light: Option<LightHandle>,
}

impl GLTFViewer {
    pub(crate) fn print_lights(&self) {
        for light in self.gltf_loader.scene().all_lights() {
            println!("{light:?}");
        }
    }
}

impl GLTFViewer {
    fn lights_ui(&mut self, ui: &mut Ui) {
        if ui.collapsing_header("Lighting settings", TreeNodeFlags::DEFAULT_OPEN) {
            ui.separator();

            ui.color_edit3(
                "Ambient light color",
                &mut self.scene_renderer.ambient_color.data.0[0],
            );
            ui.input_float(
                "Ambient light intensity",
                &mut self.scene_renderer.ambient_intensity,
            )
            .build();

            let group = ui.begin_group();
            self.gltf_loader
                .scene()
                .all_lights()
                .iter()
                .enumerate()
                .for_each(|(i, l)| {
                    let light_string = match l.ty {
                        LightType::Point => "Point",
                        LightType::Directional { .. } => "Directional",
                        LightType::Spotlight { .. } => "Spotlight",
                        LightType::Rect { .. } => "Rect",
                    };

                    let handle = LightHandle(i);

                    if ui
                        .selectable_config(&format!("{light_string} light nr. #{i}"))
                        .selected(self.camera_light.is_some_and(|l| l == handle))
                        .build()
                    {
                        self.camera_light = Some(handle);
                    }
                });

            if let Some(cl) = self.camera_light {
                let l = self.gltf_loader.scene_mut().edit_light(&cl);
                ui.indent();
                ui.checkbox("Enabled", &mut l.enabled);
                ui.input_float3("Position", &mut l.position.coords.data.0[0])
                    .build();
                let mut degrees_rot = l.direction().map(|v| v.to_degrees());
                if ui
                    .input_float3("Rotation", &mut degrees_rot.data.0[0])
                    .build()
                {
                    let rad_rot = degrees_rot.map(|v| v.to_radians());
                    l.set_direction(rad_rot);
                }

                ui.color_edit3("Color", &mut l.color.data.0[0]);
                ui.slider("Intensity", 0.0, 1000.0, &mut l.intensity);
                ui.slider("Radius", 0.0, 1000.0, &mut l.radius);
                match &mut l.ty {
                    LightType::Point => {}
                    LightType::Directional { direction, size } => {
                        ui.input_float3("Direction", &mut direction.data.0[0])
                            .build();
                        ui.input_float2("Shadow size", &mut size.data.0[0]).build();
                    }
                    LightType::Spotlight {
                        direction,
                        inner_cone_degrees,
                        outer_cone_degrees,
                    } => {
                        ui.input_float3("Direction", &mut direction.data.0[0])
                            .build();
                        ui.slider("Outer cone", *inner_cone_degrees, 45.0, outer_cone_degrees);
                        ui.slider("Inner cone", 0.0, *outer_cone_degrees, inner_cone_degrees);
                    }
                    LightType::Rect {
                        direction,
                        width,
                        height,
                    } => {
                        ui.input_float3("Direction", &mut direction.data.0[0])
                            .build();
                        ui.slider("Width", 0.0, 100000.0, width);
                        ui.slider("Height", 0.0, 100000.0, height);
                    }
                }
                ui.unindent();
            }
            ui.separator();

            if ui.button("Add new spotlight") {
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
                    shadow_setup: Some(ShadowSetup {
                        importance: NonZeroU32::new(1).unwrap(),
                    }),
                });
            }
            if ui.button("Add new directional light") {
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
                    shadow_setup: Some(ShadowSetup {
                        importance: NonZeroU32::new(1).unwrap(),
                    }),
                });
            }
            if ui.button("Add new point light") {
                self.gltf_loader.scene_mut().add_light(Light {
                    ty: LightType::Point,
                    position: point![9.766, -0.215, 2.078],
                    radius: 100.0,
                    color: vector![1.0, 1.0, 1.0],
                    intensity: 10.0,
                    enabled: true,
                    shadow_setup: Some(ShadowSetup {
                        importance: NonZeroU32::new(1).unwrap(),
                    }),
                });
            }

            group.end();
        }
    }
}

impl App for GLTFViewer {
    fn window_name(&self, app_state: &AppState) -> String {
        format!("GLTF Viewer - FPS {}", 1.0 / app_state.time().delta_frame())
    }

    fn create(app_state: &AppState, _event_loop: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let args = GltfViewerArgs::parse();

        let mut resource_map = ResourceMap::new();

        // TODO: avoid duplicating this module creation
        let vertex_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/vertex_deferred.spirv")?;
        let skybox_fragment =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/skybox_spherical.spirv")?;

        let screen_quad_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/screen_quad.spirv")?;
        let gbuffer_combine_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/gbuffer_combine.spirv")?;
        let texture_copy_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/texture_copy.spirv")?;
        let tonemap_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/tonemap.spirv")?;

        let david_texture = utils::load_hdr_to_cubemap(
            &app_state.gpu,
            &mut resource_map,
            "images/skybox/hdr/sunflowers.hdr",
        )?;
        let david_texture = resource_map.add(david_texture);

        let cube_mesh = utils::load_cube_to_resource_map(&app_state.gpu, &mut resource_map)?;

        let mut scene_renderer = DeferredRenderingPipeline::new(
            &app_state.gpu,
            screen_quad_module,
            gbuffer_combine_module,
            texture_copy_module,
            tonemap_module,
            cube_mesh,
        )?;

        let skybox_material = scene_renderer.create_material(
            &app_state.gpu,
            engine::MaterialDescription {
                name: "skybox material",
                domain: engine::MaterialDomain::Surface,
                texture_inputs: &[TextureInput {
                    name: "Cubemap".to_owned(),
                    format: ImageFormat::Rgba8,
                }],
                material_parameters: HashMap::new(),
                fragment_module: &skybox_fragment,
                vertex_module: &vertex_module,
            },
        )?;
        let skybox_master = resource_map.add(skybox_material);

        let mut skybox_textures = HashMap::new();
        skybox_textures.insert("Cubemap".to_string(), david_texture);

        let skybox_instance = MaterialInstance::create_instance(
            &app_state.gpu,
            skybox_master,
            &resource_map,
            &engine::MaterialInstanceDescription {
                name: "david skybox",
                texture_inputs: skybox_textures,
            },
        )?;
        let skybox_instance = resource_map.add(skybox_instance);

        let mut gltf_loader = GltfLoader::load(
            &args.gltf_file,
            &app_state.gpu,
            &mut scene_renderer,
            &mut resource_map,
            GltfLoadOptions {},
        )?;

        gltf_loader
            .scene_mut()
            .set_skybox_material(Some(skybox_instance));

        engine::app_state_mut()
            .swapchain_mut()
            .select_present_mode(PresentMode::Immediate)?;

        Ok(Self {
            resource_map,
            scene_renderer,
            gltf_loader,
            input: InputState::new(),
            camera: FpsCamera::default(),
            camera_light: None,
        })
    }

    fn on_event(
        &mut self,
        event: &winit::event::Event<()>,
        _app_state: &AppState,
    ) -> anyhow::Result<()> {
        self.input.update(&event);
        Ok(())
    }

    fn input(
        &mut self,
        _app_state: &AppState,
        _event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn update(&mut self, app_state: &mut AppState, ui: &mut Ui) -> anyhow::Result<()> {
        let mut settings = self.scene_renderer.fxaa_settings();

        ui.text("Hiii");

        ui.slider("FXAA iterations", 0, 12, &mut settings.iterations);
        ui.slider("FXAA subpix", 0.0, 1.0, &mut settings.fxaa_quality_subpix);
        ui.slider(
            "FXAA Edge Threshold",
            0.0,
            1.0,
            &mut settings.fxaa_quality_edge_threshold,
        );
        ui.slider(
            "FXAA Edge Threshold min",
            0.0,
            1.0,
            &mut settings.fxaa_quality_edge_threshold_min,
        );

        ui.separator();

        if ui.collapsing_header("Shadow settings", TreeNodeFlags::DEFAULT_OPEN) {
            ui.slider(
                "Depth Bias constant",
                -10.0,
                10.0,
                &mut self.scene_renderer.depth_bias_constant,
            );
            ui.slider(
                "Depth Bias slope",
                -10.0,
                10.0,
                &mut self.scene_renderer.depth_bias_slope,
            );
        }

        self.lights_ui(ui);

        self.scene_renderer.set_fxaa_settings_mut(settings);

        if ui.io().want_capture_keyboard || ui.io().want_capture_mouse {
            return Ok(());
        }

        if self.input.is_key_just_pressed(Key::P) {
            self.print_lights();
        }

        if self.input.is_mouse_button_just_pressed(MouseButton::Right) {
            app_state
                .window()
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)?;
            app_state.window().set_cursor_visible(false);
        }
        if self.input.is_mouse_button_just_released(MouseButton::Right) {
            app_state
                .window()
                .set_cursor_grab(winit::window::CursorGrabMode::None)?;
            app_state.window().set_cursor_visible(true);
        }

        if self
            .input
            .is_mouse_button_pressed(winit::event::MouseButton::Right)
        {
            self.camera
                .update(&self.input, app_state.time.delta_frame());
            let window_size = app_state.window().inner_size();

            app_state
                .window()
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

        self.input.end_frame();
        Ok(())
    }

    fn draw(&mut self, backbuffer: &Backbuffer) -> anyhow::Result<VkCommandBuffer> {
        let command_buffer = self.scene_renderer.render(
            &self.camera.camera(),
            self.gltf_loader.scene(),
            backbuffer,
            &self.resource_map,
        )?;
        Ok(command_buffer)
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
