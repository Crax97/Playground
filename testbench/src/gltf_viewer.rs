mod app;
mod gltf_loader;
mod utils;

use app::{bootstrap, App};
use ash::vk::{
    AccessFlags, DependencyFlags, ImageAspectFlags, ImageSubresourceRange,
    PipelineStageFlags, PresentModeKHR,
};
use ash::vk::{ImageLayout, Rect2D};

use gpu::ColorAttachment;
use gpu::CommandBufferSubmitInfo;
use gpu::{BeginRenderPassInfo, ImageMemoryBarrier, PipelineBarrierInfo};
use imgui::*;
use imgui_rs_vulkan_renderer::{DynamicRendering as ImguiDynamicRendering, *};
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use crate::gltf_loader::{GltfLoadOptions, GltfLoader};
use engine::{AppState, Backbuffer, Camera, DeferredRenderingPipeline, FxaaSettings, Light, LightType, RenderingPipeline, Scene};
use nalgebra::*;
use resource_map::ResourceMap;
use winit::event::{ElementState, Event};
use winit::event::VirtualKeyCode;
use winit::event_loop::EventLoop;

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}
const SPEED: f32 = 0.01;
const ROTATION_SPEED: f32 = 3.0;
const MIN_DELTA: f32 = 1.0;

pub struct GLTFViewer {
    resource_map: ResourceMap,
    camera: Camera,
    forward_movement: f32,
    rotation_movement: f32,
    rot_x: f32,
    rot_y: f32,
    dist: f32,
    movement: Vector3<f32>,
    scene_renderer: DeferredRenderingPipeline,
    gltf_loader: GltfLoader,

    imgui: Context,
    platform: WinitPlatform,
    renderer: Renderer,
}

impl App for GLTFViewer {
    fn window_name(&self, app_state: &AppState) -> String {
        format!("GLTF Viewer - FPS {}", 1.0 / app_state.time().delta_frame())
    }

    fn create(app_state: &AppState, _event_loop: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut resource_map = ResourceMap::new();

        let camera = Camera {
            location: point![2.0, 2.0, 2.0],
            forward: vector![0.0, -1.0, -1.0].normalize(),
            near: 0.01,
            ..Default::default()
        };

        let forward_movement = 0.0;
        let rotation_movement = 0.0;

        let rot_x = 0.0;
        let rot_z = 0.0;
        let dist = 1.0;

        let movement: Vector3<f32> = vector![0.0, 0.0, 0.0];

        let screen_quad_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/screen_quad.spirv")?;
        let gbuffer_combine_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/gbuffer_combine.spirv")?;
        let texture_copy_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/texture_copy.spirv")?;
        let tonemap_module =
            utils::read_file_to_vk_module(&app_state.gpu, "./shaders/tonemap.spirv")?;

        let mut scene_renderer = DeferredRenderingPipeline::new(
            &app_state.gpu,
            screen_quad_module,
            gbuffer_combine_module,
            texture_copy_module,
            tonemap_module,
        )?;

        let mut gltf_loader = GltfLoader::load(
            "gltf_models/bottle/glTF/WaterBottle.gltf",
            &app_state.gpu,
            &mut scene_renderer,
            &mut resource_map,
            GltfLoadOptions {},
        )?;

        add_scene_lights(gltf_loader.scene_mut());

        engine::app_state_mut()
            .gpu
            .swapchain_mut()
            .select_present_mode(PresentModeKHR::IMMEDIATE)?;

        let mut imgui = Context::create();
        let mut platform = WinitPlatform::init(&mut imgui);
        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        }]);
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        platform.attach_window(
            imgui.io_mut(),
            &app_state.gpu.swapchain().window,
            HiDpiMode::Rounded,
        );
        let renderer = Renderer::with_default_allocator(
            &app_state.gpu.instance(),
            app_state.gpu.vk_physical_device(),
            app_state.gpu.vk_logical_device(),
            app_state.gpu.graphics_queue(),
            app_state.gpu.command_pool(),
            ImguiDynamicRendering {
                color_attachment_format: app_state.gpu.swapchain().present_format(),
                depth_attachment_format: None,
            },
            &mut imgui,
            Some(Options {
                in_flight_frames: 2,
                ..Default::default()
            }),
        )?;
        Ok(Self {
            resource_map,
            camera,
            forward_movement,
            rotation_movement,
            rot_x,
            rot_y: rot_z,
            dist,
            movement,
            scene_renderer,
            gltf_loader,
            imgui,
            renderer,
            platform,
        })
    }

    fn on_event(&mut self, event: &Event<()>, app_state: &AppState) -> anyhow::Result<()> {
        self.platform.handle_event(self.imgui.io_mut(), &app_state.gpu.swapchain().window, event);
        Ok(())
    }
    
    fn input(
        &mut self,
        _app_state: &AppState,
        event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()> {
        match event {
            winit::event::DeviceEvent::Button { button, state } => {
                let mul = if state == ElementState::Pressed {
                    1.0
                } else {
                    0.0
                };
                if button == 1 {
                    self.rotation_movement = mul;
                } else if button == 3 {
                    self.forward_movement = mul;
                }
            }
            winit::event::DeviceEvent::Key(input) => {
                if input.virtual_keycode.unwrap_or(VirtualKeyCode::A) == VirtualKeyCode::Key1 {
                    self.scene_renderer.set_fxaa_settings_mut(FxaaSettings {
                        fxaa_quality_subpix: 0.0,
                        fxaa_quality_edge_threshold: 0.333,
                        fxaa_quality_edge_threshold_min: 0.0833,
                    });
                } else if input.virtual_keycode.unwrap_or(VirtualKeyCode::A) == VirtualKeyCode::Key2
                {
                    self.scene_renderer.set_fxaa_settings_mut(FxaaSettings {
                        fxaa_quality_subpix: 0.25,
                        fxaa_quality_edge_threshold: 0.250,
                        fxaa_quality_edge_threshold_min: 0.0833,
                    });
                } else if input.virtual_keycode.unwrap_or(VirtualKeyCode::A) == VirtualKeyCode::Key3
                {
                    self.scene_renderer.set_fxaa_settings_mut(FxaaSettings {
                        fxaa_quality_subpix: 0.5,
                        fxaa_quality_edge_threshold: 0.166,
                        fxaa_quality_edge_threshold_min: 0.0625,
                    });
                } else if input.virtual_keycode.unwrap_or(VirtualKeyCode::A) == VirtualKeyCode::Key4
                {
                    self.scene_renderer
                        .set_fxaa_settings_mut(FxaaSettings::default());
                }
            }

            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.movement.x = (delta.0.abs() as f32 - MIN_DELTA).max(0.0)
                    * delta.0.signum() as f32
                    * ROTATION_SPEED;
                self.movement.y = (delta.1.abs() as f32 - MIN_DELTA).max(0.0)
                    * delta.1.signum() as f32
                    * ROTATION_SPEED;
            }
            _ => {}
        };
        Ok(())
    }

    fn update(&mut self, _app_state: &mut AppState) -> anyhow::Result<()> {
        if self.rotation_movement > 0.0 {
            self.rot_y += self.movement.x;
            self.rot_x += -self.movement.y;
            self.rot_x = self.rot_x.clamp(-89.0, 89.0);
        } else {
            self.dist += self.movement.y * self.forward_movement * SPEED;
        }

        let rotation = Rotation::from_euler_angles(0.0, self.rot_y.to_radians(), 0.0);
        let rotation = rotation * Rotation::from_euler_angles(0.0, 0.0, self.rot_x.to_radians());
        let new_forward = rotation.to_homogeneous();
        let new_forward = new_forward.column(0);

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        let new_position = direction * self.dist;
        let new_position = point![new_position.x, new_position.y, new_position.z];
        self.camera.location = new_position;

        let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
        self.camera.forward = -direction;
        Ok(())
    }

    fn draw(&mut self, app_state: &mut AppState) -> anyhow::Result<()> {
        
        self.imgui
            .io_mut()
            .update_delta_time(std::time::Duration::from_secs_f32(
                app_state.time.delta_frame(),
            ));
        self.platform.prepare_frame(
            self.imgui.io_mut(),
            &engine::app_state().gpu.swapchain().window,
        )?;
        let ui = self.imgui.frame();
        
        let swapchain_format = app_state.gpu.swapchain().present_format();
        let swapchain_extents = app_state.gpu.swapchain().extents();
        let (swapchain_image, swapchain_image_view) =
            app_state.gpu.swapchain_mut().acquire_next_image()?;
        
        
        let mut settings = self.scene_renderer.fxaa_settings();
        ui.text("Hiii");

        ui.slider("FXAA subpix", 0.0, 1.0, &mut settings.fxaa_quality_subpix);
        ui.slider("FXAA Edge Threshold", 0.0, 1.0, &mut settings.fxaa_quality_edge_threshold);
        ui.slider("FXAA Edge Threshold min", 0.0, 1.0, &mut settings.fxaa_quality_edge_threshold_min);
        self.scene_renderer.set_fxaa_settings_mut(settings);
        
        let mut command_buffer = self.scene_renderer.render(
            &self.camera,
            self.gltf_loader.scene(),
            Backbuffer {
                size: swapchain_extents,
                format: swapchain_format,
                image: swapchain_image,
                image_view: swapchain_image_view,
            },
            &self.resource_map,
        )?;
        
        self.platform.prepare_render(
            ui,
            &engine::app_state().gpu.swapchain().window,
        );

        
        let data = self.imgui.render();
        {
            let color = vec![ColorAttachment {
                image_view: swapchain_image_view,
                load_op: gpu::ColorLoadOp::Load,
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let render_imgui = command_buffer.begin_render_pass(&BeginRenderPassInfo {
                color_attachments: &color,
                depth_attachment: None,
                stencil_attachment: None,
                render_area: Rect2D {
                    offset: ash::vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extents,
                },
            });
            let cmd_buf = render_imgui.inner();
            self.renderer.cmd_draw(cmd_buf, data)?;
        }
        command_buffer.pipeline_barrier(&PipelineBarrierInfo {
            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dependency_flags: DependencyFlags::empty(),
            memory_barriers: &[],
            buffer_memory_barriers: &[],
            image_memory_barriers: &[ImageMemoryBarrier {
                src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: AccessFlags::COLOR_ATTACHMENT_READ,
                old_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: ImageLayout::PRESENT_SRC_KHR,
                src_queue_family_index: ash::vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ash::vk::QUEUE_FAMILY_IGNORED,
                image: swapchain_image,
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            }],
        });
        let frame = app_state.gpu.get_current_swapchain_frame();
        command_buffer.submit(&CommandBufferSubmitInfo {
            wait_semaphores: &[&frame.image_available_semaphore],
            wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            signal_semaphores: &[&frame.render_finished_semaphore],
            fence: Some(&frame.in_flight_fence),
        })?;
        Ok(())
    }
}

fn add_scene_lights(scene: &mut Scene) {
    scene.add_light(Light {
        ty: LightType::Point,
        position: vector![0.0, 10.0, 0.0],
        radius: 50.0,
        color: vector![1.0, 0.0, 0.0],
        intensity: 1.0,
        enabled: true,
    });
    scene.add_light(Light {
        ty: LightType::Directional {
            direction: vector![-0.45, -0.45, 0.0],
        },
        position: vector![100.0, 100.0, 0.0],
        radius: 10.0,
        color: vector![1.0, 1.0, 1.0],
        intensity: 1.0,
        enabled: true,
    });
}

fn main() -> anyhow::Result<()> {
    bootstrap::<GLTFViewer>()
}
