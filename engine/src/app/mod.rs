pub mod app_state;
mod console;
use console::ImguiConsole;

use crate::Backbuffer;
use gpu::{
    AccessFlags, AttachmentReference, BeginRenderPassInfo, CommandBufferSubmitInfo,
    FramebufferColorAttachment, ImageLayout, Offset2D, PipelineStageFlags, Rect2D,
    RenderPassAttachment, SubpassDependency, SubpassDescription, VkCommandBuffer, VkSwapchain,
};
use imgui::{Context, FontConfig, FontSource, Ui};
use imgui_rs_vulkan_renderer::{Options, Renderer};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use log::{info, trace};
use winit::{
    dpi::PhysicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop},
};

use app_state::{app_state, app_state_mut, AppState};

pub struct ImguiData {
    imgui: Context,
    platform: WinitPlatform,
    renderer: Renderer,
}

pub trait App {
    fn window_name(&self, app_state: &AppState) -> String;

    fn create(app_state: &mut AppState, event_loop: &EventLoop<()>) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn on_event(&mut self, _event: &Event<()>, _app_state: &AppState) -> anyhow::Result<()> {
        Ok(())
    }

    fn input(
        &mut self,
        app_state: &AppState,
        event: winit::event::DeviceEvent,
    ) -> anyhow::Result<()>;
    fn update(&mut self, app_state: &mut AppState, ui: &mut Ui) -> anyhow::Result<()>;
    fn draw<'a>(
        &'a mut self,
        app_state: &'a AppState,
        backbuffer: &Backbuffer,
    ) -> anyhow::Result<VkCommandBuffer>;
}

pub fn app_loop<A: App + 'static>(
    app: &mut A,
    event: Event<'_, ()>,
    imgui_data: &mut ImguiData,
    console: &mut ImguiConsole,
) -> anyhow::Result<ControlFlow> {
    let app_state_mut = app_state_mut();
    app.on_event(&event, app_state_mut)?;
    imgui_data
        .platform
        .handle_event(imgui_data.imgui.io_mut(), &app_state_mut.window(), &event);
    app_state_mut.input.update(&event);
    match event {
        winit::event::Event::NewEvents(_) => {}
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::CloseRequested => {
                return Ok(ControlFlow::ExitWithCode(0));
            }
            winit::event::WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    app_state_mut.new_size = Some(new_size);
                }
            }
            _ => {}
        },
        winit::event::Event::DeviceEvent { event, .. } => {
            app.input(app_state_mut, event)?;
        }
        winit::event::Event::UserEvent(_) => {}
        winit::event::Event::Suspended => {}
        winit::event::Event::Resumed => {}
        winit::event::Event::MainEventsCleared => {
            if let Some(_) = app_state_mut.new_size.take() {
                app_state_mut.swapchain_mut().recreate_swapchain()?;
            }
            app_state_mut.begin_frame()?;
            let win_size = app_state_mut.window().inner_size();
            if win_size.width > 0 && win_size.height > 0 {
                update_loop(app, imgui_data, app_state_mut, console)?;
            }
            app_state_mut.end_frame()?;
        }
        winit::event::Event::RedrawRequested(..) => {}
        winit::event::Event::RedrawEventsCleared => {}
        winit::event::Event::LoopDestroyed => {
            app_state_mut.gpu.wait_device_idle().unwrap();
            app_state_mut
                .gpu
                .save_pipeline_cache("pipeline_cache.pso")?;
        }
    }

    Ok(ControlFlow::Poll)
}

fn update_loop(
    app: &mut dyn App,
    imgui_data: &mut ImguiData,
    app_state_mut: &mut AppState,
    console: &mut ImguiConsole,
) -> anyhow::Result<()> {
    console.update(&app_state_mut.input);

    imgui_data
        .platform
        .prepare_frame(imgui_data.imgui.io_mut(), &app_state_mut.window())?;
    imgui_data
        .imgui
        .io_mut()
        .update_delta_time(std::time::Duration::from_secs_f32(
            app_state_mut.time.delta_frame(),
        ));
    let window_name = app.window_name(app_state_mut);
    let ui = imgui_data.imgui.new_frame();

    app_state_mut.window().set_title(&window_name);
    imgui_data
        .platform
        .prepare_render(ui, &app_state().window());

    app.update(app_state_mut, ui)?;
    console.imgui_update(ui, &mut app_state_mut.cvar_manager);

    let swapchain_format = app_state_mut.swapchain().present_format();
    let swapchain_extents = app_state_mut.swapchain().extents();
    let (swapchain_image, swapchain_image_view) =
        app_state_mut.swapchain_mut().acquire_next_image()?;
    let backbuffer = Backbuffer {
        size: swapchain_extents,
        format: swapchain_format.into(),
        image: swapchain_image,
        image_view: swapchain_image_view,
    };
    let mut command_buffer = app.draw(self::app_state_mut(), &backbuffer)?;

    draw_imgui(imgui_data, &backbuffer, &mut command_buffer)?;

    let frame = app_state_mut.swapchain_mut().get_current_swapchain_frame();
    command_buffer.submit(&CommandBufferSubmitInfo {
        wait_semaphores: &[&frame.image_available_semaphore],
        wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
        signal_semaphores: &[&frame.render_finished_semaphore],
        fence: Some(&frame.in_flight_fence),
    })?;
    Ok(())
}

fn draw_imgui(
    imgui_data: &mut ImguiData,
    backbuffer: &Backbuffer,
    command_buffer: &mut VkCommandBuffer<'_>,
) -> Result<(), anyhow::Error> {
    {
        let color = vec![FramebufferColorAttachment {
            image_view: backbuffer.image_view.clone(),
            load_op: gpu::ColorLoadOp::Load,
            store_op: gpu::AttachmentStoreOp::Store,
            initial_layout: ImageLayout::ColorAttachment,
            final_layout: ImageLayout::PresentSrc,
        }];
        let imgui_label = command_buffer.begin_debug_region("ImGui", [0.0, 0.0, 1.0, 1.0]);
        let render_imgui = command_buffer.begin_render_pass(&BeginRenderPassInfo {
            color_attachments: &color,
            depth_attachment: None,
            stencil_attachment: None,
            render_area: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: backbuffer.size,
            },
            label: Some("ImGUI render pass"),
            subpasses: &[SubpassDescription {
                label: None,
                input_attachments: vec![],
                color_attachments: vec![AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::ColorAttachment,
                }],
                resolve_attachments: vec![],
                depth_stencil_attachment: None,
                preserve_attachments: vec![],
            }],
            // Before writing to the swapchain
            // wait for the previous render pass to have finished writing it's color attachments
            dependencies: &[SubpassDependency {
                src_subpass: SubpassDependency::EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
            }],
        });

        let cmd_buf = render_imgui.inner();
        let data = imgui_data.imgui.render();

        imgui_data.renderer.cmd_draw(cmd_buf, data)?;
        imgui_label.end();
    }
    Ok(())
}

pub fn bootstrap<A: App + 'static>() -> anyhow::Result<()> {
    let mut console = ImguiConsole::new();
    console.add_message("Hello! :)");

    let mut env_logger_builder = env_logger::builder();

    if cfg!(debug_assertions) {
        // Enable all logging in debug configuration
        env_logger_builder.filter(None, log::LevelFilter::Trace);
    }

    env_logger_builder.init();

    info!(
        "Running in {:?}",
        std::env::current_dir().unwrap_or(".".into())
    );
    let event_loop = winit::event_loop::EventLoop::default();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1920,
            height: 1080,
        })
        .with_title("Winit App")
        .build(&event_loop)?;

    let mut imgui = Context::create();
    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);

    crate::app::app_state::init("Winit App", window)?;

    let app = Box::new(A::create(app_state_mut(), &event_loop)?);
    let app = Box::leak(app);

    trace!("Created app");

    let hidpi_factor = platform.hidpi_factor();
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(FontConfig {
            size_pixels: font_size,
            ..FontConfig::default()
        }),
    }]);
    platform.attach_window(imgui.io_mut(), &app_state().window(), HiDpiMode::Rounded);

    let render_pass = app_state().gpu.get_render_pass(
        &gpu::RenderPassAttachments {
            color_attachments: vec![RenderPassAttachment {
                format: app_state().swapchain().present_format().into(),
                samples: gpu::SampleCount::Sample1,
                load_op: gpu::ColorLoadOp::DontCare,
                store_op: gpu::AttachmentStoreOp::Store,
                stencil_load_op: gpu::StencilLoadOp::DontCare,
                stencil_store_op: gpu::AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::ColorAttachment,
                final_layout: ImageLayout::PresentSrc,
                blend_state: gpu::BlendState::default(),
            }],
            depth_attachment: None,
            stencil_attachment: None,
            subpasses: vec![SubpassDescription {
                label: None,
                input_attachments: vec![],
                color_attachments: vec![AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::ColorAttachment,
                }],
                resolve_attachments: vec![],
                depth_stencil_attachment: None,
                preserve_attachments: vec![],
            }],
            dependencies: vec![SubpassDependency {
                src_subpass: SubpassDependency::EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
            }],
        },
        Some("ImGUI render pass"),
    );
    let renderer = Renderer::with_default_allocator(
        &app_state().gpu.instance(),
        app_state().gpu.vk_physical_device(),
        app_state().gpu.vk_logical_device(),
        app_state().gpu.graphics_queue(),
        app_state().gpu.graphics_command_pool().inner,
        render_pass,
        &mut imgui,
        Some(Options {
            in_flight_frames: VkSwapchain::MAX_FRAMES_IN_FLIGHT,
            enable_depth_test: false,
            enable_depth_write: false,
        }),
    )?;
    let imgui_data = ImguiData {
        imgui,
        platform,
        renderer,
    };
    let imgui_data = Box::new(imgui_data);
    let mut imgui_data = Box::leak(imgui_data);
    event_loop.run(move |event, _, control_flow| {
        match app_loop(app, event, &mut imgui_data, &mut console) {
            Ok(flow) => {
                *control_flow = flow;
            }
            Err(e) => panic!(
                "In main body of application: {}\nBacktrace: {}",
                e,
                e.backtrace()
            ),
        }
    })
}
