use ash::vk::{
    AccessFlags, DependencyFlags, ImageAspectFlags, ImageLayout, ImageSubresourceRange, Rect2D,
};
use engine::{AppState, Backbuffer};
use gpu::{
    BeginRenderPassInfo, ColorAttachment, CommandBuffer, CommandBufferSubmitInfo,
    ImageMemoryBarrier, PipelineBarrierInfo, PipelineStageFlags,
};
use imgui::{Context, FontConfig, FontSource, Ui};
use imgui_rs_vulkan_renderer::{DynamicRendering, Options, Renderer};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use log::trace;
use winit::{
    dpi::PhysicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop},
};

pub struct ImguiData {
    imgui: Context,
    renderer: Renderer,
    platform: WinitPlatform,
}

pub trait App {
    fn window_name(&self, app_state: &engine::AppState) -> String;

    fn create(app_state: &AppState, event_loop: &EventLoop<()>) -> anyhow::Result<Self>
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
    fn draw(&mut self, backbuffer: &Backbuffer) -> anyhow::Result<CommandBuffer>;
}

pub fn app_loop<A: App + 'static>(
    app: &mut A,
    event: Event<'_, ()>,
    imgui_data: &mut ImguiData,
) -> anyhow::Result<ControlFlow> {
    let app_state_mut = engine::app_state_mut();
    app.on_event(&event, app_state_mut)?;
    imgui_data
        .platform
        .handle_event(imgui_data.imgui.io_mut(), &app_state_mut.window(), &event);
    match event {
        winit::event::Event::NewEvents(_) => {}
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::CloseRequested => {
                return Ok(ControlFlow::ExitWithCode(0));
            }
            winit::event::WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    app_state_mut.swapchain_mut().recreate_swapchain().unwrap();
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
            app_state_mut.window().request_redraw();
        }
        winit::event::Event::RedrawRequested(..) => {
            let win_size = app_state_mut.window().inner_size();
            if win_size.width > 0 && win_size.height > 0 {
                update_loop(app, imgui_data, app_state_mut)?;
            }
        }
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
) -> anyhow::Result<()> {
    app_state_mut.begin_frame().unwrap();
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

    app_state_mut.window().set_title(&window_name);

    let ui = imgui_data.imgui.frame();
    app.update(app_state_mut, ui)?;
    imgui_data
        .platform
        .prepare_render(ui, &engine::app_state().window());

    let swapchain_format = app_state_mut.swapchain().present_format();
    let swapchain_extents = app_state_mut.swapchain().extents();
    let (swapchain_image, swapchain_image_view) =
        app_state_mut.swapchain_mut().acquire_next_image()?;
    let backbuffer = Backbuffer {
        size: swapchain_extents,
        format: swapchain_format,
        image: swapchain_image,
        image_view: swapchain_image_view,
    };
    let mut command_buffer = app.draw(&backbuffer)?;

    draw_imgui(imgui_data, &backbuffer, &mut command_buffer)?;
    let frame = app_state_mut.swapchain_mut().get_current_swapchain_frame();
    command_buffer.submit(&CommandBufferSubmitInfo {
        wait_semaphores: &[&frame.image_available_semaphore],
        wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
        signal_semaphores: &[&frame.render_finished_semaphore],
        fence: Some(&frame.in_flight_fence),
    })?;
    app_state_mut.end_frame()?;
    Ok(())
}

fn draw_imgui(
    imgui_data: &mut ImguiData,
    backbuffer: &Backbuffer,
    command_buffer: &mut CommandBuffer<'_>,
) -> Result<(), anyhow::Error> {
    let data = imgui_data.imgui.render();
    {
        let color = vec![ColorAttachment {
            image_view: backbuffer.image_view,
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
                extent: backbuffer.size,
            },
        });
        let cmd_buf = render_imgui.inner();
        imgui_data.renderer.cmd_draw(cmd_buf, data)?;
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
            image: backbuffer.image,
            subresource_range: ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        }],
    });
    Ok(())
}

pub fn bootstrap<A: App + 'static>() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::default();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1240,
            height: 720,
        })
        .with_title("Winit App")
        .build(&event_loop)?;

    engine::init("Winit App", window)?;

    let app = Box::new(A::create(engine::app_state(), &event_loop)?);
    let app = Box::leak(app);

    trace!("Created app");

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
        &engine::app_state().window(),
        HiDpiMode::Rounded,
    );
    let renderer = Renderer::with_default_allocator(
        &engine::app_state().gpu.instance(),
        engine::app_state().gpu.vk_physical_device(),
        engine::app_state().gpu.vk_logical_device(),
        engine::app_state().gpu.graphics_queue(),
        engine::app_state().gpu.command_pool(),
        DynamicRendering {
            color_attachment_format: engine::app_state().swapchain().present_format(),
            depth_attachment_format: None,
        },
        &mut imgui,
        Some(Options {
            in_flight_frames: 2,
            ..Default::default()
        }),
    )?;

    let imgui_data = ImguiData {
        imgui,
        renderer,
        platform,
    };
    let imgui_data = Box::new(imgui_data);
    let mut imgui_data = Box::leak(imgui_data);
    event_loop.run(
        move |event, _, control_flow| match app_loop(app, event, &mut imgui_data) {
            Ok(flow) => {
                *control_flow = flow;
            }
            Err(e) => panic!("In main body of application: {}", e),
        },
    )
}
