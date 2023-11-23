use imgui::{Context, FontConfig, FontSource, Ui};
use imgui_rs_vulkan_renderer::{Options, Renderer};
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use gpu::{
    AccessFlags, AttachmentReference, BeginRenderPassInfo, FramebufferColorAttachment, ImageLayout,
    Offset2D, PipelineStageFlags, Rect2D, RenderPassAttachment, SubpassDependency,
    SubpassDescription, VkCommandBuffer, VkSwapchain,
};
use winit::{event::Event, window::Window};

use crate::Backbuffer;

use super::app_state::AppState;

pub struct ImguiData {
    pub imgui: Context,
    pub platform: WinitPlatform,
    pub renderer: Renderer,
}

impl ImguiData {
    pub fn new(app_state: &AppState, window: &Window) -> anyhow::Result<Self> {
        let mut imgui = Context::create();
        let mut platform = WinitPlatform::init(&mut imgui);
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Default);
        let hidpi_factor = platform.hidpi_factor();
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        }]);
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);

        let render_pass = app_state.gpu.get_render_pass(
            &gpu::RenderPassAttachments {
                color_attachments: vec![RenderPassAttachment {
                    format: app_state.swapchain().present_format().into(),
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
            &app_state.gpu.instance(),
            app_state.gpu.vk_physical_device(),
            app_state.gpu.vk_logical_device(),
            app_state.gpu.graphics_queue(),
            app_state.gpu.graphics_command_pool().inner,
            render_pass,
            &mut imgui,
            Some(Options {
                in_flight_frames: VkSwapchain::MAX_FRAMES_IN_FLIGHT,
                enable_depth_test: false,
                enable_depth_write: false,
            }),
        )?;

        Ok(ImguiData {
            imgui,
            platform,
            renderer,
        })
    }

    pub fn on_event<T>(&mut self, window: &Window, event: Event<T>) {
        self.platform
            .handle_event(&mut self.imgui.io_mut(), window, &event)
    }

    pub fn begin_frame(&mut self, window: &Window) -> anyhow::Result<&mut Ui> {
        self.platform
            .prepare_frame(&mut self.imgui.io_mut(), window)?;
        Ok(self.imgui.new_frame())
    }

    pub fn end_frame(
        &mut self,
        backbuffer: &Backbuffer,
        command_buffer: &mut VkCommandBuffer<'_>,
        ui: &mut Ui,
        window: &Window,
    ) -> Result<(), anyhow::Error> {
        self.platform
            .prepare_frame(&mut self.imgui.io_mut(), window)?;
        self.platform.prepare_render(ui, window);
        {
            let color = vec![FramebufferColorAttachment {
                image_view: backbuffer.image_view.clone(),
                load_op: gpu::ColorLoadOp::Load,
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::ColorAttachment,
                final_layout: ImageLayout::ColorAttachment,
            }];
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
            let data = self.imgui.render();

            self.renderer.cmd_draw(cmd_buf, data)?;
        }
        Ok(())
    }
}
