use std::{collections::HashMap, time::SystemTime};

use bytemuck::{Pod, Zeroable};
use egui::{
    pos2, vec2, Context, FontDefinitions, FullOutput, Pos2, RawInput, Rect, Sense, TextureId,
    TexturesDelta, Ui, Vec2, ViewportId,
};
use engine_macros::glsl;
use gpu::{
    render_pass_2::RenderPass2, BeginRenderPassInfo2, Binding2, BufferCreateInfo, BufferHandle,
    BufferUsageFlags, ColorAttachment, ColorComponentFlags, CommandBuffer, ComponentMapping,
    Extent2D, Gpu, ImageAspectFlags, ImageCreateInfo, ImageFormat, ImageHandle,
    ImageSubresourceRange, ImageUsageFlags, ImageViewCreateInfo, ImageViewHandle, MemoryDomain,
    Offset2D, PipelineColorBlendAttachmentState, Rect2D, SamplerCreateInfo, SamplerHandle,
    ShaderModuleCreateInfo, ShaderModuleHandle, ShaderStage, Swapchain, VertexBindingInfo,
};

use log::warn;
use winit::{
    event::{Touch, WindowEvent},
    window::Window,
};

use crate::{input::InputState, Backbuffer, CvarManager, Time};

use super::Console;

const VERTEX: &[u32] = glsl!(
    path = "src/shaders/egui.vert",
    kind = vertex,
    entry_point = "main"
);

const FRAGMENT: &[u32] = glsl!(
    path = "src/shaders/egui.frag",
    kind = fragment,
    entry_point = "main"
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScreenData {
    screen_size: [f32; 2],
}

#[derive(Clone, Copy)]
struct EguiBuffers {
    vertices: BufferHandle,
    indices: BufferHandle,
}

struct EguiTexture {
    view: ImageViewHandle,
    image: ImageHandle,
    sampler: Option<SamplerHandle>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EventResponse {
    /// If true, egui consumed this event, i.e. wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    ///
    /// For instance, if you use egui for a game, you should only
    /// pass on the events to your game when [`Self::consumed`] is `false.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will always be `true` for tabs.
    pub consumed: bool,

    /// Do we need an egui refresh because of this event?
    pub repaint: bool,
}

pub struct EguiSupport {
    context: Context,
    raw_input: RawInput,
    default_sampler: SamplerHandle,
    mesh_buffers: Vec<EguiBuffers>,
    textures: HashMap<TextureId, EguiTexture>,
    current_frame: usize,
    pixels_per_point: f32,
    viewport_id: ViewportId,
    pointer_pos_in_points: Option<Pos2>,
    input_method_editor_started: bool,

    vs: ShaderModuleHandle,
    fs: ShaderModuleHandle,
}

impl EguiSupport {
    const MAX_NUM_VERTICES: usize = 65535;
    const MAX_VERTICES: usize =
        Self::MAX_NUM_VERTICES * std::mem::size_of::<egui::epaint::Vertex>();
    const MAX_INDICES: usize = Self::MAX_NUM_VERTICES * std::mem::size_of::<u32>();

    pub fn new(window: &Window, gpu: &dyn Gpu) -> anyhow::Result<Self> {
        let mut mesh_buffers = Vec::with_capacity(gpu::constants::MAX_FRAMES_IN_FLIGHT);

        for i in 0..gpu::constants::MAX_FRAMES_IN_FLIGHT {
            let vertices = gpu.make_buffer(
                &BufferCreateInfo {
                    label: Some(&format!("egui Vertex Buffer #{i}")),
                    size: Self::MAX_VERTICES,
                    usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
                },
                MemoryDomain::DeviceLocal,
            )?;

            let indices = gpu.make_buffer(
                &BufferCreateInfo {
                    label: Some(&format!("egui Index Buffer #{i}")),
                    size: Self::MAX_INDICES,
                    usage: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
                },
                MemoryDomain::DeviceLocal,
            )?;
            mesh_buffers.push(EguiBuffers { vertices, indices });
        }

        let vs = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("egui vertex shader"),
            code: bytemuck::cast_slice(&VERTEX),
        })?;

        let fs = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("egui fragment shader"),
            code: bytemuck::cast_slice(&FRAGMENT),
        })?;

        let sampler = gpu.make_sampler(&SamplerCreateInfo {
            mag_filter: gpu::Filter::Linear,
            min_filter: gpu::Filter::Linear,
            address_u: gpu::SamplerAddressMode::ClampToBorder,
            address_v: gpu::SamplerAddressMode::ClampToBorder,
            address_w: gpu::SamplerAddressMode::ClampToBorder,
            mip_lod_bias: 0.0,
            compare_function: None,
            min_lod: 0.0,
            max_lod: 1.0,
            border_color: [0.0; 4],
        })?;

        let context = Context::default();
        let pixels_per_point = Self::compute_pixels_per_point(&context, window);
        context.set_fonts(FontDefinitions::default());

        context.set_pixels_per_point(pixels_per_point);
        let viewport_id = context.viewport_id();

        Ok(Self {
            context,
            raw_input: RawInput::default(),
            mesh_buffers,
            textures: HashMap::new(),
            current_frame: 0,
            default_sampler: sampler,
            vs,
            fs,
            pixels_per_point,
            viewport_id,
            pointer_pos_in_points: None,
            input_method_editor_started: false,
        })
    }

    pub fn paint_console(&mut self, console: &mut Console, cvar_manager: &mut CvarManager) {
        if console.show {
            let ctx = self.create_context();
            let screen_size = ctx.screen_rect();
            let console_size = vec2(screen_size.width(), screen_size.height() * 0.33);
            const CONSOLE_INPUT_HEIGHT: f32 = 8.0;
            const CONSOLE_INPUT_PADDING: f32 = 2.0;
            egui::Window::new("Console Window")
                .resizable(false)
                .title_bar(false)
                .fixed_rect(Rect {
                    min: pos2(0.0, 0.0),
                    max: pos2(console_size.x, console_size.y),
                })
                .collapsible(false)
                .show(&ctx, |ui| {
                    ui.add(|ui: &mut Ui| {
                        egui::ScrollArea::new([false, true])
                            .vertical_scroll_offset(1.0)
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                            )
                            .show(ui, |ui: &mut Ui| {
                                let response =
                                    ui.allocate_response(vec2(0.0, 0.0), Sense::click_and_drag());
                                egui::Grid::new("messages").show(ui, |ui| {
                                    for msg in &console.messages {
                                        ui.label(format!(
                                            "@{:?} - ",
                                            msg.timestamp
                                                .duration_since(SystemTime::UNIX_EPOCH)
                                                .expect("Time went backwards since UNIX_EPOCH")
                                                .as_millis()
                                        ));
                                        ui.label(&msg.content);
                                        ui.end_row();
                                    }
                                });
                                response
                            })
                            .inner
                    });

                    ui.put(
                        Rect {
                            min: pos2(CONSOLE_INPUT_PADDING, console_size.y - CONSOLE_INPUT_HEIGHT),
                            max: pos2(console_size.x - CONSOLE_INPUT_PADDING, console_size.y),
                        },
                        |ui: &mut Ui| {
                            let resp = egui::TextEdit::singleline(&mut console.pending_input)
                                .min_size(vec2(0.0, CONSOLE_INPUT_HEIGHT))
                                .show(ui)
                                .response;
                            if resp.lost_focus() && ui.ctx().input(|i| i.key_down(egui::Key::Enter))
                            {
                                let message = std::mem::take(&mut console.pending_input);
                                console.add_message(message.clone());
                                console.handle_cvar_command(message, cvar_manager)
                            }
                            resp
                        },
                    );
                });
        }
    }

    pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.on_window_event(window, event)
    }

    pub fn swapchain_updated(&mut self, _swapchain: &Swapchain) {}

    pub fn begin_frame(&mut self, window: &Window, time: &Time) {
        let input = self.take_input(window, time);
        self.context.begin_frame(input);
    }

    pub fn end_frame(&mut self, _window: &Window) -> FullOutput {
        self.context.end_frame()
    }

    pub fn paint_frame(
        &mut self,
        gpu: &dyn Gpu,
        command_buffer: &mut CommandBuffer,
        backbuffer: &Backbuffer,
        output: FullOutput,
    ) {
        let delta = output.textures_delta;
        let clipped_primitives = self
            .context
            .tessellate(output.shapes, self.pixels_per_point);
        self.update_textures(gpu, &delta).unwrap();
        self.paint_primitives(gpu, command_buffer, backbuffer, &clipped_primitives)
            .unwrap();
        self.free_textures(gpu, &delta);
        self.current_frame = (self.current_frame + 1) % gpu::constants::MAX_FRAMES_IN_FLIGHT;
    }

    pub fn create_context(&self) -> Context {
        self.context.clone()
    }

    fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        match event {
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let native_pixels_per_point = *scale_factor as f32;

                self.raw_input
                    .viewports
                    .entry(self.viewport_id)
                    .or_default()
                    .native_pixels_per_point = Some(native_pixels_per_point);

                EventResponse {
                    repaint: true,
                    consumed: false,
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.on_mouse_button_input(*state, *button);
                EventResponse {
                    repaint: true,
                    consumed: self.context.wants_pointer_input(),
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.on_mouse_wheel(window, *delta);
                EventResponse {
                    repaint: true,
                    consumed: self.context.wants_pointer_input(),
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.on_cursor_moved(window, *position);
                EventResponse {
                    repaint: true,
                    consumed: self.context.is_using_pointer(),
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer_pos_in_points = None;
                self.raw_input.events.push(egui::Event::PointerGone);
                EventResponse {
                    repaint: true,
                    consumed: false,
                }
            }
            // WindowEvent::TouchpadPressure {device_id, pressure, stage, ..  } => {} // TODO
            WindowEvent::Touch(touch) => {
                self.on_touch(window, touch);
                let consumed = match touch.phase {
                    winit::event::TouchPhase::Started
                    | winit::event::TouchPhase::Ended
                    | winit::event::TouchPhase::Cancelled => self.context.wants_pointer_input(),
                    winit::event::TouchPhase::Moved => self.context.is_using_pointer(),
                };
                EventResponse {
                    repaint: true,
                    consumed,
                }
            }

            WindowEvent::Ime(ime) => {
                // on Mac even Cmd-C is pressed during ime, a `c` is pushed to Preedit.
                // So no need to check is_mac_cmd.
                //
                // How winit produce `Ime::Enabled` and `Ime::Disabled` differs in MacOS
                // and Windows.
                //
                // - On Windows, before and after each Commit will produce an Enable/Disabled
                // event.
                // - On MacOS, only when user explicit enable/disable ime. No Disabled
                // after Commit.
                //
                // We use input_method_editor_started to manually insert CompositionStart
                // between Commits.
                match ime {
                    winit::event::Ime::Enabled | winit::event::Ime::Disabled => (),
                    winit::event::Ime::Commit(text) => {
                        self.input_method_editor_started = false;
                        self.raw_input
                            .events
                            .push(egui::Event::CompositionEnd(text.clone()));
                    }
                    winit::event::Ime::Preedit(text, Some(_)) => {
                        if !self.input_method_editor_started {
                            self.input_method_editor_started = true;
                            self.raw_input.events.push(egui::Event::CompositionStart);
                        }
                        self.raw_input
                            .events
                            .push(egui::Event::CompositionUpdate(text.clone()));
                    }
                    winit::event::Ime::Preedit(_, None) => {}
                };

                EventResponse {
                    repaint: true,
                    consumed: self.context.wants_keyboard_input(),
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.on_keyboard_input(event);

                // When pressing the Tab key, egui focuses the first focusable element, hence Tab always consumes.
                let consumed = self.context.wants_keyboard_input()
                    || event.logical_key
                        == winit::keyboard::Key::Named(winit::keyboard::NamedKey::Tab);
                EventResponse {
                    repaint: true,
                    consumed,
                }
            }
            WindowEvent::Focused(focused) => {
                self.raw_input.focused = *focused;
                // We will not be given a KeyboardInput event when the modifiers are released while
                // the window does not have focus. Unset all modifier state to be safe.
                self.raw_input.modifiers = egui::Modifiers::default();
                self.raw_input
                    .events
                    .push(egui::Event::WindowFocused(*focused));
                EventResponse {
                    repaint: true,
                    consumed: false,
                }
            }
            WindowEvent::HoveredFile(path) => {
                self.raw_input.hovered_files.push(egui::HoveredFile {
                    path: Some(path.clone()),
                    ..Default::default()
                });
                EventResponse {
                    repaint: true,
                    consumed: false,
                }
            }
            WindowEvent::HoveredFileCancelled => {
                self.raw_input.hovered_files.clear();
                EventResponse {
                    repaint: true,
                    consumed: false,
                }
            }
            WindowEvent::DroppedFile(path) => {
                self.raw_input.hovered_files.clear();
                self.raw_input.dropped_files.push(egui::DroppedFile {
                    path: Some(path.clone()),
                    ..Default::default()
                });
                EventResponse {
                    repaint: true,
                    consumed: false,
                }
            }
            WindowEvent::ModifiersChanged(state) => {
                let state = state.state();

                let alt = state.alt_key();
                let ctrl = state.control_key();
                let shift = state.shift_key();
                let super_ = state.super_key();

                self.raw_input.modifiers.alt = alt;
                self.raw_input.modifiers.ctrl = ctrl;
                self.raw_input.modifiers.shift = shift;
                self.raw_input.modifiers.mac_cmd = cfg!(target_os = "macos") && super_;
                self.raw_input.modifiers.command = if cfg!(target_os = "macos") {
                    super_
                } else {
                    ctrl
                };

                EventResponse {
                    repaint: true,
                    consumed: false,
                }
            }

            // Things that may require repaint:
            WindowEvent::RedrawRequested
            | WindowEvent::CursorEntered { .. }
            | WindowEvent::Destroyed
            | WindowEvent::Occluded(_)
            | WindowEvent::Resized(_)
            | WindowEvent::Moved(_)
            | WindowEvent::ThemeChanged(_)
            | WindowEvent::TouchpadPressure { .. }
            | WindowEvent::CloseRequested => EventResponse {
                repaint: true,
                consumed: false,
            },

            // Things we completely ignore:
            WindowEvent::ActivationTokenDone { .. }
            | WindowEvent::AxisMotion { .. }
            | WindowEvent::SmartMagnify { .. }
            | WindowEvent::TouchpadRotate { .. } => EventResponse {
                repaint: false,
                consumed: false,
            },

            WindowEvent::TouchpadMagnify { delta, .. } => {
                // Positive delta values indicate magnification (zooming in).
                // Negative delta values indicate shrinking (zooming out).
                let zoom_factor = (*delta as f32).exp();
                self.raw_input.events.push(egui::Event::Zoom(zoom_factor));
                EventResponse {
                    repaint: true,
                    consumed: self.context.wants_pointer_input(),
                }
            }
        }
    }

    fn update_textures(&mut self, gpu: &dyn Gpu, delta: &TexturesDelta) -> anyhow::Result<()> {
        for (id, delta) in &delta.set {
            let texture = self.textures.entry(*id).or_insert_with(|| {
                let (width, height, format) = match &delta.image {
                    egui::ImageData::Color(image) => (
                        image.width() as u32,
                        image.height() as u32,
                        ImageFormat::Rgba8,
                    ),
                    egui::ImageData::Font(image) => (
                        image.width() as u32,
                        image.height() as u32,
                        ImageFormat::Rgba8,
                    ),
                };

                let image = gpu
                    .make_image(
                        &ImageCreateInfo {
                            label: Some("egui texture image"),
                            width,
                            height,
                            depth: 1,
                            mips: 1,
                            layers: 1,
                            samples: gpu::SampleCount::Sample1,
                            format,
                            usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                        },
                        MemoryDomain::DeviceLocal,
                        None,
                    )
                    .unwrap();
                let view = gpu
                    .make_image_view(&ImageViewCreateInfo {
                        label: Some("egui texture image view"),
                        image,
                        view_type: gpu::ImageViewType::Type2D,
                        format,
                        components: ComponentMapping::default(),
                        subresource_range: ImageSubresourceRange {
                            aspect_mask: ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    })
                    .unwrap();

                EguiTexture {
                    image,
                    view,
                    sampler: None,
                }
            });

            let [x, y] = delta
                .pos
                .map(|[x, y]| [x as i32, y as i32])
                .unwrap_or_default();
            match &delta.image {
                egui::ImageData::Color(image) => {
                    gpu.write_image(
                        &texture.image,
                        image.as_raw(),
                        Rect2D {
                            extent: Extent2D {
                                width: image.width() as u32,
                                height: image.height() as u32,
                            },
                            offset: Offset2D { x, y },
                        },
                        0,
                    )?;
                }
                egui::ImageData::Font(image) => {
                    let data = image
                        .srgba_pixels(None)
                        .flat_map(|a| a.to_array())
                        .collect::<Vec<u8>>();

                    gpu.write_image(
                        &texture.image,
                        &data,
                        Rect2D {
                            extent: Extent2D {
                                width: image.width() as u32,
                                height: image.height() as u32,
                            },
                            offset: Offset2D { x, y },
                        },
                        0,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn paint_primitives(
        &self,
        gpu: &dyn Gpu,
        command_buffer: &mut CommandBuffer,
        backbuffer: &Backbuffer,
        clipped_primitives: &[egui::ClippedPrimitive],
    ) -> anyhow::Result<()> {
        let mut render_pass = command_buffer.start_render_pass_2(&BeginRenderPassInfo2 {
            label: Some("egui painting"),
            color_attachments: &[ColorAttachment {
                image_view: backbuffer.image_view,
                load_op: gpu::ColorLoadOp::Load,
                store_op: gpu::AttachmentStoreOp::Store,
            }],
            depth_attachment: None,
            stencil_attachment: None,
            render_area: backbuffer.whole_area(),
        });
        self.prepare_render_pass(
            &mut render_pass,
            [backbuffer.size.width as f32, backbuffer.size.height as f32],
            self.pixels_per_point,
        );

        let mut offset_indices = 0;
        let mut offset_vertices = 0i32;
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_primitives
        {
            Self::set_clip_rect(
                &mut render_pass,
                [backbuffer.size.width, backbuffer.size.height],
                self.pixels_per_point,
                *clip_rect,
            );

            match primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    let EguiBuffers { vertices, indices } = self.mesh_buffers[self.current_frame];

                    if let Some(texture) = self.get_image_view(mesh.texture_id) {
                        gpu.write_buffer(
                            &vertices,
                            offset_vertices as u64
                                * std::mem::size_of::<egui::epaint::Vertex>() as u64,
                            bytemuck::cast_slice(&mesh.vertices),
                        )?;
                        gpu.write_buffer(
                            &indices,
                            offset_indices as u64 * std::mem::size_of::<u32>() as u64,
                            bytemuck::cast_slice(&mesh.indices),
                        )?;

                        render_pass.bind_resources_2(
                            0,
                            &[Binding2::image_view(texture, self.default_sampler)],
                        );

                        render_pass.draw_indexed(
                            mesh.indices.len() as _,
                            1,
                            offset_indices,
                            offset_vertices,
                            0,
                        )?;
                        offset_indices += mesh.indices.len() as u32;
                        offset_vertices += mesh.vertices.len() as i32;
                    } else {
                        warn!("Could not find texture id {:?}", mesh.texture_id);
                    }
                }
                egui::epaint::Primitive::Callback(_) => todo!(),
            }
        }
        Ok(())
    }
    fn set_clip_rect(
        render_passs: &mut RenderPass2,
        [width_px, height_px]: [u32; 2],
        pixels_per_point: f32,
        clip_rect: Rect,
    ) {
        // Transform clip rect to physical pixels:
        let clip_min_x = pixels_per_point * clip_rect.min.x;
        let clip_min_y = pixels_per_point * clip_rect.min.y;
        let clip_max_x = pixels_per_point * clip_rect.max.x;
        let clip_max_y = pixels_per_point * clip_rect.max.y;

        // Round to integer:
        let clip_min_x = clip_min_x.round() as i32;
        let clip_min_y = clip_min_y.round() as i32;
        let clip_max_x = clip_max_x.round() as i32;
        let clip_max_y = clip_max_y.round() as i32;

        // Clamp:
        let clip_min_x = clip_min_x.clamp(0, width_px as i32);
        let clip_min_y = clip_min_y.clamp(0, height_px as i32);
        let clip_max_x = clip_max_x.clamp(clip_min_x, width_px as i32);
        let clip_max_y = clip_max_y.clamp(clip_min_y, height_px as i32);

        render_passs.set_scissor_rect(Rect2D {
            offset: Offset2D {
                x: clip_min_x,
                y: clip_min_y,
            },
            extent: Extent2D {
                width: (clip_max_x - clip_min_x) as u32,
                height: (clip_max_y - clip_min_y) as u32,
            },
        });
    }

    fn get_image_view(&self, texture_id: TextureId) -> Option<ImageViewHandle> {
        self.textures.get(&texture_id).map(|tex| tex.view)
    }

    fn prepare_render_pass(
        &self,
        render_pass: &mut gpu::render_pass_2::RenderPass2<'_>,
        screen_size: [f32; 2],
        pixels_per_point: f32,
    ) {
        let EguiBuffers { vertices, indices } = self.mesh_buffers[self.current_frame];
        render_pass.set_vertex_buffers(
            &[
                VertexBindingInfo {
                    handle: vertices,
                    location: 0,
                    offset: 0,
                    stride: std::mem::size_of::<egui::epaint::Vertex>() as _,
                    format: ImageFormat::RgFloat32,
                    input_rate: gpu::InputRate::PerVertex,
                },
                VertexBindingInfo {
                    handle: vertices,
                    location: 1,
                    offset: memoffset::offset_of!(egui::epaint::Vertex, uv) as _,
                    stride: std::mem::size_of::<egui::epaint::Vertex>() as _,
                    format: ImageFormat::RgFloat32,
                    input_rate: gpu::InputRate::PerVertex,
                },
                VertexBindingInfo {
                    handle: vertices,
                    location: 2,
                    offset: memoffset::offset_of!(egui::epaint::Vertex, color) as _,
                    stride: std::mem::size_of::<egui::epaint::Vertex>() as _,
                    format: ImageFormat::Rgba8,
                    input_rate: gpu::InputRate::PerVertex,
                },
            ],
            &[0, 0, 0],
        );
        let screen_size = screen_size.map(|v| v / pixels_per_point);
        render_pass.set_color_attachment_blend_state(
            0,
            PipelineColorBlendAttachmentState {
                blend_enable: true,
                src_color_blend_factor: gpu::BlendMode::One,
                dst_color_blend_factor: gpu::BlendMode::OneMinusSrcAlpha,
                color_blend_op: gpu::BlendOp::Add,
                src_alpha_blend_factor: gpu::BlendMode::One,
                dst_alpha_blend_factor: gpu::BlendMode::OneMinusSrcAlpha,
                alpha_blend_op: gpu::BlendOp::Add,
                color_write_mask: ColorComponentFlags::RGBA,
            },
        );
        render_pass.set_index_buffer(indices, gpu::IndexType::Uint32, 0);
        render_pass.set_cull_mode(gpu::CullMode::None);
        render_pass.set_vertex_shader(self.vs);
        render_pass.set_enable_depth_test(false);
        render_pass.set_fragment_shader(self.fs);
        render_pass.set_front_face(gpu::FrontFace::CounterClockWise);
        render_pass.push_constants(
            0,
            0,
            bytemuck::cast_slice(&[ScreenData { screen_size }]),
            ShaderStage::ALL_GRAPHICS,
        );
    }

    fn free_textures(&mut self, gpu: &dyn Gpu, delta: &TexturesDelta) {
        for free in &delta.free {
            if let Some(egui_texture) = self.textures.remove(&free) {
                free_texture(gpu, &egui_texture);
            }
        }
    }

    fn compute_pixels_per_point(context: &Context, window: &Window) -> f32 {
        let native_pixels_per_point = window.scale_factor() as f32;
        let egui_zoom = context.zoom_factor();
        egui_zoom * native_pixels_per_point
    }

    pub fn destroy(&mut self, gpu: &dyn Gpu) {
        self.textures.values().for_each(|tex| {
            free_texture(gpu, tex);
        });

        self.mesh_buffers.iter().for_each(|mb| {
            gpu.destroy_buffer(mb.vertices);
            gpu.destroy_buffer(mb.indices);
        });
        gpu.destroy_shader_module(self.vs);
        gpu.destroy_shader_module(self.fs);
    }

    fn on_mouse_button_input(
        &mut self,
        state: winit::event::ElementState,
        button: winit::event::MouseButton,
    ) {
        if let Some(pos) = self.pointer_pos_in_points {
            if let Some(button) = translate_mouse_button(button) {
                let pressed = state == winit::event::ElementState::Pressed;

                self.raw_input.events.push(egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    modifiers: self.raw_input.modifiers,
                });
            }
        }
    }

    fn on_mouse_wheel(&mut self, _window: &Window, delta: winit::event::MouseScrollDelta) {
        let pixels_per_point = self.pixels_per_point;

        {
            let (unit, delta) = match delta {
                winit::event::MouseScrollDelta::LineDelta(x, y) => {
                    (egui::MouseWheelUnit::Line, egui::vec2(x, y))
                }
                winit::event::MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition {
                    x,
                    y,
                }) => (
                    egui::MouseWheelUnit::Point,
                    egui::vec2(x as f32, y as f32) / pixels_per_point,
                ),
            };
            let modifiers = self.raw_input.modifiers;
            self.raw_input.events.push(egui::Event::MouseWheel {
                unit,
                delta,
                modifiers,
            });
        }
        let delta = match delta {
            winit::event::MouseScrollDelta::LineDelta(x, y) => {
                let points_per_scroll_line = 50.0; // Scroll speed decided by consensus: https://github.com/emilk/egui/issues/461
                egui::vec2(x, y) * points_per_scroll_line
            }
            winit::event::MouseScrollDelta::PixelDelta(delta) => {
                egui::vec2(delta.x as f32, delta.y as f32) / pixels_per_point
            }
        };

        if self.raw_input.modifiers.ctrl || self.raw_input.modifiers.command {
            // Treat as zoom instead:
            let factor = (delta.y / 200.0).exp();
            self.raw_input.events.push(egui::Event::Zoom(factor));
        } else if self.raw_input.modifiers.shift {
            // Treat as horizontal scrolling.
            // Note: one Mac we already get horizontal scroll events when shift is down.
            self.raw_input
                .events
                .push(egui::Event::Scroll(egui::vec2(delta.x + delta.y, 0.0)));
        } else {
            self.raw_input.events.push(egui::Event::Scroll(delta));
        }
    }

    fn on_keyboard_input(&self, _event: &winit::event::KeyEvent) {}

    fn on_cursor_moved(
        &mut self,
        _window: &Window,
        pos_in_pixels: winit::dpi::PhysicalPosition<f64>,
    ) {
        let pixels_per_point = self.pixels_per_point;

        let pos_in_points = egui::pos2(
            pos_in_pixels.x as f32 / pixels_per_point,
            pos_in_pixels.y as f32 / pixels_per_point,
        );
        self.pointer_pos_in_points = Some(pos_in_points);

        self.raw_input
            .events
            .push(egui::Event::PointerMoved(pos_in_points));
    }

    fn on_touch(&self, _window: &Window, _touch: &Touch) {}

    fn take_input(&mut self, window: &Window, time: &Time) -> RawInput {
        let mut input = self.raw_input.take();
        input.time = Some(time.since_app_start() as f64);

        let screen_size_pixels = window.inner_size().cast::<f32>();
        let screen_size_pixels = Vec2 {
            x: screen_size_pixels.width,
            y: screen_size_pixels.height,
        };
        let screen_size_points = screen_size_pixels / self.pixels_per_point;

        input.screen_rect = (screen_size_pixels.x > 0.0 && screen_size_pixels.y > 0.0)
            .then(|| Rect::from_min_size(Pos2::ZERO, screen_size_points));

        // input.viewport_id = self.viewport_id;
        // input
        //     .viewports
        //     .entry(self.viewport_id)
        //     .or_default()
        //     .native_pixels_per_point = Some(window.scale_factor() as f32);

        input
    }
}

fn translate_mouse_button(button: winit::event::MouseButton) -> Option<egui::PointerButton> {
    match button {
        winit::event::MouseButton::Left => Some(egui::PointerButton::Primary),
        winit::event::MouseButton::Right => Some(egui::PointerButton::Secondary),
        winit::event::MouseButton::Middle => Some(egui::PointerButton::Middle),
        winit::event::MouseButton::Back => Some(egui::PointerButton::Extra1),
        winit::event::MouseButton::Forward => Some(egui::PointerButton::Extra2),
        winit::event::MouseButton::Other(_) => None,
    }
}

#[allow(dead_code)]
fn parse_input_modifiers(_input_state: &InputState) -> egui::Modifiers {
    todo!()
}

fn free_texture(gpu: &dyn Gpu, egui_texture: &EguiTexture) {
    gpu.destroy_image_view(egui_texture.view);
    gpu.destroy_image(egui_texture.image);

    if let Some(sampler) = egui_texture.sampler {
        gpu.destroy_sampler(sampler);
    }
}
