use std::{collections::HashMap, time::SystemTime};

use bytemuck::{Pod, Zeroable};
use copypasta::{ClipboardContext, ClipboardProvider};
use egui::{
    epaint::{ClippedShape, Vertex},
    Context, CursorIcon, FontDefinitions, FullOutput, Pos2, RawInput, Rect, TextureId,
    TexturesDelta, Vec2, ViewportId,
};
use log::warn;
use mgpu::{
    include_spirv, AddressMode, Binding, BindingSet, BindingSetDescription, BindingSetElement,
    BindingSetLayout, BindingSetLayoutInfo, BlendFactor, BlendOp, BlendSettings, BorderColor,
    Buffer, BufferDescription, BufferUsageFlags, BufferWriteParams, ColorWriteMask,
    CommandRecorder, Device, Extents2D, Extents3D, FilterMode, FragmentStageInfo, Graphics,
    GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription, ImageFormat,
    ImageRegion, ImageUsageFlags, ImageView, MemoryDomain, MipmapMode, Offset2D, Offset3D,
    PrimitiveTopology, Rect2D, RenderPass, RenderPassDescription, RenderTarget, RenderTargetInfo,
    Sampler, SamplerDescription, ShaderModule, ShaderModuleDescription, ShaderStageFlags,
    VertexInputDescription, VertexInputFrequency, VertexStageInfo,
};
use winit::{
    event::{Touch, WindowEvent},
    window::Window,
};

const EGUI_VS: &[u8] = include_spirv!("spirv/egui.vert.spv");
const EGUI_FS: &[u8] = include_spirv!("spirv/egui.frag.spv");

pub struct EguiMgpuIntegration {
    context: Context,
    raw_input: RawInput,
    sampler: Sampler,
    buffers: Vec<EguiBuffers>,
    textures: HashMap<TextureId, EguiTexture>,
    current_frame: usize,
    pixels_per_point: f32,
    viewport_id: ViewportId,
    pointer_pos_in_points: Option<Pos2>,
    current_cursor_icon: Option<CursorIcon>,
    input_method_editor_started: bool,
    clipboard_context: ClipboardContext,

    vs: ShaderModule,
    fs: ShaderModule,
    pipeline: GraphicsPipeline,
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

struct EguiBuffers {
    vertices: Buffer,
    indices: Buffer,
}

struct EguiTexture {
    view: ImageView,
    image: Image,
    sampler: Option<Sampler>,

    binding_set: BindingSet,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ScreenData {
    screen_size: [f32; 2],
}

impl EguiMgpuIntegration {
    const MAX_NUM_VERTICES: usize = 65535;
    const MAX_VERTICES: usize = Self::MAX_NUM_VERTICES * std::mem::size_of::<Vertex>();
    const MAX_INDICES: usize = Self::MAX_NUM_VERTICES * std::mem::size_of::<u32>();

    pub fn new(device: &Device, window: &Window) -> anyhow::Result<Self> {
        let frames_in_flight = device.get_info().frames_in_flight;
        let mut mesh_buffers = Vec::with_capacity(frames_in_flight);

        for i in 0..frames_in_flight {
            let vertices = device.create_buffer(&BufferDescription {
                label: Some(&format!("egui Vertex Buffer #{i}")),
                size: Self::MAX_VERTICES,
                usage_flags: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory_domain: MemoryDomain::Gpu,
            })?;

            let indices = device.create_buffer(&BufferDescription {
                label: Some(&format!("egui Index Buffer #{i}")),
                size: Self::MAX_INDICES,
                usage_flags: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory_domain: MemoryDomain::Gpu,
            })?;
            mesh_buffers.push(EguiBuffers { vertices, indices });
        }

        let vs = device.create_shader_module(&ShaderModuleDescription {
            label: Some("egui vertex shader"),
            source: bytemuck::cast_slice(EGUI_VS),
        })?;

        let fs = device.create_shader_module(&ShaderModuleDescription {
            label: Some("egui fragment shader"),
            source: bytemuck::cast_slice(EGUI_FS),
        })?;

        let sampler = device.create_sampler(&SamplerDescription {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToBorder,
            address_mode_v: AddressMode::ClampToBorder,
            address_mode_w: AddressMode::ClampToBorder,
            lod_bias: 0.0,
            compare_op: None,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: BorderColor::White,
            label: Some("egui sampler"),
            mipmap_mode: MipmapMode::Nearest,
            unnormalized_coordinates: false,
        })?;

        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDescription {
            label: Some("egui main pipeline"),
            vertex_stage: &VertexStageInfo {
                shader: &vs,
                entry_point: "main",
                vertex_inputs: &[
                    VertexInputDescription {
                        binding: 0,
                        location: 0,
                        stride: std::mem::size_of::<Vertex>(),
                        offset: std::mem::offset_of!(egui::epaint::Vertex, pos),
                        format: mgpu::VertexAttributeFormat::Float2,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                    VertexInputDescription {
                        binding: 0,
                        location: 1,
                        stride: std::mem::size_of::<Vertex>(),
                        offset: std::mem::offset_of!(egui::epaint::Vertex, uv),
                        format: mgpu::VertexAttributeFormat::Float2,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                    VertexInputDescription {
                        binding: 0,
                        location: 2,
                        stride: std::mem::size_of::<Vertex>(),
                        offset: std::mem::offset_of!(egui::epaint::Vertex, color),
                        format: mgpu::VertexAttributeFormat::Float4Unorm,
                        frequency: VertexInputFrequency::PerVertex,
                    },
                ],
            },
            fragment_stage: Some(&FragmentStageInfo {
                shader: &fs,
                entry_point: "main",
                render_targets: &[RenderTargetInfo {
                    blend: Some(BlendSettings {
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::OneMinusSourceAlpha,
                        color_blend_op: BlendOp::Add,
                        src_alpha_blend_factor: BlendFactor::OneMinusDestAlpha,
                        dst_alpha_blend_factor: BlendFactor::One,
                        alpha_blend_op: BlendOp::Add,
                        write_mask: ColorWriteMask::RGBA,
                    }),
                }],
                depth_stencil_target: None,
            }),
            primitive_restart_enabled: false,
            primitive_topology: PrimitiveTopology::TriangleList,
            polygon_mode: mgpu::PolygonMode::Filled,
            cull_mode: mgpu::CullMode::None,
            front_face: mgpu::FrontFace::ClockWise,
            multisample_state: None,
            depth_stencil_state: mgpu::DepthStencilState::default(),
            binding_set_layouts: &[BindingSetLayoutInfo {
                set: 0,
                layout: Self::binding_set_layout(),
            }],
            push_constant_info: Some(mgpu::PushConstantInfo {
                size: std::mem::size_of::<ScreenData>(),
                visibility: ShaderStageFlags::ALL_GRAPHICS,
            }),
        })?;

        let context = Context::default();
        let pixels_per_point = Self::compute_pixels_per_point(&context, window);
        context.set_fonts(FontDefinitions::default());

        context.set_pixels_per_point(pixels_per_point);
        let viewport_id = context.viewport_id();

        Ok(Self {
            context,
            raw_input: RawInput::default(),
            buffers: mesh_buffers,
            textures: HashMap::new(),
            current_frame: 0,
            sampler,
            vs,
            fs,
            pixels_per_point,
            viewport_id,
            pointer_pos_in_points: None,
            input_method_editor_started: false,
            current_cursor_icon: None,
            clipboard_context: ClipboardContext::new()
                .map_err(|e| anyhow::format_err!("Failed to create Clipboard Context: {e:?}"))?,
            pipeline,
        })
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let time_since_start = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();
        let input = self.take_input(window, time_since_start.as_secs_f64());
        self.context.begin_frame(input);
    }

    pub fn end_frame(&mut self) -> FullOutput {
        self.context.end_frame()
    }
    pub fn context(&self) -> Context {
        self.context.clone()
    }

    pub fn destroy(&mut self, device: &Device) -> anyhow::Result<()> {
        device.destroy_graphics_pipeline(self.pipeline)?;
        for (_, texture) in self.textures.drain() {
            free_texture(device, &texture);
        }
        device.destroy_sampler(self.sampler)?;
        for buffer in self.buffers.drain(0..) {
            device.destroy_buffer(buffer.indices)?;
            device.destroy_buffer(buffer.vertices)?;
        }
        device.destroy_shader_module(self.vs)?;
        device.destroy_shader_module(self.fs)?;

        Ok(())
    }

    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
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
            | WindowEvent::PinchGesture { .. }
            | WindowEvent::PanGesture { .. }
            | WindowEvent::DoubleTapGesture { .. }
            | WindowEvent::RotationGesture { .. } => EventResponse {
                repaint: false,
                consumed: false,
            },
        }
    }

    pub fn paint_frame(
        &mut self,
        device: &Device,
        output: ImageView,
        delta: TexturesDelta,
        shapes: Vec<ClippedShape>,
    ) -> anyhow::Result<()> {
        let mut command_recorder = device.create_command_recorder::<Graphics>();
        let clipped_primitives = self.context.tessellate(shapes, self.pixels_per_point);
        self.update_textures(device, &delta).unwrap();
        self.paint_primitives(device, &mut command_recorder, output, &clipped_primitives)?;
        self.free_textures(device, &delta);

        self.current_frame = (self.current_frame + 1) % device.get_info().frames_in_flight;

        command_recorder.submit()?;

        Ok(())
    }

    fn free_textures(&mut self, device: &Device, delta: &TexturesDelta) {
        for free in &delta.free {
            if let Some(egui_texture) = self.textures.remove(free) {
                free_texture(device, &egui_texture);
            }
        }
    }

    fn paint_primitives(
        &self,
        device: &Device,
        command_recorder: &mut CommandRecorder<Graphics>,
        output: ImageView,
        clipped_primitives: &[egui::ClippedPrimitive],
    ) -> anyhow::Result<()> {
        let mut render_pass = command_recorder.begin_render_pass(&RenderPassDescription {
            label: Some("egui painting"),
            render_targets: &[RenderTarget {
                view: output,
                load_op: mgpu::RenderTargetLoadOp::Load,
                store_op: mgpu::AttachmentStoreOp::Store,
                sample_count: mgpu::SampleCount::One,
            }],
            depth_stencil_attachment: None,
            render_area: Rect2D {
                offset: Default::default(),
                extents: output.extents_2d(),
            },
            flags: Default::default(),
        })?;

        let mut offset_indices = 0;
        let mut offset_vertices = 0i32;
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_primitives
        {
            Self::set_clip_rect(
                &mut render_pass,
                [output.extents_2d().width, output.extents_2d().height],
                self.pixels_per_point,
                *clip_rect,
            );

            match primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    let EguiBuffers { vertices, indices } = self.buffers[self.current_frame];

                    if let Some(texture) = self.get_image_view(mesh.texture_id) {
                        device.write_buffer(
                            vertices,
                            &BufferWriteParams {
                                data: bytemuck::cast_slice(&mesh.vertices),
                                offset: offset_vertices as usize * std::mem::size_of::<Vertex>(),
                                size: mesh.vertices.len() * std::mem::size_of::<Vertex>(),
                            },
                        )?;
                        device.write_buffer(
                            indices,
                            &BufferWriteParams {
                                data: bytemuck::cast_slice(&mesh.indices),
                                offset: offset_indices * std::mem::size_of::<u32>(),
                                size: mesh.indices.len() * std::mem::size_of::<u32>(),
                            },
                        )?;

                        render_pass.set_pipeline(self.pipeline);
                        render_pass.set_binding_sets(&[&texture.binding_set]);
                        render_pass.set_vertex_buffers([self.buffers[self.current_frame].vertices]);
                        render_pass.set_index_buffer(self.buffers[self.current_frame].indices);
                        render_pass.set_push_constant(
                            bytemuck::cast_slice(&[ScreenData {
                                screen_size: [
                                    output.extents_2d().width as f32 / self.pixels_per_point,
                                    output.extents_2d().height as f32 / self.pixels_per_point,
                                ],
                            }]),
                            ShaderStageFlags::ALL_GRAPHICS,
                        );
                        render_pass.draw_indexed(
                            mesh.indices.len() as _,
                            1,
                            offset_indices,
                            offset_vertices,
                            0,
                        )?;
                        offset_indices += mesh.indices.len();
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

    /// Call with the output given by `egui`.
    ///
    /// This will, if needed:
    /// * update the cursor
    /// * copy text to the clipboard
    /// * open any clicked urls
    /// * update the IME
    /// *
    pub fn handle_platform_output(
        &mut self,
        window: &Window,
        platform_output: egui::PlatformOutput,
    ) {
        let egui::PlatformOutput {
            cursor_icon,
            open_url,
            copied_text,
            events: _,                    // handled elsewhere
            mutable_text_under_cursor: _, // only used in eframe web
            ime,
            #[cfg(feature = "accesskit")]
            accesskit_update,
        } = platform_output;

        self.set_cursor_icon(window, cursor_icon);

        if let Some(_open_url) = open_url {
            log::trace!("open_url_in_browser(&open_url.url);");
        }

        if !copied_text.is_empty() {
            log::trace!("Clipboard copy is not supported");
            let _ = self
                .clipboard_context
                .set_contents(copied_text)
                .inspect_err(|e| {
                    log::error!("Clipboard error: {e:?}");
                });
        }

        if let Some(ime) = ime {
            let rect = ime.rect;
            let pixels_per_point = Self::compute_pixels_per_point(&self.context, window);
            window.set_ime_cursor_area(
                winit::dpi::PhysicalPosition {
                    x: pixels_per_point * rect.min.x,
                    y: pixels_per_point * rect.min.y,
                },
                winit::dpi::PhysicalSize {
                    width: pixels_per_point * rect.width(),
                    height: pixels_per_point * rect.height(),
                },
            );
        }
    }

    fn set_clip_rect(
        render_passs: &mut RenderPass,
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
            extents: Extents2D {
                width: (clip_max_x - clip_min_x) as u32,
                height: (clip_max_y - clip_min_y) as u32,
            },
        });
    }

    fn get_image_view(&self, texture_id: TextureId) -> Option<&EguiTexture> {
        self.textures.get(&texture_id)
    }

    fn update_textures(&mut self, device: &Device, delta: &TexturesDelta) -> anyhow::Result<()> {
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

                let image = device
                    .create_image(&ImageDescription {
                        label: Some("egui texture image"),
                        usage_flags: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                        creation_flags: Default::default(),
                        extents: Extents3D {
                            width,
                            height,
                            depth: 1,
                        },
                        dimension: mgpu::ImageDimension::D2,
                        mips: 1.try_into().unwrap(),
                        array_layers: 1.try_into().unwrap(),
                        samples: mgpu::SampleCount::One,
                        format,
                        memory_domain: MemoryDomain::Gpu,
                    })
                    .unwrap();
                let view = device
                    .create_image_view(&mgpu::ImageViewDescription {
                        label: Some("egui texture image view"),
                        image,
                        view_ty: mgpu::ImageViewType::D2,
                        format,
                        aspect: mgpu::ImageAspect::Color,
                        image_subresource: image.whole_subresource(),
                    })
                    .unwrap();

                let binding_set = device
                    .create_binding_set(
                        &BindingSetDescription {
                            label: Some(&format!("egui bs {:?}", id)),
                            bindings: &[
                                Binding {
                                    binding: 0,
                                    ty: mgpu::BindingType::SampledImage { view },
                                    visibility: ShaderStageFlags::ALL_GRAPHICS,
                                },
                                Binding {
                                    binding: 1,
                                    ty: mgpu::BindingType::Sampler(self.sampler),
                                    visibility: ShaderStageFlags::ALL_GRAPHICS,
                                },
                            ],
                        },
                        Self::binding_set_layout(),
                    )
                    .unwrap();

                EguiTexture {
                    image,
                    view,
                    sampler: None,
                    binding_set,
                }
            });

            let [x, y] = delta
                .pos
                .map(|[x, y]| [x as i32, y as i32])
                .unwrap_or_default();
            match &delta.image {
                egui::ImageData::Color(image) => {
                    device.write_image_data(
                        texture.image,
                        &mgpu::ImageWriteParams {
                            data: image.as_raw(),
                            region: ImageRegion {
                                offset: Offset3D { x, y, z: 0 },
                                ..texture.image.whole_region()
                            },
                        },
                    )?;
                }
                egui::ImageData::Font(image) => {
                    let data = image
                        .srgba_pixels(None)
                        .flat_map(|a| a.to_array())
                        .collect::<Vec<u8>>();

                    device.write_image_data(
                        texture.image,
                        &mgpu::ImageWriteParams {
                            data: &data,
                            region: ImageRegion {
                                offset: Offset3D { x, y, z: 0 },
                                ..texture.image.whole_region()
                            },
                        },
                    )?;
                }
            }
        }
        Ok(())
    }

    fn compute_pixels_per_point(context: &Context, window: &Window) -> f32 {
        let native_pixels_per_point = window.scale_factor() as f32;
        let egui_zoom = context.zoom_factor();
        egui_zoom * native_pixels_per_point
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

    fn on_keyboard_input(&mut self, event: &winit::event::KeyEvent) {
        let winit::event::KeyEvent {
            // Represents the position of a key independent of the currently active layout.
            //
            // It also uniquely identifies the physical key (i.e. it's mostly synonymous with a scancode).
            // The most prevalent use case for this is games. For example the default keys for the player
            // to move around might be the W, A, S, and D keys on a US layout. The position of these keys
            // is more important than their label, so they should map to Z, Q, S, and D on an "AZERTY"
            // layout. (This value is `KeyCode::KeyW` for the Z key on an AZERTY layout.)
            physical_key,

            // Represents the results of a keymap, i.e. what character a certain key press represents.
            // When telling users "Press Ctrl-F to find", this is where we should
            // look for the "F" key, because they may have a dvorak layout on
            // a qwerty keyboard, and so the logical "F" character may not be located on the physical `KeyCode::KeyF` position.
            logical_key,

            text,

            state,

            location: _, // e.g. is it on the numpad?
            repeat: _,   // egui will figure this out for us
            ..
        } = event;

        let pressed = *state == winit::event::ElementState::Pressed;

        let physical_key = if let winit::keyboard::PhysicalKey::Code(keycode) = *physical_key {
            key_from_key_code(keycode)
        } else {
            None
        };

        let logical_key = key_from_winit_key(logical_key);

        // Helpful logging to enable when adding new key support
        log::trace!(
            "logical {:?} -> {:?},  physical {:?} -> {:?}",
            event.logical_key,
            logical_key,
            event.physical_key,
            physical_key
        );

        if let Some(logical_key) = logical_key {
            if pressed {
                if is_cut_command(self.raw_input.modifiers, logical_key) {
                    self.raw_input.events.push(egui::Event::Cut);
                    return;
                } else if is_copy_command(self.raw_input.modifiers, logical_key) {
                    self.raw_input.events.push(egui::Event::Copy);
                    return;
                } else if is_paste_command(self.raw_input.modifiers, logical_key) {
                    log::trace!("Paste is currently not supported");
                    if let Ok(contents) = self.clipboard_context.get_contents() {
                        if !contents.is_empty() {
                            self.raw_input.events.push(egui::Event::Paste(contents));
                        }
                    }
                    return;
                }
            }

            self.raw_input.events.push(egui::Event::Key {
                key: logical_key,
                physical_key,
                pressed,
                repeat: false, // egui will fill this in for us!
                modifiers: self.raw_input.modifiers,
            });
        }

        if let Some(text) = &text {
            // Make sure there is text, and that it is not control characters
            // (e.g. delete is sent as "\u{f728}" on macOS).
            if !text.is_empty() && text.chars().all(is_printable_char) {
                // On some platforms we get here when the user presses Cmd-C (copy), ctrl-W, etc.
                // We need to ignore these characters that are side-effects of commands.
                // Also make sure the key is pressed (not released). On Linux, text might
                // contain some data even when the key is released.
                let is_cmd = self.raw_input.modifiers.ctrl
                    || self.raw_input.modifiers.command
                    || self.raw_input.modifiers.mac_cmd;
                if pressed && !is_cmd {
                    self.raw_input
                        .events
                        .push(egui::Event::Text(text.to_string()));
                }
            }
        }
    }

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

    fn take_input(&mut self, window: &Window, time_since_start: f64) -> RawInput {
        let mut input = self.raw_input.take();
        input.time = Some(time_since_start as f64);

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

    fn set_cursor_icon(&mut self, window: &Window, cursor_icon: egui::CursorIcon) {
        if self.current_cursor_icon == Some(cursor_icon) {
            // Prevent flickering near frame boundary when Windows OS tries to control cursor icon for window resizing.
            // On other platforms: just early-out to save CPU.
            return;
        }

        let is_pointer_in_window = self.pointer_pos_in_points.is_some();
        if is_pointer_in_window {
            self.current_cursor_icon = Some(cursor_icon);

            if let Some(winit_cursor_icon) = translate_cursor(cursor_icon) {
                window.set_cursor_visible(true);
                window.set_cursor(winit_cursor_icon);
            } else {
                window.set_cursor_visible(false);
            }
        } else {
            // Remember to set the cursor again once the cursor returns to the screen:
            self.current_cursor_icon = None;
        }
    }

    fn binding_set_layout() -> &'static BindingSetLayout<'static> {
        const BS: BindingSetLayout = BindingSetLayout {
            binding_set_elements: &[
                BindingSetElement {
                    binding: 0,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::SampledImage,
                    shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
                },
                BindingSetElement {
                    binding: 1,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::Sampler,
                    shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
                },
            ],
        };

        &BS
    }
}

fn free_texture(device: &Device, egui_texture: &EguiTexture) {
    device
        .destroy_binding_set(egui_texture.binding_set.clone())
        .unwrap();
    device.destroy_image_view(egui_texture.view).unwrap();
    device.destroy_image(egui_texture.image).unwrap();

    if let Some(sampler) = egui_texture.sampler {
        device.destroy_sampler(sampler).unwrap();
    }
}

fn is_printable_char(chr: char) -> bool {
    let is_in_private_use_area = ('\u{e000}'..='\u{f8ff}').contains(&chr)
        || ('\u{f0000}'..='\u{ffffd}').contains(&chr)
        || ('\u{100000}'..='\u{10fffd}').contains(&chr);

    !is_in_private_use_area && !chr.is_ascii_control()
}

fn is_cut_command(modifiers: egui::Modifiers, keycode: egui::Key) -> bool {
    keycode == egui::Key::Cut
        || (modifiers.command && keycode == egui::Key::X)
        || (cfg!(target_os = "windows") && modifiers.shift && keycode == egui::Key::Delete)
}

fn is_copy_command(modifiers: egui::Modifiers, keycode: egui::Key) -> bool {
    keycode == egui::Key::Copy
        || (modifiers.command && keycode == egui::Key::C)
        || (cfg!(target_os = "windows") && modifiers.ctrl && keycode == egui::Key::Insert)
}

fn is_paste_command(modifiers: egui::Modifiers, keycode: egui::Key) -> bool {
    keycode == egui::Key::Paste
        || (modifiers.command && keycode == egui::Key::V)
        || (cfg!(target_os = "windows") && modifiers.shift && keycode == egui::Key::Insert)
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

fn key_from_key_code(key: winit::keyboard::KeyCode) -> Option<egui::Key> {
    use egui::Key;
    use winit::keyboard::KeyCode;

    Some(match key {
        KeyCode::ArrowDown => Key::ArrowDown,
        KeyCode::ArrowLeft => Key::ArrowLeft,
        KeyCode::ArrowRight => Key::ArrowRight,
        KeyCode::ArrowUp => Key::ArrowUp,

        KeyCode::Escape => Key::Escape,
        KeyCode::Tab => Key::Tab,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Enter | KeyCode::NumpadEnter => Key::Enter,

        KeyCode::Insert => Key::Insert,
        KeyCode::Delete => Key::Delete,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,

        // Punctuation
        KeyCode::Space => Key::Space,
        KeyCode::Comma => Key::Comma,
        KeyCode::Period => Key::Period,
        // KeyCode::Colon => Key::Colon, // NOTE: there is no physical colon key on an american keyboard
        KeyCode::Semicolon => Key::Semicolon,
        KeyCode::Backslash => Key::Backslash,
        KeyCode::Slash | KeyCode::NumpadDivide => Key::Slash,
        KeyCode::BracketLeft => Key::OpenBracket,
        KeyCode::BracketRight => Key::CloseBracket,
        KeyCode::Backquote => Key::Backtick,

        KeyCode::Cut => Key::Cut,
        KeyCode::Copy => Key::Copy,
        KeyCode::Paste => Key::Paste,
        KeyCode::Minus | KeyCode::NumpadSubtract => Key::Minus,
        KeyCode::NumpadAdd => Key::Plus,
        KeyCode::Equal => Key::Equals,

        KeyCode::Digit0 | KeyCode::Numpad0 => Key::Num0,
        KeyCode::Digit1 | KeyCode::Numpad1 => Key::Num1,
        KeyCode::Digit2 | KeyCode::Numpad2 => Key::Num2,
        KeyCode::Digit3 | KeyCode::Numpad3 => Key::Num3,
        KeyCode::Digit4 | KeyCode::Numpad4 => Key::Num4,
        KeyCode::Digit5 | KeyCode::Numpad5 => Key::Num5,
        KeyCode::Digit6 | KeyCode::Numpad6 => Key::Num6,
        KeyCode::Digit7 | KeyCode::Numpad7 => Key::Num7,
        KeyCode::Digit8 | KeyCode::Numpad8 => Key::Num8,
        KeyCode::Digit9 | KeyCode::Numpad9 => Key::Num9,

        KeyCode::KeyA => Key::A,
        KeyCode::KeyB => Key::B,
        KeyCode::KeyC => Key::C,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G,
        KeyCode::KeyH => Key::H,
        KeyCode::KeyI => Key::I,
        KeyCode::KeyJ => Key::J,
        KeyCode::KeyK => Key::K,
        KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M,
        KeyCode::KeyN => Key::N,
        KeyCode::KeyO => Key::O,
        KeyCode::KeyP => Key::P,
        KeyCode::KeyQ => Key::Q,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyT => Key::T,
        KeyCode::KeyU => Key::U,
        KeyCode::KeyV => Key::V,
        KeyCode::KeyW => Key::W,
        KeyCode::KeyX => Key::X,
        KeyCode::KeyY => Key::Y,
        KeyCode::KeyZ => Key::Z,

        KeyCode::F1 => Key::F1,
        KeyCode::F2 => Key::F2,
        KeyCode::F3 => Key::F3,
        KeyCode::F4 => Key::F4,
        KeyCode::F5 => Key::F5,
        KeyCode::F6 => Key::F6,
        KeyCode::F7 => Key::F7,
        KeyCode::F8 => Key::F8,
        KeyCode::F9 => Key::F9,
        KeyCode::F10 => Key::F10,
        KeyCode::F11 => Key::F11,
        KeyCode::F12 => Key::F12,
        KeyCode::F13 => Key::F13,
        KeyCode::F14 => Key::F14,
        KeyCode::F15 => Key::F15,
        KeyCode::F16 => Key::F16,
        KeyCode::F17 => Key::F17,
        KeyCode::F18 => Key::F18,
        KeyCode::F19 => Key::F19,
        KeyCode::F20 => Key::F20,

        _ => {
            return None;
        }
    })
}
fn key_from_winit_key(key: &winit::keyboard::Key) -> Option<egui::Key> {
    match key {
        winit::keyboard::Key::Named(named_key) => key_from_named_key(*named_key),
        winit::keyboard::Key::Character(str) => egui::Key::from_name(str.as_str()),
        winit::keyboard::Key::Unidentified(_) | winit::keyboard::Key::Dead(_) => None,
    }
}

fn key_from_named_key(named_key: winit::keyboard::NamedKey) -> Option<egui::Key> {
    use egui::Key;
    use winit::keyboard::NamedKey;

    Some(match named_key {
        NamedKey::Enter => Key::Enter,
        NamedKey::Tab => Key::Tab,
        NamedKey::ArrowDown => Key::ArrowDown,
        NamedKey::ArrowLeft => Key::ArrowLeft,
        NamedKey::ArrowRight => Key::ArrowRight,
        NamedKey::ArrowUp => Key::ArrowUp,
        NamedKey::End => Key::End,
        NamedKey::Home => Key::Home,
        NamedKey::PageDown => Key::PageDown,
        NamedKey::PageUp => Key::PageUp,
        NamedKey::Backspace => Key::Backspace,
        NamedKey::Delete => Key::Delete,
        NamedKey::Insert => Key::Insert,
        NamedKey::Escape => Key::Escape,
        NamedKey::Cut => Key::Cut,
        NamedKey::Copy => Key::Copy,
        NamedKey::Paste => Key::Paste,

        NamedKey::Space => Key::Space,

        NamedKey::F1 => Key::F1,
        NamedKey::F2 => Key::F2,
        NamedKey::F3 => Key::F3,
        NamedKey::F4 => Key::F4,
        NamedKey::F5 => Key::F5,
        NamedKey::F6 => Key::F6,
        NamedKey::F7 => Key::F7,
        NamedKey::F8 => Key::F8,
        NamedKey::F9 => Key::F9,
        NamedKey::F10 => Key::F10,
        NamedKey::F11 => Key::F11,
        NamedKey::F12 => Key::F12,
        NamedKey::F13 => Key::F13,
        NamedKey::F14 => Key::F14,
        NamedKey::F15 => Key::F15,
        NamedKey::F16 => Key::F16,
        NamedKey::F17 => Key::F17,
        NamedKey::F18 => Key::F18,
        NamedKey::F19 => Key::F19,
        NamedKey::F20 => Key::F20,

        _ => {
            log::trace!("Unknown key: {named_key:?}");
            return None;
        }
    })
}

fn translate_cursor(cursor_icon: egui::CursorIcon) -> Option<winit::window::CursorIcon> {
    match cursor_icon {
        egui::CursorIcon::None => None,

        egui::CursorIcon::Alias => Some(winit::window::CursorIcon::Alias),
        egui::CursorIcon::AllScroll => Some(winit::window::CursorIcon::AllScroll),
        egui::CursorIcon::Cell => Some(winit::window::CursorIcon::Cell),
        egui::CursorIcon::ContextMenu => Some(winit::window::CursorIcon::ContextMenu),
        egui::CursorIcon::Copy => Some(winit::window::CursorIcon::Copy),
        egui::CursorIcon::Crosshair => Some(winit::window::CursorIcon::Crosshair),
        egui::CursorIcon::Default => Some(winit::window::CursorIcon::Default),
        egui::CursorIcon::Grab => Some(winit::window::CursorIcon::Grab),
        egui::CursorIcon::Grabbing => Some(winit::window::CursorIcon::Grabbing),
        egui::CursorIcon::Help => Some(winit::window::CursorIcon::Help),
        egui::CursorIcon::Move => Some(winit::window::CursorIcon::Move),
        egui::CursorIcon::NoDrop => Some(winit::window::CursorIcon::NoDrop),
        egui::CursorIcon::NotAllowed => Some(winit::window::CursorIcon::NotAllowed),
        egui::CursorIcon::PointingHand => Some(winit::window::CursorIcon::Pointer),
        egui::CursorIcon::Progress => Some(winit::window::CursorIcon::Progress),

        egui::CursorIcon::ResizeHorizontal => Some(winit::window::CursorIcon::EwResize),
        egui::CursorIcon::ResizeNeSw => Some(winit::window::CursorIcon::NeswResize),
        egui::CursorIcon::ResizeNwSe => Some(winit::window::CursorIcon::NwseResize),
        egui::CursorIcon::ResizeVertical => Some(winit::window::CursorIcon::NsResize),

        egui::CursorIcon::ResizeEast => Some(winit::window::CursorIcon::EResize),
        egui::CursorIcon::ResizeSouthEast => Some(winit::window::CursorIcon::SeResize),
        egui::CursorIcon::ResizeSouth => Some(winit::window::CursorIcon::SResize),
        egui::CursorIcon::ResizeSouthWest => Some(winit::window::CursorIcon::SwResize),
        egui::CursorIcon::ResizeWest => Some(winit::window::CursorIcon::WResize),
        egui::CursorIcon::ResizeNorthWest => Some(winit::window::CursorIcon::NwResize),
        egui::CursorIcon::ResizeNorth => Some(winit::window::CursorIcon::NResize),
        egui::CursorIcon::ResizeNorthEast => Some(winit::window::CursorIcon::NeResize),
        egui::CursorIcon::ResizeColumn => Some(winit::window::CursorIcon::ColResize),
        egui::CursorIcon::ResizeRow => Some(winit::window::CursorIcon::RowResize),

        egui::CursorIcon::Text => Some(winit::window::CursorIcon::Text),
        egui::CursorIcon::VerticalText => Some(winit::window::CursorIcon::VerticalText),
        egui::CursorIcon::Wait => Some(winit::window::CursorIcon::Wait),
        egui::CursorIcon::ZoomIn => Some(winit::window::CursorIcon::ZoomIn),
        egui::CursorIcon::ZoomOut => Some(winit::window::CursorIcon::ZoomOut),
    }
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
