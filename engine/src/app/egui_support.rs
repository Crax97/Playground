use std::{collections::HashMap, sync::Arc, time::SystemTime};

use bytemuck::{Pod, Zeroable};
use egui::{
    pos2, vec2, Context, FontDefinitions, FullOutput, RawInput, Rect, Sense, TextureId,
    TexturesDelta, Ui,
};
use engine_macros::glsl;
use gpu::{
    render_pass_2::RenderPass2, BeginRenderPassInfo2, Binding2, BufferCreateInfo, BufferHandle,
    BufferUsageFlags, ColorAttachment, CommandBuffer, ComponentMapping, Extent2D, Gpu,
    ImageAspectFlags, ImageCreateInfo, ImageFormat, ImageHandle, ImageSubresourceRange,
    ImageUsageFlags, ImageViewCreateInfo, ImageViewHandle, MemoryDomain, Offset2D, Rect2D,
    SamplerCreateInfo, SamplerHandle, ShaderModuleCreateInfo, ShaderModuleHandle, ShaderStage,
    Swapchain, VertexBindingInfo,
};

use log::warn;
use winit::{event::WindowEvent, window::Window};

use crate::{Backbuffer, CvarManager};

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

pub struct EguiSupport {
    context: Context,
    raw_input: RawInput,
    default_sampler: SamplerHandle,
    mesh_buffers: Vec<EguiBuffers>,
    textures: HashMap<TextureId, EguiTexture>,
    current_frame: usize,
    pixels_per_point: f32,

    vs: ShaderModuleHandle,
    fs: ShaderModuleHandle,
}

impl EguiSupport {
    const MAX_NUM_VERTICES: usize = 65535;
    const MAX_VERTICES: usize =
        Self::MAX_NUM_VERTICES * std::mem::size_of::<egui::epaint::Vertex>();
    const MAX_INDICES: usize = Self::MAX_NUM_VERTICES * std::mem::size_of::<u32>();

    pub fn new(window: &Window, gpu: &Arc<dyn Gpu>) -> anyhow::Result<Self> {
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
        context.set_fonts(FontDefinitions::default());

        let pixels_per_point = Self::compute_pixels_per_point(&context, window);
        context.set_pixels_per_point(pixels_per_point);

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

    pub fn handle_event(&mut self, winit_event: &WindowEvent) {
        let mut new_input = RawInput::default();

        self.raw_input = new_input;
    }

    pub fn swapchain_updated(&mut self, swapchain: &Swapchain) {}

    pub fn begin_frame(&mut self, window: &Window) {
        self.context.begin_frame(self.raw_input.clone());
    }

    pub fn end_frame(&mut self, window: &Window) -> FullOutput {
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
                y: height_px as i32 - clip_max_y,
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
}

fn free_texture(gpu: &dyn Gpu, egui_texture: &EguiTexture) {
    gpu.destroy_image_view(egui_texture.view);
    gpu.destroy_image(egui_texture.image);

    if let Some(sampler) = egui_texture.sampler {
        gpu.destroy_sampler(sampler);
    }
}
