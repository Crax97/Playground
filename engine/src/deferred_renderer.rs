use engine_macros::glsl;
use std::{collections::HashMap, mem::size_of};

use gpu::{
    AttachmentStoreOp, BindingType, BufferCreateInfo, BufferUsageFlags, ColorComponentFlags,
    ColorLoadOp, CompareOp, DepthStencilState, Extent2D, FragmentStageInfo, ImageAspectFlags,
    ImageCreateInfo, ImageFormat, ImageLayout, ImageUsageFlags, IndexType, MemoryDomain,
    PushConstantRange, SampleCount, ShaderModuleCreateInfo, ShaderStage, StencilLoadOp,
    StencilOpState, Swapchain, VertexStageInfo, VkBuffer, VkCommandBuffer, VkGpu, VkImage,
    VkImageView, VkShaderModule,
};
use nalgebra::{vector, Matrix4, Point4, Vector2, Vector3, Vector4};
use resource_map::{ResourceHandle, ResourceMap};

const FXAA_FS: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/fxaa_fs.frag",
    entry_point = "main"
);

const FXAA_VS: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/fxaa_vs.vert",
    entry_point = "main"
);

const SHADOW_MAP_WIDTH: u32 = 7680;
const SHADOW_MAP_HEIGHT: u32 = 4320;

#[repr(C)]
#[derive(Clone, Copy)]
struct FxaaShaderParams {
    rcp_frame: Vector2<f32>,
    fxaa_quality_subpix: f32,
    fxaa_quality_edge_threshold: f32,
    fxaa_quality_edge_threshold_min: f32,
    iterations: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ObjectDrawInfo {
    model: Matrix4<f32>,
    camera_index: u32,
}

#[derive(Clone, Copy)]
pub struct FxaaSettings {
    pub fxaa_quality_subpix: f32,
    pub fxaa_quality_edge_threshold: f32,
    pub fxaa_quality_edge_threshold_min: f32,
    pub iterations: u32,
}

impl Default for FxaaSettings {
    fn default() -> Self {
        Self {
            fxaa_quality_subpix: 0.75,
            fxaa_quality_edge_threshold: 0.166,
            fxaa_quality_edge_threshold_min: 0.0833,
            iterations: 12,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameData {
    eye: Point4<f32>,
    view: nalgebra::Matrix4<f32>,
    projection: nalgebra::Matrix4<f32>,
    viewport_size_offset: Vector4<f32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuLightInfo {
    position_radius: Vector4<f32>,
    direction: Vector4<f32>,
    color: Vector4<f32>,
    extras: Vector4<f32>,
    ty_shadowcaster: [i32; 4],
}

impl From<&Light> for GpuLightInfo {
    fn from(light: &Light) -> Self {
        let (direction, extras, ty) = match light.ty {
            LightType::Point => (Default::default(), Default::default(), 0),
            LightType::Directional { direction, .. } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                Default::default(),
                1,
            ),
            LightType::Spotlight {
                direction,
                inner_cone_degrees,
                outer_cone_degrees,
            } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                vector![
                    (90.0 - inner_cone_degrees).to_radians(),
                    (90.0 - outer_cone_degrees).to_radians(),
                    0.0,
                    0.0
                ],
                2,
            ),
            LightType::Rect {
                direction,
                width,
                height,
            } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                vector![width, height, 0.0, 0.0],
                3,
            ),
        };
        Self {
            position_radius: vector![
                light.position.x,
                light.position.y,
                light.position.z,
                light.radius
            ],
            color: vector![light.color.x, light.color.y, light.color.z, light.intensity],
            direction,
            extras,
            ty_shadowcaster: [ty, -1, 0, 0],
        }
    }
}

use crate::{
    app_state, app_state_mut,
    camera::Camera,
    material::{MasterMaterial, MasterMaterialDescription},
    Backbuffer, BufferDescription, BufferType, ClearValue, FragmentState, GpuRunner,
    GraphRunContext, Image2DInfo, ImageArrayInfo, ImageDescription, ImageViewDescription, Light,
    LightType, MaterialDescription, MaterialDomain, MaterialInstance, MeshPrimitive, ModuleInfo,
    PipelineTarget, RenderGraph, RenderGraphPipelineDescription, RenderPassContext, RenderStage,
    RenderingPipeline, SamplerState, Scene,
};

use gpu::{BlendMode, BlendOp, BlendState, RenderPassAttachment};

struct FrameBuffers {
    camera_buffer: VkBuffer,
    light_buffer: VkBuffer,
}

struct DrawCall<'a> {
    prim: &'a MeshPrimitive,
    transform: Matrix4<f32>,
    material: ResourceHandle<MaterialInstance>,
}

pub struct DeferredRenderingPipeline {
    frame_buffers: Vec<FrameBuffers>,
    render_graph: RenderGraph,
    screen_quad: VkShaderModule,
    texture_copy: VkShaderModule,
    gbuffer_combine: VkShaderModule,
    tonemap_fs: VkShaderModule,

    fxaa_settings: FxaaSettings,

    runner: GpuRunner,
    fxaa_vs: VkShaderModule,
    fxaa_fs: VkShaderModule,
    in_flight_frame: usize,
    max_frames_in_flight: usize,
    light_iteration: u64,
    active_lights: Vec<GpuLightInfo>,
    light_povs: Vec<PerFrameData>,
    shadow_map: VkImage,
    shadow_map_view: VkImageView,

    pub depth_bias_constant: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope: f32,

    pub ambient_color: Vector3<f32>,
    pub ambient_intensity: f32,
}

impl DeferredRenderingPipeline {
    pub fn new(
        gpu: &VkGpu,
        screen_quad: VkShaderModule,
        gbuffer_combine: VkShaderModule,
        texture_copy: VkShaderModule,
        tonemap_fs: VkShaderModule,
    ) -> anyhow::Result<Self> {
        let mut frame_buffers = vec![];
        for _ in 0..Swapchain::MAX_FRAMES_IN_FLIGHT {
            let camera_buffer = {
                let create_info = BufferCreateInfo {
                    label: Some("Deferred Renderer - Camera buffer"),
                    size: std::mem::size_of::<PerFrameData>() * 100,
                    usage: BufferUsageFlags::STORAGE_BUFFER
                        | BufferUsageFlags::UNIFORM_BUFFER
                        | BufferUsageFlags::TRANSFER_DST,
                };
                gpu.create_buffer(
                    &create_info,
                    MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
                )?
            };
            let light_buffer = {
                let create_info = BufferCreateInfo {
                    label: Some("Light Buffer"),
                    size: std::mem::size_of::<GpuLightInfo>() * 1000,
                    usage: BufferUsageFlags::UNIFORM_BUFFER
                        | BufferUsageFlags::STORAGE_BUFFER
                        | BufferUsageFlags::TRANSFER_DST,
                };
                gpu.create_buffer(
                    &create_info,
                    MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
                )?
            };
            frame_buffers.push(FrameBuffers {
                camera_buffer,
                light_buffer,
            })
        }

        let shadow_map = gpu.create_image(
            &ImageCreateInfo {
                label: Some("Shadow Map image"),
                width: SHADOW_MAP_WIDTH,
                height: SHADOW_MAP_HEIGHT,
                format: ImageFormat::Depth,
                usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
            },
            MemoryDomain::DeviceLocal,
            None,
        )?;
        let shadow_map_view = gpu.create_image_view(&gpu::ImageViewCreateInfo {
            image: &shadow_map,
            view_type: gpu::ImageViewType::Type2D,
            format: ImageFormat::Depth,
            components: gpu::ComponentMapping::default(),
            subresource_range: gpu::ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        })?;

        let render_graph = RenderGraph::new();

        let fxaa_vs = gpu.create_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(FXAA_VS),
        })?;
        let fxaa_fs = gpu.create_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(FXAA_FS),
        })?;

        Ok(Self {
            render_graph,
            screen_quad,
            gbuffer_combine,
            texture_copy,
            frame_buffers,
            tonemap_fs,
            fxaa_vs,
            fxaa_fs,
            fxaa_settings: Default::default(),
            light_iteration: 0,
            runner: GpuRunner::new(),
            in_flight_frame: 0,
            max_frames_in_flight: Swapchain::MAX_FRAMES_IN_FLIGHT,
            depth_bias_constant: 1.25,
            depth_bias_clamp: 0.0,
            depth_bias_slope: 1.75,
            ambient_color: vector![1.0, 1.0, 1.0],
            ambient_intensity: 0.3,
            active_lights: vec![],
            light_povs: vec![],
            shadow_map,
            shadow_map_view,
        })
    }

    pub fn fxaa_settings(&self) -> FxaaSettings {
        self.fxaa_settings
    }
    pub fn fxaa_settings_mut(&mut self) -> &mut FxaaSettings {
        &mut self.fxaa_settings
    }
    pub fn set_fxaa_settings_mut(&mut self, settings: FxaaSettings) {
        self.fxaa_settings = settings;
    }

    fn main_render_loop(
        resource_map: &ResourceMap,
        pipeline_target: PipelineTarget,
        draw_hashmap: &HashMap<&MasterMaterial, Vec<DrawCall>>,
        camera_index: u32,
        ctx: &mut RenderPassContext,
    ) {
        let mut total_primitives_rendered = 0;
        for (master, material_draw_calls) in draw_hashmap.iter() {
            {
                let pipeline = master
                    .get_pipeline(pipeline_target)
                    .expect("failed to fetch pipeline {pipeline_target:?}");
                ctx.render_pass_command.bind_pipeline(pipeline);
                ctx.render_pass_command.bind_descriptor_sets(
                    pipeline,
                    0,
                    &[ctx.read_descriptor_set.expect("No descriptor set???")],
                );

                for (idx, draw_call) in material_draw_calls.iter().enumerate() {
                    let material = &draw_call.material;
                    let material = resource_map.get(material);
                    let primitive_label = ctx.render_pass_command.begin_debug_region(
                        &format!(
                            "{} - {}, total primitives rendered {total_primitives_rendered}",
                            material.name, idx
                        ),
                        [0.0, 0.3, 0.4, 1.0],
                    );
                    ctx.render_pass_command.bind_descriptor_sets(
                        pipeline,
                        1,
                        &[&material.user_descriptor_set],
                    );
                    ctx.render_pass_command.bind_index_buffer(
                        &draw_call.prim.index_buffer,
                        0,
                        IndexType::Uint32,
                    );
                    ctx.render_pass_command.bind_vertex_buffer(
                        0,
                        &[
                            &draw_call.prim.position_component,
                            &draw_call.prim.color_component,
                            &draw_call.prim.normal_component,
                            &draw_call.prim.tangent_component,
                            &draw_call.prim.uv_component,
                        ],
                        &[0, 0, 0, 0, 0],
                    );
                    ctx.render_pass_command.push_constant(
                        pipeline,
                        &ObjectDrawInfo {
                            model: draw_call.transform,
                            camera_index,
                        },
                        0,
                    );
                    ctx.render_pass_command
                        .draw_indexed(draw_call.prim.index_count, 1, 0, 0, 0);

                    primitive_label.end();
                    total_primitives_rendered += 1;
                }
                ctx.render_pass_command.insert_debug_label(
                    &format!("Total primtives drawn this frame: {total_primitives_rendered}"),
                    [0.0, 0.3, 0.4, 1.0],
                );
            }
        }
    }

    fn generate_draw_calls<'r, 's>(
        resource_map: &'r ResourceMap,
        scene: &'s Scene,
    ) -> HashMap<&'s MasterMaterial, Vec<DrawCall<'s>>>
    where
        'r: 's,
    {
        let mut draw_hashmap: HashMap<&MasterMaterial, Vec<DrawCall>> = HashMap::new();

        for primitive in scene.primitives.iter() {
            let mesh = resource_map.get(&primitive.mesh);
            for (idx, mesh_prim) in mesh.primitives.iter().enumerate() {
                let material_handle = primitive.materials[idx].clone();
                let material = resource_map.get(&material_handle);
                let master = resource_map.get(&material.owner);
                draw_hashmap.entry(master).or_default().push(DrawCall {
                    prim: mesh_prim,
                    transform: primitive.transform,
                    material: material_handle,
                });
            }
        }
        draw_hashmap
    }

    fn update_lights(&mut self, scene: &Scene) {
        self.light_iteration = scene.lights_iteration();
        self.active_lights.clear();
        self.light_povs.clear();

        let mut x = 0.0;
        let mut y = 0.0;
        let mut shadow_caster_idx = 0;
        let collected_active_lights: Vec<GpuLightInfo> = scene
            .all_enabled_lights()
            .enumerate()
            .map(|(_, l)| {
                let mut light: GpuLightInfo = l.into();
                if l.shadow_setup.is_some() {
                    light.ty_shadowcaster[1] = shadow_caster_idx;
                    let povs = l
                        .shadow_view_matrices()
                        .iter()
                        .map(|v| {
                            let w = l.shadow_setup.unwrap().width as f32;
                            let h = l.shadow_setup.unwrap().height as f32;

                            let pfd = Some(PerFrameData {
                                eye: Point4::new(l.position.x, l.position.y, l.position.z, 0.0),
                                view: crate::utils::constants::MATRIX_COORDINATE_X_FLIP * v,
                                projection: l.projection_matrix(),
                                viewport_size_offset: vector![x, y, w, h],
                            });

                            x += w;
                            if x > SHADOW_MAP_WIDTH as f32 {
                                if (y + h) <= SHADOW_MAP_HEIGHT as f32 {
                                    x = 0.0;
                                    y += h;
                                } else {
                                    return None;
                                }
                            }
                            shadow_caster_idx += 1;
                            pfd
                        })
                        .take_while(|l| l.is_some())
                        .flatten()
                        .collect::<Vec<_>>();
                    self.light_povs.extend(povs);
                }
                light
            })
            .collect();
        self.active_lights = collected_active_lights;
    }
}

impl RenderingPipeline for DeferredRenderingPipeline {
    fn render(
        &mut self,
        pov: &Camera,
        scene: &Scene,
        backbuffer: &Backbuffer,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<VkCommandBuffer> {
        let projection = pov.projection();

        if self.light_iteration != scene.lights_iteration() {
            self.update_lights(scene);
        }

        let current_buffers = &self.frame_buffers[self.in_flight_frame];

        self.in_flight_frame = (1 + self.in_flight_frame) % self.max_frames_in_flight;

        let mut per_frame_data = vec![PerFrameData {
            eye: Point4::new(pov.location[0], pov.location[1], pov.location[2], 0.0),
            view: crate::utils::constants::MATRIX_COORDINATE_X_FLIP * pov.view(),
            projection,
            viewport_size_offset: vector![
                0.0,
                0.0,
                backbuffer.size.width as f32,
                backbuffer.size.height as f32
            ],
        }];

        per_frame_data.extend_from_slice(&self.light_povs);

        super::app_state()
            .gpu
            .write_buffer_data_with_offset(
                &current_buffers.camera_buffer,
                0,
                &[per_frame_data.len() as u32 - 1],
            )
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer_data_with_offset(
                &current_buffers.camera_buffer,
                size_of::<u32>() as u64 * 4,
                &per_frame_data,
            )
            .unwrap();

        let ambient = vector![
            self.ambient_color.x,
            self.ambient_color.y,
            self.ambient_color.z,
            self.ambient_intensity
        ];

        super::app_state()
            .gpu
            .write_buffer_data_with_offset(&current_buffers.light_buffer, 0, &[ambient])
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer_data_with_offset(
                &current_buffers.light_buffer,
                std::mem::size_of::<Vector4<f32>>() as _,
                &[self.active_lights.len() as u32],
            )
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer_data_with_offset(
                &current_buffers.light_buffer,
                std::mem::size_of::<Vector4<f32>>() as u64 + size_of::<u32>() as u64 * 4,
                &self.active_lights,
            )
            .unwrap();

        app_state_mut().begin_frame()?;

        let draw_hashmap = Self::generate_draw_calls(resource_map, scene);

        //#region render graph resources
        let framebuffer_rgba_desc = ImageDescription {
            view_description: ImageViewDescription::Image2D {
                info: Image2DInfo {
                    width: backbuffer.size.width,
                    height: backbuffer.size.height,
                    present: false,
                },
            },
            format: ImageFormat::Rgba8,
            samples: 1,
            clear_value: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
            sampler_state: None,
        };
        let framebuffer_normal_desc = ImageDescription {
            view_description: ImageViewDescription::Image2D {
                info: Image2DInfo {
                    height: backbuffer.size.height,
                    width: backbuffer.size.width,
                    present: false,
                },
            },

            format: ImageFormat::Rgba8,
            samples: 1,

            clear_value: ClearValue::Color([0.5, 0.5, 0.5, 1.0]),
            sampler_state: None,
        };
        let framebuffer_vector_desc = ImageDescription {
            view_description: ImageViewDescription::Image2D {
                info: Image2DInfo {
                    height: backbuffer.size.height,
                    width: backbuffer.size.width,
                    present: false,
                },
            },

            format: ImageFormat::RgbaFloat32,
            samples: 1,

            clear_value: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
            sampler_state: None,
        };
        let framebuffer_depth_desc = ImageDescription {
            view_description: ImageViewDescription::Image2D {
                info: Image2DInfo {
                    height: backbuffer.size.height,
                    width: backbuffer.size.width,
                    present: false,
                },
            },

            format: ImageFormat::Depth,
            samples: 1,

            clear_value: ClearValue::Depth(1.0),
            sampler_state: None,
        };
        let shadow_map_desc = ImageDescription {
            view_description: ImageViewDescription::Image2D {
                info: Image2DInfo {
                    height: SHADOW_MAP_HEIGHT,
                    width: SHADOW_MAP_WIDTH,
                    present: false,
                },
            },

            format: ImageFormat::Depth,
            samples: 1,

            clear_value: ClearValue::Depth(1.0),
            sampler_state: Some(SamplerState {
                compare_op: Some(gpu::CompareOp::LessEqual),
            }),
        };

        let _shadow_2d_array = ImageDescription {
            format: ImageFormat::Depth,
            view_description: ImageViewDescription::Array {
                info: ImageArrayInfo {
                    format: gpu::ImageViewType::Type2D,
                },
            },
            samples: 1,

            clear_value: ClearValue::Depth(1.0),
            sampler_state: Some(SamplerState {
                compare_op: Some(gpu::CompareOp::LessEqual),
            }),
        };
        let _shadow_cube_array = ImageDescription {
            format: ImageFormat::Depth,
            view_description: ImageViewDescription::Array {
                info: ImageArrayInfo {
                    format: gpu::ImageViewType::Cube,
                },
            },
            samples: 1,

            clear_value: ClearValue::Depth(1.0),
            sampler_state: Some(SamplerState {
                compare_op: Some(gpu::CompareOp::LessEqual),
            }),
        };
        let framebuffer_swapchain_desc = ImageDescription {
            view_description: ImageViewDescription::Image2D {
                info: Image2DInfo {
                    height: backbuffer.size.height,
                    width: backbuffer.size.width,
                    present: false,
                },
            },

            format: backbuffer.format.into(),
            samples: 1,

            clear_value: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
            sampler_state: None,
        };

        let camera_buffer = self.render_graph.use_buffer(
            "camera-buffer",
            &BufferDescription {
                length: std::mem::size_of::<PerFrameData>() as u64,
                ty: BufferType::Storage,
            },
            true,
        )?;

        let light_buffer = self.render_graph.use_buffer(
            "light-buffer",
            &BufferDescription {
                length: std::mem::size_of::<PerFrameData>() as u64,
                ty: BufferType::Storage,
            },
            true,
        )?;

        let swapchain_image =
            self.render_graph
                .use_image("swapchain", &framebuffer_swapchain_desc, true)?;
        let depth_target =
            self.render_graph
                .use_image("depth-buffer", &framebuffer_depth_desc, false)?;
        let shadow_map = self
            .render_graph
            .use_image("shadow_map", &shadow_map_desc, true)?;
        let color_target =
            self.render_graph
                .use_image("color-buffer", &framebuffer_vector_desc, false)?;
        let tonemap_output =
            self.render_graph
                .use_image("tonemap-buffer", &framebuffer_rgba_desc, false)?;
        let fxaa_output =
            self.render_graph
                .use_image("fxaa-buffer", &framebuffer_rgba_desc, false)?;

        let position_target =
            self.render_graph
                .use_image("position-buffer", &framebuffer_vector_desc, false)?;
        let normal_target =
            self.render_graph
                .use_image("normal_buffer", &framebuffer_normal_desc, false)?;
        let diffuse_target =
            self.render_graph
                .use_image("diffuse_buffer", &framebuffer_rgba_desc, false)?;
        let emissive_target =
            self.render_graph
                .use_image("emissive_buffer", &framebuffer_rgba_desc, false)?;
        let pbr_target =
            self.render_graph
                .use_image("pbr_buffer", &framebuffer_rgba_desc, false)?;

        self.render_graph.persist_resource(&swapchain_image);

        let dbuffer_pass = self
            .render_graph
            .begin_render_pass("EarlyZPass", backbuffer.size)?
            .writes_attachments(&[depth_target])
            .shader_reads(&[camera_buffer])
            .mark_external()
            .commit();
        let shadow_map_pass = self
            .render_graph
            .begin_render_pass(
                "ShadowMapRendering",
                Extent2D {
                    width: SHADOW_MAP_WIDTH,
                    height: SHADOW_MAP_HEIGHT,
                },
            )?
            .writes_attachments(&[shadow_map])
            .shader_reads(&[camera_buffer])
            .mark_external()
            .commit();

        let gbuffer_pass = self
            .render_graph
            .begin_render_pass("GBuffer", backbuffer.size)?
            .writes_attachments(&[
                position_target,
                normal_target,
                diffuse_target,
                emissive_target,
                pbr_target,
            ])
            .reads_attachments(&[depth_target])
            .shader_reads(&[camera_buffer])
            .mark_external()
            .with_blend_state(BlendState {
                blend_enable: false,
                src_color_blend_factor: BlendMode::One,
                dst_color_blend_factor: BlendMode::Zero,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendMode::One,
                dst_alpha_blend_factor: BlendMode::Zero,
                alpha_blend_op: BlendOp::Add,
                color_write_mask: ColorComponentFlags::RGBA,
            })
            .commit();

        let combine_pass = self
            .render_graph
            .begin_render_pass("GBufferCombine", backbuffer.size)?
            .writes_attachments(&[color_target])
            .shader_reads(&[
                position_target,
                normal_target,
                diffuse_target,
                emissive_target,
                pbr_target,
                shadow_map,
                camera_buffer,
                light_buffer,
            ])
            .with_blend_state(BlendState {
                blend_enable: false,
                src_color_blend_factor: BlendMode::One,
                dst_color_blend_factor: BlendMode::Zero,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendMode::One,
                dst_alpha_blend_factor: BlendMode::Zero,
                alpha_blend_op: BlendOp::Add,
                color_write_mask: ColorComponentFlags::RGBA,
            })
            .commit();

        let tonemap_pass = self
            .render_graph
            .begin_render_pass("Tonemapping", backbuffer.size)?
            .shader_reads(&[color_target])
            .writes_attachments(&[tonemap_output])
            .with_blend_state(BlendState {
                blend_enable: false,
                src_color_blend_factor: BlendMode::One,
                dst_color_blend_factor: BlendMode::Zero,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendMode::One,
                dst_alpha_blend_factor: BlendMode::Zero,
                alpha_blend_op: BlendOp::Add,
                color_write_mask: ColorComponentFlags::RGBA,
            })
            .commit();
        let fxaa_pass = self
            .render_graph
            .begin_render_pass("Fxaa", backbuffer.size)?
            .shader_reads(&[tonemap_output])
            .writes_attachments(&[fxaa_output])
            .with_blend_state(BlendState {
                blend_enable: false,
                src_color_blend_factor: BlendMode::One,
                dst_color_blend_factor: BlendMode::Zero,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendMode::One,
                dst_alpha_blend_factor: BlendMode::Zero,
                alpha_blend_op: BlendOp::Add,
                color_write_mask: ColorComponentFlags::RGBA,
            })
            .commit();

        let present_render_pass = self
            .render_graph
            .begin_render_pass("Present", backbuffer.size)?
            .shader_reads(&[fxaa_output])
            .writes_attachments(&[swapchain_image])
            .with_blend_state(BlendState {
                blend_enable: false,
                src_color_blend_factor: BlendMode::One,
                dst_color_blend_factor: BlendMode::Zero,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendMode::One,
                dst_alpha_blend_factor: BlendMode::Zero,
                alpha_blend_op: BlendOp::Add,
                color_write_mask: ColorComponentFlags::RGBA,
            })
            .commit();

        self.render_graph.compile()?;

        self.render_graph.define_pipeline_for_renderpass(
            &crate::app_state().gpu,
            &combine_pass,
            "CombinePipeline",
            &RenderGraphPipelineDescription {
                vertex_inputs: &[],
                stage: RenderStage::Graphics {
                    vertex: ModuleInfo {
                        module: &self.screen_quad,
                        entry_point: "main",
                    },
                    fragment: ModuleInfo {
                        module: &self.gbuffer_combine,
                        entry_point: "main",
                    },
                },
                fragment_state: FragmentState {
                    input_topology: gpu::PrimitiveTopology::TriangleStrip,
                    primitive_restart: false,
                    polygon_mode: gpu::PolygonMode::Fill,
                    cull_mode: gpu::CullMode::None,
                    front_face: gpu::FrontFace::ClockWise,
                    depth_stencil_state: DepthStencilState {
                        depth_test_enable: false,
                        depth_write_enable: false,
                        depth_compare_op: CompareOp::Always,
                        stencil_test_enable: false,
                        front: StencilOpState::default(),
                        back: StencilOpState::default(),
                        min_depth_bounds: 0.0,
                        max_depth_bounds: 1.0,
                    },
                    logic_op: None,
                    push_constant_ranges: &[],
                },
            },
        )?;

        self.render_graph.define_pipeline_for_renderpass(
            &crate::app_state().gpu,
            &tonemap_pass,
            "TonemapPipeline",
            &RenderGraphPipelineDescription {
                vertex_inputs: &[],
                stage: RenderStage::Graphics {
                    vertex: ModuleInfo {
                        module: &self.screen_quad,
                        entry_point: "main",
                    },
                    fragment: ModuleInfo {
                        module: &self.tonemap_fs,
                        entry_point: "main",
                    },
                },
                fragment_state: FragmentState {
                    input_topology: gpu::PrimitiveTopology::TriangleStrip,
                    primitive_restart: false,
                    polygon_mode: gpu::PolygonMode::Fill,
                    cull_mode: gpu::CullMode::None,
                    front_face: gpu::FrontFace::ClockWise,
                    depth_stencil_state: DepthStencilState {
                        depth_test_enable: false,
                        depth_write_enable: false,
                        depth_compare_op: CompareOp::Always,
                        stencil_test_enable: false,
                        front: StencilOpState::default(),
                        back: StencilOpState::default(),
                        min_depth_bounds: 0.0,
                        max_depth_bounds: 1.0,
                    },
                    logic_op: None,
                    push_constant_ranges: &[],
                },
            },
        )?;

        self.render_graph.define_pipeline_for_renderpass(
            &crate::app_state().gpu,
            &fxaa_pass,
            "FxaaPipeline",
            &RenderGraphPipelineDescription {
                vertex_inputs: &[],
                stage: RenderStage::Graphics {
                    vertex: ModuleInfo {
                        module: &self.fxaa_vs,
                        entry_point: "main",
                    },
                    fragment: ModuleInfo {
                        module: &self.fxaa_fs,
                        entry_point: "main",
                    },
                },
                fragment_state: FragmentState {
                    input_topology: gpu::PrimitiveTopology::TriangleList,
                    primitive_restart: false,
                    polygon_mode: gpu::PolygonMode::Fill,
                    cull_mode: gpu::CullMode::None,
                    front_face: gpu::FrontFace::ClockWise,
                    depth_stencil_state: DepthStencilState {
                        depth_test_enable: false,
                        depth_write_enable: false,
                        depth_compare_op: CompareOp::Always,
                        stencil_test_enable: false,
                        front: StencilOpState::default(),
                        back: StencilOpState::default(),
                        min_depth_bounds: 0.0,
                        max_depth_bounds: 1.0,
                    },
                    logic_op: None,
                    push_constant_ranges: &[PushConstantRange {
                        stage_flags: ShaderStage::ALL,
                        offset: 0,
                        size: std::mem::size_of::<FxaaShaderParams>() as _,
                    }],
                },
            },
        )?;
        self.render_graph.define_pipeline_for_renderpass(
            &app_state().gpu,
            &present_render_pass,
            "Present",
            &RenderGraphPipelineDescription {
                vertex_inputs: &[],
                stage: RenderStage::Graphics {
                    vertex: ModuleInfo {
                        module: &self.screen_quad,
                        entry_point: "main",
                    },
                    fragment: ModuleInfo {
                        module: &self.texture_copy,
                        entry_point: "main",
                    },
                },
                fragment_state: FragmentState {
                    input_topology: gpu::PrimitiveTopology::TriangleStrip,
                    primitive_restart: false,
                    polygon_mode: gpu::PolygonMode::Fill,
                    cull_mode: gpu::CullMode::None,
                    front_face: gpu::FrontFace::ClockWise,
                    depth_stencil_state: DepthStencilState {
                        depth_test_enable: false,
                        depth_write_enable: false,
                        depth_compare_op: CompareOp::Always,
                        stencil_test_enable: false,
                        front: StencilOpState::default(),
                        back: StencilOpState::default(),
                        min_depth_bounds: 0.0,
                        max_depth_bounds: 1.0,
                    },
                    logic_op: None,
                    push_constant_ranges: &[],
                },
            },
        )?;

        //#endregion

        let mut graphics_command_buffer = app_state()
            .gpu
            .create_command_buffer(gpu::QueueType::Graphics)?;
        let mut context = GraphRunContext::new(
            &crate::app_state().gpu,
            &mut graphics_command_buffer,
            crate::app_state().time().frames_since_start(),
        );

        //#region context setup
        context.register_callback(&dbuffer_pass, |_: &VkGpu, ctx| {
            Self::main_render_loop(
                resource_map,
                PipelineTarget::DepthOnly,
                &draw_hashmap,
                0,
                ctx,
            );
        });
        context.register_callback(&shadow_map_pass, |_: &VkGpu, ctx| {
            ctx.render_pass_command.set_depth_bias(
                self.depth_bias_constant,
                self.depth_bias_clamp,
                self.depth_bias_slope,
            );
            for (i, pov) in per_frame_data.iter().enumerate().skip(1) {
                ctx.render_pass_command.set_viewport(gpu::Viewport {
                    x: pov.viewport_size_offset.x,
                    y: pov.viewport_size_offset.y,
                    width: pov.viewport_size_offset.z,
                    height: pov.viewport_size_offset.w,
                    min_depth: 0.0,
                    max_depth: 1.0,
                });

                Self::main_render_loop(
                    resource_map,
                    PipelineTarget::DepthOnly,
                    &draw_hashmap,
                    i as _,
                    ctx,
                );
            }
        });
        context.register_callback(&gbuffer_pass, |_: &VkGpu, ctx| {
            Self::main_render_loop(
                resource_map,
                PipelineTarget::ColorAndDepth,
                &draw_hashmap,
                0,
                ctx,
            );
        });

        context.register_callback(&combine_pass, |_: &VkGpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });
        context.register_callback(&tonemap_pass, |_: &VkGpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });
        context.register_callback(&fxaa_pass, |_: &VkGpu, ctx| {
            let rcp_frame = vector![backbuffer.size.width as f32, backbuffer.size.height as f32];
            let rcp_frame = vector![1.0 / rcp_frame.x, 1.0 / rcp_frame.y];

            let params = FxaaShaderParams {
                rcp_frame,
                fxaa_quality_subpix: self.fxaa_settings.fxaa_quality_subpix,
                fxaa_quality_edge_threshold: self.fxaa_settings.fxaa_quality_edge_threshold,
                fxaa_quality_edge_threshold_min: self.fxaa_settings.fxaa_quality_edge_threshold_min,
                iterations: self.fxaa_settings.iterations,
            };

            ctx.render_pass_command.push_constant(
                ctx.pipeline.expect("No FXAA pipeline"),
                &params,
                0,
            );
            ctx.render_pass_command.draw(3, 1, 0, 0);
        });
        context.register_callback(&present_render_pass, |_: &VkGpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });

        context.inject_external_image(&swapchain_image, backbuffer.image, backbuffer.image_view);
        context.inject_external_image(&shadow_map, &self.shadow_map, &self.shadow_map_view);
        context.injext_external_buffer(&camera_buffer, &current_buffers.camera_buffer);
        context.injext_external_buffer(&light_buffer, &current_buffers.light_buffer);
        //#endregion
        self.render_graph.run(context, &mut self.runner)?;

        Ok(graphics_command_buffer)
    }

    fn create_material(
        &mut self,
        gpu: &VkGpu,
        material_description: MaterialDescription,
    ) -> anyhow::Result<MasterMaterial> {
        let color_attachments = &[
            // Position
            RenderPassAttachment {
                format: ImageFormat::RgbaFloat32,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Load,
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Normals
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Diffuse
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Emissive
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Metal/Roughness
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
        ];
        let master_description = MasterMaterialDescription {
            name: material_description.name,
            domain: material_description.domain,
            global_inputs: match material_description.domain {
                MaterialDomain::Surface => &[BindingType::Storage],
                MaterialDomain::PostProcess => &[
                    BindingType::Storage,              // Camera buffer
                    BindingType::CombinedImageSampler, // Previous post process result/ Initial scene color,
                    BindingType::CombinedImageSampler, // Scene position,
                    BindingType::CombinedImageSampler, // Scene diffuse,
                    BindingType::CombinedImageSampler, // Scene normal,
                ],
            },
            texture_inputs: material_description.texture_inputs,
            material_parameters: material_description.material_parameters,
            vertex_info: &VertexStageInfo {
                entry_point: "main",
                module: material_description.vertex_module,
            },
            fragment_info: &FragmentStageInfo {
                entry_point: "main",
                module: material_description.fragment_module,
                color_attachments,
                depth_stencil_attachments: &[],
            },
            primitive_restart: false,
            polygon_mode: gpu::PolygonMode::Fill,
            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::CounterClockWise,
            push_constant_ranges: &[PushConstantRange {
                stage_flags: ShaderStage::ALL,
                offset: 0,
                size: std::mem::size_of::<ObjectDrawInfo>() as u32,
            }],
            logic_op: None,
        };

        MasterMaterial::new(gpu, &master_description)
    }
}
