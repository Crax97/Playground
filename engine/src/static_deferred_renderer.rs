use engine_macros::glsl;
use std::{collections::HashMap, mem::size_of};

use ash::{
    prelude::VkResult,
    vk::{
        BufferUsageFlags, CompareOp, IndexType, PipelineBindPoint, PipelineStageFlags,
        PushConstantRange, ShaderModuleCreateFlags, ShaderStageFlags, StencilOpState,
    },
};
use gpu::{
    BindingType, BufferCreateInfo, CommandBuffer, DepthStencilState, FragmentStageInfo, Gpu,
    GpuBuffer, GpuShaderModule, ImageFormat, MemoryDomain, ShaderModuleCreateInfo, Swapchain, ToVk,
    VertexStageInfo,
};
use nalgebra::{vector, Matrix4, Vector2, Vector4};
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
#[repr(C)]
#[derive(Clone, Copy)]
struct FxaaShaderParams {
    rcp_frame: Vector2<f32>,
    fxaa_quality_subpix: f32,
    fxaa_quality_edge_threshold: f32,
    fxaa_quality_edge_threshold_min: f32,
}

#[derive(Clone, Copy)]
pub struct FxaaSettings {
    pub fxaa_quality_subpix: f32,
    pub fxaa_quality_edge_threshold: f32,
    pub fxaa_quality_edge_threshold_min: f32,
}

impl Default for FxaaSettings {
    fn default() -> Self {
        Self {
            fxaa_quality_subpix: 0.75,
            fxaa_quality_edge_threshold: 0.166,
            fxaa_quality_edge_threshold_min: 0.0833,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameData {
    eye: Vector4<f32>,
    view: nalgebra::Matrix4<f32>,
    projection: nalgebra::Matrix4<f32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuLightInfo {
    position_radius: Vector4<f32>,
    direction: Vector4<f32>,
    color: Vector4<f32>,
    extras: Vector4<f32>,
    ty: [u32; 4],
}

impl From<&Light> for GpuLightInfo {
    fn from(light: &Light) -> Self {
        let (direction, extras, ty) = match light.ty {
            LightType::Point => (Default::default(), Default::default(), 0),
            LightType::Directional { direction } => (
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
                vector![inner_cone_degrees, outer_cone_degrees, 0.0, 0.0],
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
            ty: [ty, 0, 0, 0],
        }
    }
}

use crate::{
    app_state,
    camera::Camera,
    material::{MasterMaterial, MasterMaterialDescription},
    Backbuffer, BufferDescription, BufferType, ClearValue, FragmentState, GpuRunner,
    GraphRunContext, Light, LightType, MaterialDescription, MaterialDomain, MaterialInstance,
    MeshPrimitive, ModuleInfo, PipelineTarget, RenderGraph, RenderGraphPipelineDescription,
    RenderPassContext, RenderStage, RenderingPipeline, Scene,
};

use ash::vk::{
    AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor, BlendOp,
    ColorComponentFlags, DependencyFlags, ImageLayout, SampleCountFlags, SubpassDependency,
    SubpassDescriptionFlags, SUBPASS_EXTERNAL,
};
use gpu::{
    BlendState, RenderPass, RenderPassAttachment, RenderPassDescription, SubpassDescription,
};

struct FrameBuffers {
    camera_buffer: GpuBuffer,
    light_buffer: GpuBuffer,
}

struct DrawCall<'a> {
    prim: &'a MeshPrimitive,
    transform: Matrix4<f32>,
    material: ResourceHandle<MaterialInstance>,
}

pub struct DeferredRenderingPipeline {
    frame_buffers: Vec<FrameBuffers>,
    material_context: DeferredRenderingMaterialContext,
    render_graph: RenderGraph,
    screen_quad: GpuShaderModule,
    texture_copy: GpuShaderModule,
    gbuffer_combine: GpuShaderModule,
    tonemap_fs: GpuShaderModule,

    fxaa_settings: FxaaSettings,

    runner: GpuRunner,
    fxaa_vs: GpuShaderModule,
    fxaa_fs: GpuShaderModule,
    in_flight_frame: usize,
    max_frames_in_flight: usize,
}

impl DeferredRenderingPipeline {
    pub fn new(
        gpu: &Gpu,
        screen_quad: GpuShaderModule,
        gbuffer_combine: GpuShaderModule,
        texture_copy: GpuShaderModule,
        tonemap_fs: GpuShaderModule,
    ) -> anyhow::Result<Self> {
        let mut frame_buffers = vec![];
        for _ in 0..Swapchain::MAX_FRAMES_IN_FLIGHT {
            let camera_buffer = {
                let create_info = BufferCreateInfo {
                    label: Some("Deferred Renderer - Camera buffer"),
                    size: std::mem::size_of::<PerFrameData>(),
                    usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
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

        let material_context = DeferredRenderingMaterialContext::new(gpu)?;

        let render_graph = RenderGraph::new();

        let fxaa_vs = gpu.create_shader_module(&ShaderModuleCreateInfo {
            flags: ShaderModuleCreateFlags::empty(),
            code: bytemuck::cast_slice(FXAA_VS),
        })?;
        let fxaa_fs = gpu.create_shader_module(&ShaderModuleCreateInfo {
            flags: ShaderModuleCreateFlags::empty(),
            code: bytemuck::cast_slice(FXAA_FS),
        })?;

        Ok(Self {
            material_context,
            render_graph,
            screen_quad,
            gbuffer_combine,
            texture_copy,
            frame_buffers,
            tonemap_fs,
            fxaa_vs,
            fxaa_fs,
            fxaa_settings: Default::default(),
            runner: GpuRunner::new(),
            in_flight_frame: 0,
            max_frames_in_flight: Swapchain::MAX_FRAMES_IN_FLIGHT,
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
                    PipelineBindPoint::GRAPHICS,
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
                        PipelineBindPoint::GRAPHICS,
                        pipeline,
                        1,
                        &[&material.user_descriptor_set],
                    );
                    ctx.render_pass_command.bind_index_buffer(
                        &draw_call.prim.index_buffer,
                        0,
                        IndexType::UINT32,
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
                    ctx.render_pass_command
                        .push_constant(pipeline, &draw_call.transform, 0);
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
}

pub struct DeferredRenderingMaterialContext {
    render_passes: HashMap<PipelineTarget, RenderPass>,
}

impl DeferredRenderingMaterialContext {
    pub fn new(gpu: &Gpu) -> VkResult<Self> {
        let mut render_passes: HashMap<PipelineTarget, RenderPass> = HashMap::new();

        let depth_only_render_pass = RenderPass::new(
            gpu,
            &RenderPassDescription {
                attachments: &[RenderPassAttachment {
                    format: ImageFormat::Depth.to_vk(),
                    samples: SampleCountFlags::TYPE_1,
                    load_op: AttachmentLoadOp::CLEAR,
                    store_op: AttachmentStoreOp::STORE,
                    stencil_load_op: AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: AttachmentStoreOp::DONT_CARE,
                    initial_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    blend_state: BlendState {
                        blend_enable: true,
                        src_color_blend_factor: BlendFactor::ONE,
                        dst_color_blend_factor: BlendFactor::ZERO,
                        color_blend_op: BlendOp::ADD,
                        src_alpha_blend_factor: BlendFactor::ONE,
                        dst_alpha_blend_factor: BlendFactor::ZERO,
                        alpha_blend_op: BlendOp::ADD,
                        color_write_mask: ColorComponentFlags::RGBA,
                    },
                }],
                subpasses: &[SubpassDescription {
                    flags: SubpassDescriptionFlags::empty(),
                    pipeline_bind_point: PipelineBindPoint::GRAPHICS,
                    input_attachments: &[],
                    color_attachments: &[],
                    resolve_attachments: &[],
                    depth_stencil_attachment: &[AttachmentReference {
                        attachment: 0,
                        layout: ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
                    }],
                    preserve_attachments: &[],
                }],
                dependencies: &[SubpassDependency {
                    src_subpass: SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                    dst_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    src_access_mask: AccessFlags::empty(),
                    dst_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dependency_flags: DependencyFlags::empty(),
                }],
            },
        )?;
        let surface_render_pass = Self::make_surface_color_depth_pass(gpu)?;
        let post_process_render_pass = Self::make_post_process_pass(gpu)?;
        render_passes.insert(PipelineTarget::DepthOnly, depth_only_render_pass);
        render_passes.insert(PipelineTarget::ColorAndDepth, surface_render_pass);
        render_passes.insert(PipelineTarget::PostProcess, post_process_render_pass);

        Ok(Self { render_passes })
    }

    fn make_surface_color_depth_pass(gpu: &Gpu) -> VkResult<RenderPass> {
        let surface_attachments = &[
            // Position
            RenderPassAttachment {
                format: ImageFormat::RgbaFloat.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Normals
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Diffuse
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Emissive
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Metal/Roughness
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Depth
            RenderPassAttachment {
                format: ImageFormat::Depth.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::LOAD,
                store_op: AttachmentStoreOp::NONE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                final_layout: ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: false,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
        ];
        let surface_render_pass = RenderPass::new(
            gpu,
            &RenderPassDescription {
                attachments: surface_attachments,
                subpasses: &[SubpassDescription {
                    flags: SubpassDescriptionFlags::empty(),
                    pipeline_bind_point: PipelineBindPoint::GRAPHICS,
                    input_attachments: &[],
                    color_attachments: &[
                        AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        },
                        AttachmentReference {
                            attachment: 1,
                            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        },
                        AttachmentReference {
                            attachment: 2,
                            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        },
                        AttachmentReference {
                            attachment: 3,
                            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        },
                        AttachmentReference {
                            attachment: 4,
                            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        },
                    ],
                    resolve_attachments: &[],
                    depth_stencil_attachment: &[AttachmentReference {
                        attachment: 5,
                        layout: ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    }],
                    preserve_attachments: &[],
                }],
                dependencies: &[SubpassDependency {
                    src_subpass: SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: AccessFlags::empty(),
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: DependencyFlags::empty(),
                }],
            },
        )?;
        Ok(surface_render_pass)
    }

    fn make_post_process_pass(gpu: &Gpu) -> VkResult<RenderPass> {
        let post_process_attachments = &[RenderPassAttachment {
            format: ImageFormat::Rgba8.to_vk(),
            samples: SampleCountFlags::TYPE_1,
            load_op: AttachmentLoadOp::CLEAR,
            store_op: AttachmentStoreOp::STORE,
            stencil_load_op: AttachmentLoadOp::DONT_CARE,
            stencil_store_op: AttachmentStoreOp::DONT_CARE,
            initial_layout: ImageLayout::UNDEFINED,
            final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            blend_state: BlendState {
                blend_enable: true,
                src_color_blend_factor: BlendFactor::ONE,
                dst_color_blend_factor: BlendFactor::ZERO,
                color_blend_op: BlendOp::ADD,
                src_alpha_blend_factor: BlendFactor::ONE,
                dst_alpha_blend_factor: BlendFactor::ZERO,
                alpha_blend_op: BlendOp::ADD,
                color_write_mask: ColorComponentFlags::RGBA,
            },
        }];
        let post_process_render_pass = RenderPass::new(
            gpu,
            &RenderPassDescription {
                attachments: post_process_attachments,
                subpasses: &[SubpassDescription {
                    flags: SubpassDescriptionFlags::empty(),
                    pipeline_bind_point: PipelineBindPoint::GRAPHICS,
                    input_attachments: &[],
                    color_attachments: &[AttachmentReference {
                        attachment: 0,
                        layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    }],
                    resolve_attachments: &[],
                    depth_stencil_attachment: &[],
                    preserve_attachments: &[],
                }],
                dependencies: &[SubpassDependency {
                    src_subpass: SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: AccessFlags::empty(),
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: DependencyFlags::empty(),
                }],
            },
        )?;
        Ok(post_process_render_pass)
    }
}

impl RenderingPipeline for DeferredRenderingPipeline {
    fn render(
        &mut self,
        pov: &Camera,
        scene: &Scene,
        backbuffer: &Backbuffer,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<CommandBuffer> {
        let projection = pov.projection();

        let current_buffers = &self.frame_buffers[self.in_flight_frame];

        self.in_flight_frame = (1 + self.in_flight_frame) % self.max_frames_in_flight;

        super::app_state()
            .gpu
            .write_buffer_data(
                &current_buffers.camera_buffer,
                &[PerFrameData {
                    eye: Vector4::new(pov.location[0], pov.location[1], pov.location[2], 0.0),
                    view: crate::utils::constants::MATRIX_COORDINATE_X_FLIP * pov.view(),
                    projection,
                }],
            )
            .unwrap();

        let collected_active_lights: Vec<GpuLightInfo> =
            scene.all_enabled_lights().map(|l| l.into()).collect();

        super::app_state()
            .gpu
            .write_buffer_data_with_offset(
                &current_buffers.light_buffer,
                0,
                &[collected_active_lights.len() as u64],
            )
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer_data_with_offset(
                &current_buffers.light_buffer,
                size_of::<u32>() as u64 * 4,
                &collected_active_lights,
            )
            .unwrap();

        app_state().gpu.begin_frame()?;

        let draw_hashmap = Self::generate_draw_calls(resource_map, scene);

        //#region render graph resources
        let framebuffer_rgba_desc = crate::ImageDescription {
            width: backbuffer.size.width,
            height: backbuffer.size.height,
            format: ImageFormat::Rgba8,
            samples: 1,
            present: false,
            clear_value: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
        };
        let framebuffer_normal_desc = crate::ImageDescription {
            width: backbuffer.size.width,
            height: backbuffer.size.height,
            format: ImageFormat::Rgba8,
            samples: 1,
            present: false,
            clear_value: ClearValue::Color([0.5, 0.5, 0.5, 1.0]),
        };
        let framebuffer_vector_desc = crate::ImageDescription {
            width: backbuffer.size.width,
            height: backbuffer.size.height,
            format: ImageFormat::RgbaFloat,
            samples: 1,
            present: false,
            clear_value: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
        };
        let framebuffer_depth_desc = crate::ImageDescription {
            width: backbuffer.size.width,
            height: backbuffer.size.height,
            format: ImageFormat::Depth,
            samples: 1,
            present: false,
            clear_value: ClearValue::Depth(1.0),
        };
        let framebuffer_swapchain_desc = crate::ImageDescription {
            width: backbuffer.size.width,
            height: backbuffer.size.height,
            format: backbuffer.format.into(),
            samples: 1,
            present: false,
            clear_value: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
        };

        let camera_buffer = self.render_graph.use_buffer(
            "camera-buffer",
            &BufferDescription {
                length: std::mem::size_of::<PerFrameData>() as u64,
                ty: BufferType::Uniform,
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
                src_color_blend_factor: BlendFactor::ONE,
                dst_color_blend_factor: BlendFactor::ZERO,
                color_blend_op: BlendOp::ADD,
                src_alpha_blend_factor: BlendFactor::ONE,
                dst_alpha_blend_factor: BlendFactor::ZERO,
                alpha_blend_op: BlendOp::ADD,
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
                camera_buffer,
                light_buffer,
            ])
            .with_blend_state(BlendState {
                blend_enable: false,
                src_color_blend_factor: BlendFactor::ONE,
                dst_color_blend_factor: BlendFactor::ZERO,
                color_blend_op: BlendOp::ADD,
                src_alpha_blend_factor: BlendFactor::ONE,
                dst_alpha_blend_factor: BlendFactor::ZERO,
                alpha_blend_op: BlendOp::ADD,
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
                src_color_blend_factor: BlendFactor::ONE,
                dst_color_blend_factor: BlendFactor::ZERO,
                color_blend_op: BlendOp::ADD,
                src_alpha_blend_factor: BlendFactor::ONE,
                dst_alpha_blend_factor: BlendFactor::ZERO,
                alpha_blend_op: BlendOp::ADD,
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
                src_color_blend_factor: BlendFactor::ONE,
                dst_color_blend_factor: BlendFactor::ZERO,
                color_blend_op: BlendOp::ADD,
                src_alpha_blend_factor: BlendFactor::ONE,
                dst_alpha_blend_factor: BlendFactor::ZERO,
                alpha_blend_op: BlendOp::ADD,
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
                src_color_blend_factor: BlendFactor::ONE,
                dst_color_blend_factor: BlendFactor::ZERO,
                color_blend_op: BlendOp::ADD,
                src_alpha_blend_factor: BlendFactor::ONE,
                dst_alpha_blend_factor: BlendFactor::ZERO,
                alpha_blend_op: BlendOp::ADD,
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
                        depth_compare_op: CompareOp::ALWAYS,
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
                        depth_compare_op: CompareOp::ALWAYS,
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
                        depth_compare_op: CompareOp::ALWAYS,
                        stencil_test_enable: false,
                        front: StencilOpState::default(),
                        back: StencilOpState::default(),
                        min_depth_bounds: 0.0,
                        max_depth_bounds: 1.0,
                    },
                    logic_op: None,
                    push_constant_ranges: &[PushConstantRange {
                        stage_flags: ShaderStageFlags::ALL,
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
                        depth_compare_op: CompareOp::ALWAYS,
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

        let mut graphics_command_buffer =
            CommandBuffer::new(&crate::app_state().gpu, gpu::QueueType::Graphics)?;
        let mut context = GraphRunContext::new(
            &crate::app_state().gpu,
            &mut graphics_command_buffer,
            crate::app_state().time().frames_since_start(),
        );

        //#region context setup
        context.register_callback(&dbuffer_pass, |_: &Gpu, ctx| {
            Self::main_render_loop(resource_map, PipelineTarget::DepthOnly, &draw_hashmap, ctx);
        });
        context.register_callback(&gbuffer_pass, |_: &Gpu, ctx| {
            Self::main_render_loop(
                resource_map,
                PipelineTarget::ColorAndDepth,
                &draw_hashmap,
                ctx,
            );
        });

        context.register_callback(&combine_pass, |_: &Gpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });
        context.register_callback(&tonemap_pass, |_: &Gpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });
        context.register_callback(&fxaa_pass, |_: &Gpu, ctx| {
            let rcp_frame = vector![backbuffer.size.width as f32, backbuffer.size.height as f32];
            let rcp_frame = vector![1.0 / rcp_frame.x, 1.0 / rcp_frame.y];

            let params = FxaaShaderParams {
                rcp_frame,
                fxaa_quality_subpix: self.fxaa_settings.fxaa_quality_subpix,
                fxaa_quality_edge_threshold: self.fxaa_settings.fxaa_quality_edge_threshold,
                fxaa_quality_edge_threshold_min: self.fxaa_settings.fxaa_quality_edge_threshold_min,
            };

            ctx.render_pass_command.push_constant(
                ctx.pipeline.expect("No FXAA pipeline"),
                &params,
                0,
            );
            ctx.render_pass_command.draw(3, 1, 0, 0);
        });
        context.register_callback(&present_render_pass, |_: &Gpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });

        context.inject_external_renderpass(
            &gbuffer_pass,
            self.material_context
                .render_passes
                .get(&PipelineTarget::ColorAndDepth)
                .unwrap(),
        );
        context.inject_external_renderpass(
            &dbuffer_pass,
            self.material_context
                .render_passes
                .get(&PipelineTarget::DepthOnly)
                .unwrap(),
        );

        context.inject_external_image(&swapchain_image, backbuffer.image, backbuffer.image_view);
        context.injext_external_buffer(&camera_buffer, &current_buffers.camera_buffer);
        context.injext_external_buffer(&light_buffer, &current_buffers.light_buffer);
        //#endregion
        self.render_graph.run(context, &mut self.runner)?;

        Ok(graphics_command_buffer)
    }

    fn create_material(
        &mut self,
        gpu: &Gpu,
        material_description: MaterialDescription,
    ) -> anyhow::Result<MasterMaterial> {
        let color_attachments = &[
            // Position
            RenderPassAttachment {
                format: ImageFormat::RgbaFloat.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Normals
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Diffuse
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Emissive
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Metal/Roughness
            RenderPassAttachment {
                format: ImageFormat::Rgba8.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendFactor::ONE,
                    dst_color_blend_factor: BlendFactor::ZERO,
                    color_blend_op: BlendOp::ADD,
                    src_alpha_blend_factor: BlendFactor::ONE,
                    dst_alpha_blend_factor: BlendFactor::ZERO,
                    alpha_blend_op: BlendOp::ADD,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
        ];
        let master_description = MasterMaterialDescription {
            name: material_description.name,
            domain: material_description.domain,
            global_inputs: match material_description.domain {
                MaterialDomain::Surface => &[BindingType::Uniform],
                MaterialDomain::PostProcess => &[
                    BindingType::Uniform,              // Camera buffer
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
                stage_flags: ShaderStageFlags::ALL,
                offset: 0,
                size: std::mem::size_of::<Matrix4<f32>>() as u32,
            }],
            logic_op: None,
        };

        MasterMaterial::new(gpu, &master_description)
    }
}
