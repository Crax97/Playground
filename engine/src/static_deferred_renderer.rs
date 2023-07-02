use std::{collections::HashMap, mem::size_of, rc::Rc};

use ash::{
    prelude::VkResult,
    vk::{
        self, BufferUsageFlags, CompareOp, IndexType, PipelineBindPoint, PipelineStageFlags,
        PushConstantRange, ShaderStageFlags, StencilOpState,
    },
};
use gpu::{
    BufferCreateInfo, BufferRange, DepthStencilState, DescriptorType, FragmentStageInfo, Gpu,
    GpuBuffer, GpuShaderModule, ImageFormat, MemoryDomain, Swapchain, ToVk, VertexStageInfo,
};
use nalgebra::{vector, Matrix4, Vector4};
use resource_map::ResourceMap;

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
    material::{MasterMaterial, MasterMaterialDescription, MaterialDomain},
    BufferDescription, BufferType, FragmentState, GpuRunner, GraphRunContext, Light, LightType,
    MaterialDescription, ModuleInfo, PipelineTarget, RenderGraph, RenderGraphPipelineDescription,
    RenderStage, RenderingPipeline, Scene, ScenePrimitive,
};

use ash::vk::{
    AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor, BlendOp,
    ColorComponentFlags, DependencyFlags, ImageLayout, SampleCountFlags, SubpassDependency,
    SubpassDescriptionFlags, SUBPASS_EXTERNAL,
};
use gpu::{
    BlendState, RenderPass, RenderPassAttachment, RenderPassDescription, SubpassDescription,
};

pub struct DeferredRenderingPipeline {
    resource_map: Rc<ResourceMap>,

    camera_buffer: GpuBuffer,
    light_buffer: GpuBuffer,
    material_context: DeferredRenderingMaterialContext,
    render_graph: RenderGraph,
    screen_quad: GpuShaderModule,
    texture_copy: GpuShaderModule,
    gbuffer_combine: GpuShaderModule,

    runner: GpuRunner,
}

impl DeferredRenderingPipeline {
    pub fn new(
        gpu: &Gpu,
        resource_map: Rc<ResourceMap>,
        screen_quad: GpuShaderModule,
        gbuffer_combine: GpuShaderModule,
        texture_copy: GpuShaderModule,
    ) -> anyhow::Result<Self> {
        let camera_buffer = {
            let create_info = BufferCreateInfo {
                label: Some("Deferred Renderer - Camera buffer"),
                size: std::mem::size_of::<PerFrameData>(),
                usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
            };
            let buffer = gpu.create_buffer(
                &create_info,
                MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
            )?;
            buffer
        };
        let light_buffer = {
            let create_info = BufferCreateInfo {
                label: Some("Light Buffer"),
                size: std::mem::size_of::<GpuLightInfo>() * 1000,
                usage: BufferUsageFlags::UNIFORM_BUFFER
                    | BufferUsageFlags::STORAGE_BUFFER
                    | BufferUsageFlags::TRANSFER_DST,
            };
            let buffer = gpu.create_buffer(
                &create_info,
                MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
            )?;
            buffer
        };

        let material_context = DeferredRenderingMaterialContext::new(gpu)?;

        let render_graph = RenderGraph::new();

        Ok(Self {
            camera_buffer,
            light_buffer,
            resource_map,
            material_context,
            render_graph,
            screen_quad,
            gbuffer_combine,
            texture_copy,
            runner: GpuRunner::new(),
        })
    }
}

pub struct DeferredRenderingMaterialContext {
    render_passes: HashMap<PipelineTarget, RenderPass>,
}

impl DeferredRenderingMaterialContext {
    pub fn new(gpu: &Gpu) -> VkResult<Self> {
        let mut render_passes: HashMap<PipelineTarget, RenderPass> = HashMap::new();

        let depth_only_render_pass = RenderPass::new(
            &gpu,
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
        let attachments = &[
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
            &gpu,
            &RenderPassDescription {
                attachments,
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
        render_passes.insert(PipelineTarget::DepthOnly, depth_only_render_pass);
        render_passes.insert(PipelineTarget::ColorAndDepth, surface_render_pass);

        Ok(Self { render_passes })
    }
}

impl RenderingPipeline for DeferredRenderingPipeline {
    fn render(
        &mut self,
        pov: &Camera,
        scene: &Scene,
        swapchain: &mut Swapchain,
    ) -> anyhow::Result<()> {
        let projection = pov.projection();
        super::app_state()
            .gpu
            .write_buffer_data(
                &self.camera_buffer,
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
                &self.light_buffer,
                0,
                &[collected_active_lights.len() as u64],
            )
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer_data_with_offset(
                &self.light_buffer,
                size_of::<u32>() as u64 * 4,
                &collected_active_lights,
            )
            .unwrap();

        let swapchain_extents = swapchain.extents();
        let swapchain_format = swapchain.present_format();

        let (image, view) = swapchain.acquire_next_image()?;
        app_state().gpu.begin_frame()?;

        let mut draw_hashmap: HashMap<&MasterMaterial, Vec<ScenePrimitive>> = HashMap::new();

        for primitive in scene.primitives.iter() {
            for material in &primitive.materials {
                let material = self.resource_map.get(material);
                let master = self.resource_map.get(&material.owner);
                draw_hashmap
                    .entry(master)
                    .or_default()
                    .push(primitive.clone());
            }
        }
        //#region render graph resources
        let framebuffer_rgba_desc = crate::ImageDescription {
            width: swapchain_extents.width,
            height: swapchain_extents.height,
            format: ImageFormat::Rgba8,
            samples: 1,
            present: false,
        };
        let framebuffer_vector_desc = crate::ImageDescription {
            width: swapchain_extents.width,
            height: swapchain_extents.height,
            format: ImageFormat::RgbaFloat,
            samples: 1,
            present: false,
        };
        let framebuffer_depth_desc = crate::ImageDescription {
            width: swapchain_extents.width,
            height: swapchain_extents.height,
            format: ImageFormat::Depth,
            samples: 1,
            present: false,
        };
        let framebuffer_swapchain_desc = crate::ImageDescription {
            width: swapchain_extents.width,
            height: swapchain_extents.height,
            format: swapchain_format.into(),
            samples: 1,
            present: true,
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
                .use_image("color-buffer", &framebuffer_rgba_desc, false)?;

        let position_target =
            self.render_graph
                .use_image("position-buffer", &framebuffer_vector_desc, false)?;
        let normal_target =
            self.render_graph
                .use_image("normal_buffer", &framebuffer_rgba_desc, false)?;
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
            .begin_render_pass("EarlyZPass", swapchain_extents)?
            .writes_attachments(&[depth_target])
            .shader_reads(&[camera_buffer])
            .mark_external()
            .commit();

        let gbuffer_pass = self
            .render_graph
            .begin_render_pass("GBuffer", swapchain_extents)?
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
            .begin_render_pass("GBufferCombine", swapchain_extents)?
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

        let present_render_pass = self
            .render_graph
            .begin_render_pass("Present", swapchain_extents)?
            .shader_reads(&[color_target])
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
                    ..Default::default()
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
                    ..Default::default()
                },
            },
        )?;

        //#endregion

        let mut context = GraphRunContext::new(
            &crate::app_state().gpu,
            crate::app_state().time().frames_since_start(),
        );

        //#region context setup
        context.register_callback(&dbuffer_pass, |_: &Gpu, ctx| {
            for (master, primitives) in draw_hashmap.iter() {
                {
                    let pipeline = master
                        .get_pipeline(PipelineTarget::DepthOnly)
                        .expect("Failed to fetch depth-only pipeline");
                    ctx.render_pass_command.bind_descriptor_sets(
                        PipelineBindPoint::GRAPHICS,
                        pipeline,
                        0,
                        &[&ctx.read_descriptor_set.expect("No descriptor set???")],
                    );
                    for (idx, primitive) in primitives.iter().enumerate() {
                        let primitive_label = ctx.render_pass_command.begin_debug_region(
                            &format!("Rendering mesh {}", idx),
                            [0.0, 0.3, 0.4, 1.0],
                        );
                        let mesh = self.resource_map.get(&primitive.mesh);

                        ctx.render_pass_command.bind_pipeline(pipeline);

                        for (prim_idx, mesh_prim) in mesh.primitives.iter().enumerate() {
                            let primitive_label = ctx.render_pass_command.begin_debug_region(
                                &format!("Rendering mesh {} primitive {}", idx, prim_idx),
                                [0.0, 0.3, 0.4, 1.0],
                            );
                            let material = &primitive.materials[prim_idx];
                            let material = self.resource_map.get(material);
                            ctx.render_pass_command.bind_descriptor_sets(
                                PipelineBindPoint::GRAPHICS,
                                pipeline,
                                1,
                                &[&material.user_descriptor_set],
                            );
                            ctx.render_pass_command.bind_index_buffer(
                                &mesh_prim.index_buffer,
                                0,
                                IndexType::UINT32,
                            );
                            ctx.render_pass_command.bind_vertex_buffer(
                                0,
                                &[
                                    &mesh_prim.position_component,
                                    &mesh_prim.color_component,
                                    &mesh_prim.normal_component,
                                    &mesh_prim.tangent_component,
                                    &mesh_prim.uv_component,
                                ],
                                &[0, 0, 0, 0, 0],
                            );
                            ctx.render_pass_command.push_constant(
                                &pipeline,
                                &primitive.transform,
                                0,
                            );
                            ctx.render_pass_command
                                .draw_indexed(mesh_prim.index_count, 1, 0, 0, 0);

                            primitive_label.end();
                        }
                        primitive_label.end();
                    }
                }
            }
        });
        context.register_callback(&gbuffer_pass, |_: &Gpu, ctx| {
            for (master, primitives) in draw_hashmap.iter() {
                {
                    let pipeline = master
                        .get_pipeline(PipelineTarget::ColorAndDepth)
                        .expect("Failed to fetch color-and-depth pipeline");
                    ctx.render_pass_command.bind_descriptor_sets(
                        PipelineBindPoint::GRAPHICS,
                        pipeline,
                        0,
                        &[&ctx.read_descriptor_set.expect("No descriptor set???")],
                    );
                    for (idx, primitive) in primitives.iter().enumerate() {
                        let primitive_label = ctx.render_pass_command.begin_debug_region(
                            &format!("Rendering mesh {}", idx),
                            [0.0, 0.3, 0.4, 1.0],
                        );
                        let mesh = self.resource_map.get(&primitive.mesh);

                        ctx.render_pass_command.bind_pipeline(pipeline);

                        for (prim_idx, mesh_prim) in mesh.primitives.iter().enumerate() {
                            let primitive_label = ctx.render_pass_command.begin_debug_region(
                                &format!("Rendering mesh {} primitive {}", idx, prim_idx),
                                [0.0, 0.3, 0.4, 1.0],
                            );
                            let material = &primitive.materials[prim_idx];
                            let material = self.resource_map.get(material);
                            ctx.render_pass_command.bind_descriptor_sets(
                                PipelineBindPoint::GRAPHICS,
                                pipeline,
                                1,
                                &[&material.user_descriptor_set],
                            );
                            ctx.render_pass_command.bind_index_buffer(
                                &mesh_prim.index_buffer,
                                0,
                                IndexType::UINT32,
                            );
                            ctx.render_pass_command.bind_vertex_buffer(
                                0,
                                &[
                                    &mesh_prim.position_component,
                                    &mesh_prim.color_component,
                                    &mesh_prim.normal_component,
                                    &mesh_prim.tangent_component,
                                    &mesh_prim.uv_component,
                                ],
                                &[0, 0, 0, 0, 0],
                            );
                            ctx.render_pass_command.push_constant(
                                &pipeline,
                                &primitive.transform,
                                0,
                            );
                            ctx.render_pass_command
                                .draw_indexed(mesh_prim.index_count, 1, 0, 0, 0);

                            primitive_label.end();
                        }
                        primitive_label.end();
                    }
                }
            }
        });

        context.register_callback(&combine_pass, |_: &Gpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });
        context.register_callback(&present_render_pass, |_: &Gpu, ctx| {
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });

        context.set_clear_callback(&gbuffer_pass, |handle| {
            if handle == &normal_target {
                ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue {
                        float32: [0.5, 0.5, 0.5, 1.0],
                    },
                }
            } else if handle == &depth_target {
                ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 255,
                    },
                }
            } else {
                ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }
            }
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

        context.inject_external_image(&swapchain_image, image, view);
        context.injext_external_buffer(&camera_buffer, &self.camera_buffer);
        context.injext_external_buffer(&light_buffer, &self.light_buffer);
        //#endregion
        self.render_graph.run(context, &mut self.runner)?;

        Ok(())
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
            domain: MaterialDomain::Surface,
            global_inputs: &[DescriptorType::UniformBuffer(BufferRange {
                handle: &self.camera_buffer,
                offset: 0,
                size: vk::WHOLE_SIZE,
            })],
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

        MasterMaterial::new(
            gpu,
            &master_description,
            &self.material_context.render_passes,
        )
    }
}
