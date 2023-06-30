use std::{collections::HashMap, mem::size_of, rc::Rc};

use ash::{
    prelude::VkResult,
    vk::{
        self, BufferUsageFlags, CompareOp, IndexType, PipelineBindPoint, PipelineStageFlags,
        PushConstantRange, ShaderStageFlags, StencilOpState,
    },
};
use gpu::{
    BindingElement, BufferCreateInfo, BufferRange, DepthStencilState, DescriptorInfo,
    DescriptorSetInfo, FragmentStageInfo, GlobalBinding, Gpu, GpuBuffer, GpuDescriptorSet,
    GpuShaderModule, ImageFormat, MemoryDomain, Pipeline, PipelineDescription, Swapchain, ToVk,
    VertexAttributeDescription, VertexBindingDescription, VertexStageInfo,
};
use nalgebra::{Matrix4, vector, Vector2, Vector3, Vector4};
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
            LightType::Point => { (Default::default(), Default::default(), 0) }
            LightType::Directional { direction } => { 
                (vector![-direction.x, -direction.y, direction.z, 0.0], 
                 Default::default(), 
                 1) 
            }
            LightType::Spotlight {
                direction,
                inner_cone_degrees,
                outer_cone_degrees } => {
                (vector![-direction.x, -direction.y, direction.z, 0.0],
                 vector![inner_cone_degrees, outer_cone_degrees, 0.0, 0.0],
                 2)
            }
            LightType::Rect { direction, width, height } => {
                (vector![-direction.x, -direction.y, direction.z, 0.0], vector![width, height, 0.0, 0.0], 3)
            }
        };
        Self {
            position_radius: vector![-light.position.x, -light.position.y, light.position.z, light.radius],
            color: vector![light.color.x, light.color.y, light.color.z, 0.0],
            direction,
            extras,
            ty: [ty, 0, 0, 0],
        }
    }
}

use crate::{app_state, camera::Camera, material::{Material, MaterialContext, MaterialDescription, MaterialDomain}, FragmentState, GpuRunner, GraphRunContext, ModuleInfo, PipelineTarget, RenderGraph, RenderGraphPipelineDescription, RenderStage, RenderingPipeline, Scene, ScenePrimitive, Texture, BufferDescription, BufferType, Light, LightType};

use ash::vk::{
    AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor, BlendOp,
    ColorComponentFlags, DependencyFlags, Format, ImageLayout, SampleCountFlags, SubpassDependency,
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
                usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
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
    render_passes: HashMap<MaterialDomain, RenderPass>,
}

impl DeferredRenderingMaterialContext {
    pub fn new(gpu: &Gpu) -> VkResult<Self> {
        let mut render_passes: HashMap<MaterialDomain, RenderPass> = HashMap::new();

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
        render_passes.insert(MaterialDomain::DepthOnly, depth_only_render_pass);
        render_passes.insert(MaterialDomain::Surface, surface_render_pass);

        Ok(Self { render_passes })
    }

    fn create_surface_material_pipeline<'a>(
        &self,
        gpu: &Gpu,
        material_description: &'a MaterialDescription<'a>,
    ) -> VkResult<Pipeline> {
        let mut bindings: Vec<_> = material_description
            .input_textures
            .iter()
            .enumerate()
            .map(|(i, _)| BindingElement {
                binding_type: gpu::BindingType::CombinedImageSampler,
                index: i as _,
                stage: gpu::ShaderStage::VertexFragment,
            })
            .collect();

        let mut idx = bindings.len();
        for _ in &material_description.uniform_buffers {
            bindings.push(BindingElement {
                binding_type: gpu::BindingType::Uniform,
                index: idx as _,
                stage: gpu::ShaderStage::VertexFragment,
            });
            idx = idx + 1;
        }

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

        let description = PipelineDescription {
            global_bindings: &[
                // camera
                GlobalBinding {
                    set_index: 0,
                    elements: &[BindingElement {
                        binding_type: gpu::BindingType::Uniform,
                        index: 0,
                        stage: gpu::ShaderStage::VertexFragment,
                    }],
                },
                // shader inputs
                GlobalBinding {
                    set_index: 1,
                    elements: &bindings,
                },
            ],
            vertex_inputs: &[
                VertexBindingDescription {
                    binding: 0,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 0,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 1,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 1,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 2,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 2,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 3,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 3,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 4,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector2<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 4,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 0,
                    }],
                },
            ],
            vertex_stage: Some(VertexStageInfo {
                entry_point: "main",
                module: &material_description.vertex_module,
            }),
            fragment_stage: Some(FragmentStageInfo {
                entry_point: "main",
                module: &material_description.fragment_module,
                color_attachments,
                depth_stencil_attachments: &[],
            }),
            input_topology: gpu::PrimitiveTopology::TriangleList,
            primitive_restart: false,
            polygon_mode: gpu::PolygonMode::Fill,
            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::CounterClockWise,
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: false,
                depth_compare_op: CompareOp::EQUAL,
                stencil_test_enable: false,
                front: StencilOpState::default(),
                back: StencilOpState::default(),
                min_depth_bounds: 0.0,
                max_depth_bounds: 1.0,
            },
            push_constant_ranges: &[PushConstantRange {
                stage_flags: ShaderStageFlags::ALL,
                offset: 0,
                size: std::mem::size_of::<Matrix4<f32>>() as u32,
            }],
            ..Default::default()
        };
        Pipeline::new(
            gpu,
            self.get_material_render_pass(MaterialDomain::Surface),
            &description,
        )
    }
    fn create_surface_depth_only_material_pipeline<'a>(
        &self,
        gpu: &Gpu,
        material_description: &'a MaterialDescription<'a>,
    ) -> VkResult<Pipeline> {
        let mut bindings: Vec<_> = material_description
            .input_textures
            .iter()
            .enumerate()
            .map(|(i, _)| BindingElement {
                binding_type: gpu::BindingType::CombinedImageSampler,
                index: i as _,
                stage: gpu::ShaderStage::VertexFragment,
            })
            .collect();
        let mut idx = bindings.len();
        for _ in &material_description.uniform_buffers {
            bindings.push(BindingElement {
                binding_type: gpu::BindingType::Uniform,
                index: idx as _,
                stage: gpu::ShaderStage::VertexFragment,
            });
            idx = idx + 1;
        }

        let color_attachments = &[];

        let descfription = PipelineDescription {
            global_bindings: &[
                GlobalBinding {
                    set_index: 0,
                    elements: &[BindingElement {
                        binding_type: gpu::BindingType::Uniform,
                        index: 0,
                        stage: gpu::ShaderStage::VertexFragment,
                    }],
                },
                GlobalBinding {
                    set_index: 1,
                    elements: &bindings,
                },
            ],
            vertex_inputs: &[
                VertexBindingDescription {
                    binding: 0,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 0,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 1,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 1,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 2,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 2,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 3,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector3<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 3,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: 0,
                    }],
                },
                VertexBindingDescription {
                    binding: 4,
                    input_rate: gpu::InputRate::PerVertex,
                    stride: size_of::<Vector2<f32>>() as u32,
                    attributes: &[VertexAttributeDescription {
                        location: 4,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 0,
                    }],
                },
            ],
            vertex_stage: Some(VertexStageInfo {
                entry_point: "main",
                module: &material_description.vertex_module,
            }),
            fragment_stage: Some(FragmentStageInfo {
                entry_point: "main",
                module: &material_description.fragment_module,
                color_attachments,
                depth_stencil_attachments: &[],
            }),
            input_topology: gpu::PrimitiveTopology::TriangleList,
            primitive_restart: false,
            polygon_mode: gpu::PolygonMode::Fill,
            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::CounterClockWise,
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: CompareOp::LESS,
                stencil_test_enable: false,
                front: StencilOpState::default(),
                back: StencilOpState::default(),
                min_depth_bounds: 0.0,
                max_depth_bounds: 1.0,
            },
            push_constant_ranges: &[PushConstantRange {
                stage_flags: ShaderStageFlags::ALL,
                offset: 0,
                size: std::mem::size_of::<Matrix4<f32>>() as u32,
            }],
            ..Default::default()
        };
        Pipeline::new(
            gpu,
            self.get_material_render_pass(MaterialDomain::DepthOnly),
            &descfription,
        )
    }
}

impl MaterialContext for DeferredRenderingMaterialContext {
    fn create_material(
        &self,
        gpu: &Gpu,
        resource_map: &ResourceMap,
        material_description: MaterialDescription,
    ) -> VkResult<Material> {
        let color_pipeline = self.create_surface_material_pipeline(gpu, &material_description)?;

        let depth_pipeline =
            self.create_surface_depth_only_material_pipeline(gpu, &material_description)?;

        let mut pipelines = HashMap::new();
        pipelines.insert(PipelineTarget::ColorAndDepth, color_pipeline);
        pipelines.insert(PipelineTarget::DepthOnly, depth_pipeline);
        Material::new(
            gpu,
            resource_map,
            pipelines,
            material_description.uniform_buffers,
            material_description.input_textures,
        )
    }
    fn get_material_render_pass(&self, domain: MaterialDomain) -> &RenderPass {
        self.render_passes.get(&domain).unwrap()
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
                    eye: Vector4::new(pov.forward[0], pov.forward[1], pov.forward[2], 0.0),
                    view: crate::utils::constants::MATRIX_COORDINATE_FLIP * pov.view(),
                    projection,
                }],
            )
            .unwrap();
        
        let collected_active_lights : Vec<GpuLightInfo> = scene.all_enabled_lights()
            .map(|l| { l.into() })
            .collect();
        
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

        let mut color_depth_hashmap: HashMap<&Pipeline, Vec<ScenePrimitive>> = HashMap::new();
        let mut depth_hashmap: HashMap<&Pipeline, Vec<ScenePrimitive>> = HashMap::new();

        for primitive in scene.primitives.iter() {
            let material = self.resource_map.get(&primitive.material);
            color_depth_hashmap
                .entry(&material.pipelines[&PipelineTarget::ColorAndDepth])
                .or_default()
                .push(primitive.clone());
            depth_hashmap
                .entry(&material.pipelines[&PipelineTarget::DepthOnly])
                .or_default()
                .push(primitive.clone());
        }

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
        
        let camera_buffer = self
            .render_graph
            .use_buffer("camera-buffer", &BufferDescription { 
                length: std::mem::size_of::<PerFrameData>() as u64, 
                ty: BufferType::Uniform
            }, 
    true)?;
        
        let light_buffer = self
            .render_graph
            .use_buffer("light-buffer", &BufferDescription {
                length: std::mem::size_of::<PerFrameData>() as u64,
                ty: BufferType::Storage
            },
    true)?;
        
        let swapchain_image = self
            .render_graph
            .use_image("swapchain", &framebuffer_swapchain_desc, true)?;
        let depth_target = self
            .render_graph
            .use_image("depth-buffer", &framebuffer_depth_desc, false)?;
        let color_target = self
            .render_graph
            .use_image("color-buffer", &framebuffer_rgba_desc, false)?;

        let position_target = self
            .render_graph
            .use_image("position-buffer", &framebuffer_vector_desc, false)?;
        let normal_target = self
            .render_graph
            .use_image("normal_buffer", &framebuffer_rgba_desc, false)?;
        let diffuse_target = self
            .render_graph
            .use_image("diffuse_buffer", &framebuffer_rgba_desc, false)?;
        let emissive_target = self
            .render_graph
            .use_image("emissive_buffer", &framebuffer_rgba_desc, false)?;
        let pbr_target = self
            .render_graph
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
                light_buffer
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

        let combine_handle = self.render_graph.create_pipeline_for_render_pass(
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
                    front_face: gpu::FrontFace::CounterClockWise,
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
                    push_constant_ranges: &[ash::vk::PushConstantRange {
                        offset: 0,
                        size: std::mem::size_of::<Vector4<f32>>() as _,
                        stage_flags: ShaderStageFlags::ALL,
                    }],
                    ..Default::default()
                },
            },
        )?;

        let present_handle = self.render_graph.create_pipeline_for_render_pass(
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
                    front_face: gpu::FrontFace::CounterClockWise,
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

        let mut context = GraphRunContext::new(
            &crate::app_state().gpu,
            &mut crate::app_state_mut().swapchain,
            crate::app_state().time().frames_since_start(),
        );
        context.register_callback(&dbuffer_pass, |_: &Gpu, ctx| {
            for (pipeline, primitives) in depth_hashmap.iter() {
                {
                    ctx.render_pass_command.bind_descriptor_sets(
                        PipelineBindPoint::GRAPHICS,
                        &pipeline,
                        0,
                        &[&ctx.read_descriptor_set.expect("No descriptor set???")],
                    );
                    for (idx, primitive) in primitives.iter().enumerate() {
                        let primitive_label = ctx.render_pass_command.begin_debug_region(
                            &format!("Rendering primitive {}", idx),
                            [0.0, 0.3, 0.4, 1.0],
                        );
                        let mesh = self.resource_map.get(&primitive.mesh);
                        let material = self.resource_map.get(&primitive.material);

                        ctx.render_pass_command.bind_pipeline(&pipeline);
                        ctx.render_pass_command.bind_descriptor_sets(
                            PipelineBindPoint::GRAPHICS,
                            &pipeline,
                            1,
                            &[&material.resources_descriptor_set],
                        );

                        ctx.render_pass_command.bind_index_buffer(
                            &mesh.index_buffer,
                            0,
                            IndexType::UINT32,
                        );
                        ctx.render_pass_command.bind_vertex_buffer(
                            0,
                            &[
                                &mesh.position_component,
                                &mesh.color_component,
                                &mesh.normal_component,
                                &mesh.tangent_component,
                                &mesh.uv_component,
                            ],
                            &[0, 0, 0, 0, 0],
                        );
                        let index_count = self.resource_map.get(&primitive.mesh).index_count;
                        ctx.render_pass_command
                            .push_constant(&pipeline, &primitive.transform, 0);
                        ctx.render_pass_command
                            .draw_indexed(index_count, 1, 0, 0, 0);
                        primitive_label.end();
                    }
                }
            }
        });
        context.register_callback(&gbuffer_pass, |_: &Gpu, ctx| {
            for (pipeline, primitives) in color_depth_hashmap.iter() {
                {
                    // ctx.render_pass_command.bind_descriptor_sets(
                    //     PipelineBindPoint::GRAPHICS,
                    //     &pipeline,
                    //     0,
                    //     &[&ctx.read_descriptor_set.expect("No descriptor set???")],
                    // );
                    for (idx, primitive) in primitives.iter().enumerate() {
                        let primitive_label = ctx.render_pass_command.begin_debug_region(
                            &format!("Rendering primitive {}", idx),
                            [0.0, 0.3, 0.4, 1.0],
                        );
                        let mesh = self.resource_map.get(&primitive.mesh);
                        let material = self.resource_map.get(&primitive.material);

                        ctx.render_pass_command.bind_pipeline(&pipeline);
                        ctx.render_pass_command.bind_descriptor_sets(
                            PipelineBindPoint::GRAPHICS,
                            &pipeline,
                            1,
                            &[&material.resources_descriptor_set],
                        );

                        ctx.render_pass_command.bind_index_buffer(
                            &mesh.index_buffer,
                            0,
                            IndexType::UINT32,
                        );
                        ctx.render_pass_command.bind_vertex_buffer(
                            0,
                            &[
                                &mesh.position_component,
                                &mesh.color_component,
                                &mesh.normal_component,
                                &mesh.tangent_component,
                                &mesh.uv_component,
                            ],
                            &[0, 0, 0, 0, 0],
                        );

                        let index_count = self.resource_map.get(&primitive.mesh).index_count;
                        ctx.render_pass_command
                            .push_constant(&pipeline, &primitive.transform, 0);
                        ctx.render_pass_command
                            .draw_indexed(index_count, 1, 0, 0, 0);
                        primitive_label.end();
                    }
                }
            }
        });

        context.register_callback(&combine_pass, |_: &Gpu, ctx| {
            let pipeline = ctx.render_graph.get_pipeline(&combine_handle).unwrap();
            ctx.render_pass_command.push_constant(&pipeline, &pov.location, 0);
            ctx.render_pass_command.bind_pipeline(&pipeline);
            ctx.render_pass_command.bind_descriptor_sets(
                PipelineBindPoint::GRAPHICS,
                &pipeline,
                0,
                &[ctx.read_descriptor_set.unwrap()],
            );
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });
        context.register_callback(&present_render_pass, |_: &Gpu, ctx| {
            let pipeline = ctx.render_graph.get_pipeline(&present_handle).unwrap();
            ctx.render_pass_command.bind_pipeline(pipeline);
            ctx.render_pass_command.bind_descriptor_sets(
                PipelineBindPoint::GRAPHICS,
                pipeline,
                0,
                &[ctx.read_descriptor_set.unwrap()],
            );
            ctx.render_pass_command.draw(4, 1, 0, 0);
        });

        context.set_clear_callback(&gbuffer_pass, |handle| {
            if handle == &normal_target {
                ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue {
                        float32: [0.5, 0.5, 0.5, 1.0]
                    }
                }
            } else if handle == &depth_target {
                ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 255,
                    }
                }
            } else {
                ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0]
                    }
                }
            }
        });

        context.inject_external_renderpass(
            &gbuffer_pass,
            self.material_context
                .get_material_render_pass(MaterialDomain::Surface),
        );
        context.inject_external_renderpass(
            &dbuffer_pass,
            self.material_context
                .get_material_render_pass(MaterialDomain::DepthOnly),
        );

        context.inject_external_image(&swapchain_image, image, view);
        context.injext_external_buffer(&camera_buffer, &self.camera_buffer);
        context.injext_external_buffer(&light_buffer, &self.light_buffer);
        self.render_graph.run(context, &mut self.runner)?;

        Ok(())
    }

    fn get_context(&self) -> &dyn MaterialContext {
        &self.material_context
    }
}
