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
use nalgebra::{Matrix4, Vector2, Vector3};
use resource_map::{ResourceHandle, ResourceMap};

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameData {
    view: nalgebra::Matrix4<f32>,
    projection: nalgebra::Matrix4<f32>,
}

use crate::{
    app_state,
    camera::Camera,
    gpu_pipeline::GpuPipeline,
    material::{Material, MaterialContext, MaterialDescription, MaterialDomain},
    FragmentState, GpuRunner, GraphRunContext, ModuleInfo, RenderGraph,
    RenderGraphPipelineDescription, RenderStage, RenderingPipeline, Scene, ScenePrimitive,
};

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
    camera_buffer_descriptor_set: GpuDescriptorSet,
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

        let camera_buffer_descriptor_set = gpu.create_descriptor_set(&DescriptorSetInfo {
            descriptors: &[DescriptorInfo {
                binding: 0,
                element_type: gpu::DescriptorType::UniformBuffer(BufferRange {
                    handle: &camera_buffer,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }),
                binding_stage: gpu::ShaderStage::VertexFragment,
            }],
        })?;

        let material_context = DeferredRenderingMaterialContext::new(gpu)?;

        let render_graph = RenderGraph::new();

        Ok(Self {
            camera_buffer,
            resource_map,
            camera_buffer_descriptor_set,
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
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: ImageFormat::RgbaFloat.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: Format::R8G8B8A8_UNORM,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: Format::R8G8B8A8_UNORM,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: Format::R8G8B8A8_UNORM,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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

        render_passes.insert(MaterialDomain::Surface, surface_render_pass);

        Ok(Self { render_passes })
    }

    fn create_pipeline<'a>(
        &self,
        gpu: &Gpu,
        material_description: &'a MaterialDescription<'a>,
    ) -> VkResult<Pipeline> {
        match material_description.domain {
            MaterialDomain::Surface => {
                self.create_surface_material_pipeline(gpu, material_description)
            }
        }
    }

    fn create_surface_material_pipeline<'a>(
        &self,
        gpu: &Gpu,
        material_description: &'a MaterialDescription<'a>,
    ) -> VkResult<Pipeline> {
        let texture_bindings: Vec<_> = material_description
            .input_textures
            .iter()
            .enumerate()
            .map(|(i, _)| BindingElement {
                binding_type: gpu::BindingType::CombinedImageSampler,
                index: i as _,
                stage: gpu::ShaderStage::VertexFragment,
            })
            .collect();

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
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: ImageFormat::RgbaFloat.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: Format::R8G8B8A8_UNORM,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: Format::R8G8B8A8_UNORM,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                format: Format::R8G8B8A8_UNORM,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
                    elements: &texture_bindings,
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
            front_face: gpu::FrontFace::ClockWise,
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: false,
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
            self.get_material_render_pass(MaterialDomain::Surface),
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
        let pipeline = self.create_pipeline(gpu, &material_description)?;

        let pipeline = resource_map.add(GpuPipeline(pipeline));

        Material::new(
            gpu,
            resource_map,
            pipeline,
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
        super::app_state()
            .gpu
            .write_buffer_data(
                &self.camera_buffer,
                &[PerFrameData {
                    view: crate::utils::constants::Z_INVERT_MATRIX * pov.view(),
                    projection: pov.projection(),
                }],
            )
            .unwrap();

        let swapchain_extents = swapchain.extents();
        let swapchain_format = swapchain.present_format();

        let (image, view) = swapchain.acquire_next_image()?;

        let mut pipeline_hashmap: HashMap<ResourceHandle<GpuPipeline>, Vec<ScenePrimitive>> =
            HashMap::new();

        for primitive in scene.primitives.iter() {
            let material = self.resource_map.get(&primitive.material);
            pipeline_hashmap
                .entry(material.pipeline.clone())
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

        let swapchain_buffer = self
            .render_graph
            .use_image("swapchain", &framebuffer_swapchain_desc)?;

        let depth_buffer = self
            .render_graph
            .use_image("depth-buffer", &framebuffer_depth_desc)?;
        let color_buffer = self
            .render_graph
            .use_image("color-buffer", &framebuffer_rgba_desc)?;

        let position_buffer = self
            .render_graph
            .use_image("position-buffer", &framebuffer_vector_desc)?;
        let normal_buffer = self
            .render_graph
            .use_image("normal_buffer", &framebuffer_vector_desc)?;
        let diffuse_buffer = self
            .render_graph
            .use_image("diffuse_buffer", &framebuffer_rgba_desc)?;
        let emissive_buffer = self
            .render_graph
            .use_image("emissive_buffer", &framebuffer_rgba_desc)?;
        let pbr_buffer = self
            .render_graph
            .use_image("pbr_buffer", &framebuffer_rgba_desc)?;

        self.render_graph.persist_resource(&swapchain_buffer);

        let dbuffer_pass = self
            .render_graph
            .begin_render_pass("EarlyZPass", swapchain_extents)?
            .writes_attachments(&[depth_buffer])
            .commit();

        let gbuffer_pass = self
            .render_graph
            .begin_render_pass("GBuffer", swapchain_extents)?
            .writes_attachments(&[
                position_buffer,
                normal_buffer,
                diffuse_buffer,
                emissive_buffer,
                pbr_buffer,
            ])
            .reads_attachments(&[depth_buffer])
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
            .writes_attachments(&[color_buffer])
            .reads(&[
                position_buffer,
                normal_buffer,
                diffuse_buffer,
                emissive_buffer,
                pbr_buffer,
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
            .reads(&[color_buffer])
            .writes_attachments(&[swapchain_buffer])
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

        self.render_graph.compile()?;

        let mut context = GraphRunContext::new(
            &crate::app_state().gpu,
            &mut crate::app_state_mut().swapchain,
            crate::app_state().time().frames_since_start(),
        );
        context.register_callback(&dbuffer_pass, |_: &Gpu, ctx| {
            //for (pipeline, primitives) in pipeline_hashmap.iter() {
            //    {
            //        let pipeline = self.resource_map.get(pipeline);
            //        ctx.render_pass_command.bind_descriptor_sets(
            //            PipelineBindPoint::GRAPHICS,
            //            &pipeline.0,
            //            0,
            //            &[&self.camera_buffer_descriptor_set],
            //        );
            //        for (idx, primitive) in primitives.iter().enumerate() {
            //            let primitive_label = ctx.render_pass_command.begin_debug_region(
            //                &format!("Rendering primitive {}", idx),
            //                [0.0, 0.3, 0.4, 1.0],
            //            );
            //            let mesh = self.resource_map.get(&primitive.mesh);
            //            let material = self.resource_map.get(&primitive.material);
            //
            //            ctx.render_pass_command.bind_pipeline(&pipeline.0);
            //            ctx.render_pass_command.bind_descriptor_sets(
            //                PipelineBindPoint::GRAPHICS,
            //                &pipeline.0,
            //                1,
            //                &[&material.resources_descriptor_set],
            //            );
            //
            //            ctx.render_pass_command.bind_index_buffer(
            //                &mesh.index_buffer,
            //                0,
            //                IndexType::UINT32,
            //            );
            //            ctx.render_pass_command.bind_vertex_buffer(
            //                0,
            //                &[
            //                    &mesh.position_component,
            //                    &mesh.color_component,
            //                    &mesh.normal_component,
            //                    &mesh.tangent_component,
            //                    &mesh.uv_component,
            //                ],
            //                &[0, 0, 0, 0, 0],
            //            );
            //            ctx.render_pass_command
            //                .push_constant(&pipeline.0, &primitive.transform, 0);
            //            ctx.render_pass_command.draw_indexed(6, 1, 0, 0, 0);
            //            primitive_label.end();
            //        }
            //    }
            //}
        });
        context.register_callback(&gbuffer_pass, |_: &Gpu, ctx| {
            for (pipeline, primitives) in pipeline_hashmap.iter() {
                {
                    let pipeline = self.resource_map.get(pipeline);
                    ctx.render_pass_command.bind_descriptor_sets(
                        PipelineBindPoint::GRAPHICS,
                        &pipeline.0,
                        0,
                        &[&self.camera_buffer_descriptor_set],
                    );
                    for (idx, primitive) in primitives.iter().enumerate() {
                        let primitive_label = ctx.render_pass_command.begin_debug_region(
                            &format!("Rendering primitive {}", idx),
                            [0.0, 0.3, 0.4, 1.0],
                        );
                        let mesh = self.resource_map.get(&primitive.mesh);
                        let material = self.resource_map.get(&primitive.material);

                        ctx.render_pass_command.bind_pipeline(&pipeline.0);
                        ctx.render_pass_command.bind_descriptor_sets(
                            PipelineBindPoint::GRAPHICS,
                            &pipeline.0,
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
                        ctx.render_pass_command
                            .push_constant(&pipeline.0, &primitive.transform, 0);
                        ctx.render_pass_command.draw_indexed(6, 1, 0, 0, 0);
                        primitive_label.end();
                    }
                }
            }
        });

        context.register_callback(&combine_pass, |_: &Gpu, ctx| {
            let pipeline = ctx.render_graph.get_pipeline(&combine_handle).unwrap();

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

        context.inject_external_renderpass(
            &gbuffer_pass,
            self.material_context
                .get_material_render_pass(MaterialDomain::Surface),
        );

        context.inject_external_image(&swapchain_buffer, image, view);
        self.render_graph.run(context, &mut self.runner)?;

        Ok(())
    }

    fn get_context(&self) -> &dyn MaterialContext {
        &self.material_context
    }
}
