use std::{collections::HashMap, rc::Rc};

use ash::{
    prelude::VkResult,
    vk::{
        self, BufferUsageFlags, ClearColorValue, ClearDepthStencilValue, ClearValue, Extent2D,
        IndexType, Offset2D, PipelineBindPoint, PipelineStageFlags, Rect2D,
    },
};
use gpu::{
    BeginRenderPassInfo, BufferCreateInfo, BufferRange, DescriptorInfo, DescriptorSetInfo, Gpu,
    GpuBuffer, GpuDescriptorSet, GpuFramebuffer, MemoryDomain, Swapchain,
};
use nalgebra::{point, vector, Matrix4};
use resource_map::{ResourceHandle, ResourceMap};

use crate::{
    gpu_pipeline::GpuPipeline,
    material::{Material, MaterialContext, MaterialDomain},
    mesh::Mesh,
    PerFrameData,
};

use ash::vk::{
    AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor, BlendOp,
    ColorComponentFlags, DependencyFlags, Format, ImageLayout, SampleCountFlags, SubpassDependency,
    SubpassDescriptionFlags, SUBPASS_EXTERNAL,
};
use gpu::{
    BlendState, RenderPass, RenderPassAttachment, RenderPassDescription, SubpassDescription,
};
#[derive(Clone)]
pub struct ScenePrimitive {
    pub mesh: ResourceHandle<Mesh>,
    pub material: ResourceHandle<Material>,
    pub transform: Matrix4<f32>,
}

pub struct Scene {
    primitives: Vec<ScenePrimitive>,
}

impl Scene {
    pub fn new() -> Self {
        Self { primitives: vec![] }
    }

    pub fn add(&mut self, primitive: ScenePrimitive) -> usize {
        let idx = self.primitives.len();
        self.primitives.push(primitive);
        idx
    }

    pub fn edit(&mut self, idx: usize) -> &mut ScenePrimitive {
        &mut self.primitives[idx]
    }

    pub fn all_primitives(&self) -> &[ScenePrimitive] {
        &self.primitives
    }
    pub fn edit_all_primitives(&mut self) -> &mut [ScenePrimitive] {
        &mut self.primitives
    }
}

pub trait SceneRenderer {
    type ConcreteMaterialContext: MaterialContext;
    fn render(
        &mut self,
        gpu: &Gpu,
        scene: &Scene,
        framebuffer: &GpuFramebuffer,
        swapchain: &mut Swapchain,
    );

    fn get_context(&self) -> &Self::ConcreteMaterialContext;
}

pub struct ForwardNaiveRenderer {
    resource_map: Rc<ResourceMap>,
    extents: Extent2D,

    camera_buffer: GpuBuffer,
    camera_buffer_descriptor_set: GpuDescriptorSet,
    material_context: ForwardRendererMaterialContext,
}
impl ForwardNaiveRenderer {
    pub fn new(gpu: &Gpu, resource_map: Rc<ResourceMap>, swapchain: &Swapchain) -> VkResult<Self> {
        let camera_buffer = {
            let create_info = BufferCreateInfo {
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
                binding_stage: gpu::ShaderStage::Vertex,
            }],
        })?;

        let material_context = ForwardRendererMaterialContext::new(gpu, swapchain)?;
        Ok(Self {
            camera_buffer,
            resource_map,
            extents: swapchain.extents(),
            camera_buffer_descriptor_set,
            material_context,
        })
    }
}

pub struct ForwardRendererMaterialContext {
    render_passes: HashMap<MaterialDomain, RenderPass>,
}

impl ForwardRendererMaterialContext {
    pub fn new(gpu: &Gpu, swapchain: &Swapchain) -> VkResult<Self> {
        let mut render_passes: HashMap<MaterialDomain, RenderPass> = HashMap::new();

        let attachments = &[
            RenderPassAttachment {
                format: swapchain.present_format(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::PRESENT_SRC_KHR,
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
            RenderPassAttachment {
                format: Format::D16_UNORM,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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
                    color_attachments: &[AttachmentReference {
                        attachment: 0,
                        layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    }],
                    resolve_attachments: &[],
                    depth_stencil_attachment: &[AttachmentReference {
                        attachment: 1,
                        layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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
}
impl MaterialContext for ForwardRendererMaterialContext {
    fn get_material_render_pass(&self, domain: MaterialDomain) -> &RenderPass {
        self.render_passes.get(&domain).unwrap()
    }
}

impl SceneRenderer for ForwardNaiveRenderer {
    type ConcreteMaterialContext = ForwardRendererMaterialContext;
    fn render(
        &mut self,
        gpu: &Gpu,
        scene: &Scene,
        framebuffer: &GpuFramebuffer,
        swapchain: &mut Swapchain,
    ) {
        let mut pipeline_hashmap: HashMap<ResourceHandle<GpuPipeline>, Vec<ScenePrimitive>> =
            HashMap::new();

        for primitive in scene.primitives.iter() {
            let material = self.resource_map.get(&primitive.material);
            pipeline_hashmap
                .entry(material.pipeline.clone())
                .or_default()
                .push(primitive.clone());
        }
        let mut command_buffer = gpu::CommandBuffer::new(gpu, gpu::QueueType::Graphics).unwrap();
        for (pipeline, primitives) in pipeline_hashmap.iter() {
            {
                let pipeline = self.resource_map.get(pipeline);
                command_buffer.bind_descriptor_sets(
                    PipelineBindPoint::GRAPHICS,
                    &pipeline.0,
                    0,
                    &[&self.camera_buffer_descriptor_set],
                );
                let mut render_pass = command_buffer.begin_render_pass(&BeginRenderPassInfo {
                    framebuffer,
                    render_pass: self
                        .material_context
                        .get_material_render_pass(MaterialDomain::Surface),
                    clear_color_values: &[
                        ClearValue {
                            color: ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        ClearValue {
                            depth_stencil: ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ],
                    render_area: Rect2D {
                        offset: Offset2D { x: 0, y: 0 },
                        extent: self.extents,
                    },
                });
                for primitive in primitives.iter() {
                    let mesh = self.resource_map.get(&primitive.mesh);
                    let material = self.resource_map.get(&primitive.material);
                    gpu.write_buffer_data(
                        &self.camera_buffer,
                        &[PerFrameData {
                            view: nalgebra::Matrix4::look_at_rh(
                                &point![2.0, 2.0, 2.0],
                                &point![0.0, 0.0, 0.0],
                                &vector![0.0, 0.0, -1.0],
                            ),
                            projection: nalgebra::Matrix4::new_perspective(
                                1240.0 / 720.0,
                                45.0,
                                0.1,
                                10.0,
                            ),
                        }],
                    )
                    .unwrap();
                    render_pass.bind_pipeline(&pipeline.0);
                    render_pass.bind_descriptor_sets(
                        PipelineBindPoint::GRAPHICS,
                        &pipeline.0,
                        1,
                        &[&material.resources_descriptor_set],
                    );

                    render_pass.bind_index_buffer(&mesh.index_buffer, 0, IndexType::UINT32);
                    render_pass.bind_vertex_buffer(
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
                    render_pass.push_constant(&pipeline.0, &primitive.transform, 0);
                    render_pass.draw_indexed(6, 1, 0, 0, 0);
                }
            }
        }

        command_buffer
            .submit(&gpu::CommandBufferSubmitInfo {
                wait_semaphores: &[&swapchain.image_available_semaphore],
                wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                signal_semaphores: &[&swapchain.render_finished_semaphore],
                fence: Some(&swapchain.in_flight_fence),
            })
            .unwrap();
    }

    fn get_context(&self) -> &Self::ConcreteMaterialContext {
        &self.material_context
    }
}
