use std::{collections::HashMap, mem::size_of, rc::Rc};

use ash::{
    prelude::VkResult,
    vk::{
        self, BufferUsageFlags, ClearColorValue, ClearDepthStencilValue, ClearValue, CompareOp,
        Extent2D, IndexType, Offset2D, PipelineBindPoint, PipelineStageFlags, PushConstantRange,
        Rect2D, ShaderStageFlags, StencilOpState,
    },
};
use gpu::{
    BeginRenderPassInfo, BindingElement, BufferCreateInfo, BufferRange, DepthStencilState,
    DescriptorInfo, DescriptorSetInfo, FragmentStageInfo, GlobalBinding, Gpu, GpuBuffer,
    GpuDescriptorSet, GpuFramebuffer, MemoryDomain, Pipeline, PipelineDescription, Swapchain,
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
    camera::Camera,
    gpu_pipeline::GpuPipeline,
    material::{Material, MaterialContext, MaterialDescription, MaterialDomain},
    mesh::Mesh,
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

pub trait RenderingPipeline {
    fn render(
        &mut self,
        pov: &Camera,
        scene: &Scene,
        framebuffer: &GpuFramebuffer,
    );

    fn get_context(&self) -> &dyn MaterialContext;
}

pub struct ForwardRenderingPipeline {
    resource_map: Rc<ResourceMap>,
    extents: Extent2D,

    camera_buffer: GpuBuffer,
    camera_buffer_descriptor_set: GpuDescriptorSet,
    material_context: ForwardRendererMaterialContext,
}
impl ForwardRenderingPipeline {
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
    swapchain_present_format: Format,
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

        Ok(Self {
            render_passes,
            swapchain_present_format: swapchain.present_format(),
        })
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
                stage: gpu::ShaderStage::Fragment,
            })
            .collect();

        let color_attachments = &[RenderPassAttachment {
            format: self.swapchain_present_format,
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
        }];

        let descfription = PipelineDescription {
            global_bindings: &[
                GlobalBinding {
                    set_index: 0,
                    elements: &[BindingElement {
                        binding_type: gpu::BindingType::Uniform,
                        index: 0,
                        stage: gpu::ShaderStage::Vertex,
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
            self.get_material_render_pass(MaterialDomain::Surface),
            &descfription,
        )
    }
}
impl MaterialContext for ForwardRendererMaterialContext {
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
#[rustfmt::skip]
mod constants {
    use nalgebra::Matrix4;
    pub(super) const Z_INVERT_MATRIX: Matrix4<f32> = 

        Matrix4::<f32>::new(
        -1.0, 0.0, 0.0, 0.0, 
        0.0, -1.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 1.0,
    );
}

impl RenderingPipeline for ForwardRenderingPipeline {
    fn render(
        &mut self,
        pov: &Camera,
        scene: &Scene,
        framebuffer: &GpuFramebuffer,
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
        let mut command_buffer =
            gpu::CommandBuffer::new(&super::app_state() .gpu, gpu::QueueType::Graphics).unwrap();
            super::app_state()
            .gpu
            .write_buffer_data(
                &self.camera_buffer,
                &[PerFrameData {
                    view: constants::Z_INVERT_MATRIX * pov.view(),
                    projection: pov.projection(),
                }],
            )
            .unwrap();
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
                wait_semaphores: &[&super::app_state().swapchain.image_available_semaphore],
                wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                signal_semaphores: &[&super::app_state().swapchain.render_finished_semaphore],
                fence: Some(&super::app_state().swapchain.in_flight_fence),
            })
            .unwrap();
    }

    fn get_context(&self) -> &dyn MaterialContext {
        &self.material_context
    }
}
