use ash::vk::{Extent2D, Format};
use gpu::{CommandBuffer, Gpu, GpuImage, GpuImageView};
use nalgebra::{vector, Matrix4, Point3, Vector2, Vector3};
use resource_map::{ResourceHandle, ResourceMap};

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameData {
    view: nalgebra::Matrix4<f32>,
    projection: nalgebra::Matrix4<f32>,
}

use crate::{mesh::Mesh, Camera, MasterMaterial, MaterialDescription, MaterialInstance};

#[derive(Clone)]
pub struct ScenePrimitive {
    pub mesh: ResourceHandle<Mesh>,
    pub materials: Vec<ResourceHandle<MaterialInstance>>,
    pub transform: Matrix4<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Point,
    Directional {
        direction: Vector3<f32>,
        size: Vector2<f32>,
    },
    Spotlight {
        direction: Vector3<f32>,
        inner_cone_degrees: f32,
        outer_cone_degrees: f32,
    },
    Rect {
        direction: Vector3<f32>,
        width: f32,
        height: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ShadowSetup {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub ty: LightType,
    pub position: Point3<f32>,
    pub radius: f32,
    pub color: Vector3<f32>,
    pub intensity: f32,

    pub enabled: bool,
    pub shadow_setup: Option<ShadowSetup>,
}
impl Light {
    pub fn set_direction(&mut self, forward: Vector3<f32>) {
        match &mut self.ty {
            LightType::Point => {}
            LightType::Directional { direction, .. }
            | LightType::Spotlight { direction, .. }
            | LightType::Rect { direction, .. } => *direction = forward,
        }
    }

    pub fn direction(&self) -> Vector3<f32> {
        match self.ty {
            LightType::Point => {
                todo!()
            }
            LightType::Directional { direction, .. }
            | LightType::Spotlight { direction, .. }
            | LightType::Rect { direction, .. } => direction,
        }
    }

    pub(crate) fn shadow_view_matrices(&self) -> Vec<Matrix4<f32>> {
        match self.ty {
            LightType::Point => vec![
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, 1.0, 0.0]),
                    &vector![0.0, 0.0, 1.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, -1.0, 0.0]),
                    &vector![0.0, 0.0, 1.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![1.0, 0.0, 0.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![-1.0, 0.0, 0.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, 0.0, 1.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
                Matrix4::look_at_rh(
                    &self.position,
                    &(self.position + vector![0.0, 0.0, -1.0]),
                    &vector![0.0, 1.0, 0.0],
                ),
            ],
            LightType::Directional { direction, .. } => vec![Matrix4::look_at_rh(
                &self.position,
                &(self.position + direction),
                &vector![0.0, 1.0, 0.0],
            )],
            _ => vec![Matrix4::look_at_rh(
                &self.position,
                &(self.position + self.direction()),
                &vector![0.0, 1.0, 0.0],
            )],
        }
    }

    pub(crate) fn projection_matrix(&self) -> Matrix4<f32> {
        const ZNEAR: f32 = 1.0;
        match self.ty {
            LightType::Point => {
                Matrix4::new_perspective(1.0, 90.0f32.to_radians(), ZNEAR, self.radius.max(ZNEAR + 0.1))
            }
            LightType::Directional { size, .. } => Matrix4::new_orthographic(
                -size.x * 0.5,
                size.x * 0.5,
                -size.y * 0.5,
                size.y * 0.5,
                -self.radius * 0.5,
                self.radius * 0.5,
            ),
            LightType::Spotlight {
                outer_cone_degrees, ..
            } => Matrix4::new_perspective(1.0, outer_cone_degrees.to_radians(), ZNEAR, self.radius.max(ZNEAR + 0.1)),
            LightType::Rect { width, height, .. } => {
                Matrix4::new_perspective(width / height, 90.0, ZNEAR, self.radius.max(ZNEAR + 0.1))
            }
        }
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq)]
pub struct LightHandle(pub usize);

#[derive(Default)]
pub struct Scene {
    pub primitives: Vec<ScenePrimitive>,
    pub lights: Vec<Light>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            primitives: vec![],
            lights: vec![],
        }
    }

    pub fn add(&mut self, primitive: ScenePrimitive) -> usize {
        let idx = self.primitives.len();
        self.primitives.push(primitive);
        idx
    }

    pub fn add_light(&mut self, light: Light) -> LightHandle {
        let idx = self.lights.len();
        self.lights.push(light);
        LightHandle(idx)
    }

    pub fn edit(&mut self, idx: usize) -> &mut ScenePrimitive {
        &mut self.primitives[idx]
    }
    pub fn edit_light(&mut self, handle: &LightHandle) -> &mut Light {
        &mut self.lights[handle.0]
    }

    pub fn all_primitives(&self) -> &[ScenePrimitive] {
        &self.primitives
    }
    pub fn all_lights(&self) -> &[Light] {
        &self.lights
    }

    pub fn all_lights_mut(&mut self) -> &mut [Light] {
        &mut self.lights
    }

    pub fn all_enabled_lights(&self) -> impl Iterator<Item = &Light> {
        self.lights.iter().filter(|l| l.enabled)
    }

    pub fn edit_all_primitives(&mut self) -> &mut [ScenePrimitive] {
        &mut self.primitives
    }
}

pub struct Backbuffer<'a> {
    pub size: Extent2D,
    pub format: Format,
    pub image: &'a GpuImage,
    pub image_view: &'a GpuImageView,
}

pub trait RenderingPipeline {
    fn render(
        &mut self,
        pov: &Camera,
        scene: &Scene,
        backbuffer: &Backbuffer,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<CommandBuffer>;

    fn create_material(
        &mut self,
        gpu: &Gpu,
        material_description: MaterialDescription,
    ) -> anyhow::Result<MasterMaterial>;
}

/*
pub struct ForwardRenderingPipeline {
    resource_map: Rc<ResourceMap>,

    camera_buffer: GpuBuffer,
    camera_buffer_descriptor_set: GpuDescriptorSet,
    material_context: ForwardRendererMaterialContext,
    render_graph: RenderGraph,
    runner: GpuRunner,
}
impl ForwardRenderingPipeline {
    pub fn new(
        gpu: &Gpu,
        resource_map: Rc<ResourceMap>,
        swapchain: &Swapchain,
    ) -> anyhow::Result<Self> {
        let camera_buffer = {
            let create_info = BufferCreateInfo {
                label: Some("Forward Renderer - Camera buffer"),
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

        let material_context = ForwardRendererMaterialContext::new(gpu, swapchain)?;

        let render_graph = RenderGraph::new();

        Ok(Self {
            camera_buffer,
            resource_map,
            camera_buffer_descriptor_set,
            material_context,
            render_graph,
            runner: GpuRunner::new(),
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
        let pipeline = self.create_surface_material_pipeline(gpu, &material_description)?;

        let mut pipelines = HashMap::new();
        pipelines.insert(PipelineTarget::ColorAndDepth, pipeline);

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
impl RenderingPipeline for ForwardRenderingPipeline {
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
                    view: pov.view(),
                    projection: pov.projection(),
                }],
            )
            .unwrap();

        let swapchain_extents = swapchain.extents();
        let swapchain_format = swapchain.present_format();

        let (image, view) = swapchain.acquire_next_image()?;

        let mut color_depth_hashmap: HashMap<&Pipeline, Vec<ScenePrimitive>> = HashMap::new();

        for primitive in scene.primitives.iter() {
            for mat in &primitive.materials {
                let material = self.resource_map.get(mat);
                color_depth_hashmap
                    .entry(&material.pipelines[&PipelineTarget::ColorAndDepth])
                    .or_default()
                    .push(primitive.clone());
            }
        }

        let depth_buffer = self.render_graph.use_image(
            "depth-buffer",
            &crate::ImageDescription {
                width: swapchain_extents.width,
                height: swapchain_extents.height,
                format: gpu::ImageFormat::Depth,
                samples: 1,
                present: false,
            },
            false,
        )?;
        let color_buffer = self.render_graph.use_image(
            "color-buffer",
            &crate::ImageDescription {
                width: swapchain_extents.width,
                height: swapchain_extents.height,
                format: swapchain_format.into(),
                samples: 1,
                present: true,
            },
            true,
        )?;
        self.render_graph.persist_resource(&color_buffer);

        let forward_pass_handle = self
            .render_graph
            .begin_render_pass("ForwardPass", swapchain_extents)?
            .writes_attachments(&[color_buffer, depth_buffer])
            .mark_external()
            .commit();

        self.render_graph.compile()?;

        let mut context = GraphRunContext::new(
            &crate::app_state().gpu,
            &mut crate::app_state_mut().swapchain,
            crate::app_state().time().frames_since_start(),
        );

        context.register_callback(&forward_pass_handle, |_: &Gpu, ctx| {
            for (pipeline, primitives) in color_depth_hashmap.iter() {
                {
                    ctx.render_pass_command.bind_descriptor_sets(
                        PipelineBindPoint::GRAPHICS,
                        &pipeline,
                        0,
                        &[&self.camera_buffer_descriptor_set],
                    );
                    for (idx, primitive) in primitives.iter().enumerate() {
                        let primitive_label = ctx.render_pass_command.begin_debug_region(
                            &format!("Rendering primitive {}", idx),
                            [0.0, 0.3, 0.4, 1.0],
                        );
                        let mesh = self.resource_map.get(&primitive.mesh);

                        for (idx, mesh_prim) in mesh.primitives.iter().enumerate() {
                            let material = &primitive.materials[idx];
                            let material = self.resource_map.get(material);

                            ctx.render_pass_command.bind_pipeline(&pipeline);
                            ctx.render_pass_command.bind_descriptor_sets(
                                PipelineBindPoint::GRAPHICS,
                                &pipeline,
                                1,
                                &[&material.resources_descriptor_set],
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
                        }
                        primitive_label.end();
                    }
                }
            }
        });

        context.inject_external_renderpass(
            &forward_pass_handle,
            self.material_context
                .get_material_render_pass(MaterialDomain::Surface),
        );
        context.inject_external_image(&color_buffer, image, view);
        self.render_graph.run(context, &mut self.runner)?;

        Ok(())
    }

    fn get_context(&self) -> &dyn MaterialContext {
        &self.material_context
    }
}
 */
