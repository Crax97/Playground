mod gpu_pipeline;
mod material;
mod mesh;
mod scene;
mod static_deferred_renderer;
mod texture;
mod utils;

use std::{io::BufReader, mem::size_of, rc::Rc};

use ash::vk::{
    self, AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor,
    BlendOp, BufferUsageFlags, ColorComponentFlags, CompareOp, ComponentMapping, DependencyFlags,
    Format, ImageAspectFlags, ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageViewType,
    PipelineBindPoint, PipelineStageFlags, PresentModeKHR, PushConstantRange, SampleCountFlags,
    ShaderStageFlags, StencilOpState, SubpassDependency, SubpassDescriptionFlags, SUBPASS_EXTERNAL,
};

use gpu::{
    BlendState, BufferCreateInfo, DepthStencilState, FragmentStageInfo, FramebufferCreateInfo,
    GlobalBinding, Gpu, GpuConfiguration, ImageCreateInfo, MemoryDomain, Pipeline,
    PipelineDescription, RenderPass, RenderPassAttachment, RenderPassDescription,
    SubpassDescription, TransitionInfo, VertexAttributeDescription, VertexBindingDescription,
    VertexStageInfo,
};

use gpu_pipeline::GpuPipeline;
use material::Material;
use mesh::{Mesh, MeshCreateInfo};
use nalgebra::*;
use resource_map::ResourceMap;
use scene::{ForwardNaiveRenderer, Scene, ScenePrimitive, SceneRenderer};
use texture::Texture;
use winit::{dpi::PhysicalSize, event_loop::ControlFlow};

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameData {
    view: nalgebra::Matrix4<f32>,
    projection: nalgebra::Matrix4<f32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}
fn main() -> anyhow::Result<()> {
    env_logger::init();

    let resource_map = Rc::new(ResourceMap::new());

    let event_loop = winit::event_loop::EventLoop::default();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1240,
            height: 720,
        })
        .build(&event_loop)?;

    let gpu = Gpu::new(GpuConfiguration {
        app_name: "Hello World!",
        engine_name: "Hello Engine!",
        enable_validation_layer: if cfg!(debug_assertions) { true } else { false },
        window: &window,
    })?;

    let mut swapchain = gpu::Swapchain::new(&gpu, window)?;

    let cpu_image = image::load(
        BufReader::new(std::fs::File::open("images/texture.jpg")?),
        image::ImageFormat::Jpeg,
    )?;
    let cpu_image = cpu_image.into_rgba8();

    let vertex_module = utils::read_file_to_vk_module(&gpu, "./shaders/vertex.spirv")?;
    let fragment_module = utils::read_file_to_vk_module(&gpu, "./shaders/fragment.spirv")?;

    let mesh_data = MeshCreateInfo {
        indices: &[0, 1, 2, 2, 3, 0],
        positions: &[
            vector![-0.5, -0.5, 0.0],
            vector![0.5, -0.5, 0.0],
            vector![0.5, 0.5, 0.0],
            vector![-0.5, 0.5, 0.0],
        ],
        colors: &[
            vector![1.0, 0.0, 0.0],
            vector![0.0, 1.0, 0.0],
            vector![0.0, 0.0, 1.0],
            vector![1.0, 1.0, 1.0],
        ],
        normals: &[
            vector![0.0, 1.0, 0.0],
            vector![0.0, 1.0, 0.0],
            vector![0.0, 1.0, 0.0],
            vector![0.0, 1.0, 0.0],
        ],
        tangents: &[
            vector![0.0, 0.0, 1.0],
            vector![0.0, 0.0, 1.0],
            vector![0.0, 0.0, 1.0],
            vector![0.0, 0.0, 1.0],
        ],
        uvs: &[
            vector![1.0, 0.0],
            vector![0.0, 0.0],
            vector![0.0, 1.0],
            vector![1.0, 1.0],
        ],
    };

    let mesh = Mesh::new(&gpu, &mesh_data)?;
    let mesh = resource_map.add(mesh);

    let vertex_data = &[
        VertexData {
            position: vector![-0.5, -0.5],
            color: vector![1.0, 0.0, 0.0],
            uv: vector![1.0, 0.0],
        },
        VertexData {
            position: vector![0.5, -0.5],
            color: vector![0.0, 1.0, 0.0],
            uv: vector![0.0, 0.0],
        },
        VertexData {
            position: vector![0.5, 0.5],
            color: vector![0.0, 0.0, 1.0],
            uv: vector![0.0, 1.0],
        },
        VertexData {
            position: vector![-0.5, 0.5],
            color: vector![1.0, 1.0, 1.0],
            uv: vector![1.0, 1.0],
        },
    ];
    let indices: &[u32] = &[0, 1, 2, 2, 3, 0];
    let data_size = size_of::<VertexData>() * vertex_data.len();
    let vertex_buffer = {
        let create_info = BufferCreateInfo {
            size: data_size,
            usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
        };
        let buffer = gpu.create_buffer(&create_info, MemoryDomain::DeviceLocal)?;
        buffer
    };

    let index_size = size_of::<u32>() * indices.len();

    let index_buffer = {
        let create_info = BufferCreateInfo {
            size: index_size,
            usage: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
        };
        let buffer = gpu.create_buffer(&create_info, MemoryDomain::DeviceLocal)?;
        buffer
    };

    let uniform_buffer_1 = {
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

    let uniform_buffer_2 = {
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

    let texture = Texture::new_with_data(&gpu, cpu_image.width(), cpu_image.height(), &cpu_image)?;
    let texture = resource_map.add(texture);

    let depth_image = gpu.create_image(
        &ImageCreateInfo {
            width: swapchain.extents().width,
            height: swapchain.extents().height,
            format: vk::Format::D16_UNORM,
            usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        },
        MemoryDomain::DeviceLocal,
    )?;

    let depth_image_view = gpu.create_image_view(&gpu::ImageViewCreateInfo {
        image: &depth_image,
        view_type: ImageViewType::TYPE_2D,
        format: Format::D16_UNORM,
        components: ComponentMapping::default(),
        subresource_range: ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    })?;

    gpu.write_buffer_data(&vertex_buffer, vertex_data)?;
    gpu.write_buffer_data(&index_buffer, indices)?;

    gpu.transition_image_layout(
        &depth_image,
        TransitionInfo {
            layout: ImageLayout::UNDEFINED,
            access_mask: AccessFlags::empty(),
            stage_mask: PipelineStageFlags::TOP_OF_PIPE,
        },
        TransitionInfo {
            layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        },
        ImageAspectFlags::DEPTH,
    )?;

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

    let render_pass = RenderPass::new(
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

    let pipeline = GpuPipeline(
        Pipeline::new(
            &gpu,
            &render_pass,
            &PipelineDescription {
                global_bindings: &[
                    GlobalBinding {
                        binding_type: gpu::BindingType::Uniform,
                        index: 0,
                        stage: gpu::ShaderStage::Vertex,
                    },
                    GlobalBinding {
                        binding_type: gpu::BindingType::CombinedImageSampler,
                        index: 1,
                        stage: gpu::ShaderStage::Fragment,
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
                    module: &vertex_module,
                }),
                fragment_stage: Some(FragmentStageInfo {
                    entry_point: "main",
                    module: &fragment_module,
                    color_attachments: &[RenderPassAttachment {
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
                    }],
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
            },
        )?,
        render_pass,
    );

    let pipeline = resource_map.add(pipeline);

    let material_1 = Material::new(
        &gpu,
        resource_map.clone(),
        pipeline.clone(),
        vec![uniform_buffer_1],
        vec![texture.clone()],
    )?;
    let material_2 = Material::new(
        &gpu,
        resource_map.clone(),
        pipeline.clone(),
        vec![uniform_buffer_2],
        vec![texture.clone()],
    )?;
    let material_1 = resource_map.add(material_1);
    let material_2 = resource_map.add(material_2);

    swapchain.select_present_mode(PresentModeKHR::MAILBOX)?;

    let mut scene = Scene::new();

    scene.add(ScenePrimitive {
        mesh: mesh.clone(),
        material: material_1.clone(),
        transform: Matrix4::identity(),
    });
    scene.add(ScenePrimitive {
        mesh: mesh.clone(),
        material: material_2.clone(),
        transform: Matrix4::new_translation(&vector![0.0, 0.0, 1.0]),
    });

    let mut scene_renderer = ForwardNaiveRenderer::new(resource_map.clone(), swapchain.extents());
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            winit::event::Event::NewEvents(_) => {}
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::ExitWithCode(0)
                }
                winit::event::WindowEvent::Resized(_) => {
                    swapchain.recreate_swapchain().unwrap();
                }
                _ => {}
            },
            winit::event::Event::DeviceEvent { .. } => {}
            winit::event::Event::UserEvent(_) => {}
            winit::event::Event::Suspended => {}
            winit::event::Event::Resumed => {}
            winit::event::Event::MainEventsCleared => {
                swapchain.window.request_redraw();
            }
            winit::event::Event::RedrawRequested(..) => {
                for (idx, primitive) in scene.edit_all_primitives().iter_mut().enumerate() {
                    let mul = if idx % 2 == 0 { 1.0 } else { -1.0 };
                    primitive.transform =
                        primitive.transform * Matrix4::new_rotation(vector![0.0, 0.0, 0.002 * mul]);
                }

                let pipeline = resource_map.get(&pipeline);
                let sw_extents = swapchain.extents();
                let next_image = swapchain.acquire_next_image().unwrap();
                let framebuffer = gpu
                    .create_framebuffer(&FramebufferCreateInfo {
                        render_pass: &pipeline.1,
                        attachments: &[&next_image, &depth_image_view],
                        width: sw_extents.width,
                        height: sw_extents.height,
                    })
                    .unwrap();
                gpu.reset_state().unwrap();
                scene_renderer.render(&gpu, &scene, &framebuffer, &mut swapchain);
                let _ = swapchain.present();
            }
            winit::event::Event::RedrawEventsCleared => {}
            winit::event::Event::LoopDestroyed => gpu.wait_device_idle().unwrap(),
        }
    })
}
