mod mesh;
mod utils;

use std::{io::BufReader, mem::size_of};

use ash::vk::{
    self, AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor,
    BlendOp, BorderColor, BufferUsageFlags, ClearColorValue, ClearDepthStencilValue, ClearValue,
    ColorComponentFlags, CompareOp, ComponentMapping, DependencyFlags, Filter, Format,
    ImageAspectFlags, ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageViewType,
    IndexType, Offset2D, PipelineBindPoint, PipelineStageFlags, PresentModeKHR, Rect2D,
    SampleCountFlags, SamplerAddressMode, SamplerCreateFlags, SamplerCreateInfo, SamplerMipmapMode,
    StencilOpState, StructureType, SubpassDependency, SubpassDescriptionFlags, SUBPASS_EXTERNAL,
};

use gpu::{
    BeginRenderPassInfo, BlendState, BufferCreateInfo, BufferRange, DepthStencilState,
    DescriptorInfo, DescriptorSetInfo, FragmentStageInfo, FramebufferCreateInfo, GlobalBinding,
    Gpu, GpuBuffer, GpuConfiguration, GpuDescriptorSet, GpuFramebuffer, ImageCreateInfo, Material,
    MaterialDescription, MemoryDomain, RenderPass, RenderPassAttachment, RenderPassDescription,
    ResourceHandle, SamplerState, SubpassDescription, TransitionInfo, VertexAttributeDescription,
    VertexBindingDescription, VertexStageInfo,
};
use image::EncodableLayout;
use mesh::{Mesh, MeshCreateInfo};
use nalgebra::*;
use winit::{dpi::PhysicalSize, event_loop::ControlFlow};

#[repr(C)]
#[derive(Clone, Copy)]
struct PerObjectData {
    model: nalgebra::Matrix4<f32>,
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

    let device = gpu.vk_logical_device();
    let vertex_module = utils::read_file_to_vk_module(&device, "./shaders/vertex.spirv")?;
    let fragment_module = utils::read_file_to_vk_module(&device, "./shaders/fragment.spirv")?;

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
            size: std::mem::size_of::<PerObjectData>(),
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
            size: std::mem::size_of::<PerObjectData>(),
            usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
        };
        let buffer = gpu.create_buffer(
            &create_info,
            MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
        )?;
        buffer
    };

    let image = gpu.create_image(
        &ImageCreateInfo {
            width: cpu_image.width(),
            height: cpu_image.height(),
            format: vk::Format::R8G8B8A8_UNORM,
            usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
        },
        MemoryDomain::DeviceLocal,
    )?;

    let image_view = gpu.create_image_view(&gpu::ImageViewCreateInfo {
        image: &image,
        view_type: ImageViewType::TYPE_2D,
        format: Format::R8G8B8A8_UNORM,
        components: ComponentMapping::default(),
        subresource_range: ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    })?;

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
        &image,
        TransitionInfo {
            layout: ImageLayout::UNDEFINED,
            access_mask: AccessFlags::empty(),
            stage_mask: PipelineStageFlags::TOP_OF_PIPE,
        },
        TransitionInfo {
            layout: ImageLayout::TRANSFER_DST_OPTIMAL,
            access_mask: AccessFlags::TRANSFER_WRITE,
            stage_mask: PipelineStageFlags::TRANSFER,
        },
        ImageAspectFlags::COLOR,
    )?;

    let mb_16 = 1024 * 1024 * 16;

    let staging_buffer = {
        let create_info = BufferCreateInfo {
            size: mb_16,
            usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_SRC,
        };
        gpu.create_buffer(
            &create_info,
            MemoryDomain::HostCached | MemoryDomain::HostVisible,
        )?
    };
    gpu.write_buffer_data(&staging_buffer, cpu_image.as_bytes())?;
    gpu.copy_buffer_to_image(
        &staging_buffer,
        &image,
        cpu_image.width(),
        cpu_image.height(),
    )?;

    gpu.transition_image_layout(
        &image,
        TransitionInfo {
            layout: ImageLayout::TRANSFER_DST_OPTIMAL,
            access_mask: AccessFlags::TRANSFER_WRITE,
            stage_mask: PipelineStageFlags::TRANSFER,
        },
        TransitionInfo {
            layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            access_mask: AccessFlags::SHADER_READ,
            stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
        },
        ImageAspectFlags::COLOR,
    )?;

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

    let sampler = gpu.create_sampler(&SamplerCreateInfo {
        s_type: StructureType::SAMPLER_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: SamplerCreateFlags::empty(),
        mag_filter: Filter::LINEAR,
        min_filter: Filter::LINEAR,
        mipmap_mode: SamplerMipmapMode::LINEAR,
        address_mode_u: SamplerAddressMode::REPEAT,
        address_mode_v: SamplerAddressMode::REPEAT,
        address_mode_w: SamplerAddressMode::REPEAT,
        mip_lod_bias: 0.0,
        anisotropy_enable: vk::TRUE,
        max_anisotropy: gpu
            .physical_device_properties()
            .limits
            .max_sampler_anisotropy,
        compare_enable: vk::FALSE,
        compare_op: CompareOp::ALWAYS,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: BorderColor::default(),
        unnormalized_coordinates: vk::FALSE,
    })?;

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

    let material = Material::new(
        &gpu,
        &render_pass,
        &MaterialDescription {
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

            ..Default::default()
        },
    )?;

    let descriptor_set_1 = gpu.create_descriptor_set(&DescriptorSetInfo {
        descriptors: &[
            DescriptorInfo {
                binding: 0,
                element_type: gpu::DescriptorType::UniformBuffer(BufferRange {
                    handle: uniform_buffer_1.clone(),
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }),
                binding_stage: gpu::ShaderStage::Vertex,
            },
            DescriptorInfo {
                binding: 1,
                element_type: gpu::DescriptorType::CombinedImageSampler(SamplerState {
                    sampler: sampler.clone(),
                    image_view: image_view.clone(),
                    image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }),
                binding_stage: gpu::ShaderStage::Fragment,
            },
        ],
    })?;
    let descriptor_set_2 = gpu.create_descriptor_set(&DescriptorSetInfo {
        descriptors: &[
            DescriptorInfo {
                binding: 0,
                element_type: gpu::DescriptorType::UniformBuffer(BufferRange {
                    handle: uniform_buffer_2.clone(),
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }),
                binding_stage: gpu::ShaderStage::Vertex,
            },
            DescriptorInfo {
                binding: 1,
                element_type: gpu::DescriptorType::CombinedImageSampler(SamplerState {
                    sampler: sampler.clone(),
                    image_view: image_view.clone(),
                    image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }),
                binding_stage: gpu::ShaderStage::Fragment,
            },
        ],
    })?;

    swapchain.select_present_mode(PresentModeKHR::MAILBOX)?;

    let mut time = 0.0;
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
                time += 0.001;

                let next_image = swapchain.acquire_next_image().unwrap();
                let depth_image_view = gpu
                    .resource_map
                    .get(&depth_image_view)
                    .unwrap()
                    .inner_image_view();
                let next_image = gpu
                    .resource_map
                    .get(&next_image)
                    .unwrap()
                    .inner_image_view();

                let framebuffer = GpuFramebuffer::create(
                    gpu.vk_logical_device(),
                    &FramebufferCreateInfo {
                        render_pass: &render_pass,
                        attachments: &[next_image, depth_image_view],
                        width: swapchain.extents().width,
                        height: swapchain.extents().height,
                    },
                )
                .unwrap();
                gpu.reset_state().unwrap();
                render_textured_quads(
                    &gpu,
                    &mesh,
                    &[
                        (&uniform_buffer_1, &Vector3::zeros(), &descriptor_set_1),
                        (
                            &uniform_buffer_2,
                            &vector![0.0, 0.0, (time * 10.0).sin()],
                            &descriptor_set_2,
                        ),
                    ],
                    time,
                    &material,
                    &render_pass,
                    &framebuffer,
                    &mut swapchain,
                );
                let _ = swapchain.present();
            }
            winit::event::Event::RedrawEventsCleared => {}
            winit::event::Event::LoopDestroyed => gpu.wait_device_idle().unwrap(),
        }
    })
}

fn render_textured_quads(
    gpu: &Gpu,
    mesh: &Mesh,
    infos: &[(
        &ResourceHandle<GpuBuffer>,
        &Vector3<f32>,
        &ResourceHandle<GpuDescriptorSet>,
    )],
    time: f32,
    material: &Material,
    render_pass: &RenderPass,
    framebuffer: &GpuFramebuffer,
    swapchain: &mut gpu::Swapchain,
) {
    for (buf, off, _) in infos {
        gpu.write_buffer_data(
            buf,
            &[PerObjectData {
                model: nalgebra::Matrix4::new_rotation(vector![0.0, 0.0, time])
                    * nalgebra::Matrix4::new_translation(off),
                view: nalgebra::Matrix4::look_at_rh(
                    &point![2.0, 2.0, 2.0],
                    &point![0.0, 0.0, 0.0],
                    &vector![0.0, 0.0, -1.0],
                ),
                projection: nalgebra::Matrix4::new_perspective(1240.0 / 720.0, 45.0, 0.1, 10.0),
            }],
        )
        .unwrap();
    }
    {
        let mut command_buffer = gpu::CommandBuffer::new(
            gpu,
            gpu::CommandBufferSubmitInfo {
                wait_semaphores: vec![swapchain.image_available_semaphore.clone()],
                wait_stages: vec![PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                signal_semaphores: vec![swapchain.render_finished_semaphore.clone()],
                fence: Some(swapchain.in_flight_fence.clone()),
                target_queue: gpu::QueueType::Graphics,
            },
        )
        .unwrap();

        {
            let mut render_pass = command_buffer.begin_render_pass(&BeginRenderPassInfo {
                framebuffer,
                render_pass,
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
                    extent: swapchain.extents(),
                },
            });
            render_pass.bind_material(material);
            for (_, _, set) in infos {
                render_pass.bind_descriptor_sets(PipelineBindPoint::GRAPHICS, material, 0, &[set]);

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
                render_pass.draw_indexed(6, 1, 0, 0, 0);
            }
        }
    }
}
