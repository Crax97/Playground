mod gpu;
mod utils;

use std::{
    ffi::CString,
    io::BufReader,
    mem::size_of,
    ops::Deref,
    ptr::{addr_of, null},
    sync::Arc,
};

use ash::{
    extensions::khr::Swapchain,
    vk::{
        self, AccessFlags, AttachmentDescription, AttachmentDescriptionFlags, AttachmentLoadOp,
        AttachmentReference, AttachmentStoreOp, BlendFactor, BlendOp, BorderColor,
        BufferCreateFlags, BufferCreateInfo, BufferUsageFlags, ClearColorValue,
        ColorComponentFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo,
        CommandBufferLevel, CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo,
        CommandPoolResetFlags, CompareOp, CullModeFlags, DependencyFlags, DescriptorBufferInfo,
        DescriptorImageInfo, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
        DescriptorPoolSize, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags,
        DescriptorSetLayoutCreateInfo, DescriptorType, DynamicState, Filter, Format,
        FramebufferCreateFlags, FramebufferCreateInfo, FrontFace, GraphicsPipelineCreateInfo,
        ImageLayout, ImageUsageFlags, ImageView, IndexType, LogicOp, MappedMemoryRange,
        MemoryAllocateInfo, MemoryMapFlags, MemoryPropertyFlags, Offset2D, Pipeline,
        PipelineBindPoint, PipelineCache, PipelineColorBlendAttachmentState,
        PipelineColorBlendStateCreateFlags, PipelineColorBlendStateCreateInfo, PipelineCreateFlags,
        PipelineDepthStencilStateCreateFlags, PipelineDepthStencilStateCreateInfo,
        PipelineDynamicStateCreateFlags, PipelineDynamicStateCreateInfo,
        PipelineInputAssemblyStateCreateFlags, PipelineInputAssemblyStateCreateInfo,
        PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, PipelineMultisampleStateCreateFlags,
        PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateFlags,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateFlags,
        PipelineShaderStageCreateInfo, PipelineStageFlags, PipelineTessellationStateCreateFlags,
        PipelineTessellationStateCreateInfo, PipelineVertexInputStateCreateFlags,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateFlags,
        PipelineViewportStateCreateInfo, PolygonMode, PresentModeKHR, PrimitiveTopology, Rect2D,
        RenderPassCreateFlags, RenderPassCreateInfo, SampleCountFlags, SamplerAddressMode,
        SamplerCreateFlags, SamplerCreateInfo, SamplerMipmapMode, Semaphore, ShaderStageFlags,
        SharingMode, StencilOp, StencilOpState, StructureType, SubmitInfo, SubpassContents,
        SubpassDependency, SubpassDescription, SubpassDescriptionFlags,
        VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, Viewport,
        WriteDescriptorSet, SUBPASS_EXTERNAL,
    },
};

use gpu::{
    BlendState, ColorAttachment, FragmentStageInfo, GlobalBinding, Gpu, GpuBuffer,
    GpuConfiguration, ImageCreateInfo, Material, MaterialDescription, MemoryDomain,
    PasstroughAllocator, ResourceHandle, TransitionInfo, VertexAttributeDescription,
    VertexBindingDescription, VertexStageInfo,
};
use image::{EncodableLayout, RgbaImage};
use memoffset::offset_of;
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

    let command_pool = unsafe {
        device.create_command_pool(
            &CommandPoolCreateInfo {
                s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                p_next: null(),
                flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index: gpu.graphics_queue_family_index(),
            },
            None,
        )
    }?;

    let command_buffer = unsafe {
        device.allocate_command_buffers(&CommandBufferAllocateInfo {
            s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: null(),
            command_pool,
            level: CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        })?[0]
    };

    let mb_16 = 1024 * 1024 * 16;

    let staging_buffer = {
        let create_info = BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: null(),
            flags: BufferCreateFlags::empty(),
            size: mb_16,
            usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: null(),
        };
        gpu.create_buffer(
            &create_info,
            MemoryDomain::HostCached | MemoryDomain::HostVisible,
        )?
    };
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
    let data_size = (size_of::<VertexData>() * vertex_data.len()) as u64;
    let vertex_buffer = {
        let create_info = BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: null(),
            flags: BufferCreateFlags::empty(),
            size: data_size,
            usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: null(),
        };
        let buffer = gpu.create_buffer(&create_info, MemoryDomain::DeviceLocal)?;
        buffer
    };

    let index_size = (size_of::<u32>() * indices.len()) as u64;

    let index_buffer = {
        let create_info = BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: null(),
            flags: BufferCreateFlags::empty(),
            size: index_size,
            usage: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: null(),
        };
        let buffer = gpu.create_buffer(&create_info, MemoryDomain::DeviceLocal)?;
        buffer
    };

    let uniform_buffer = {
        let create_info = BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: null(),
            flags: BufferCreateFlags::empty(),
            size: std::mem::size_of::<PerObjectData>() as u64,
            usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
            sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: null(),
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

    gpu.resource_map
        .get(&staging_buffer)
        .unwrap()
        .write_data(vertex_data);

    gpu.copy_buffer(&staging_buffer, &vertex_buffer, data_size)?;
    gpu.resource_map
        .get(&staging_buffer)
        .unwrap()
        .write_data(indices);

    gpu.copy_buffer(&staging_buffer, &index_buffer, index_size)?;

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
    )?;

    gpu.resource_map
        .get(&staging_buffer)
        .unwrap()
        .write_data(cpu_image.as_bytes());
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
            .state
            .physical_device
            .device_properties
            .limits
            .max_sampler_anisotropy,
        compare_enable: vk::FALSE,
        compare_op: CompareOp::ALWAYS,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: BorderColor::default(),
        unnormalized_coordinates: vk::FALSE,
    })?;

    let material = Material::new(
        &gpu,
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
            vertex_inputs: &[VertexBindingDescription {
                binding: 0,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<VertexData>() as u32,
                attributes: &[
                    VertexAttributeDescription {
                        location: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: offset_of!(VertexData, position) as u32,
                    },
                    VertexAttributeDescription {
                        location: 1,
                        format: vk::Format::R32G32B32_SFLOAT,
                        offset: offset_of!(VertexData, color) as u32,
                    },
                    VertexAttributeDescription {
                        location: 2,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: offset_of!(VertexData, uv) as u32,
                    },
                ],
            }],
            vertex_stage: Some(VertexStageInfo {
                entry_point: "main",
                module: vertex_module,
            }),
            fragment_stage: Some(FragmentStageInfo {
                entry_point: "main",
                module: fragment_module,
                color_attachments: &[ColorAttachment {
                    format: swapchain.present_format.format,
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
                dependencies: &[SubpassDependency {
                    src_subpass: SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: AccessFlags::empty(),
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: DependencyFlags::empty(),
                }],
            }),
            input_topology: gpu::PrimitiveTopology::TriangleList,
            primitive_restart: false,
            polygon_mode: gpu::PolygonMode::Fill,
            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::ClockWise,
            ..Default::default()
        },
    )?;

    let descriptor_set_layout = {
        let uniform_buffer_binding = DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::VERTEX,
            p_immutable_samplers: null(),
        };
        let sampler_binding = DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: null(),
        };
        let create_info = DescriptorSetLayoutCreateInfo {
            s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: null(),
            flags: DescriptorSetLayoutCreateFlags::empty(),
            binding_count: 2,
            p_bindings: [uniform_buffer_binding, sampler_binding].as_ptr(),
        };
        unsafe { device.create_descriptor_set_layout(&create_info, None) }?
    };

    let descriptor_pool = unsafe {
        let pool_size_uniform_buffer = DescriptorPoolSize {
            ty: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        };
        let pool_size_sampler = DescriptorPoolSize {
            ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };
        device.create_descriptor_pool(
            &DescriptorPoolCreateInfo {
                s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                p_next: null(),
                flags: DescriptorPoolCreateFlags::empty(),
                max_sets: 1,
                pool_size_count: 2,
                p_pool_sizes: [pool_size_uniform_buffer, pool_size_sampler].as_ptr(),
            },
            None,
        )?
    };

    let descriptor_set = unsafe {
        let descriptor_set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
            s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: addr_of!(descriptor_set_layout),
        })?[0];

        let buffer_info = DescriptorBufferInfo {
            buffer: *gpu.resource_map.get(&uniform_buffer).unwrap().deref(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let image_info = DescriptorImageInfo {
            sampler: *gpu.resource_map.get(&sampler).unwrap().deref(),
            image_view: gpu.resource_map.get(&image).unwrap().view,
            image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        device.update_descriptor_sets(
            &[
                WriteDescriptorSet {
                    s_type: StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: DescriptorType::UNIFORM_BUFFER,
                    p_image_info: null(),
                    p_buffer_info: addr_of!(buffer_info),
                    p_texel_buffer_view: null(),
                },
                WriteDescriptorSet {
                    s_type: StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: addr_of!(image_info),
                    p_buffer_info: null(),
                    p_texel_buffer_view: null(),
                },
            ],
            &[],
        );

        descriptor_set
    };

    swapchain.select_present_mode(PresentModeKHR::MAILBOX)?;

    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }
    let mut time = 0.0;
    event_loop.run(move |event, event_loop, mut control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            winit::event::Event::NewEvents(_) => {}
            winit::event::Event::WindowEvent { window_id, event } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::ExitWithCode(0)
                }
                winit::event::WindowEvent::Resized(_) => unsafe {
                    swapchain.recreate_swapchain();
                },
                _ => {}
            },
            winit::event::Event::DeviceEvent { device_id, event } => {}
            winit::event::Event::UserEvent(_) => {}
            winit::event::Event::Suspended => {}
            winit::event::Event::Resumed => {}
            winit::event::Event::MainEventsCleared => {
                swapchain.window.request_redraw();
            }
            winit::event::Event::RedrawRequested(window) => {
                time += 0.001;
                render_frame(
                    &gpu,
                    &uniform_buffer,
                    time,
                    &material,
                    &mut swapchain,
                    &device,
                    command_pool,
                    command_buffer,
                    descriptor_set,
                    &vertex_buffer,
                    &index_buffer,
                );
            }
            winit::event::Event::RedrawEventsCleared => {}
            winit::event::Event::LoopDestroyed => unsafe {
                device.device_wait_idle().unwrap();
                //                device.free_memory(device_memory, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.free_command_buffers(command_pool, &[command_buffer]);
                device.destroy_command_pool(command_pool, None);
            },
        }
    })
}

fn render_frame(
    gpu: &Gpu,
    uniform_buffer: &ResourceHandle<GpuBuffer>,
    time: f32,
    material: &Material,
    swapchain: &mut gpu::Swapchain,
    device: &ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: CommandBuffer,
    descriptor_set: vk::DescriptorSet,
    vertex_buffer: &ResourceHandle<GpuBuffer>,
    index_buffer: &ResourceHandle<GpuBuffer>,
) {
    gpu.resource_map
        .get(&uniform_buffer)
        .unwrap()
        .write_data(&[PerObjectData {
            model: nalgebra::Matrix4::new_rotation(vector![0.0, 0.0, time]),
            view: nalgebra::Matrix4::look_at_rh(
                &point![2.0, 2.0, 2.0],
                &point![0.0, 0.0, 0.0],
                &vector![0.0, 0.0, -1.0],
            ),
            projection: nalgebra::Matrix4::new_perspective(1240.0 / 720.0, 45.0, 0.1, 10.0),
        }]);

    let next_image = swapchain.acquire_next_image().unwrap();
    let framebuffer = unsafe {
        let create_info = FramebufferCreateInfo {
            s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: null(),
            flags: FramebufferCreateFlags::empty(),
            render_pass: material.render_pass,
            attachment_count: 1,
            p_attachments: &next_image as *const ImageView,
            width: swapchain.extents().width,
            height: swapchain.extents().height,
            layers: 1,
        };

        gpu.state
            .logical_device
            .create_framebuffer(&create_info, None)
            .unwrap()
    };
    unsafe {
        render_textured_quad(
            device,
            command_pool,
            command_buffer,
            material,
            framebuffer,
            swapchain,
            descriptor_set,
            gpu,
            vertex_buffer,
            index_buffer,
        );

        gpu.state
            .logical_device
            .destroy_framebuffer(framebuffer, None);
    };
}

unsafe fn render_textured_quad(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: CommandBuffer,
    material: &Material,
    framebuffer: vk::Framebuffer,
    swapchain: &mut gpu::Swapchain,
    descriptor_set: vk::DescriptorSet,
    gpu: &Gpu,
    vertex_buffer: &ResourceHandle<GpuBuffer>,
    index_buffer: &ResourceHandle<GpuBuffer>,
) {
    device
        .reset_command_pool(command_pool, CommandPoolResetFlags::empty())
        .unwrap();
    device
        .begin_command_buffer(
            command_buffer,
            &CommandBufferBeginInfo {
                s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: null(),
                flags: CommandBufferUsageFlags::empty(),
                p_inheritance_info: null(),
            },
        )
        .unwrap();
    device.cmd_begin_render_pass(
        command_buffer,
        &vk::RenderPassBeginInfo {
            s_type: StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: null(),
            render_pass: material.render_pass,
            framebuffer,
            render_area: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: swapchain.extents(),
            },
            clear_value_count: 1,
            p_clear_values: &vk::ClearValue {
                color: ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        SubpassContents::INLINE,
    );
    device.cmd_bind_pipeline(
        command_buffer,
        PipelineBindPoint::GRAPHICS,
        material.pipeline,
    );
    device.cmd_set_viewport(
        command_buffer,
        0,
        &[Viewport {
            x: 0 as f32,
            y: 0 as f32,
            width: swapchain.extents().width as f32,
            height: swapchain.extents().height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
    );
    device.cmd_set_scissor(
        command_buffer,
        0,
        &[Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: swapchain.extents(),
        }],
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        PipelineBindPoint::GRAPHICS,
        material.pipeline_layout,
        0,
        &[descriptor_set],
        &[],
    );
    device.cmd_bind_vertex_buffers(
        command_buffer,
        0,
        &[*gpu.resource_map.get(&vertex_buffer).unwrap().deref()],
        &[0],
    );
    device.cmd_bind_index_buffer(
        command_buffer,
        *gpu.resource_map.get(&index_buffer).unwrap().deref(),
        0,
        IndexType::UINT32,
    );
    device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 0);
    device.cmd_end_render_pass(command_buffer);

    device.end_command_buffer(command_buffer).unwrap();
    device
        .queue_submit(
            gpu.graphics_queue(),
            &[SubmitInfo {
                s_type: StructureType::SUBMIT_INFO,
                p_next: null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: swapchain.image_available_semaphore.deref() as *const Semaphore,
                p_wait_dst_stage_mask: &PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    as *const PipelineStageFlags,
                command_buffer_count: 1,
                p_command_buffers: &command_buffer as *const CommandBuffer,
                signal_semaphore_count: 1,
                p_signal_semaphores: swapchain.render_finished_semaphore.deref()
                    as *const Semaphore,
            }],
            *swapchain.in_flight_fence,
        )
        .unwrap();

    let _ = swapchain.present();
}
