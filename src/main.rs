mod gpu;
mod gpu_extension;
mod utils;

use std::{
    ffi::{c_void, CString},
    mem::size_of,
    ptr::null,
};

use ash::vk::{
    self, BlendFactor, BlendOp, BufferCreateFlags, BufferCreateInfo, BufferUsageFlags,
    ClearColorValue, ColorComponentFlags, CommandBuffer, CommandBufferAllocateInfo,
    CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags, CommandPoolCreateFlags,
    CommandPoolCreateInfo, CommandPoolResetFlags, CompareOp, CullModeFlags, DynamicState, Extent2D,
    Format, FrontFace, GraphicsPipelineCreateInfo, LogicOp, MappedMemoryRange, MemoryAllocateInfo,
    MemoryMapFlags, MemoryPropertyFlags, Offset2D, Pipeline, PipelineBindPoint, PipelineCache,
    PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateFlags,
    PipelineColorBlendStateCreateInfo, PipelineCreateFlags, PipelineDepthStencilStateCreateFlags,
    PipelineDepthStencilStateCreateInfo, PipelineDynamicStateCreateFlags,
    PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateFlags,
    PipelineInputAssemblyStateCreateInfo, PipelineLayoutCreateFlags, PipelineLayoutCreateInfo,
    PipelineMultisampleStateCreateFlags, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateFlags, PipelineRasterizationStateCreateInfo,
    PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo, PipelineStageFlags,
    PipelineTessellationStateCreateFlags, PipelineTessellationStateCreateInfo,
    PipelineVertexInputStateCreateFlags, PipelineVertexInputStateCreateInfo,
    PipelineViewportStateCreateFlags, PipelineViewportStateCreateInfo, PolygonMode, PresentModeKHR,
    PrimitiveTopology, Rect2D, SampleCountFlags, Semaphore, ShaderStageFlags, SharingMode,
    StencilOp, StencilOpState, StructureType, SubmitInfo, SubpassContents,
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, Viewport,
};

use gpu::{Gpu, GpuConfiguration};
use gpu_extension::{DefaultExtensions, SurfaceParamters, SwapchainExtension};
use memoffset::offset_of;
use nalgebra::*;
use winit::{dpi::PhysicalSize, event_loop::ControlFlow};

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
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

    let mut gpu = Gpu::<SwapchainExtension<DefaultExtensions>>::new(
        GpuConfiguration {
            app_name: "Hello World!",
            engine_name: "Hello Engine!",
            enable_validation_layer: if cfg!(debug_assertions) { true } else { false },
            ..Default::default()
        },
        SurfaceParamters {
            inner_params: (),
            window,
            window_size: Extent2D {
                width: 1240,
                height: 720,
            },
        },
    )?;

    let surface = gpu.presentation_surface();
    let device = gpu.vk_logical_device();
    let physical_device = gpu.vk_physical_device();
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

    let find_memory_type = |type_filter: u32, mem_properties: MemoryPropertyFlags| -> u32 {
        let memory_properties = unsafe {
            gpu.info()
                .instance
                .get_physical_device_memory_properties(physical_device)
        };

        for i in 0..memory_properties.memory_type_count {
            if (type_filter & (1 << i)) > 0
                && memory_properties.memory_types[i as usize]
                    .property_flags
                    .intersects(mem_properties)
            {
                return i;
            }
        }

        panic!("No memory type found!")
    };

    let vertex_buffer = unsafe {
        let vertex_data = &[
            VertexData {
                position: vector![0.0, -0.5],
                color: vector![1.0, 0.0, 0.0],
            },
            VertexData {
                position: vector![0.5, 0.5],
                color: vector![0.0, 1.0, 0.0],
            },
            VertexData {
                position: vector![-0.5, 0.5],
                color: vector![0.0, 0.0, 1.0],
            },
        ];
        let create_info = BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: null(),
            flags: BufferCreateFlags::empty(),
            size: (size_of::<VertexData>() * vertex_data.len()) as u64,
            usage: BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: null(),
        };
        let buffer = device.create_buffer(&create_info, None).unwrap();

        let memory_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type = find_memory_type(
            memory_requirements.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        let memory_allocate_info = MemoryAllocateInfo {
            s_type: StructureType::MEMORY_ALLOCATE_INFO,
            p_next: null(),
            allocation_size: memory_requirements.size,
            memory_type_index: memory_type,
        };

        let device_memory = device
            .allocate_memory(&memory_allocate_info, None)
            .expect("Failed to allocate device memory");
        device
            .bind_buffer_memory(buffer, device_memory, 0)
            .expect("Failed to bind buffer memory!");

        let address = device
            .map_memory(
                device_memory,
                0,
                (std::mem::size_of::<VertexData>() * 3) as u64,
                MemoryMapFlags::empty(),
            )
            .expect("Failed to map memory!");
        let address = address as *mut VertexData;
        let address = std::slice::from_raw_parts_mut(address, 3);

        address.copy_from_slice(vertex_data);

        device
            .flush_mapped_memory_ranges(&[MappedMemoryRange {
                s_type: StructureType::MAPPED_MEMORY_RANGE,
                p_next: null(),
                memory: device_memory,
                offset: 0,
                size: memory_requirements.size,
            }])
            .expect("Failed to flush memory ranges");

        device.unmap_memory(device_memory);

        buffer
    };

    let pipeline = unsafe {
        let layout_infos = PipelineLayoutCreateInfo {
            s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: null(),
            flags: PipelineLayoutCreateFlags::empty(),
            set_layout_count: 0,
            p_set_layouts: null(),
            push_constant_range_count: 0,
            p_push_constant_ranges: null(),
        };
        let pipeline_layout = device.create_pipeline_layout(&layout_infos, None)?;

        let main_name = CString::new("main")?;

        let stages: [PipelineShaderStageCreateInfo; 2] = [
            PipelineShaderStageCreateInfo {
                s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: null(),
                flags: PipelineShaderStageCreateFlags::empty(),
                stage: ShaderStageFlags::VERTEX,
                module: vertex_module,
                p_name: main_name.as_ptr(),
                p_specialization_info: null(),
            },
            PipelineShaderStageCreateInfo {
                s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: null(),
                flags: PipelineShaderStageCreateFlags::empty(),
                stage: ShaderStageFlags::FRAGMENT,
                module: fragment_module,
                p_name: main_name.as_ptr(),
                p_specialization_info: null(),
            },
        ];

        let input_binding_descriptions = &[VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<VertexData>() as u32,
            input_rate: VertexInputRate::VERTEX,
        }];

        let input_attribute_descriptions = &[
            VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: Format::R32G32_SFLOAT,
                offset: offset_of!(VertexData, position) as u32,
            },
            VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: Format::R32G32B32_SFLOAT,
                offset: offset_of!(VertexData, color) as u32,
            },
        ];

        let input_stage = PipelineVertexInputStateCreateInfo {
            s_type: StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: input_binding_descriptions.as_ptr(),
            vertex_attribute_description_count: 2,
            p_vertex_attribute_descriptions: input_attribute_descriptions.as_ptr(),
        };

        let assembly_state = PipelineInputAssemblyStateCreateInfo {
            s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineInputAssemblyStateCreateFlags::empty(),
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
        };

        let tessellation_state: PipelineTessellationStateCreateInfo =
            PipelineTessellationStateCreateInfo {
                s_type: StructureType::PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                p_next: null(),
                flags: PipelineTessellationStateCreateFlags::empty(),
                patch_control_points: 0,
            };

        let viewport_state = PipelineViewportStateCreateInfo {
            s_type: StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineViewportStateCreateFlags::empty(),
            viewport_count: 1,
            p_viewports: null(),
            scissor_count: 1,
            p_scissors: null(),
        };

        let raster_state = PipelineRasterizationStateCreateInfo {
            s_type: StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: PolygonMode::FILL,
            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };

        let multisample_state = PipelineMultisampleStateCreateInfo {
            s_type: StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 1.0,
            p_sample_mask: null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
        };

        let stencil_state = PipelineDepthStencilStateCreateInfo {
            s_type: StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: CompareOp::GREATER,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: StencilOpState {
                fail_op: StencilOp::KEEP,
                pass_op: StencilOp::KEEP,
                depth_fail_op: StencilOp::KEEP,
                compare_op: CompareOp::ALWAYS,
                compare_mask: 0xFFFFFFF,
                write_mask: 0x0,
                reference: 0,
            },
            back: StencilOpState {
                fail_op: StencilOp::KEEP,
                pass_op: StencilOp::KEEP,
                depth_fail_op: StencilOp::KEEP,
                compare_op: CompareOp::ALWAYS,
                compare_mask: 0xFFFFFFF,
                write_mask: 0x0,
                reference: 0,
            },
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        };

        let color_blend = PipelineColorBlendStateCreateInfo {
            s_type: StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: LogicOp::COPY,
            attachment_count: 1,
            p_attachments: &[PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                src_color_blend_factor: BlendFactor::ONE,
                dst_color_blend_factor: BlendFactor::ZERO,
                color_blend_op: BlendOp::ADD,
                src_alpha_blend_factor: BlendFactor::ONE,
                dst_alpha_blend_factor: BlendFactor::ZERO,
                alpha_blend_op: BlendOp::ADD,
                color_write_mask: ColorComponentFlags::RGBA,
            }] as *const PipelineColorBlendAttachmentState,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };

        let dynamic_state = PipelineDynamicStateCreateInfo {
            s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            p_next: null(),
            flags: PipelineDynamicStateCreateFlags::empty(),
            dynamic_state_count: 2,
            p_dynamic_states: &[DynamicState::VIEWPORT, DynamicState::SCISSOR]
                as *const DynamicState,
        };

        let create_infos = [GraphicsPipelineCreateInfo {
            s_type: StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: null(),
            flags: PipelineCreateFlags::ALLOW_DERIVATIVES,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &input_stage as *const PipelineVertexInputStateCreateInfo,
            p_input_assembly_state: &assembly_state as *const PipelineInputAssemblyStateCreateInfo,
            p_tessellation_state: &tessellation_state as *const PipelineTessellationStateCreateInfo,
            p_viewport_state: &viewport_state as *const PipelineViewportStateCreateInfo,
            p_rasterization_state: &raster_state as *const PipelineRasterizationStateCreateInfo,
            p_multisample_state: &multisample_state as *const PipelineMultisampleStateCreateInfo,
            p_depth_stencil_state: &stencil_state as *const PipelineDepthStencilStateCreateInfo,
            p_color_blend_state: &color_blend as *const PipelineColorBlendStateCreateInfo,
            p_dynamic_state: &dynamic_state as *const PipelineDynamicStateCreateInfo,
            layout: pipeline_layout,
            render_pass: gpu.render_pass().clone(),
            subpass: 0,
            base_pipeline_handle: Pipeline::null(),
            base_pipeline_index: 0,
        }];

        device
            .create_graphics_pipelines(PipelineCache::null(), &create_infos, None)
            .unwrap()
    }[0];
    gpu.select_present_mode(PresentModeKHR::MAILBOX)?;

    unsafe {
        gpu.as_ref().destroy_shader_module(vertex_module, None);
        gpu.as_ref().destroy_shader_module(fragment_module, None);
    }
    event_loop.run(move |event, event_loop, mut control_flow| match event {
        winit::event::Event::NewEvents(_) => {}
        winit::event::Event::WindowEvent { window_id, event } => match event {
            winit::event::WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::ExitWithCode(0)
            }
            _ => {}
        },
        winit::event::Event::DeviceEvent { device_id, event } => {}
        winit::event::Event::UserEvent(_) => {}
        winit::event::Event::Suspended => {}
        winit::event::Event::Resumed => {}
        winit::event::Event::MainEventsCleared => {}
        winit::event::Event::RedrawRequested(_) => {
            let (index, framebuffer, next_image) = gpu.get_next_swapchain_image().unwrap();
            unsafe {
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
                        render_pass: gpu.render_pass().clone(),
                        framebuffer,
                        render_area: Rect2D {
                            offset: Offset2D { x: 0, y: 0 },
                            extent: gpu.extents(),
                        },
                        clear_value_count: 1,
                        p_clear_values: &vk::ClearValue {
                            color: ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        },
                    },
                    SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(command_buffer, PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_set_viewport(
                    command_buffer,
                    0,
                    &[Viewport {
                        x: 0 as f32,
                        y: 0 as f32,
                        width: gpu.extents().width as f32,
                        height: gpu.extents().height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                device.cmd_set_scissor(
                    command_buffer,
                    0,
                    &[Rect2D {
                        offset: Offset2D { x: 0, y: 0 },
                        extent: gpu.extents(),
                    }],
                );
                device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
                device.cmd_draw(command_buffer, 3, 1, 0, 0);
                device.cmd_end_render_pass(command_buffer);

                device.end_command_buffer(command_buffer).unwrap();
                device
                    .queue_submit(
                        gpu.graphics_queue(),
                        &[SubmitInfo {
                            s_type: StructureType::SUBMIT_INFO,
                            p_next: null(),
                            wait_semaphore_count: 0,
                            p_wait_semaphores: null(),
                            p_wait_dst_stage_mask: &PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                                as *const PipelineStageFlags,
                            command_buffer_count: 1,
                            p_command_buffers: &command_buffer as *const CommandBuffer,
                            signal_semaphore_count: 1,
                            p_signal_semaphores: gpu.render_finished_semaphore()
                                as *const Semaphore,
                        }],
                        gpu.in_flight_fence(),
                    )
                    .unwrap();

                let _ = gpu.present(index);
            };
        }
        winit::event::Event::RedrawEventsCleared => {}
        winit::event::Event::LoopDestroyed => *control_flow = ControlFlow::ExitWithCode(0),
    })
}
