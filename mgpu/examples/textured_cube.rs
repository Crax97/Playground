mod util;

use std::num::NonZeroU32;

use bytemuck::offset_of;
use glam::{vec3, Vec3};
use mgpu::{
    AttachmentStoreOp, Binding, BindingSetDescription, BindingSetElement, BindingSetLayout,
    BindingSetLayoutInfo, BindingType, BufferDescription, BufferUsageFlags, CompareOp,
    DepthStencilState, DepthStencilTarget, DepthStencilTargetInfo, DepthStencilTargetLoadOp,
    DeviceConfiguration, DeviceFeatures, DevicePreference, Extents2D, Extents3D, FilterMode,
    FragmentStageInfo, Graphics, GraphicsPipelineDescription, ImageDescription, ImageDimension,
    ImageFormat, ImageUsageFlags, ImageViewDescription, ImageWriteParams, MemoryDomain, MipmapMode,
    Rect2D, RenderPassDescription, RenderTarget, RenderTargetInfo, RenderTargetLoadOp, SampleCount,
    SamplerDescription, ShaderModuleDescription, ShaderStageFlags, SwapchainCreationInfo,
    VertexInputDescription, VertexInputFrequency, VertexStageInfo,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use shaderc::ShaderKind;
use util::*;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;

const VERTEX_SHADER: &str = "
#version 460
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 fs_uv;

layout(set = 0, binding = 2, std140) uniform ObjectData {
    mat4 mvp;
};

void main() {
    vec4 vs_pos = mvp * vec4(pos, 1.0);
    gl_Position = vs_pos;
    fs_uv = uv;
}
";
const FRAGMENT_SHADER: &str = "
#version 460
layout(set = 0, binding = 0) uniform sampler tex_sampler;
layout(set = 0, binding = 1) uniform texture2D tex;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 color;

void main() {
    color = texture(sampler2D(tex, tex_sampler), uv);
}
";

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("Textured Cube")
        .build(&event_loop)
        .unwrap();

    let device = mgpu::Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::HAL_DEBUG_LAYERS,
        device_preference: Some(DevicePreference::HighPerformance),
        display_handle: Some(event_loop.display_handle().unwrap().as_raw()),
        desired_frames_in_flight: 3,
    })
    .expect("Failed to create gpu device");
    let mut swapchain = device
        .create_swapchain(&SwapchainCreationInfo {
            display_handle: window.display_handle().unwrap(),
            window_handle: window.window_handle().unwrap(),
            preferred_format: None,
            preferred_present_mode: None,
        })
        .expect("Failed to create swapchain");
    let cube_data = vec![
        vertex(-1.0, -1.0, 1.0, 0.0, 0.0),
        vertex(1.0, 1.0, 1.0, 1.0, 1.0),
        vertex(-1.0, 1.0, 1.0, 0.0, 1.0),
        vertex(1.0, -1.0, 1.0, 1.0, 0.0),
        vertex(-1.0, -1.0, -1.0, 0.0, 0.0),
        vertex(1.0, 1.0, -1.0, 1.0, 1.0),
        vertex(-1.0, 1.0, -1.0, 0.0, 1.0),
        vertex(1.0, -1.0, -1.0, 1.0, 0.0),
        vertex(1.0, -1.0, -1.0, 0.0, 0.0),
        vertex(1.0, 1.0, 1.0, 1.0, 1.0),
        vertex(1.0, 1.0, -1.0, 0.0, 1.0),
        vertex(1.0, -1.0, 1.0, 1.0, 0.0),
        vertex(-1.0, -1.0, -1.0, 0.0, 0.0),
        vertex(-1.0, 1.0, 1.0, 1.0, 1.0),
        vertex(-1.0, 1.0, -1.0, 0.0, 1.0),
        vertex(-1.0, -1.0, 1.0, 1.0, 0.0),
        vertex(-1.0, 1.0, -1.0, 0.0, 0.0),
        vertex(1.0, 1.0, 1.0, 1.0, 1.0),
        vertex(1.0, 1.0, -1.0, 0.0, 1.0),
        vertex(-1.0, 1.0, 1.0, 1.0, 0.0),
        vertex(-1.0, -1.0, -1.0, 0.0, 0.0),
        vertex(1.0, -1.0, 1.0, 1.0, 1.0),
        vertex(1.0, -1.0, -1.0, 0.0, 1.0),
        vertex(-1.0, -1.0, 1.0, 1.0, 0.0),
    ];

    let cube_indices = [
        0, 1, 2, 3, 1, 0, //Bottom
        6, 5, 4, 4, 5, 7, // Front
        10, 9, 8, 8, 9, 11, // Left
        12, 13, 14, 15, 13, 12, // Right
        16, 17, 18, 19, 17, 16, // Up
        22, 21, 20, 20, 21, 23, // Down
    ];
    let mut rotation = 0.0f32;
    let view = glam::Mat4::look_at_rh(vec3(-5.0, 10.0, -5.0), Vec3::default(), vec3(0.0, 1.0, 0.0));
    let projection = glam::Mat4::perspective_rh(75.0f32.to_radians(), 800.0 / 600.0, 0.01, 1000.0);

    let texture_data = util::read_image_data("mgpu/examples/assets/david.jpg");
    let mut depth_image = device
        .create_image(&ImageDescription {
            label: Some("Depth image"),
            usage_flags: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            extents: Extents3D {
                width: 800,
                height: 600,
                depth: 1,
            },
            dimension: ImageDimension::D2,
            mips: NonZeroU32::new(1).unwrap(),
            array_layers: NonZeroU32::new(1).unwrap(),
            samples: SampleCount::One,
            format: ImageFormat::Depth32,
            memory_domain: MemoryDomain::Gpu,
        })
        .unwrap();
    let mut depth_image_view = device
        .create_image_view(&ImageViewDescription {
            label: Some("Depth image view"),
            format: ImageFormat::Depth32,
            aspect: mgpu::ImageAspect::Depth,
            image: depth_image,
            image_subresource: depth_image.whole_subresource(),
            dimension: ImageDimension::D2,
        })
        .unwrap();
    let texture_image = device
        .create_image(&ImageDescription {
            label: Some("Cube Texture"),
            usage_flags: ImageUsageFlags::SAMPLED
                | ImageUsageFlags::TRANSFER_DST
                | ImageUsageFlags::TRANSFER_SRC,
            extents: Extents3D {
                width: 512,
                height: 512,
                depth: 1,
            },
            dimension: ImageDimension::D2,
            mips: NonZeroU32::new(9).unwrap(),
            array_layers: NonZeroU32::new(1).unwrap(),
            samples: SampleCount::One,
            format: ImageFormat::Rgba8,
            memory_domain: MemoryDomain::Gpu,
        })
        .unwrap();

    device
        .write_image_data(
            texture_image,
            &ImageWriteParams {
                data: &texture_data,
                region: texture_image.mip_region(0),
            },
        )
        .unwrap();

    // The order of operations is important!
    // Requesting a mip chain BEFORE we write to the image doesn't guarantee that
    // the mip chain will be created from the new image data
    device
        .generate_mip_chain(texture_image, FilterMode::Linear)
        .unwrap();

    let texture_image_view = device
        .create_image_view(&ImageViewDescription {
            label: Some("David image view"),
            format: ImageFormat::Rgba8,
            aspect: mgpu::ImageAspect::Color,
            image: texture_image,
            image_subresource: texture_image.whole_subresource(),
            dimension: ImageDimension::D2,
        })
        .unwrap();

    let cube_data_buffer = device
        .create_buffer(&BufferDescription {
            label: Some("Triangle data"),
            usage_flags: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of_val(cube_data.as_slice()),
            memory_domain: MemoryDomain::Gpu,
        })
        .unwrap();

    let cube_index_buffer = device
        .create_buffer(&BufferDescription {
            label: Some("Triangle index buffer"),
            usage_flags: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of_val(cube_indices.as_slice()),
            memory_domain: MemoryDomain::Gpu,
        })
        .unwrap();

    let cube_object_data = device
        .create_buffer(&BufferDescription {
            label: Some("Cube Object Data"),
            usage_flags: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of::<glam::Mat4>(),
            memory_domain: MemoryDomain::Gpu,
        })
        .unwrap();

    device
        .write_buffer(
            cube_data_buffer,
            &cube_data_buffer.write_all_params(bytemuck::cast_slice(&cube_data)),
        )
        .unwrap();

    device
        .write_buffer(
            cube_index_buffer,
            &cube_index_buffer.write_all_params(bytemuck::cast_slice(&cube_indices)),
        )
        .unwrap();

    let vertex_shader_source = compile_glsl(VERTEX_SHADER, ShaderKind::Vertex);
    let fragment_shader_source = compile_glsl(FRAGMENT_SHADER, ShaderKind::Fragment);
    let vertex_shader_module = device
        .create_shader_module(&ShaderModuleDescription {
            label: Some("Triangle Vertex Shader"),
            source: &vertex_shader_source,
        })
        .unwrap();
    let fragment_shader_module = device
        .create_shader_module(&ShaderModuleDescription {
            label: Some("Triangle Fragment Shader"),
            source: &fragment_shader_source,
        })
        .unwrap();
    let binding_set_layout = BindingSetLayout {
        binding_set_elements: vec![
            BindingSetElement {
                binding: 0,
                array_length: 1,
                ty: mgpu::BindingSetElementKind::Sampler,
                shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
            },
            BindingSetElement {
                binding: 1,
                array_length: 1,
                ty: mgpu::BindingSetElementKind::SampledImage,
                shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
            },
            BindingSetElement {
                binding: 2,
                array_length: 1,
                ty: mgpu::BindingSetElementKind::Buffer {
                    ty: mgpu::BufferType::Uniform,
                    access_mode: mgpu::StorageAccessMode::Read,
                },
                shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
            },
        ],
    };
    let pipeline = device
        .create_graphics_pipeline(
            &GraphicsPipelineDescription::new(
                Some("Textured Cube"),
                &VertexStageInfo {
                    shader: &vertex_shader_module,
                    entry_point: "main",
                    vertex_inputs: &[
                        VertexInputDescription {
                            location: 0,
                            stride: std::mem::size_of::<Vertex>(),
                            offset: 0,
                            frequency: VertexInputFrequency::PerVertex,
                            format: mgpu::VertexAttributeFormat::Float3,
                        },
                        VertexInputDescription {
                            location: 1,
                            stride: std::mem::size_of::<Vertex>(),
                            offset: offset_of!(Vertex, uv),
                            frequency: VertexInputFrequency::PerVertex,
                            format: mgpu::VertexAttributeFormat::Float2,
                        },
                    ],
                },
            )
            .fragment_stage(&FragmentStageInfo {
                shader: &fragment_shader_module,
                entry_point: "main",
                render_targets: &[RenderTargetInfo {
                    format: ImageFormat::Rgba8,
                    blend: None,
                }],
                depth_stencil_target: Some(&DepthStencilTargetInfo {
                    format: ImageFormat::Depth32,
                }),
            })
            .binding_set_layouts(&[BindingSetLayoutInfo {
                layout: binding_set_layout.clone(),
                set: 0,
            }])
            .depth_stencil_state(DepthStencilState {
                depth_test_enabled: true,
                depth_write_enabled: true,
                depth_compare_op: CompareOp::Less,
            }),
        )
        .unwrap();

    let sampler = device
        .create_sampler(&SamplerDescription {
            label: Some("Cube Texture Sampler"),
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_mode: MipmapMode::Linear,
            ..Default::default()
        })
        .unwrap();

    let binding_set = device
        .create_binding_set(
            &BindingSetDescription {
                label: Some("Cube Parameters"),
                bindings: &[
                    Binding {
                        binding: 0,
                        ty: BindingType::Sampler(sampler),
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: 1,
                        ty: BindingType::SampledImage {
                            view: texture_image_view,
                        },
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: 2,
                        ty: cube_object_data.bind_whole_range_uniform_buffer(),
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                ],
            },
            &binding_set_layout,
        )
        .unwrap();

    // device.destroy_shader_module(vertex_shader_module).unwrap();
    // device
    //     .destroy_shader_module(fragment_shader_module)
    //     .unwrap();

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    event_loop
        .run(|event, event_loop| match event {
            Event::NewEvents(_) => {}
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    let model =
                        glam::Mat4::from_rotation_y(rotation) * glam::Mat4::from_scale(Vec3::ONE);
                    let mvp = projection * view * model;
                    let mvp = [mvp];
                    device
                        .write_buffer(
                            cube_object_data,
                            &cube_object_data.write_all_params(bytemuck::cast_slice(&mvp)),
                        )
                        .unwrap();

                    let swapchain_image = swapchain.acquire_next_image().unwrap();

                    let mut command_recorder = device.create_command_recorder::<Graphics>();
                    {
                        let mut render_pass = command_recorder
                            .begin_render_pass(&RenderPassDescription {
                                label: Some("Cube Rendering"),
                                render_targets: &[RenderTarget {
                                    view: swapchain_image.view,
                                    sample_count: SampleCount::One,
                                    load_op: RenderTargetLoadOp::Clear([0.3, 0.0, 0.5, 1.0]),
                                    store_op: AttachmentStoreOp::Store,
                                }],
                                depth_stencil_attachment: Some(&DepthStencilTarget {
                                    view: depth_image_view,
                                    sample_count: SampleCount::One,
                                    load_op: DepthStencilTargetLoadOp::Clear(1.0, 0),
                                    store_op: AttachmentStoreOp::Store,
                                }),
                                render_area: Rect2D {
                                    offset: Default::default(),
                                    extents: swapchain_image.extents,
                                },
                            })
                            .unwrap();
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_vertex_buffers([cube_data_buffer, cube_data_buffer]);
                        render_pass.set_index_buffer(cube_index_buffer);
                        render_pass.set_binding_sets(&[binding_set.clone()]);
                        render_pass.draw_indexed(36, 1, 0, 0, 0).unwrap();
                        rotation += 0.01;
                    }
                    command_recorder.submit().unwrap();

                    swapchain.present().unwrap();

                    device.submit().unwrap();
                    window.request_redraw();
                }
                WindowEvent::Resized(new_size) => {
                    swapchain
                        .resized(
                            Extents2D {
                                width: new_size.width,
                                height: new_size.height,
                            },
                            window.window_handle().unwrap(),
                            window.display_handle().unwrap(),
                        )
                        .unwrap();

                    device.destroy_image_view(depth_image_view).unwrap();
                    device.destroy_image(depth_image).unwrap();
                    depth_image = device
                        .create_image(&ImageDescription {
                            label: Some("Depth image"),
                            usage_flags: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                            extents: Extents3D {
                                width: new_size.width,
                                height: new_size.height,
                                depth: 1,
                            },
                            dimension: ImageDimension::D2,
                            mips: NonZeroU32::new(1).unwrap(),
                            array_layers: NonZeroU32::new(1).unwrap(),
                            samples: SampleCount::One,
                            format: ImageFormat::Depth32,
                            memory_domain: MemoryDomain::Gpu,
                        })
                        .unwrap();
                    depth_image_view = device
                        .create_image_view(&ImageViewDescription {
                            label: Some("Depth image view"),
                            format: ImageFormat::Depth32,
                            aspect: mgpu::ImageAspect::Depth,
                            image: depth_image,
                            image_subresource: depth_image.whole_subresource(),
                            dimension: ImageDimension::D2,
                        })
                        .unwrap();
                }
                _ => {}
            },
            Event::DeviceEvent { .. } => {}
            Event::UserEvent(_) => {}
            Event::Suspended => {}
            Event::Resumed => {}
            Event::AboutToWait => {}
            Event::LoopExiting => {
                event_loop.exit();
            }
            Event::MemoryWarning => {}
        })
        .unwrap();

    swapchain.destroy().unwrap();

    device.destroy_graphics_pipeline(pipeline).unwrap();
    device.destroy_shader_module(vertex_shader_module).unwrap();
    device
        .destroy_shader_module(fragment_shader_module)
        .unwrap();
    device.destroy_binding_set(binding_set).unwrap();
    device.destroy_image_view(depth_image_view).unwrap();
    device.destroy_image_view(texture_image_view).unwrap();
    device.destroy_image(depth_image).unwrap();
    device.destroy_image(texture_image).unwrap();
    device.destroy_sampler(sampler).unwrap();
    device.destroy_buffer(cube_data_buffer).unwrap();
    device.destroy_buffer(cube_index_buffer).unwrap();
    device.destroy_buffer(cube_object_data).unwrap();
}
