mod util;

use bytemuck::offset_of;
use mgpu::{
    AttachmentStoreOp, BlendFactor, BlendOp, BlendSettings, BufferDescription, BufferUsageFlags,
    ColorWriteMask, DeviceConfiguration, DeviceFeatures, DevicePreference, Extents2D,
    FragmentStageInfo, Graphics, GraphicsPipelineDescription, ImageFormat, MemoryDomain, Rect2D,
    RenderPassDescription, RenderTarget, RenderTargetInfo, RenderTargetLoadOp, SampleCount,
    ShaderModuleDescription, SwapchainCreationInfo, VertexInputDescription, VertexInputFrequency,
    VertexStageInfo,
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

void main() {
    vec4 vs_pos = vec4(pos, 1.0);
    gl_Position = vs_pos;
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
        .with_title("Triangle")
        .build(&event_loop)
        .unwrap();

    let device = mgpu::Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::DEBUG_FEATURES,
        device_preference: Some(DevicePreference::HighPerformance),
        display_handle: event_loop.display_handle().unwrap().as_raw(),
        desired_frames_in_flight: 3,
    })
    .expect("Failed to create gpu device");
    let mut swapchain = device
        .create_swapchain(&SwapchainCreationInfo {
            display_handle: window.display_handle().unwrap(),
            window_handle: window.window_handle().unwrap(),
            preferred_format: None,
        })
        .expect("Failed to create swapchain");
    let cube_data = vec![
        vertex(-1.0, -1.0, 0.0, 0.0, 0.0),
        vertex(1.0, -1.0, 0.0, 0.0, 0.0),
        vertex(0.0, 1.0, 0.0, 0.0, 0.0),
    ];

    let cube_data_buffer = device
        .create_buffer(&BufferDescription {
            label: Some("Triangle data"),
            usage_flags: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of_val(cube_data.as_slice()),
            memory_domain: MemoryDomain::DeviceLocal,
        })
        .unwrap();

    device
        .write_buffer(
            cube_data_buffer,
            &cube_data_buffer.write_all_params(bytemuck::cast_slice(&cube_data)),
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
                depth_stencil_target: None,
            }),
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
                    let swapchain_image = swapchain.acquire_next_image().unwrap();

                    let mut command_recorder = device.create_command_recorder::<Graphics>();
                    {
                        let mut render_pass = command_recorder
                            .begin_render_pass(&RenderPassDescription {
                                label: Some("Triangle rendering"),
                                render_targets: &[RenderTarget {
                                    view: swapchain_image.view,
                                    sample_count: SampleCount::One,
                                    load_op: RenderTargetLoadOp::Clear([0.3, 0.0, 0.5, 1.0]),
                                    store_op: AttachmentStoreOp::Store,
                                }],
                                depth_stencil_attachment: None,
                                render_area: Rect2D {
                                    offset: Default::default(),
                                    extents: swapchain_image.extents,
                                },
                            })
                            .unwrap();
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_vertex_buffers([cube_data_buffer, cube_data_buffer]);
                        render_pass.draw(3, 1, 0, 0).unwrap();
                    }
                    command_recorder.submit().unwrap();

                    swapchain.present().unwrap();

                    device.submit().unwrap();
                    window.request_redraw();
                }
                WindowEvent::Resized(new_size) => swapchain
                    .resized(
                        Extents2D {
                            width: new_size.width,
                            height: new_size.height,
                        },
                        window.window_handle().unwrap(),
                        window.display_handle().unwrap(),
                    )
                    .unwrap(),
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

    // device.destroy_image_view(image_view);
    device.destroy_buffer(cube_data_buffer).unwrap();
}
