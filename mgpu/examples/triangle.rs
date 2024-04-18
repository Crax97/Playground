use mgpu::{
    AttachmentStoreOp, BufferDescription, BufferUsageFlags, BufferWriteParams, ColorWriteMask,
    DeviceConfiguration, DeviceFeatures, DevicePreference, Extents2D, FragmentStageInfo, Graphics,
    GraphicsPipelineDescription, ImageFormat, MemoryDomain, Rect2D, RenderPassDescription,
    RenderTarget, RenderTargetInfo, RenderTargetLoadOp, SampleCount, ShaderModuleDescription,
    SwapchainCreationInfo, VertexInputDescription, VertexInputFrequency, VertexStageInfo,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use shaderc::ShaderKind;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;

const VERTEX_SHADER: &str = "
#version 460
layout(location = 0) in vec3 pos;
layout(location = 0) out vec4 vs_pos;

void main() {
    vs_pos = vec4(pos, 1.0);
}
";
const FRAGMENT_SHADER: &str = "
#version 460
layout(location = 0) in vec4 vs_pos;
layout(location = 0) out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
";

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
}

fn vertex(x: f32, y: f32, z: f32) -> Vertex {
    Vertex { pos: [x, y, z] }
}

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
    let triangle_data = vec![
        vertex(-1.0, -1.0, 0.0),
        vertex(1.0, -1.0, 0.0),
        vertex(0.0, 1.0, 0.0),
    ];

    let triangle_buffer = device
        .create_buffer(&BufferDescription {
            label: Some("Triangle data"),
            usage_flags: BufferUsageFlags::VERTEX_BUFFER,
            size: std::mem::size_of_val(&triangle_data),
            memory_domain: MemoryDomain::DeviceLocal,
        })
        .unwrap();

    // device
    //     .write_buffer(
    //         triangle_buffer,
    //         &triangle_buffer.write_all_params(bytemuck::cast_slice(&triangle_data)),
    //     )
    //     .unwrap();

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
                Some("Triangle Pipeline"),
                &VertexStageInfo {
                    shader: &vertex_shader_module,
                    entry_point: "main",
                    vertex_inputs: &[VertexInputDescription {
                        location: 0,
                        stride: std::mem::size_of::<Vertex>(),
                        offset: 0,
                        frequency: VertexInputFrequency::PerVertex,
                        format: mgpu::VertexAttributeFormat::Float3,
                    }],
                },
            )
            .fragment_stage(&FragmentStageInfo {
                shader: &fragment_shader_module,
                entry_point: "main",
                render_targets: &[RenderTargetInfo {
                    format: ImageFormat::Rgba8,
                    blend: None,
                    write_mask: ColorWriteMask::RGBA,
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
                        render_pass.set_vertex_buffers([triangle_buffer]);
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
    device.destroy_buffer(triangle_buffer).unwrap();
}

fn compile_glsl(shader_source: &str, shader_kind: ShaderKind) -> Vec<u32> {
    let compiler = shaderc::Compiler::new().unwrap();
    let compiled = compiler
        .compile_into_spirv(shader_source, shader_kind, "none", "main", None)
        .unwrap();

    compiled.as_binary().to_vec()
}

fn read_image_data() -> Vec<u8> {
    let image_content = std::fs::read("examples/assets/david.jpg").unwrap();
    image::load_from_memory(&image_content)
        .unwrap()
        .to_rgb8()
        .to_vec()
}
