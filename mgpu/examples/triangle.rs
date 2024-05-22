mod util;

use mgpu::{
    AttachmentStoreOp, Buffer, BufferDescription, BufferUsageFlags, Device, DeviceConfiguration,
    DeviceFeatures, DevicePreference, Extents2D, FragmentStageInfo, Graphics, GraphicsPipeline,
    GraphicsPipelineDescription, ImageFormat, MemoryDomain, Rect2D, RenderPassDescription,
    RenderTarget, RenderTargetInfo, RenderTargetLoadOp, SampleCount, ShaderModule,
    ShaderModuleDescription, Swapchain, SwapchainCreationInfo, VertexInputDescription,
    VertexInputFrequency, VertexStageInfo,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use shaderc::ShaderKind;
use util::*;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes};

const VERTEX_SHADER: &str = "
#version 460
layout(location = 0) in vec3 pos;

void main() {
    vec4 vs_pos = vec4(pos, 1.0);
    gl_Position = vs_pos;
}
";
const FRAGMENT_SHADER: &str = "
#version 460
layout(location = 0) out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
";

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let device = mgpu::Device::new(DeviceConfiguration {
        app_name: Some("Triangle Application"),
        features: DeviceFeatures::HAL_DEBUG_LAYERS,
        device_preference: Some(DevicePreference::HighPerformance),
        display_handle: Some(event_loop.display_handle().unwrap().as_raw()),
        desired_frames_in_flight: 3,
    })
    .expect("Failed to create gpu device");

    let triangle_data = vec![
        pos(-1.0, -1.0, 0.0),
        pos(1.0, -1.0, 0.0),
        pos(0.0, 1.0, 0.0),
    ];

    let triangle_buffer = device
        .create_buffer(&BufferDescription {
            label: Some("Triangle data"),
            usage_flags: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of_val(triangle_data.as_slice()),
            memory_domain: MemoryDomain::Gpu,
        })
        .unwrap();

    device
        .write_buffer(
            triangle_buffer,
            &triangle_buffer.write_all_params(bytemuck::cast_slice(&triangle_data)),
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
                Some("Triangle Pipeline"),
                &VertexStageInfo {
                    shader: &vertex_shader_module,
                    entry_point: "main",
                    vertex_inputs: &[VertexInputDescription {
                        location: 0,
                        stride: std::mem::size_of::<Position>(),
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
                }],
                depth_stencil_target: None,
            }),
        )
        .unwrap();

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    struct TriangleApplication {
        device: Device,
        swapchain: Option<Swapchain>,
        window: Option<Window>,

        vertex_shader_module: ShaderModule,
        fragment_shader_module: ShaderModule,
        triangle_buffer: Buffer,
        pipeline: GraphicsPipeline,
    }

    impl ApplicationHandler for TriangleApplication {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            let window = event_loop
                .create_window(WindowAttributes::default().with_title("Triangle"))
                .unwrap();
            self.swapchain = Some(
                self.device
                    .create_swapchain(&SwapchainCreationInfo {
                        display_handle: window.display_handle().unwrap(),
                        window_handle: window.window_handle().unwrap(),
                        preferred_format: None,
                        preferred_present_mode: None,
                        extents: Extents2D {
                            width: window.inner_size().width,
                            height: window.inner_size().height,
                        },
                    })
                    .expect("Failed to create swapchain!"),
            );
            self.window = Some(window);
        }

        fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
            self.swapchain.as_mut().unwrap().destroy().unwrap();
            self.device
                .destroy_graphics_pipeline(self.pipeline)
                .unwrap();
            self.device
                .destroy_shader_module(self.vertex_shader_module)
                .unwrap();
            self.device
                .destroy_shader_module(self.fragment_shader_module)
                .unwrap();
            self.device.destroy_buffer(self.triangle_buffer).unwrap();
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::Resized(size) => self
                    .swapchain
                    .as_mut()
                    .unwrap()
                    .resized(
                        Extents2D {
                            width: size.width,
                            height: size.height,
                        },
                        self.window.as_ref().unwrap().window_handle().unwrap(),
                        self.window.as_ref().unwrap().display_handle().unwrap(),
                    )
                    .unwrap(),

                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    let swapchain_image = self
                        .swapchain
                        .as_mut()
                        .unwrap()
                        .acquire_next_image()
                        .unwrap();

                    let mut command_recorder = self.device.create_command_recorder::<Graphics>();
                    {
                        let mut render_pass = command_recorder
                            .begin_render_pass(&RenderPassDescription {
                                label: Some("Triangle rendering"),
                                flags: Default::default(),
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
                        render_pass.set_pipeline(self.pipeline);
                        render_pass.set_vertex_buffers([self.triangle_buffer]);
                        render_pass.draw(3, 1, 0, 0).unwrap();
                    }
                    command_recorder.submit().unwrap();
                    self.swapchain.as_mut().unwrap().present().unwrap();

                    self.device.submit().unwrap();
                    self.window.as_ref().unwrap().request_redraw();
                }
                _ => {}
            };
        }
    }
    event_loop
        .run_app(&mut TriangleApplication {
            device,
            swapchain: None,
            window: None,
            vertex_shader_module,
            fragment_shader_module,
            triangle_buffer,
            pipeline,
        })
        .unwrap();
}
