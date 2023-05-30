mod utils;

use std::{f32::consts::PI, io::BufReader, mem::size_of, rc::Rc};

use ash::vk::{
    self, AccessFlags, BufferUsageFlags, ComponentMapping, Format, ImageAspectFlags, ImageLayout,
    ImageSubresourceRange, ImageUsageFlags, ImageViewType, PipelineStageFlags, PresentModeKHR,
};

use gpu::{
    BufferCreateInfo, FramebufferCreateInfo, Gpu, GpuConfiguration, ImageCreateInfo, MemoryDomain,
    TransitionInfo,
};

use engine::{
    AppState, Camera, ForwardRenderingPipeline, MaterialDescription, MaterialDomain, Mesh,
    MeshCreateInfo, RenderingPipeline, Scene, ScenePrimitive, Texture,
};
use nalgebra::*;
use resource_map::ResourceMap;
use winit::{
    dpi::PhysicalSize,
    event::{ButtonId, ElementState, VirtualKeyCode},
    event_loop::ControlFlow,
};

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

    let mut scene_renderer = ForwardRenderingPipeline::new(&gpu, resource_map.clone(), &swapchain)?;

    let material_1 = scene_renderer.get_context().create_material(
        &gpu,
        &resource_map,
        MaterialDescription {
            domain: MaterialDomain::Surface,
            uniform_buffers: vec![],
            input_textures: vec![texture.clone()],
            fragment_module: &fragment_module,
            vertex_module: &vertex_module,
        },
    )?;
    let material = resource_map.add(material_1);

    swapchain.select_present_mode(PresentModeKHR::MAILBOX)?;

    let mut app_state = AppState::new(gpu, swapchain);

    let mut scene = Scene::new();

    scene.add(ScenePrimitive {
        mesh: mesh.clone(),
        material: material.clone(),
        transform: Matrix4::identity(),
    });
    scene.add(ScenePrimitive {
        mesh: mesh.clone(),
        material: material.clone(),
        transform: Matrix4::new_translation(&vector![0.0, 0.0, 1.0]),
    });
    scene.add(ScenePrimitive {
        mesh: mesh.clone(),
        material: material.clone(),
        transform: Matrix4::new_translation(&vector![0.0, 0.0, -1.0]),
    });

    let mut camera = Camera {
        location: point![2.0, 2.0, 2.0],
        forward: vector![0.0, -1.0, -1.0].normalize(),
        ..Default::default()
    };

    let mut forward_movement = 0.0;
    let mut rotation_movement = 0.0;

    let mut rot_x = 45.0;
    let mut rot_z = 55.0;
    let mut dist = 5.0;

    let mut movement: Vector3<f32> = vector![0.0, 0.0, 0.0];
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        const SPEED: f32 = 0.1;
        const ROTATION_SPEED: f32 = 3.0;
        const MIN_DELTA: f32 = 1.0;

        match event {
            winit::event::Event::NewEvents(_) => {}
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::ExitWithCode(0)
                }
                winit::event::WindowEvent::Resized(_) => {
                    app_state.swapchain.recreate_swapchain().unwrap();
                }
                _ => {}
            },
            winit::event::Event::DeviceEvent { event, .. } => match event {
                winit::event::DeviceEvent::Button { button, state } => {
                    let mul = if state == ElementState::Pressed {
                        1.0
                    } else {
                        0.0
                    };
                    if button == 3 {
                        rotation_movement = mul;
                    } else if button == 1 {
                        forward_movement = mul;
                    }
                }
                winit::event::DeviceEvent::MouseMotion { delta } => {
                    movement.x = (delta.0.abs() as f32 - MIN_DELTA).max(0.0)
                        * delta.0.signum() as f32
                        * ROTATION_SPEED;
                    movement.y = (delta.1.abs() as f32 - MIN_DELTA).max(0.0)
                        * delta.1.signum() as f32 as f32
                        * ROTATION_SPEED;
                }
                _ => {}
            },
            winit::event::Event::UserEvent(_) => {}
            winit::event::Event::Suspended => {}
            winit::event::Event::Resumed => {}
            winit::event::Event::MainEventsCleared => {
                app_state.swapchain.window.request_redraw();
            }
            winit::event::Event::RedrawRequested(..) => {
                if rotation_movement > 0.0 {
                    rot_z += movement.y;
                    rot_z = rot_z.clamp(-89.0, 89.0);
                    rot_x += movement.x;
                } else {
                    dist += movement.y * forward_movement * SPEED;
                }

                let new_forward = Rotation::<f32, 3>::from_axis_angle(
                    &Unit::new_normalize(vector![0.0, 0.0, 1.0]),
                    rot_x.to_radians(),
                ) * Rotation::<f32, 3>::from_axis_angle(
                    &Unit::new_normalize(vector![0.0, 1.0, 0.0]),
                    -rot_z.to_radians(),
                );
                let new_forward = new_forward.to_homogeneous();
                let new_forward = new_forward.column(0);

                let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
                let new_position = direction * dist;
                let new_position = point![new_position.x, new_position.y, new_position.z];
                camera.location = new_position;

                let direction = vector![new_forward[0], new_forward[1], new_forward[2]];
                camera.forward = -direction;

                app_state.begin_frame().unwrap();

                let sw_extents = app_state.swapchain.extents();
                let next_image = app_state.swapchain.acquire_next_image().unwrap();
                let framebuffer = app_state
                    .gpu
                    .create_framebuffer(&FramebufferCreateInfo {
                        render_pass: scene_renderer
                            .get_context()
                            .get_material_render_pass(MaterialDomain::Surface),
                        attachments: &[&next_image, &depth_image_view],
                        width: sw_extents.width,
                        height: sw_extents.height,
                    })
                    .unwrap();
                app_state.end_frame().unwrap();
                scene_renderer.render(&mut app_state, &camera, &scene, &framebuffer);
                let _ = app_state.swapchain.present();
            }
            winit::event::Event::RedrawEventsCleared => {}
            winit::event::Event::LoopDestroyed => app_state.gpu.wait_device_idle().unwrap(),
        }
    })
}
