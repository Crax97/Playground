use std::mem::MaybeUninit;

use glam::{vec2, vec3};
use mgpu::{
    Device, DeviceConfiguration, DeviceFeatures, Extents2D, MgpuResult, Swapchain,
    SwapchainCreationInfo, SwapchainImage,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

use crate::{
    asset_map::AssetMap,
    assets::{
        mesh::{Mesh, MeshDescription},
        texture::{Texture, TextureDescription, TextureSamplerConfiguration, TextureUsageFlags},
    },
    constants::{BRDF_LUT_HANDLE, CUBE_MESH_HANDLE, DEFAULT_ENV_WHITE_HANDLE},
    core::Time,
    cubemap_utils,
    fps_limiter::FpsLimiter,
    input::InputState,
    sampler_allocator::SamplerAllocator,
};

pub struct AppRunner {}

pub struct AppDescription {
    pub window_size: Extents2D,
    pub initial_title: Option<&'static str>,
    pub app_identifier: &'static str,
}

pub struct AppContext {
    pub device: Device,
    pub swapchain: MaybeUninit<Swapchain>,
    pub window: MaybeUninit<Window>,
    pub input: InputState,
    pub time: Time,
    pub fps_limiter: FpsLimiter,
}
impl AppContext {
    pub fn window(&self) -> &Window {
        unsafe { self.window.assume_init_ref() }
    }
}

pub struct RenderContext {
    pub swapchain_image: SwapchainImage,
}

pub trait App {
    fn create(context: &AppContext) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn on_window_created(&mut self, context: &AppContext) -> anyhow::Result<()> {
        let _ = context;
        Ok(())
    }

    fn handle_window_event(
        &mut self,
        event: &WindowEvent,
        context: &AppContext,
    ) -> anyhow::Result<()> {
        let _ = (event, context);
        Ok(())
    }
    fn handle_device_event(
        &mut self,
        event: &DeviceEvent,
        context: &AppContext,
    ) -> anyhow::Result<()> {
        let _ = (event, context);
        Ok(())
    }
    fn update(&mut self, context: &AppContext) -> anyhow::Result<()>;
    fn render(&mut self, context: &AppContext, render_context: RenderContext)
        -> anyhow::Result<()>;
    fn resized(&mut self, context: &AppContext, new_extents: Extents2D) -> MgpuResult<()>;
    fn shutdown(&mut self, context: &AppContext) -> anyhow::Result<()>;
}

pub fn bootstrap<A: App>(description: AppDescription) -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let device = Device::new(DeviceConfiguration {
        app_name: Some(description.app_identifier),
        features: DeviceFeatures::HAL_DEBUG_LAYERS,
        device_preference: Some(mgpu::DevicePreference::HighPerformance),
        desired_frames_in_flight: 3,
        display_handle: Some(event_loop.display_handle()?.as_raw()),
    })?;

    let context = AppContext {
        device,
        swapchain: MaybeUninit::uninit(),
        window: MaybeUninit::uninit(),
        input: InputState::default(),
        time: Time::default(),
        fps_limiter: FpsLimiter::new(60),
    };

    struct AppRunner<A: App> {
        app: A,
        context: AppContext,
        window_attributes: WindowAttributes,
        description: AppDescription,
    }

    impl<A: App> ApplicationHandler for AppRunner<A> {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            let window = event_loop
                .create_window(self.window_attributes.clone())
                .unwrap();
            let swapchain = self
                .context
                .device
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
                .expect("Failed to create swapchain");

            self.context.window = MaybeUninit::new(window);
            self.context.swapchain = MaybeUninit::new(swapchain);

            self.app.on_window_created(&self.context).unwrap();
        }

        fn device_event(
            &mut self,
            _event_loop: &winit::event_loop::ActiveEventLoop,
            _device_id: winit::event::DeviceId,
            event: winit::event::DeviceEvent,
        ) {
            self.context.input.device_event(event);
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: winit::event::WindowEvent,
        ) {
            handle_window_event(
                &mut self.app,
                &mut self.context,
                event,
                event_loop,
                &self.description,
            )
            .expect("Failed to handle window event");
        }
    }

    let app = A::create(&context)?;

    event_loop.run_app(&mut AppRunner {
        app,
        context,
        window_attributes: WindowAttributes::default()
            .with_inner_size(PhysicalSize {
                width: description.window_size.width,
                height: description.window_size.height,
            })
            .with_title(description.initial_title.unwrap_or("Engine App")),
        description,
    })?;
    Ok(())
}

fn handle_window_event<A: App>(
    app: &mut A,
    app_context: &mut AppContext,
    event: WindowEvent,
    target: &ActiveEventLoop,
    description: &AppDescription,
) -> anyhow::Result<()> {
    app.handle_window_event(&event, app_context)?;
    app_context.input.update(&event);

    match event {
        winit::event::WindowEvent::Resized(new_size) => {
            let swapchain = unsafe { app_context.swapchain.assume_init_mut() };
            let window = unsafe { app_context.window.assume_init_mut() };
            let new_extents = Extents2D {
                width: new_size.width,
                height: new_size.height,
            };
            swapchain.resized(
                new_extents,
                window.window_handle()?,
                window.display_handle()?,
            )?;
            app.resized(app_context, new_extents)?;
        }

        winit::event::WindowEvent::CloseRequested => target.exit(),
        winit::event::WindowEvent::RedrawRequested => {
            app_context.time.begin_frame();
            app.update(app_context)?;
            let next_image = {
                let swapchain = unsafe { app_context.swapchain.assume_init_mut() };
                swapchain.acquire_next_image()?
            };

            let render_context = RenderContext {
                swapchain_image: next_image,
            };

            app.render(app_context, render_context)?;

            let swapchain = unsafe { app_context.swapchain.assume_init_mut() };
            let window = unsafe { app_context.window.assume_init_mut() };
            swapchain.present()?;
            app_context.device.submit()?;

            let frame_time = app_context.time.delta_from_frame_begin();
            app_context.fps_limiter.update(frame_time);
            app_context.time.end_frame();

            app_context.input.end_frame();
            let fps = 1.0 / app_context.time.delta_seconds();
            let app_title = format!(
                "{} - FPS {}",
                description.initial_title.unwrap_or("Engine App"),
                fps as u64
            );
            window.set_title(&app_title);
            window.request_redraw();
        }
        _ => {}
    };
    Ok(())
}

impl Default for AppDescription {
    fn default() -> Self {
        Self {
            window_size: Extents2D {
                width: 800,
                height: 600,
            },
            initial_title: Default::default(),
            app_identifier: "EngineApp",
        }
    }
}

pub fn asset_map_with_defaults(
    device: &Device,
    sampler_allocator: &SamplerAllocator,
) -> anyhow::Result<AssetMap> {
    let mut map = AssetMap::new();
    map.add(
        create_cube_mesh(device)?,
        CUBE_MESH_HANDLE.identifier().clone(),
    );
    map.add(
        create_default_env(device, sampler_allocator)?,
        DEFAULT_ENV_WHITE_HANDLE.identifier().clone(),
    );
    map.add(
        cubemap_utils::generate_ibl_lut(device, sampler_allocator)?,
        BRDF_LUT_HANDLE.identifier().clone(),
    );
    Ok(map)
}

fn create_cube_mesh(device: &mgpu::Device) -> anyhow::Result<Mesh> {
    let mesh_description = MeshDescription {
        label: Some("Cube mesh"),
        indices: &[
            0, 1, 2, 3, 1, 0, //Bottom
            6, 5, 4, 4, 5, 7, // Front
            10, 9, 8, 8, 9, 11, // Left
            12, 13, 14, 15, 13, 12, // Right
            16, 17, 18, 19, 17, 16, // Up
            22, 21, 20, 20, 21, 23, // Down
        ],
        positions: &[
            // Back
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(1.0, -1.0, 1.0),
            // Front
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(1.0, -1.0, -1.0),
            // Left
            vec3(1.0, -1.0, -1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(1.0, -1.0, 1.0),
            // Right
            vec3(-1.0, -1.0, -1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, 1.0),
            // Up
            vec3(-1.0, 1.0, -1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, 1.0, 1.0),
            // Down
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(-1.0, -1.0, 1.0),
        ],
        colors: &[vec3(1.0, 0.0, 0.0)],
        normals: &[
            // Back
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            // Front
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            // Left
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            // Right
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            // Up
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            // Down
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
        ],
        tangents: &[
            // Back
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            // Front
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            // Left
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            // Right
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            // Up
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            // Down
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
        ],
        uvs: &[
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
        ],
    };
    let cube_mesh = Mesh::new(device, &mesh_description)?;
    Ok(cube_mesh)
}

fn create_default_env(
    device: &mgpu::Device,
    sampler_allocator: &SamplerAllocator,
) -> anyhow::Result<Texture> {
    let texture = Texture::new(
        device,
        &TextureDescription {
            label: Some("Default env texture"),
            data: &[&[255; 4 * 6]],
            ty: crate::assets::texture::TextureType::Cubemap(Extents2D {
                width: 1,
                height: 1,
            }),
            format: mgpu::ImageFormat::Rgba8,
            usage_flags: TextureUsageFlags::default(),
            num_mips: 1.try_into().unwrap(),
            auto_generate_mips: false,
            sampler_configuration: TextureSamplerConfiguration::default(),
        },
        sampler_allocator,
    )?;
    Ok(texture)
}
