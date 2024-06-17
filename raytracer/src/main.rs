use engine::{
    app::{bootstrap, App, AppDescription},
    glam::{Vec3, Vec4},
    shader_parameter_writer::{ScalarParameterType, ScalarParameterWriter},
};
use mgpu::{
    include_spirv, Binding, BindingSetDescription, BindingSetElement, BindingSetLayout,
    BindingSetLayoutInfo, BlitParams, Buffer, BufferDescription, BufferUsageFlags,
    BufferWriteParams, ComputePassDescription, ComputePipeline, ComputePipelineDescription,
    Extents2D, Extents3D, Graphics, Image, ImageCreationFlags, ImageDescription, ImageFormat,
    ImageUsageFlags, ImageView, ImageViewDescription, PushConstantInfo, ShaderModule,
    ShaderModuleDescription, ShaderStageFlags,
};

const COMPUTE_SHADER_SOURCE: &[u8] = include_spirv!("../spirv/raytracer.comp.spv");

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Copy, Clone, Default)]
struct Sphere {
    position_radius: Vec4,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Copy, Clone, Default)]
struct Plane {
    position: Vec4,
    normal: Vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Primitive {
    color: [f32; 4],
    ty_index: [u32; 4],
}

#[derive(Default)]
struct Scene {
    primitives: Vec<Primitive>,
    spheres: Vec<Sphere>,
    planes: Vec<Plane>,
}

unsafe impl bytemuck::Zeroable for Primitive {}
unsafe impl bytemuck::Pod for Primitive {}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Copy, Clone)]
struct CameraData {
    position_fov: [f32; 4],
    direction: [f32; 4],
    image_size: [f32; 4],
}

struct RaytracerApp {
    primitives_buffer: Buffer,
    spheres_buffer: Buffer,
    planes_buffer: Buffer,
    compute_shader: ShaderModule,
    compute_pipeline: ComputePipeline,
    output_image: Image,
    output_image_view: ImageView,
    binding_set: mgpu::BindingSet,
    parameter_writer: ScalarParameterWriter,
    scene: Scene,
    camera_info: CameraData,
    time: f32,
}

impl App for RaytracerApp {
    fn create(context: &engine::app::AppContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let device = &context.device;
        let primitives_buffer = device.create_buffer(&BufferDescription {
            label: Some("Primitives Buffer"),
            usage_flags: BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::UNIFORM_BUFFER
                | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of::<Primitive>() * 5000,
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;
        let spheres_buffer = device.create_buffer(&BufferDescription {
            label: Some("Spheres Buffer"),
            usage_flags: BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::UNIFORM_BUFFER
                | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of::<Sphere>() * 5000,
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;
        let planes_buffer = device.create_buffer(&BufferDescription {
            label: Some("Planes Buffer"),
            usage_flags: BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::UNIFORM_BUFFER
                | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of::<Plane>() * 5000,
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;

        let bsl = BindingSetLayout {
            binding_set_elements: &[
                BindingSetElement {
                    binding: 0,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::Buffer {
                        ty: mgpu::BufferType::Storage,
                        access_mode: mgpu::StorageAccessMode::Read,
                    },
                    shader_stage_flags: ShaderStageFlags::COMPUTE,
                },
                BindingSetElement {
                    binding: 1,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::StorageImage {
                        format: ImageFormat::Rgba8,
                        access_mode: mgpu::StorageAccessMode::ReadWrite,
                    },
                    shader_stage_flags: ShaderStageFlags::COMPUTE,
                },
                BindingSetElement {
                    binding: 2,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::Buffer {
                        ty: mgpu::BufferType::Storage,
                        access_mode: mgpu::StorageAccessMode::Read,
                    },
                    shader_stage_flags: ShaderStageFlags::COMPUTE,
                },
                BindingSetElement {
                    binding: 3,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::Buffer {
                        ty: mgpu::BufferType::Storage,
                        access_mode: mgpu::StorageAccessMode::Read,
                    },
                    shader_stage_flags: ShaderStageFlags::COMPUTE,
                },
            ],
        };

        let output_image = device.create_image(&ImageDescription {
            label: Some("Output Image"),
            usage_flags: ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC,
            creation_flags: ImageCreationFlags::empty(),
            extents: Extents3D {
                width: 1240,
                height: 720,
                depth: 1,
            },
            dimension: mgpu::ImageDimension::D2,
            mips: 1.try_into().unwrap(),
            array_layers: 1.try_into().unwrap(),
            samples: mgpu::SampleCount::One,
            format: ImageFormat::Rgba8,
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;

        let output_image_view = device.create_image_view(&ImageViewDescription {
            label: Some("Output Image View"),
            image: output_image,
            format: ImageFormat::Rgba8,
            view_ty: mgpu::ImageViewType::D2,
            aspect: mgpu::ImageAspect::Color,
            image_subresource: output_image.whole_subresource(),
        })?;

        let compute_shader = device.create_shader_module(&ShaderModuleDescription {
            label: Some("Main CS"),
            source: bytemuck::cast_slice(COMPUTE_SHADER_SOURCE),
        })?;
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescription {
            label: Some("Main Pipeline"),
            shader: compute_shader,
            entry_point: "main",
            binding_set_layouts: &[BindingSetLayoutInfo {
                set: 0,
                layout: &bsl,
            }],
            push_constant_info: Some(PushConstantInfo {
                size: std::mem::size_of::<CameraData>(),
                visibility: ShaderStageFlags::COMPUTE,
            }),
        })?;

        let binding_set = device.create_binding_set(
            &BindingSetDescription {
                label: Some("Main Binding Set"),
                bindings: &[
                    Binding {
                        binding: 0,
                        ty: mgpu::BindingType::StorageBuffer {
                            buffer: primitives_buffer,
                            offset: 0,
                            range: primitives_buffer.size(),
                            access_mode: mgpu::StorageAccessMode::Read,
                        },
                        visibility: ShaderStageFlags::COMPUTE,
                    },
                    Binding {
                        binding: 1,
                        ty: mgpu::BindingType::StorageImage {
                            view: output_image_view,
                            access_mode: mgpu::StorageAccessMode::ReadWrite,
                        },
                        visibility: ShaderStageFlags::COMPUTE,
                    },
                    Binding {
                        binding: 2,
                        ty: mgpu::BindingType::StorageBuffer {
                            buffer: spheres_buffer,
                            offset: 0,
                            range: spheres_buffer.size(),
                            access_mode: mgpu::StorageAccessMode::Read,
                        },
                        visibility: ShaderStageFlags::COMPUTE,
                    },
                    Binding {
                        binding: 3,
                        ty: mgpu::BindingType::StorageBuffer {
                            buffer: planes_buffer,
                            offset: 0,
                            range: planes_buffer.size(),
                            access_mode: mgpu::StorageAccessMode::Read,
                        },
                        visibility: ShaderStageFlags::COMPUTE,
                    },
                ],
            },
            &bsl,
        )?;

        let parameter_writer = ScalarParameterWriter::new(
            device,
            &[&device.get_shader_module_layout(compute_shader)?],
            0,
            0,
        )?;

        let mut scene = Scene::default();

        scene.add_sphere([1.0, 0.0, 0.0, 0.0], Vec3::new(0.0, 0.0, 20.0), 5.0);
        scene.add_sphere([0.0, 1.0, 0.0, 0.0], Vec3::new(-10.0, 0.0, 25.0), 3.0);
        scene.add_sphere([0.0, 0.0, 1.0, 0.0], Vec3::new(10.0, 0.0, 25.0), 7.0);
        scene.add_plane(
            [1.0, 1.0, 1.0, 0.0],
            Vec3::new(0.0, -50.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        scene.add_plane(
            [0.0, 0.3, 0.7, 0.0],
            Vec3::new(0.0, 50.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );

        Ok(Self {
            primitives_buffer,
            spheres_buffer,
            planes_buffer,
            compute_pipeline,
            compute_shader,
            output_image,
            output_image_view,
            binding_set,
            parameter_writer,
            scene,
            camera_info: CameraData {
                position_fov: [0.0, 0.0, 0.0, 90.0f32.to_radians()],
                direction: [0.0, 0.0, 1.0, 0.0],
                image_size: [1024.0, 720.0, 0.0, 0.0],
            },
            time: 0f32,
        })
    }

    fn update(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        if !self.scene.spheres.is_empty() {
            context.device.write_buffer(
                self.spheres_buffer,
                &BufferWriteParams {
                    data: bytemuck::cast_slice(&self.scene.spheres),
                    offset: 0,
                    size: std::mem::size_of_val(self.scene.spheres.as_slice()),
                },
            )?;
        }
        if !self.scene.planes.is_empty() {
            context.device.write_buffer(
                self.planes_buffer,
                &BufferWriteParams {
                    data: bytemuck::cast_slice(&self.scene.planes),
                    offset: 0,
                    size: std::mem::size_of_val(self.scene.planes.as_slice()),
                },
            )?;
        }
        // self.parameter_writer.write(
        //     "num_prims",
        //     ScalarParameterType::ScalarU32(self.scene.primitives.len() as u32),
        // );
        // // self.parameter_writer
        // //     .write_array("prims", bytemuck::cast_slice(&self.scene.primitives));
        // self.parameter_writer
        //     .update_buffer(&context.device, self.primitives_buffer)?;

        context.device.write_buffer(
            self.primitives_buffer,
            &BufferWriteParams {
                data: bytemuck::cast_slice(&[self.scene.primitives.len() as u32]),
                offset: 0,
                size: std::mem::size_of::<u32>(),
            },
        )?;
        context.device.write_buffer(
            self.primitives_buffer,
            &BufferWriteParams {
                data: bytemuck::cast_slice(&self.scene.primitives),
                offset: 16,
                size: std::mem::size_of::<Primitive>() * self.scene.primitives.len(),
            },
        )?;

        self.scene.spheres[0].position_radius = [0.0, self.time.sin() * 10.0, 15.0, 7.0].into();
        self.scene.spheres[1].position_radius =
            [7.0, (self.time + 7.0).sin() * 10.0, 5.0, 3.0].into();
        self.scene.spheres[2].position_radius =
            [-7.0, (self.time + 3.0).sin() * 10.0, 25.0, 5.0].into();
        self.time += 0.01;
        Ok(())
    }

    fn render(
        &mut self,
        context: &engine::app::AppContext,
        render_context: engine::app::RenderContext,
    ) -> anyhow::Result<()> {
        let device = &context.device;

        const GROUP_SIZE_X: u32 = 16;
        const GROUP_SIZE_Y: u32 = 16;

        let mut command_recorder = device.create_command_recorder::<Graphics>();
        {
            let mut pass = command_recorder.begin_compute_pass(&ComputePassDescription {
                label: Some("Main Pass"),
            });

            pass.set_pipeline(self.compute_pipeline);
            pass.set_binding_sets(&[&self.binding_set]);
            pass.set_push_constant(
                bytemuck::cast_slice(&[self.camera_info]),
                ShaderStageFlags::COMPUTE,
            );
            pass.dispatch(
                (self.output_image.extents().width / GROUP_SIZE_X) + 1,
                (self.output_image.extents().height / GROUP_SIZE_Y) + 1,
                1,
            )?;
        }

        command_recorder.blit(&BlitParams {
            src_image: self.output_image,
            src_region: self.output_image.mip_region(0),
            dst_image: render_context.swapchain_image.image,
            dst_region: render_context.swapchain_image.image.mip_region(0),
            filter: mgpu::FilterMode::Linear,
        });

        command_recorder.submit()?;

        Ok(())
    }

    fn resized(
        &mut self,
        context: &engine::app::AppContext,
        new_extents: mgpu::Extents2D,
    ) -> mgpu::MgpuResult<()> {
        Ok(())
    }

    fn shutdown(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        Ok(())
    }
}

impl Scene {
    pub const SPHERE: u32 = 0x00000000;
    pub const PLANE: u32 = 0x00000001;

    pub const TYPE_MASK: u32 = 0xF0000000;
    pub fn add_sphere(&mut self, color: [f32; 4], origin: Vec3, radius: f32) {
        let sphere = Sphere {
            position_radius: [origin.x, origin.y, origin.z, radius].into(),
        };
        let prim_type = Self::SPHERE << 28;
        let prim_index = self.spheres.len() as u32;

        let ty_index = [prim_type | prim_index, 0, 0, 0];
        self.spheres.push(sphere);
        self.primitives.push(Primitive { color, ty_index });
    }

    pub fn add_plane(&mut self, color: [f32; 4], origin: Vec3, normal: Vec3) {
        let plane = Plane {
            position: origin.extend(0.0),
            normal: normal.extend(0.0),
        };
        let prim_type = Self::PLANE << 28;
        let prim_index = self.planes.len() as u32;

        let ty_index = [prim_type | prim_index, 0, 0, 0];
        self.planes.push(plane);
        self.primitives.push(Primitive { color, ty_index });
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::<RaytracerApp>(AppDescription {
        window_size: Extents2D {
            width: 800,
            height: 600,
        },
        initial_title: Some("Raytracer"),
        app_identifier: "RaytracerApp",
    })
}
