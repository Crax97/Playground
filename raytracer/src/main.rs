use engine::{
    app::{bootstrap, App, AppDescription},
    shader_parameter_writer::{
        ScalarMaterialParameter, ScalarParameterType, ScalarParameterWriter,
    },
};
use mgpu::{
    include_spirv, Binding, BindingSetDescription, BindingSetElement, BindingSetLayout,
    BindingSetLayoutInfo, BlitParams, Buffer, BufferDescription, BufferUsageFlags,
    ComputePassDescription, ComputePipeline, ComputePipelineDescription, Extents2D, Extents3D,
    Graphics, Image, ImageCreationFlags, ImageDescription, ImageFormat, ImageUsageFlags, ImageView,
    ImageViewDescription, PushConstantInfo, ShaderModule, ShaderModuleDescription,
    ShaderStageFlags,
};

const COMPUTE_SHADER_SOURCE: &[u8] = include_spirv!("../spirv/raytracer.comp.spv");

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Copy, Clone)]
struct Sphere {
    color: [f32; 4],
    position_radius: [f32; 4],
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Copy, Clone)]
struct CameraData {
    position_fov: [f32; 4],
    direction: [f32; 4],
    image_size: [f32; 4],
}

struct RaytracerApp {
    scene_buffer: Buffer,
    compute_shader: ShaderModule,
    compute_pipeline: ComputePipeline,
    output_image: Image,
    output_image_view: ImageView,
    binding_set: mgpu::BindingSet,
    parameter_writer: ScalarParameterWriter,
    spheres: Vec<Sphere>,
    camera_info: CameraData,
}

impl App for RaytracerApp {
    fn create(context: &engine::app::AppContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let device = &context.device;
        let scene_buffer = device.create_buffer(&BufferDescription {
            label: Some("Scene Buffer"),
            usage_flags: BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::UNIFORM_BUFFER
                | BufferUsageFlags::TRANSFER_DST,
            size: std::mem::size_of::<Sphere>() * 5000,
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
                            buffer: scene_buffer,
                            offset: 0,
                            range: scene_buffer.size(),
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

        Ok(Self {
            scene_buffer,
            compute_pipeline,
            compute_shader,
            output_image,
            output_image_view,
            binding_set,
            parameter_writer,
            spheres: vec![
                Sphere {
                    color: [1.0, 0.0, 0.0, 1.0],
                    position_radius: [0.0, 0.0, 20.0, 7.0],
                },
                Sphere {
                    color: [0.0, 1.0, 0.0, 1.0],
                    position_radius: [-10.0, 0.0, 20.0, 3.0],
                },
                Sphere {
                    color: [0.0, 0.0, 1.0, 1.0],
                    position_radius: [10.0, 0.0, 20.0, 6.0],
                },
            ],
            camera_info: CameraData {
                position_fov: [0.0, 0.0, 0.0, 90.0f32.to_radians()],
                direction: [0.0, 0.0, 1.0, 0.0],
                image_size: [1024.0, 720.0, 0.0, 0.0],
            },
        })
    }

    fn update(&mut self, context: &engine::app::AppContext) -> anyhow::Result<()> {
        self.parameter_writer.write(
            "num_prims",
            ScalarParameterType::ScalarU32(self.spheres.len() as u32),
        );
        self.parameter_writer
            .write_array("prims", bytemuck::cast_slice(&self.spheres));
        self.parameter_writer
            .update_buffer(&context.device, self.scene_buffer)?;
        Ok(())
    }

    fn render(
        &mut self,
        context: &engine::app::AppContext,
        render_context: engine::app::RenderContext,
    ) -> anyhow::Result<()> {
        let device = &context.device;
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
            pass.dispatch((1024 / 16) + 1, (720 / 16) + 1, 1)?;
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
