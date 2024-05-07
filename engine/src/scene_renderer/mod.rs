use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use mgpu::{
    AttachmentStoreOp, Binding, BindingSet, BindingSetDescription, BindingSetElement,
    BindingSetLayout, BlitParams, Buffer, BufferDescription, BufferUsageFlags, BufferWriteParams,
    DepthStencilTarget, DepthStencilTargetLoadOp, Device, Extents3D, Graphics, Image,
    ImageDescription, ImageUsageFlags, ImageView, ImageViewDescription, Offset2D, Rect2D,
    RenderPassDescription, RenderTarget, RenderTargetLoadOp, ShaderStageFlags,
};

use crate::{
    assert_size_does_not_exceed,
    asset_map::AssetMap,
    math::{color::LinearColor, constants::UP, Transform},
    scene::Scene,
};

pub struct SceneRenderer {
    frames: Vec<FrameData>,
    clear_color: LinearColor,

    current_frame: usize,
}

#[derive(Clone)]
struct FrameData {
    point_of_view_buffer: Buffer,
    frame_binding_set: BindingSet,

    depth_stencil_image: Image,
    depth_stencil_image_view: ImageView,

    final_render_image: Image,
    final_render_image_view: ImageView,
}

pub enum ProjectionMode {
    Orthographic {
        width: f32,
        height: f32,
    },
    Perspective {
        fov_y_radians: f32,
        aspect_ratio: f32,
    },
}

pub struct PointOfView {
    pub transform: Transform,
    pub projection_mode: ProjectionMode,
    near_plane: f32,
    far_plane: f32,
}

pub struct SceneRenderingParams<'a> {
    pub device: &'a Device,
    pub output_image: ImageView,
    pub scene: &'a Scene,
    pub pov: &'a PointOfView,
    pub asset_map: &'a mut AssetMap,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct GPUGlobalFrameData {
    projection: Mat4,
    view: Mat4,
    delta_time: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub(crate) struct GPUPerObjectDrawData {
    pub model_matrix: Mat4,
}

assert_size_does_not_exceed!(
    GPUPerObjectDrawData,
    mgpu::MAX_PUSH_CONSTANT_RANGE_SIZE_BYTES
);

impl SceneRenderer {
    pub fn scene_data_binding_set_layout() -> BindingSetLayout {
        BindingSetLayout {
            binding_set_elements: vec![BindingSetElement {
                binding: 0,
                array_length: 1,
                ty: mgpu::BindingSetElementKind::Buffer {
                    ty: mgpu::BufferType::Uniform,
                    access_mode: mgpu::StorageAccessMode::Read,
                },
                shader_stage_flags: ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
            }],
        }
    }
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        let mut frame_data = vec![];
        let device_info = device.get_info();
        for i in 0..device_info.frames_in_flight {
            let point_of_view_buffer = device.create_buffer(&BufferDescription {
                label: Some(&format!("POV buffer for frame {}", i)),
                usage_flags: BufferUsageFlags::TRANSFER_DST
                    | BufferUsageFlags::UNIFORM_BUFFER
                    | BufferUsageFlags::STORAGE_BUFFER,
                size: std::mem::size_of::<GPUGlobalFrameData>(),
                memory_domain: mgpu::MemoryDomain::Gpu,
            })?;
            let final_render_image = device.create_image(&ImageDescription {
                label: Some(&format!("Final render image for frame {}", i)),
                usage_flags: ImageUsageFlags::TRANSFER_SRC
                    | ImageUsageFlags::SAMPLED
                    | ImageUsageFlags::COLOR_ATTACHMENT,
                extents: Extents3D {
                    width: 1920,
                    height: 1080,
                    depth: 1,
                },
                dimension: mgpu::ImageDimension::D2,
                mips: 1.try_into().unwrap(),
                array_layers: 1.try_into().unwrap(),
                samples: mgpu::SampleCount::One,
                format: mgpu::ImageFormat::Rgba8,
                memory_domain: mgpu::MemoryDomain::Gpu,
            })?;
            let final_render_image_view = device.create_image_view(&ImageViewDescription {
                label: Some(&format!("Final render image view for frame {}", i)),
                image: final_render_image,
                format: mgpu::ImageFormat::Rgba8,
                dimension: mgpu::ImageDimension::D2,
                aspect: mgpu::ImageAspect::Color,
                image_subresource: final_render_image.whole_subresource(),
            })?;

            let depth_stencil_image = device.create_image(&ImageDescription {
                label: Some(&format!("Depth/Stencil image for frame {}", i)),
                usage_flags: ImageUsageFlags::TRANSFER_SRC
                    | ImageUsageFlags::SAMPLED
                    | ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                extents: Extents3D {
                    width: 1920,
                    height: 1080,
                    depth: 1,
                },
                dimension: mgpu::ImageDimension::D2,
                mips: 1.try_into().unwrap(),
                array_layers: 1.try_into().unwrap(),
                samples: mgpu::SampleCount::One,
                format: mgpu::ImageFormat::Depth32,
                memory_domain: mgpu::MemoryDomain::Gpu,
            })?;
            let depth_stencil_image_view = device.create_image_view(&ImageViewDescription {
                label: Some(&format!("Depth/Stencil image view for frame {}", i)),
                image: depth_stencil_image,
                format: mgpu::ImageFormat::Depth32,
                dimension: mgpu::ImageDimension::D2,
                aspect: mgpu::ImageAspect::Depth,
                image_subresource: depth_stencil_image.whole_subresource(),
            })?;

            let frame_binding_set = device.create_binding_set(
                &BindingSetDescription {
                    label: Some("Frame binding set"),
                    bindings: &[Binding {
                        binding: 0,
                        ty: point_of_view_buffer.bind_whole_range_uniform_buffer(),
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    }],
                },
                &Self::scene_data_binding_set_layout(),
            )?;

            let current_frame_data = FrameData {
                point_of_view_buffer,
                frame_binding_set,
                depth_stencil_image,
                depth_stencil_image_view,
                final_render_image,
                final_render_image_view,
            };
            frame_data.push(current_frame_data)
        }

        Ok(Self {
            frames: frame_data,
            clear_color: LinearColor::VIOLET,
            current_frame: 0,
        })
    }
    pub fn render(&mut self, params: SceneRenderingParams) -> anyhow::Result<()> {
        let device = params.device;
        let current_frame = &self.frames[self.current_frame];
        self.update_buffers(params.device, current_frame, &params.pov)?;
        let mut command_recorder = device.create_command_recorder::<Graphics>();

        {
            let mut scene_output_pass =
                command_recorder.begin_render_pass(&RenderPassDescription {
                    label: Some("Scene Rendering"),
                    render_targets: &[RenderTarget {
                        view: current_frame.final_render_image_view,
                        sample_count: mgpu::SampleCount::One,
                        load_op: RenderTargetLoadOp::Clear(self.clear_color.data),
                        store_op: AttachmentStoreOp::Store,
                    }],
                    depth_stencil_attachment: Some(&DepthStencilTarget {
                        view: current_frame.depth_stencil_image_view,
                        sample_count: mgpu::SampleCount::One,
                        load_op: DepthStencilTargetLoadOp::Clear(1.0, 0),
                        store_op: AttachmentStoreOp::Store,
                    }),
                    render_area: Rect2D {
                        offset: Offset2D::default(),
                        extents: current_frame.final_render_image.extents().to_2d(),
                    },
                })?;
            for item in params.scene.iter() {
                match &item.primitive_type {
                    crate::scene::ScenePrimitive::Group => {}
                    crate::scene::ScenePrimitive::Mesh(info) => {
                        let mesh = params.asset_map.get(&info.handle).expect("No mesh");
                        let material = params.asset_map.get(&info.material).expect("No material");
                        let model_matrix = item.transform.matrix();

                        scene_output_pass.set_pipeline(material.pipeline);
                        scene_output_pass.set_push_constant(
                            bytemuck::cast_slice(&[model_matrix]),
                            ShaderStageFlags::ALL_GRAPHICS,
                        );
                        scene_output_pass.set_binding_sets(&[
                            &current_frame.frame_binding_set,
                            &material.binding_set,
                        ]);

                        scene_output_pass.set_vertex_buffers([
                            mesh.position_component,
                            mesh.normal_component,
                            mesh.tangent_component,
                            mesh.color_component,
                            mesh.uv_component,
                        ]);
                        scene_output_pass.set_index_buffer(mesh.index_buffer);
                        scene_output_pass.draw_indexed(mesh.info.num_indices, 1, 0, 0, 0)?;
                    }
                }
            }
        }
        command_recorder.blit(&BlitParams {
            src_image: current_frame.final_render_image,
            src_region: current_frame.final_render_image.whole_region(),
            dst_image: params.output_image.owner(),
            dst_region: params.output_image.owner().whole_region(),
            filter: mgpu::FilterMode::Linear,
        });
        command_recorder.submit()?;

        self.current_frame = (self.current_frame + 1) % params.device.get_info().frames_in_flight;

        Ok(())
    }

    fn update_buffers(
        &self,
        device: &Device,
        current_frame: &FrameData,
        pov: &PointOfView,
    ) -> anyhow::Result<()> {
        let pov_projection_matrix = pov.projection_matrix();
        let pov_view_matrix = pov.view_matrix();
        let gpu_global_data = GPUGlobalFrameData {
            projection: pov_projection_matrix,
            view: pov_view_matrix,
            delta_time: [1.0 / 60.0, 0.0, 0.0, 0.0],
        };

        device.write_buffer(
            current_frame.point_of_view_buffer,
            &BufferWriteParams {
                data: bytemuck::cast_slice(&[gpu_global_data]),
                offset: 0,
                size: std::mem::size_of::<GPUGlobalFrameData>(),
            },
        )?;
        Ok(())
    }
}

impl PointOfView {
    pub fn new_perspective(
        near_plane: f32,
        far_plane: f32,
        fov_y_degrees: f32,
        aspect_ratio: f32,
    ) -> Self {
        Self {
            transform: Transform::default(),
            projection_mode: ProjectionMode::Perspective {
                fov_y_radians: fov_y_degrees.to_radians(),
                aspect_ratio,
            },
            near_plane,
            far_plane,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(
            self.transform.location,
            self.transform.location + self.transform.forward(),
            UP,
        )
    }

    pub fn projection_matrix(&self) -> Mat4 {
        match self.projection_mode {
            ProjectionMode::Orthographic { width, height } => Mat4::orthographic_rh(
                -width * 0.5,
                width * 0.5,
                -height * 0.5,
                height * 0.5,
                self.near_plane,
                self.far_plane,
            ),
            ProjectionMode::Perspective {
                fov_y_radians,
                aspect_ratio,
            } => Mat4::perspective_rh(fov_y_radians, aspect_ratio, self.near_plane, self.far_plane),
        }
    }
}
