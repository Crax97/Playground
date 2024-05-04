mod material_manager;
mod shader_cache;
use bytemuck::{Pod, Zeroable};
use glam::{vec3, Mat4, Vec3, Vec4};
use mgpu::{
    AttachmentStoreOp, BindingSetElement, BindingSetLayout, Buffer, BufferDescription,
    BufferUsageFlags, DepthStencilTarget, DepthStencilTargetLoadOp, Device, Extents3D, Graphics,
    Image, ImageDescription, ImageUsageFlags, ImageView, ImageViewDescription, MgpuResult,
    Offset2D, Rect2D, RenderPassDescription, RenderTarget, RenderTargetLoadOp, ShaderStageFlags,
};

use crate::{
    asset_map::AssetMap,
    math::{constants::UP, Transform},
    scene::Scene,
};

#[derive(Default)]
pub struct SceneRenderer {
    frames: Vec<FrameData>,
    current_frame: usize,
}

#[derive(Clone, Copy)]
struct FrameData {
    point_of_view_buffer: Buffer,

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
    transform: Transform,
    projection_mode: ProjectionMode,
    near_plane: f32,
    far_plane: f32,
}

pub struct SceneRenderingParams<'a> {
    pub device: &'a Device,
    pub output_image: ImageView,
    pub scene: &'a Scene,
    pub pov: PointOfView,
    pub asset_map: &'a mut AssetMap,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct GPUPointOfView {
    matrix: Mat4,
    position: Vec4,
    direction: Vec4,
}
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
        for i in 0..device.get_info().frames_in_flight {
            let point_of_view_buffer = device.create_buffer(&BufferDescription {
                label: Some(&format!("POV buffer for frame {}", i)),
                usage_flags: BufferUsageFlags::TRANSFER_DST
                    | BufferUsageFlags::UNIFORM_BUFFER
                    | BufferUsageFlags::STORAGE_BUFFER,
                size: std::mem::size_of::<[GPUPointOfView; 1000]>(),
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
            let current_frame_data = FrameData {
                point_of_view_buffer,
                depth_stencil_image,
                depth_stencil_image_view,
                final_render_image,
                final_render_image_view,
            };
            frame_data.push(current_frame_data)
        }

        Ok(Self {
            frames: frame_data,
            current_frame: 0,
        })
    }
    pub fn render(&mut self, params: SceneRenderingParams) -> MgpuResult<()> {
        let device = params.device;
        let current_frame = &self.frames[self.current_frame];
        // self.update_buffers(params.device, current_frame);
        let mut command_recorder = device.create_command_recorder::<Graphics>();

        {
            let mut scene_output_pass =
                command_recorder.begin_render_pass(&RenderPassDescription {
                    label: Some("Scene Rendering"),
                    render_targets: &[RenderTarget {
                        view: current_frame.final_render_image_view,
                        sample_count: mgpu::SampleCount::One,
                        load_op: RenderTargetLoadOp::Clear([0.0; 4]),
                        store_op: AttachmentStoreOp::Store,
                    }],
                    depth_stencil_attachment: Some(&DepthStencilTarget {
                        view: current_frame.depth_stencil_image_view,
                        sample_count: mgpu::SampleCount::One,
                        load_op: DepthStencilTargetLoadOp::Clear(0.0, 0),
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

                        scene_output_pass.set_vertex_buffers([
                            mesh.position_component,
                            mesh.normal_component,
                            mesh.tangent_component,
                            mesh.color_component,
                            mesh.uv_component,
                        ]);
                        scene_output_pass.set_index_buffer(mesh.index_buffer);
                    }
                }
            }
        }
        command_recorder.submit();
        self.current_frame =
            (self.current_frame + 1) % params.device.get_info().frames_in_flight as usize;
        todo!()
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
        Mat4::look_at_rh(self.transform.location, self.transform.forward(), UP)
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
