use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use mgpu::{
    AttachmentStoreOp, Binding, BindingSet, BindingSetDescription, BindingSetElement,
    BindingSetLayout, BindingSetLayoutInfo, BlitParams, Buffer, BufferDescription,
    BufferUsageFlags, BufferWriteParams, DepthStencilState, DepthStencilTarget,
    DepthStencilTargetLoadOp, Device, Extents2D, Extents3D, FragmentStageInfo, Graphics,
    GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription, ImageFormat,
    ImageUsageFlags, ImageView, ImageViewDescription, Offset2D, Rect2D, RenderPassDescription,
    RenderPassFlags, RenderTarget, RenderTargetInfo, RenderTargetLoadOp, Sampler,
    SamplerDescription, ShaderModule, ShaderModuleDescription, ShaderStageFlags, VertexStageInfo,
};

use crate::{
    assert_size_does_not_exceed,
    asset_map::{AssetHandle, AssetMap},
    assets::texture::Texture,
    include_spirv,
    math::{color::LinearColor, constants::UP, Transform},
    scene::Scene,
    shader_parameter_writer::ScalarParameterWriter,
};

const QUAD_VERTEX: &[u8] = include_spirv!("../spirv/quad_vertex.vert.spv");
const SCENE_LIGHTNING_FRAGMENT: &[u8] = include_spirv!("../spirv/scene_lightning.frag.spv");

pub struct SceneRenderer {
    frames: Vec<FrameData>,
    clear_color: LinearColor,

    quad_vertex_shader: ShaderModule,
    scene_lightning_fragment: ShaderModule,

    scene_lightning_pipeline: GraphicsPipeline,

    current_frame: usize,
    scene_lightning_parameters_writer: ScalarParameterWriter,
    scene_setup: SceneSetup,
}

#[derive(Clone, Copy, Default)]
pub struct SceneSetup {
    pub ambient_color: LinearColor,
    pub ambient_intensity: f32,
}

#[derive(Clone)]
struct RenderImage {
    image: Image,
    view: ImageView,
}

#[derive(Clone)]
pub struct SceneImages {
    depth: RenderImage,
    diffuse: RenderImage,
    emissive_ao: RenderImage,
    normal: RenderImage,
    position: RenderImage,
    metallic_roughness: RenderImage,

    binding_set: BindingSet,
}

struct FrameData {
    point_of_view_buffer: Buffer,
    frame_binding_set: BindingSet,
    scene_images_sampler: Sampler,
    scene_images: SceneImages,
    final_image: RenderImage,
    scene_params_binding_set: BindingSet,
    scene_lightning_parameter_buffer: Buffer,
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

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SceneOutput {
    #[default]
    FinalImage,
    BaseColor,
    Normal,
    WorldPosition,
    EmissiveAO,
    MetallicRoughness,
}

pub struct SceneRenderingParams<'a> {
    pub device: &'a Device,
    pub output_image: ImageView,
    pub scene: &'a Scene,
    pub pov: &'a PointOfView,
    pub asset_map: &'a mut AssetMap,
    pub output: SceneOutput,
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
    pub fn per_object_scene_binding_set_layout() -> &'static BindingSetLayout<'static> {
        const LAYOUT: BindingSetLayout = BindingSetLayout {
            binding_set_elements: &[BindingSetElement {
                binding: 0,
                array_length: 1,
                ty: mgpu::BindingSetElementKind::Buffer {
                    ty: mgpu::BufferType::Uniform,
                    access_mode: mgpu::StorageAccessMode::Read,
                },
                shader_stage_flags: ShaderStageFlags::from_bits_retain(
                    ShaderStageFlags::VERTEX.bits() | ShaderStageFlags::FRAGMENT.bits(),
                ),
            }],
        };
        &LAYOUT
    }

    pub fn scene_lightning_binding_set_layout() -> &'static BindingSetLayout<'static> {
        const LAYOUT: BindingSetLayout = BindingSetLayout {
            binding_set_elements: &[
                BindingSetElement {
                    binding: 0,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::Buffer {
                        ty: mgpu::BufferType::Uniform,
                        access_mode: mgpu::StorageAccessMode::Read,
                    },
                    shader_stage_flags: ShaderStageFlags::from_bits_retain(
                        ShaderStageFlags::VERTEX.bits() | ShaderStageFlags::FRAGMENT.bits(),
                    ),
                },
                BindingSetElement {
                    binding: 1,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::Buffer {
                        ty: mgpu::BufferType::Uniform,
                        access_mode: mgpu::StorageAccessMode::Read,
                    },
                    shader_stage_flags: ShaderStageFlags::from_bits_retain(
                        ShaderStageFlags::VERTEX.bits() | ShaderStageFlags::FRAGMENT.bits(),
                    ),
                },
                // BindingSetElement {
                //     binding: 2,
                //     array_length: 1,
                //     ty: mgpu::BindingSetElementKind::SampledImage,
                //     shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
                // },
            ],
        };
        &LAYOUT
    }
    pub fn new(
        device: &Device,
        // env_map: AssetHandle<Texture>,
        // asset_map: &AssetMap,
    ) -> anyhow::Result<Self> {
        let mut frame_data = vec![];
        let device_info = device.get_info();

        let quad_vertex_shader = device.create_shader_module(&ShaderModuleDescription {
            label: Some("Quad VS"),
            source: bytemuck::cast_slice(QUAD_VERTEX),
        })?;

        let scene_lightning_fragment = device.create_shader_module(&ShaderModuleDescription {
            label: Some("SceneLightning FS"),
            source: bytemuck::cast_slice(SCENE_LIGHTNING_FRAGMENT),
        })?;

        let scene_images_sampler = device.create_sampler(&SamplerDescription {
            label: Some("GBuffer sampler"),
            mag_filter: mgpu::FilterMode::Linear,
            min_filter: mgpu::FilterMode::Linear,
            mipmap_mode: mgpu::MipmapMode::Linear,
            address_mode_u: mgpu::AddressMode::ClampToEdge,
            address_mode_v: mgpu::AddressMode::ClampToEdge,
            address_mode_w: mgpu::AddressMode::ClampToEdge,
            lod_bias: 0.0,
            compare_op: None,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: mgpu::BorderColor::White,
            unnormalized_coordinates: false,
        })?;

        let scene_lightning_parameters_writer = ScalarParameterWriter::new(
            device,
            &[&device.get_shader_module_layout(scene_lightning_fragment)?],
            1,
            1,
        )?;
        for i in 0..device_info.frames_in_flight {
            let point_of_view_buffer = device.create_buffer(&BufferDescription {
                label: Some(&format!("POV buffer for frame {}", i)),
                usage_flags: BufferUsageFlags::TRANSFER_DST
                    | BufferUsageFlags::UNIFORM_BUFFER
                    | BufferUsageFlags::STORAGE_BUFFER,
                size: std::mem::size_of::<GPUGlobalFrameData>(),
                memory_domain: mgpu::MemoryDomain::Gpu,
            })?;

            let scene_lightning_parameter_buffer = device.create_buffer(&BufferDescription {
                label: Some("Material user buffer"),
                usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM_BUFFER,
                size: scene_lightning_parameters_writer.binary_blob.len(),
                memory_domain: mgpu::MemoryDomain::Gpu,
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
                Self::per_object_scene_binding_set_layout(),
            )?;

            let scene_params_binding_set = device.create_binding_set(
                &BindingSetDescription {
                    label: Some("Scene parameters binding set"),
                    bindings: &[
                        Binding {
                            binding: 0,
                            ty: point_of_view_buffer.bind_whole_range_uniform_buffer(),
                            visibility: ShaderStageFlags::ALL_GRAPHICS,
                        },
                        Binding {
                            binding: 1,
                            ty: scene_lightning_parameter_buffer.bind_whole_range_uniform_buffer(),
                            visibility: ShaderStageFlags::ALL_GRAPHICS,
                        },
                        // Binding {
                        //     binding: 2,
                        //     ty: mgpu::BindingType::SampledImage {
                        //         view: asset_map.get(&env_map).unwrap().view,
                        //     },
                        //     visibility: ShaderStageFlags::ALL_GRAPHICS,
                        // },
                    ],
                },
                Self::scene_lightning_binding_set_layout(),
            )?;

            let extents = Extents2D {
                width: 1920,
                height: 1080,
            };
            let current_frame_data = FrameData {
                point_of_view_buffer,
                frame_binding_set,
                scene_images: SceneImages::new(device, extents, scene_images_sampler, i)?,
                final_image: RenderImage::new(
                    device,
                    ImageFormat::Rgba32f,
                    extents,
                    "Final render image",
                    i,
                )?,
                scene_images_sampler,
                scene_lightning_parameter_buffer,
                scene_params_binding_set,
            };
            frame_data.push(current_frame_data)
        }

        let scene_lightning_pipeline =
            device.create_graphics_pipeline(&GraphicsPipelineDescription {
                label: Some("SceneLightning Pipeline"),
                vertex_stage: &VertexStageInfo {
                    shader: &quad_vertex_shader,
                    entry_point: "main",
                    vertex_inputs: &[],
                },
                fragment_stage: Some(&FragmentStageInfo {
                    shader: &scene_lightning_fragment,
                    entry_point: "main",
                    render_targets: &[RenderTargetInfo {
                        format: ImageFormat::Rgba8,
                        blend: None,
                    }],
                    depth_stencil_target: None,
                }),
                primitive_restart_enabled: false,
                primitive_topology: mgpu::PrimitiveTopology::TriangleList,
                polygon_mode: mgpu::PolygonMode::Filled,
                cull_mode: mgpu::CullMode::None,
                front_face: mgpu::FrontFace::ClockWise,
                multisample_state: None,
                depth_stencil_state: DepthStencilState::default(),
                binding_set_layouts: &[
                    BindingSetLayoutInfo {
                        set: 0,
                        layout: SceneImages::binding_set_layout(),
                    },
                    BindingSetLayoutInfo {
                        set: 1,
                        layout: SceneRenderer::scene_lightning_binding_set_layout(),
                    },
                ],
                push_constant_info: None,
            })?;

        Ok(Self {
            frames: frame_data,
            clear_color: LinearColor::VIOLET,

            quad_vertex_shader,
            scene_lightning_fragment,
            scene_lightning_pipeline,

            scene_lightning_parameters_writer,
            scene_setup: SceneSetup {
                ambient_color: LinearColor::new(0.3, 0.3, 0.3, 1.0),
                ambient_intensity: 1.0,
            },

            current_frame: 0,
        })
    }
    pub fn render(&mut self, params: SceneRenderingParams) -> anyhow::Result<()> {
        let device = params.device;
        self.update_buffers(params.device, params.pov)?;

        let current_frame = &self.frames[self.current_frame];

        let mut command_recorder = device.create_command_recorder::<Graphics>();

        {
            let mut scene_output_pass =
                command_recorder.begin_render_pass(&RenderPassDescription {
                    label: Some("Scene Rendering"),
                    flags: Default::default(),
                    render_targets: &[
                        RenderTarget {
                            view: current_frame.scene_images.diffuse.view,
                            sample_count: mgpu::SampleCount::One,
                            load_op: RenderTargetLoadOp::Clear(self.clear_color.data),
                            store_op: AttachmentStoreOp::Store,
                        },
                        RenderTarget {
                            view: current_frame.scene_images.emissive_ao.view,
                            sample_count: mgpu::SampleCount::One,
                            load_op: RenderTargetLoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                            store_op: AttachmentStoreOp::Store,
                        },
                        RenderTarget {
                            view: current_frame.scene_images.position.view,
                            sample_count: mgpu::SampleCount::One,
                            load_op: RenderTargetLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                        },
                        RenderTarget {
                            view: current_frame.scene_images.normal.view,
                            sample_count: mgpu::SampleCount::One,
                            load_op: RenderTargetLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                        },
                        RenderTarget {
                            view: current_frame.scene_images.metallic_roughness.view,
                            sample_count: mgpu::SampleCount::One,
                            load_op: RenderTargetLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                        },
                    ],
                    depth_stencil_attachment: Some(&DepthStencilTarget {
                        view: current_frame.scene_images.depth.view,
                        sample_count: mgpu::SampleCount::One,
                        load_op: DepthStencilTargetLoadOp::Clear(1.0, 0),
                        store_op: AttachmentStoreOp::Store,
                    }),
                    render_area: Rect2D {
                        offset: Offset2D::default(),
                        extents: current_frame.scene_images.diffuse.image.extents().to_2d(),
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

        {
            let mut scene_lightning_pass =
                command_recorder.begin_render_pass(&RenderPassDescription {
                    label: Some("Scene lightning"),
                    flags: RenderPassFlags::DONT_FLIP_VIEWPORT,
                    render_targets: &[RenderTarget {
                        view: current_frame.final_image.view,
                        sample_count: mgpu::SampleCount::One,
                        load_op: RenderTargetLoadOp::DontCare,
                        store_op: AttachmentStoreOp::Store,
                    }],
                    depth_stencil_attachment: None,
                    render_area: Rect2D {
                        offset: Default::default(),
                        extents: current_frame.final_image.image.extents().to_2d(),
                    },
                })?;
            scene_lightning_pass.set_pipeline(self.scene_lightning_pipeline);
            scene_lightning_pass.set_binding_sets(&[
                &current_frame.scene_images.binding_set,
                &current_frame.scene_params_binding_set,
            ]);
            scene_lightning_pass.draw(6, 1, 0, 0)?;
        }

        let final_image = match params.output {
            SceneOutput::FinalImage => &current_frame.final_image,
            SceneOutput::BaseColor => &current_frame.scene_images.diffuse,
            SceneOutput::Normal => &current_frame.scene_images.normal,
            SceneOutput::EmissiveAO => &current_frame.scene_images.emissive_ao,
            SceneOutput::MetallicRoughness => &current_frame.scene_images.metallic_roughness,
            SceneOutput::WorldPosition => &current_frame.scene_images.position,
        };

        command_recorder.blit(&BlitParams {
            src_image: final_image.image,
            src_region: final_image.image.whole_region(),
            dst_image: params.output_image.owner(),
            dst_region: params.output_image.owner().whole_region(),
            filter: mgpu::FilterMode::Linear,
        });
        command_recorder.submit()?;

        self.current_frame = (self.current_frame + 1) % params.device.get_info().frames_in_flight;

        Ok(())
    }

    fn update_buffers(&mut self, device: &Device, pov: &PointOfView) -> anyhow::Result<()> {
        let current_frame = &self.frames[self.current_frame];
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

        self.scene_lightning_parameters_writer.write(
            "ambient_color",
            [
                self.scene_setup.ambient_color.r(),
                self.scene_setup.ambient_color.g(),
                self.scene_setup.ambient_color.g(),
            ],
        );
        self.scene_lightning_parameters_writer
            .write("ambient_intensity", self.scene_setup.ambient_intensity);
        self.scene_lightning_parameters_writer
            .write("eye_location", pov.transform.location.to_array());
        self.scene_lightning_parameters_writer
            .update_buffer(device, current_frame.scene_lightning_parameter_buffer)?;
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

impl RenderImage {
    pub fn new(
        device: &Device,
        format: ImageFormat,
        extents: Extents2D,
        label: &str,
        frame: usize,
    ) -> anyhow::Result<Self> {
        let attachment_flag = match format.aspect() {
            mgpu::ImageAspect::Color => ImageUsageFlags::COLOR_ATTACHMENT,
            mgpu::ImageAspect::Depth => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        };
        let image = device.create_image(&ImageDescription {
            label: Some(&format!("{}/{}", label, frame)),
            creation_flags: Default::default(),
            usage_flags: ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::SAMPLED | attachment_flag,
            extents: Extents3D {
                width: extents.width,
                height: extents.height,
                depth: 1,
            },
            dimension: mgpu::ImageDimension::D2,
            mips: 1.try_into().unwrap(),
            array_layers: 1.try_into().unwrap(),
            samples: mgpu::SampleCount::One,
            format,
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;
        let view = device.create_image_view(&ImageViewDescription {
            label: Some(&format!("{}/{} View", label, frame)),
            image,
            view_ty: mgpu::ImageViewType::D2,
            aspect: format.aspect(),
            image_subresource: image.whole_subresource(),
            format,
        })?;

        Ok(Self { image, view })
    }
}

impl SceneImages {
    pub fn new(
        device: &Device,
        extents: Extents2D,
        scene_images_samplers: Sampler,
        frame: usize,
    ) -> anyhow::Result<Self> {
        let diffuse = RenderImage::new(
            device,
            ImageFormat::Rgba32f,
            extents,
            "GBuffer diffuse",
            frame,
        )?;
        let emissive_ao = RenderImage::new(
            device,
            ImageFormat::Rgba32f,
            extents,
            "GBuffer emissive/ao",
            frame,
        )?;
        let position = RenderImage::new(
            device,
            ImageFormat::Rgba32f,
            extents,
            "GBuffer position",
            frame,
        )?;
        let normal = RenderImage::new(
            device,
            ImageFormat::Rgba32f,
            extents,
            "GBuffer normal",
            frame,
        )?;
        let metallic_roughness = RenderImage::new(
            device,
            ImageFormat::Rgba32f,
            extents,
            "GBuffer metallic/roughness",
            frame,
        )?;
        let binding_set = device.create_binding_set(
            &BindingSetDescription {
                label: Some(&format!("SceneImages BindingSet {}", frame)),
                bindings: &[
                    Binding {
                        binding: 0,
                        ty: mgpu::BindingType::Sampler(scene_images_samplers),
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: 1,
                        ty: mgpu::BindingType::SampledImage { view: diffuse.view },
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: 2,
                        ty: mgpu::BindingType::SampledImage {
                            view: emissive_ao.view,
                        },
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: 3,
                        ty: mgpu::BindingType::SampledImage {
                            view: position.view,
                        },
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: 4,
                        ty: mgpu::BindingType::SampledImage { view: normal.view },
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                    Binding {
                        binding: 5,
                        ty: mgpu::BindingType::SampledImage {
                            view: metallic_roughness.view,
                        },
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                    },
                ],
            },
            Self::binding_set_layout(),
        )?;

        Ok(Self {
            depth: RenderImage::new(
                device,
                ImageFormat::Depth32,
                extents,
                "GBuffer depth",
                frame,
            )?,
            diffuse,
            emissive_ao,
            normal,
            position,
            metallic_roughness,
            binding_set,
        })
    }

    pub fn binding_set_layout() -> &'static BindingSetLayout<'static> {
        const BINDING_SET_LAYOUT: BindingSetLayout = BindingSetLayout {
            binding_set_elements: &[
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
                    ty: mgpu::BindingSetElementKind::SampledImage,
                    shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
                },
                BindingSetElement {
                    binding: 3,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::SampledImage,
                    shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
                },
                BindingSetElement {
                    binding: 4,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::SampledImage,
                    shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
                },
                BindingSetElement {
                    binding: 5,
                    array_length: 1,
                    ty: mgpu::BindingSetElementKind::SampledImage,
                    shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
                },
            ],
        };
        &BINDING_SET_LAYOUT
    }
}
