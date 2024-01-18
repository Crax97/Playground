mod cascaded_shadow_map;
mod gbuffer;
mod render_image;
mod sampler_allocator;

use crate::render_scene::render_structs::*;
use cascaded_shadow_map::*;
use gbuffer::*;
use render_image::*;
use sampler_allocator::*;

use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use engine_macros::glsl;
use std::mem::size_of;

use crate::{
    material::{MasterMaterial, MasterMaterialDescription},
    math::shape::BoundingShape,
    post_process_pass::{PostProcessPass, PostProcessResources},
    Camera, CvarManager, Frustum, Light, LightType, MaterialDescription, MaterialInstance, Mesh,
    MeshPrimitive, PipelineTarget, RenderScene, RenderingPipeline, ScenePrimitive, Texture,
    TiledTexture2DPacker,
};

use crate::resource_map::{ResourceHandle, ResourceMap};
use gpu::{
    AccessFlags, AttachmentReference, AttachmentStoreOp, BeginRenderPassInfo, Binding,
    BufferCreateInfo, BufferHandle, BufferUsageFlags, ColorLoadOp, CommandBuffer,
    CommandBufferSubmitInfo, ComponentMapping, Extent2D, FragmentStageInfo,
    FramebufferColorAttachment, FramebufferDepthAttachment, Gpu, ImageAspectFlags, ImageCreateInfo,
    ImageFormat, ImageHandle, ImageLayout, ImageMemoryBarrier, ImageSubresourceRange,
    ImageUsageFlags, ImageViewCreateInfo, ImageViewHandle, ImageViewType, IndexType, InputRate,
    MemoryDomain, Offset2D, PipelineBarrierInfo, PipelineStageFlags, Rect2D, RenderPass,
    SampleCount, SamplerHandle, ShaderModuleCreateInfo, ShaderModuleHandle, ShaderStage,
    SubpassDependency, SubpassDescription, VertexBindingInfo, VertexStageInfo, Viewport,
    VkSwapchain,
};
use nalgebra::{point, vector, Matrix4, Point3, Point4, Vector2, Vector3, Vector4};

const SCREEN_QUAD: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/screen_quad.vert",
    entry_point = "main"
);

const SCREEN_QUAD_FLIPPED: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/screen_quad_flipped.vert",
    entry_point = "main"
);

const COMBINE_SHADER_3D: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/main_combine_shader_3d.frag",
    entry_point = "main"
);

const COMBINE_SHADER_2D: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/main_combine_shader_2d.frag",
    entry_point = "main"
);

const TEXTURE_COPY: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/texture_copy.frag",
    entry_point = "main"
);
pub struct DeferredRenderingPipeline {
    frame_buffers: Vec<FrameBuffers>,
    image_allocator: ImageAllocator,
    sampler_allocator: SamplerAllocator,
    screen_quad: ShaderModuleHandle,
    texture_copy: ShaderModuleHandle,
    combine_shader: ShaderModuleHandle,

    post_process_stack: Vec<Box<dyn PostProcessPass>>,

    in_flight_frame: usize,
    max_frames_in_flight: usize,
    active_lights: Vec<GpuLightInfo>,
    light_povs: Vec<PointOfViewData>,
    pub(crate) irradiance_map: Option<ResourceHandle<Texture>>,
    default_irradiance_map: ResourceHandle<Texture>,

    cube_mesh: ResourceHandle<Mesh>,

    pub csm_slices: u8,

    pub ambient_color: Vector3<f32>,
    pub ambient_intensity: f32,

    pub update_frustum: bool,
    pub drawcalls_last_frame: u64,

    csm_buffers: Vec<CsmBuffers>,
    shadow_atlas: ImageHandle,
    shadow_atlas_view: ImageViewHandle,
    shadow_maps: Vec<ShadowMap>,
    pub csm_split_lambda: f32,
    csm_splits: Vec<f32>,
    pub z_mult: f32,

    frustum: Frustum,
    gbuffer_nearest_sampler: SamplerHandle,
    screen_quad_flipped: ShaderModuleHandle,
    early_z_pass_enabled: bool,
    view_size: Extent2D,
}

impl DeferredRenderingPipeline {
    pub const MAX_SHADOW_CASTERS: usize = 100;
    pub fn new(
        gpu: &dyn Gpu,
        resource_map: &mut ResourceMap,
        cube_mesh: ResourceHandle<Mesh>,
        combine_shader: ShaderModuleHandle,
    ) -> anyhow::Result<Self> {
        let mut frame_buffers = vec![];
        for _ in 0..VkSwapchain::MAX_FRAMES_IN_FLIGHT {
            let camera_buffer = {
                let create_info = BufferCreateInfo {
                    label: Some("Deferred Renderer - Camera buffer"),
                    size: std::mem::size_of::<PointOfViewData>() * 100,
                    usage: BufferUsageFlags::STORAGE_BUFFER
                        | BufferUsageFlags::UNIFORM_BUFFER
                        | BufferUsageFlags::TRANSFER_DST,
                };
                gpu.make_buffer(
                    &create_info,
                    MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
                )?
            };
            let light_buffer = {
                let create_info = BufferCreateInfo {
                    label: Some("Light Buffer"),
                    size: std::mem::size_of::<GpuLightInfo>() * 1000,
                    usage: BufferUsageFlags::UNIFORM_BUFFER
                        | BufferUsageFlags::STORAGE_BUFFER
                        | BufferUsageFlags::TRANSFER_DST,
                };
                gpu.make_buffer(
                    &create_info,
                    MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
                )?
            };
            frame_buffers.push(FrameBuffers {
                camera_buffer,
                light_buffer,
            })
        }

        let screen_quad = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(SCREEN_QUAD),
        })?;

        let screen_quad_flipped = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(SCREEN_QUAD_FLIPPED),
        })?;

        let texture_copy = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(TEXTURE_COPY),
        })?;

        let gbuffer_nearest_sampler = gpu.make_sampler(&gpu::SamplerCreateInfo {
            mag_filter: gpu::Filter::Linear,
            min_filter: gpu::Filter::Linear,
            address_u: gpu::SamplerAddressMode::ClampToBorder,
            address_v: gpu::SamplerAddressMode::ClampToBorder,
            address_w: gpu::SamplerAddressMode::ClampToBorder,
            mip_lod_bias: 0.0,
            compare_function: None,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: [0.0; 4],
        })?;

        let default_irradiance_map = Texture::new_with_data(
            gpu,
            1,
            1,
            bytemuck::cast_slice(&[255u8; 6 * 3]),
            Some("Default White Irradiance Map"),
            ImageFormat::Rgb8,
            gpu::ImageViewType::Cube,
        )?;
        let default_irradiance_map = resource_map.add(default_irradiance_map);

        let view_size = Extent2D {
            width: 1920,
            height: 1080,
        };

        let shadow_atlas = gpu.make_image(
            &ImageCreateInfo {
                label: Some("CSM Shadow Atlas"),
                width: SHADOW_ATLAS_WIDTH,
                height: SHADOW_ATLAS_HEIGHT,
                depth: 1,
                mips: 1,
                layers: 1,
                samples: gpu::SampleCount::Sample1,
                format: gpu::ImageFormat::Depth,
                usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
            },
            MemoryDomain::DeviceLocal,
            None,
        )?;

        let shadow_atlas_view = gpu.make_image_view(&ImageViewCreateInfo {
            image: shadow_atlas.clone(),
            view_type: gpu::ImageViewType::Type2D,
            format: gpu::ImageFormat::Depth,
            components: ComponentMapping {
                r: gpu::ComponentSwizzle::Identity,
                g: gpu::ComponentSwizzle::Identity,
                b: gpu::ComponentSwizzle::Identity,
                a: gpu::ComponentSwizzle::Identity,
            },
            subresource_range: ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        })?;

        let mut csm_buffers = Vec::with_capacity(VkSwapchain::MAX_FRAMES_IN_FLIGHT);
        for i in 0..VkSwapchain::MAX_FRAMES_IN_FLIGHT {
            let shadow_casters = gpu.make_buffer(
                &BufferCreateInfo {
                    label: Some(&format!("Shadow Casters #{}", i)),
                    size: Self::MAX_SHADOW_CASTERS * std::mem::size_of::<ShadowMap>(),
                    usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::UNIFORM_BUFFER,
                },
                MemoryDomain::HostCoherent | MemoryDomain::HostVisible,
            )?;
            let csm_splits = gpu.make_buffer(
                &BufferCreateInfo {
                    label: Some(&format!("CSM Splits #{}", i)),
                    size: Self::MAX_SHADOW_CASTERS * std::mem::size_of::<f32>(),
                    usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::UNIFORM_BUFFER,
                },
                MemoryDomain::HostCoherent | MemoryDomain::HostVisible,
            )?;

            csm_buffers.push(CsmBuffers {
                shadow_casters,
                csm_splits,
            });
        }

        Ok(Self {
            image_allocator: ImageAllocator::new(4),
            sampler_allocator: SamplerAllocator::new(4),
            screen_quad,
            screen_quad_flipped,
            combine_shader,
            texture_copy,
            frame_buffers,
            post_process_stack: vec![],
            in_flight_frame: 0,
            max_frames_in_flight: VkSwapchain::MAX_FRAMES_IN_FLIGHT,
            ambient_color: vector![1.0, 1.0, 1.0],
            ambient_intensity: 0.3,
            active_lights: vec![],
            light_povs: vec![],
            cube_mesh,
            irradiance_map: None,
            default_irradiance_map,
            gbuffer_nearest_sampler,
            early_z_pass_enabled: true,
            view_size,

            update_frustum: true,
            frustum: Frustum::default(),
            drawcalls_last_frame: 0,
            csm_slices: 4,
            csm_buffers,
            shadow_atlas,
            csm_split_lambda: 0.98,
            shadow_atlas_view,
            shadow_maps: vec![],
            csm_splits: vec![],
            z_mult: 10.0,
        })
    }

    pub fn make_2d_combine_shader(gpu: &dyn Gpu) -> anyhow::Result<ShaderModuleHandle> {
        gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(COMBINE_SHADER_2D),
        })
    }

    pub fn make_3d_combine_shader(gpu: &dyn Gpu) -> anyhow::Result<ShaderModuleHandle> {
        gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(COMBINE_SHADER_3D),
        })
    }

    /*
    The combine shader is the shader in charge of taking the GBuffer output render targets
    and combining them in the final image.
    Take a look at main_combine_shader_2d/3d to understand how it works
    */
    pub fn set_combine_shader(&mut self, shader_handle: ShaderModuleHandle) {
        self.combine_shader = shader_handle;
    }

    pub fn set_early_z_enabled(&mut self, early_z_enabled: bool) {
        self.early_z_pass_enabled = early_z_enabled;
    }

    pub(crate) fn main_render_loop(
        gpu: &dyn Gpu,
        primitives: &Vec<&ScenePrimitive>,
        resource_map: &ResourceMap,
        pipeline_target: PipelineTarget,
        render_pass: &mut RenderPass,
        camera_index: u32,
        frame_buffers: &FrameBuffers,
        sampler_allocator: &SamplerAllocator,
    ) -> anyhow::Result<()> {
        let mut total_primitives_rendered = 0;
        for primitive in primitives {
            {
                let mesh = resource_map.get(&primitive.mesh);
                for (material_idx, mesh_prim) in mesh.primitives.iter().enumerate() {
                    let material = &primitive.materials[material_idx];
                    let master = resource_map.get(&material.owner);
                    bind_master_material(master, pipeline_target, render_pass, frame_buffers);

                    // render_pass.set_cull_mode(master.cull_mode);
                    // render_pass.set_front_face(master.front_face);
                    let model = primitive.transform;
                    draw_mesh_primitive(
                        gpu,
                        render_pass,
                        material,
                        master,
                        mesh_prim,
                        model,
                        resource_map,
                        sampler_allocator,
                        camera_index,
                    )?;
                    total_primitives_rendered += 1;
                }
            }

            render_pass.insert_debug_label(
                &format!("Total primtives drawn this frame: {total_primitives_rendered}"),
                [0.0, 0.3, 0.4, 1.0],
            );
        }
        Ok(())
    }

    fn draw_skybox(
        gpu: &dyn Gpu,
        camera_location: &Point3<f32>,
        render_pass: &mut RenderPass,
        skybox_mesh: &Mesh,
        skybox_material: &MaterialInstance,
        skybox_master: &MasterMaterial,
        resource_map: &ResourceMap,
        sampler_allocator: &SamplerAllocator,
    ) -> anyhow::Result<()> {
        // let label = render_pass.begin_debug_region(
        //     &format!("Skybox - using material {}", skybox_master.name),
        //     [0.2, 0.2, 0.0, 1.0],
        // );
        const SKYBOX_SCALE: f32 = 1.0;
        let skybox_transform =
            Matrix4::new_translation(&camera_location.coords) * Matrix4::new_scaling(SKYBOX_SCALE);
        render_pass.set_enable_depth_test(false);
        render_pass.set_depth_write_enabled(false);
        render_pass.set_color_output_enabled(true);
        render_pass.set_cull_mode(gpu::CullMode::None);
        render_pass.set_depth_compare_op(gpu::CompareOp::Always);
        draw_mesh_primitive(
            gpu,
            render_pass,
            skybox_material,
            skybox_master,
            &skybox_mesh.primitives[0],
            skybox_transform,
            resource_map,
            sampler_allocator,
            0,
        )?;
        // label.end();
        Ok(())
    }

    pub fn set_irradiance_texture(
        &mut self,
        irradiance_map: Option<ResourceHandle<crate::Texture>>,
    ) {
        self.irradiance_map = irradiance_map;
    }

    fn main_pass(
        &self,
        gpu: &dyn Gpu,
        graphics_command_buffer: &mut CommandBuffer,
        final_scene_image: &RenderImage,
        gbuffer: &GBuffer,
        resource_map: &ResourceMap,
        render_size: Extent2D,
        current_buffers: &FrameBuffers,
        primitives: &Vec<&ScenePrimitive>,
        pov: &Camera,
        scene: &RenderScene,
    ) -> anyhow::Result<()> {
        let GBuffer {
            depth_component,
            position_component,
            normal_component,
            diffuse_component,
            emissive_component,
            pbr_component,
            gbuffer_sampler: _,
            viewport_size: _,
        } = gbuffer;

        {
            let early_z_enabled_descriptions: &[SubpassDescription] = &[
                SubpassDescription {
                    label: Some("Z Pass".to_owned()),
                    input_attachments: vec![],
                    color_attachments: vec![],
                    resolve_attachments: vec![],
                    depth_stencil_attachment: Some(AttachmentReference {
                        attachment: 6,
                        layout: ImageLayout::DepthStencilAttachment,
                    }),
                    preserve_attachments: vec![],
                },
                SubpassDescription {
                    label: Some("GBuffer output".to_owned()),
                    input_attachments: vec![],
                    color_attachments: vec![
                        AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::ColorAttachment,
                        },
                        AttachmentReference {
                            attachment: 1,
                            layout: ImageLayout::ColorAttachment,
                        },
                        AttachmentReference {
                            attachment: 2,
                            layout: ImageLayout::ColorAttachment,
                        },
                        AttachmentReference {
                            attachment: 3,
                            layout: ImageLayout::ColorAttachment,
                        },
                        AttachmentReference {
                            attachment: 4,
                            layout: ImageLayout::ColorAttachment,
                        },
                    ],
                    resolve_attachments: vec![],
                    depth_stencil_attachment: Some(AttachmentReference {
                        attachment: 6,
                        layout: ImageLayout::DepthStencilReadOnly,
                    }),
                    preserve_attachments: vec![],
                },
                SubpassDescription {
                    label: Some("Gbuffer Combine".to_owned()),
                    input_attachments: vec![
                        AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        AttachmentReference {
                            attachment: 1,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        AttachmentReference {
                            attachment: 2,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        AttachmentReference {
                            attachment: 3,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        AttachmentReference {
                            attachment: 4,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                    ],
                    color_attachments: vec![AttachmentReference {
                        attachment: 5,
                        layout: ImageLayout::ColorAttachment,
                    }],
                    resolve_attachments: vec![],
                    depth_stencil_attachment: None,
                    preserve_attachments: vec![],
                },
            ];
            let early_z_disabled_descriptions: &[SubpassDescription] = &[SubpassDescription {
                label: Some("GBuffer output".to_owned()),
                input_attachments: vec![],
                color_attachments: vec![
                    AttachmentReference {
                        attachment: 0,
                        layout: ImageLayout::ColorAttachment,
                    },
                    AttachmentReference {
                        attachment: 1,
                        layout: ImageLayout::ColorAttachment,
                    },
                    AttachmentReference {
                        attachment: 2,
                        layout: ImageLayout::ColorAttachment,
                    },
                    AttachmentReference {
                        attachment: 3,
                        layout: ImageLayout::ColorAttachment,
                    },
                    AttachmentReference {
                        attachment: 4,
                        layout: ImageLayout::ColorAttachment,
                    },
                ],
                resolve_attachments: vec![],
                depth_stencil_attachment: Some(AttachmentReference {
                    attachment: 6,
                    layout: ImageLayout::DepthStencilAttachment,
                }),
                preserve_attachments: vec![],
            }];
            let early_z_enabled_dependencies: &[SubpassDependency] = &[
                SubpassDependency {
                    src_subpass: SubpassDependency::EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                    dst_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        .union(PipelineStageFlags::LATE_FRAGMENT_TESTS),
                    src_access_mask: AccessFlags::empty(),
                    dst_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                },
                SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: 1,
                    src_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        .union(PipelineStageFlags::LATE_FRAGMENT_TESTS),
                    dst_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        .union(PipelineStageFlags::LATE_FRAGMENT_TESTS)
                        .union(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                    src_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        .union(AccessFlags::COLOR_ATTACHMENT_WRITE),
                },
                SubpassDependency {
                    src_subpass: 1,
                    dst_subpass: 2,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
                    src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: AccessFlags::SHADER_READ,
                },
            ];

            let early_z_disabled_dependencies: &[SubpassDependency] = &[SubpassDependency {
                src_subpass: SubpassDependency::EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    .union(PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                    .union(PipelineStageFlags::LATE_FRAGMENT_TESTS),
                src_access_mask: AccessFlags::empty(),
                dst_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    .union(AccessFlags::COLOR_ATTACHMENT_WRITE),
            }];

            let mut render_pass =
                graphics_command_buffer.start_render_pass(&gpu::BeginRenderPassInfo {
                    label: Some("Main pass"),
                    color_attachments: &[
                        FramebufferColorAttachment {
                            image_view: position_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ShaderReadOnly,
                        },
                        FramebufferColorAttachment {
                            image_view: normal_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.5; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ShaderReadOnly,
                        },
                        FramebufferColorAttachment {
                            image_view: diffuse_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ShaderReadOnly,
                        },
                        FramebufferColorAttachment {
                            image_view: emissive_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ShaderReadOnly,
                        },
                        FramebufferColorAttachment {
                            image_view: pbr_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ShaderReadOnly,
                        },
                        FramebufferColorAttachment {
                            image_view: final_scene_image.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ShaderReadOnly,
                        },
                    ],
                    depth_attachment: Some(gpu::FramebufferDepthAttachment {
                        image_view: depth_component.view.clone(),
                        load_op: gpu::DepthLoadOp::Clear(1.0),
                        store_op: AttachmentStoreOp::Store,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::DepthStencilReadOnly,
                    }),
                    stencil_attachment: None,
                    render_area: gpu::Rect2D {
                        offset: gpu::Offset2D::default(),
                        extent: render_size,
                    },
                    subpasses: if self.early_z_pass_enabled {
                        early_z_enabled_descriptions
                    } else {
                        early_z_disabled_descriptions
                    },
                    dependencies: if self.early_z_pass_enabled {
                        early_z_enabled_dependencies
                    } else {
                        early_z_disabled_dependencies
                    },
                });

            if self.early_z_pass_enabled {
                render_pass.set_cull_mode(gpu::CullMode::Back);
                render_pass.set_depth_compare_op(gpu::CompareOp::LessEqual);

                render_pass.set_color_output_enabled(false);
                render_pass.set_enable_depth_test(true);
                render_pass.set_depth_write_enabled(true);
                Self::main_render_loop(
                    gpu,
                    primitives,
                    resource_map,
                    PipelineTarget::DepthOnly,
                    &mut render_pass,
                    0,
                    current_buffers,
                    &self.sampler_allocator,
                )
                .context("Early Z Pass")?;

                render_pass.advance_to_next_subpass();
            }

            if let Some(material) = scene.get_skybox_material() {
                let cube_mesh = resource_map.get(&self.cube_mesh);
                let skybox_master = resource_map.get(&material.owner);
                bind_master_material(
                    skybox_master,
                    PipelineTarget::ColorAndDepth,
                    &mut render_pass,
                    current_buffers,
                );
                Self::draw_skybox(
                    gpu,
                    &pov.location,
                    &mut render_pass,
                    cube_mesh,
                    material,
                    skybox_master,
                    resource_map,
                    &self.sampler_allocator,
                )?;
            }

            render_pass.set_front_face(gpu::FrontFace::CounterClockWise);
            render_pass.set_enable_depth_test(true);
            render_pass.set_depth_write_enabled(!self.early_z_pass_enabled);
            render_pass.set_color_output_enabled(true);
            render_pass.set_cull_mode(gpu::CullMode::Back);
            render_pass.set_depth_compare_op(if self.early_z_pass_enabled {
                gpu::CompareOp::Equal
            } else {
                gpu::CompareOp::LessEqual
            });
            Self::main_render_loop(
                gpu,
                primitives,
                resource_map,
                PipelineTarget::ColorAndDepth,
                &mut render_pass,
                0,
                current_buffers,
                &self.sampler_allocator,
            )
            .context("Gbuffer output pass")?;
            render_pass.advance_to_next_subpass();

            // Combine
            let csm_buffers = &self.csm_buffers[self.in_flight_frame];
            gbuffer.bind_as_input_attachments(&mut render_pass, 0, 0);
            render_pass.bind_resources(
                1,
                &[
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: self.shadow_atlas_view.clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 0,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::StorageBuffer {
                            handle: csm_buffers.shadow_casters.clone(),
                            offset: 0,
                            range: gpu::WHOLE_SIZE as _,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 1,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: resource_map
                                .get(
                                    self.irradiance_map
                                        .as_ref()
                                        .unwrap_or(&self.default_irradiance_map),
                                )
                                .view
                                .clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 2,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::StorageBuffer {
                            handle: current_buffers.camera_buffer.clone(),
                            offset: 0,
                            range: gpu::WHOLE_SIZE as usize,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 3,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::StorageBuffer {
                            handle: current_buffers.light_buffer.clone(),
                            offset: 0,
                            range: gpu::WHOLE_SIZE as usize,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 4,
                    },
                ],
            );

            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct CSMConstantData {
                count: u32,
                splits: [f32; 6],
            }

            let mut data = CSMConstantData {
                count: self.csm_splits.len() as _,
                splits: [0.0; 6],
            };

            for (i, split) in self.csm_splits.iter().take(6).enumerate() {
                data.splits[i] = *split;
            }

            render_pass.push_constants(0, 0, bytemuck::cast_slice(&[data]), ShaderStage::FRAGMENT);

            render_pass.set_front_face(gpu::FrontFace::ClockWise);
            render_pass.set_cull_mode(gpu::CullMode::None);
            render_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
            render_pass.set_enable_depth_test(false);
            render_pass.set_depth_write_enabled(false);
            render_pass.set_vertex_shader(self.screen_quad.clone());
            render_pass.set_fragment_shader(self.combine_shader.clone());
            render_pass.draw(4, 1, 0, 0)?;
        }

        Ok(())
    }

    pub fn draw_textured_quad(
        &self,
        graphics_command_buffer: &mut CommandBuffer,
        destination: &ImageViewHandle,
        source: &ImageViewHandle,
        viewport: Rect2D,
        flip_render_target: bool,
        override_shader: Option<ShaderModuleHandle>,
    ) -> anyhow::Result<()> {
        let mut present_render_pass =
            graphics_command_buffer.start_render_pass(&gpu::BeginRenderPassInfo {
                label: Some("Copy to backbuffer"),
                color_attachments: &[FramebufferColorAttachment {
                    image_view: destination.clone(),
                    load_op: ColorLoadOp::DontCare,
                    store_op: AttachmentStoreOp::Store,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ColorAttachment,
                }],
                depth_attachment: None,
                stencil_attachment: None,
                render_area: viewport,
                subpasses: &[SubpassDescription {
                    label: None,
                    input_attachments: vec![],
                    color_attachments: vec![AttachmentReference {
                        attachment: 0,
                        layout: ImageLayout::ColorAttachment,
                    }],
                    resolve_attachments: vec![],
                    depth_stencil_attachment: None,
                    preserve_attachments: vec![],
                }],
                dependencies: &[],
            });
        present_render_pass.bind_resources(
            0,
            &[Binding {
                ty: gpu::DescriptorBindingType::ImageView {
                    image_view_handle: source.clone(),
                    sampler_handle: self.gbuffer_nearest_sampler.clone(),
                    layout: gpu::ImageLayout::ShaderReadOnly,
                },
                binding_stage: ShaderStage::FRAGMENT,
                location: 0,
            }],
        );

        let screen_quad = if flip_render_target {
            self.screen_quad_flipped.clone()
        } else {
            self.screen_quad.clone()
        };

        present_render_pass.set_front_face(gpu::FrontFace::ClockWise);
        present_render_pass.set_cull_mode(gpu::CullMode::None);
        present_render_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
        present_render_pass.set_enable_depth_test(false);
        present_render_pass.set_depth_write_enabled(false);
        present_render_pass.set_vertex_shader(screen_quad);
        present_render_pass
            .set_fragment_shader(override_shader.unwrap_or(self.texture_copy.clone()));
        present_render_pass.draw(4, 1, 0, 0)
    }

    fn post_process_pass(
        &self,
        gpu: &dyn Gpu,
        graphics_command_buffer: &mut CommandBuffer,
        color_output: RenderImage,
        color_desc: RenderImageDescription,
        render_size: Extent2D,
        cvar_manager: &CvarManager,
    ) -> anyhow::Result<RenderImage> {
        if self.post_process_stack.is_empty() {
            return Ok(color_output);
        }

        let subpasses = self
            .post_process_stack
            .iter()
            .enumerate()
            .map(|(i, pass)| SubpassDescription {
                label: Some(pass.name()),
                input_attachments: if i == 0 {
                    vec![]
                } else {
                    vec![AttachmentReference {
                        attachment: (i as u32) % 2,
                        layout: ImageLayout::ShaderReadOnly,
                    }]
                },
                color_attachments: vec![AttachmentReference {
                    attachment: (i + 1) as u32 % 2,
                    layout: ImageLayout::ColorAttachment,
                }],
                resolve_attachments: vec![],
                depth_stencil_attachment: None,
                preserve_attachments: vec![],
            })
            .collect::<Vec<_>>();

        let mut dependencies = self
            .post_process_stack
            .iter()
            .enumerate()
            .map(|(i, _)| SubpassDependency {
                src_subpass: i as u32,
                dst_subpass: i as u32 + 1,
                src_stage_mask: PipelineStageFlags::FRAGMENT_SHADER
                    | PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER
                    | PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: AccessFlags::SHADER_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: AccessFlags::SHADER_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
            })
            .collect::<Vec<_>>();
        dependencies.pop();

        let final_color_output = {
            let final_color_output = {
                let post_process_backbuffer_1 = color_output;
                let post_process_backbuffer_2 =
                    self.image_allocator.get(gpu, "post_process2", &color_desc);

                let mut current_postprocess = 0;
                let mut post_process_pass =
                    graphics_command_buffer.start_render_pass(&BeginRenderPassInfo {
                        label: Some("Post Process"),
                        color_attachments: &[
                            FramebufferColorAttachment {
                                image_view: post_process_backbuffer_1.view.clone(),
                                load_op: ColorLoadOp::Clear([0.0; 4]),
                                store_op: AttachmentStoreOp::Store,
                                initial_layout: ImageLayout::ShaderReadOnly,
                                final_layout: ImageLayout::ColorAttachment,
                            },
                            FramebufferColorAttachment {
                                image_view: post_process_backbuffer_2.view.clone(),
                                load_op: ColorLoadOp::Clear([0.0; 4]),
                                store_op: AttachmentStoreOp::Store,
                                initial_layout: ImageLayout::Undefined,
                                final_layout: ImageLayout::ColorAttachment,
                            },
                        ],
                        depth_attachment: None,
                        stencil_attachment: None,
                        render_area: Rect2D {
                            offset: Offset2D::default(),
                            extent: render_size,
                        },
                        subpasses: &subpasses,
                        dependencies: &dependencies,
                    });
                post_process_pass.set_cull_mode(gpu::CullMode::None);
                post_process_pass.set_enable_depth_test(false);
                post_process_pass.set_depth_write_enabled(false);
                post_process_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);

                for pass in &self.post_process_stack {
                    let previous_pass_result = if current_postprocess == 0 {
                        &post_process_backbuffer_1.view
                    } else {
                        &post_process_backbuffer_2.view
                    };

                    pass.apply(
                        &mut post_process_pass,
                        &PostProcessResources {
                            screen_quad: &self.screen_quad,
                            previous_pass_result,
                            sampler: &self.gbuffer_nearest_sampler,
                            cvar_manager,
                            render_size,
                        },
                    )?;
                    current_postprocess = (current_postprocess + 1) % 2;
                    post_process_pass.advance_to_next_subpass();
                }

                if current_postprocess == 0 {
                    post_process_backbuffer_1
                } else {
                    post_process_backbuffer_2
                }
            };
            graphics_command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
                memory_barriers: &[],
                buffer_memory_barriers: &[],
                image_memory_barriers: &[ImageMemoryBarrier {
                    src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: AccessFlags::SHADER_READ,
                    old_layout: ImageLayout::ColorAttachment,
                    new_layout: ImageLayout::ShaderReadOnly,
                    src_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
                    image: final_color_output.image.clone(),
                    subresource_range: ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                }],
            });
            final_color_output
        };
        Ok(final_color_output)
    }

    pub fn add_post_process_pass(&mut self, pass: impl PostProcessPass) {
        self.post_process_stack.push(Box::new(pass))
    }

    pub fn get_shadow_texture(&self, gpu: &dyn Gpu) -> ImageViewHandle {
        self.shadow_atlas_view.clone()
    }

    fn render_shadow_atlas(
        &self,
        gpu: &dyn Gpu,
        camera: &crate::Camera,
        scene: &crate::RenderScene,
        command_buffer: &mut gpu::CommandBuffer,
        frame_buffers: &FrameBuffers,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<()> {
        {
            let mut shadow_atlas_pass =
                command_buffer.start_render_pass(&gpu::BeginRenderPassInfo {
                    label: Some("CSM Shadow Atlas Emit"),
                    color_attachments: &[],
                    depth_attachment: Some(FramebufferDepthAttachment {
                        image_view: self.shadow_atlas_view.clone(),
                        load_op: gpu::DepthLoadOp::Clear(1.0),
                        store_op: gpu::AttachmentStoreOp::Store,
                        initial_layout: gpu::ImageLayout::Undefined,
                        final_layout: gpu::ImageLayout::ShaderReadOnly,
                    }),
                    stencil_attachment: None,
                    render_area: Rect2D {
                        offset: Default::default(),
                        extent: Extent2D {
                            width: SHADOW_ATLAS_WIDTH,
                            height: SHADOW_ATLAS_HEIGHT,
                        },
                    },
                    subpasses: &[SubpassDescription {
                        label: None,
                        input_attachments: vec![],
                        color_attachments: vec![],
                        resolve_attachments: vec![],
                        depth_stencil_attachment: Some(AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::DepthStencilAttachment,
                        }),
                        preserve_attachments: vec![],
                    }],
                    dependencies: &[SubpassDependency {
                        src_subpass: SubpassDependency::EXTERNAL,
                        dst_subpass: 0,
                        src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        dst_stage_mask: PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    }],
                });

            for shadow_map in &self.shadow_maps {
                let caster_pov = shadow_map.type_num_maps_pov_lightid[2];
                // We subtract 1 because index 0 is reserved for the camera
                let pov = &self.light_povs[caster_pov as usize - 1];
                let frustum = Frustum::from_view_proj(pov.view, pov.projection);
                let primitives = scene.intersect_frustum(&frustum);
                let pov_idx = shadow_map.type_num_maps_pov_lightid[2];
                let light = &scene.lights[shadow_map.type_num_maps_pov_lightid[3] as usize];
                let setup = light
                    .shadow_configuration
                    .expect("Bug: a light is set to render a shadow map, but has no shadow setup");

                shadow_atlas_pass.set_depth_bias(setup.depth_bias, setup.depth_slope);
                shadow_atlas_pass.set_cull_mode(gpu::CullMode::Back);
                shadow_atlas_pass.set_depth_compare_op(gpu::CompareOp::LessEqual);

                shadow_atlas_pass.set_color_output_enabled(false);
                shadow_atlas_pass.set_enable_depth_test(true);
                shadow_atlas_pass.set_depth_write_enabled(true);
                let width = shadow_map.offset_size[2] as f32;
                shadow_atlas_pass.set_viewport(Viewport {
                    x: shadow_map.offset_size[0] as f32,
                    y: shadow_map.offset_size[1] as f32,
                    width,
                    height: shadow_map.offset_size[3] as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                });

                DeferredRenderingPipeline::main_render_loop(
                    gpu,
                    &primitives,
                    resource_map,
                    PipelineTarget::DepthOnly,
                    &mut shadow_atlas_pass,
                    pov_idx,
                    frame_buffers,
                    &self.sampler_allocator,
                )?;
            }
        }

        Ok(())
    }

    fn get_gbuffer(&self, gpu: &dyn Gpu) -> GBuffer {
        let vector_desc = RenderImageDescription {
            format: ImageFormat::RgbaFloat32,
            samples: SampleCount::Sample1,
            width: self.view_size.width,
            height: self.view_size.height,
            view_type: ImageViewType::Type2D,
        };
        let depth_desc = RenderImageDescription {
            format: ImageFormat::Depth,
            samples: SampleCount::Sample1,
            width: self.view_size.width,
            height: self.view_size.height,
            view_type: ImageViewType::Type2D,
        };

        let depth_component = self.image_allocator.get(gpu, "depth", &depth_desc);
        let position_component = self.image_allocator.get(gpu, "position", &vector_desc);
        let normal_component = self.image_allocator.get(gpu, "normal", &vector_desc);
        let diffuse_component = self.image_allocator.get(gpu, "diffuse", &vector_desc);
        let emissive_component = self.image_allocator.get(gpu, "emissive", &vector_desc);
        let pbr_component = self.image_allocator.get(gpu, "pbr_component", &vector_desc);
        GBuffer {
            depth_component,
            position_component,
            normal_component,
            diffuse_component,
            emissive_component,
            pbr_component,
            gbuffer_sampler: self.gbuffer_nearest_sampler.clone(),
            viewport_size: self.view_size,
        }
    }

    fn update_light_data(
        &mut self,
        gpu: &dyn Gpu,
        scene_camera: &Camera,
        scene: &RenderScene,
    ) -> anyhow::Result<()> {
        self.active_lights.clear();
        self.light_povs.clear();
        self.csm_splits.clear();

        let camera_splits = scene_camera.split_into_slices(self.csm_slices, self.csm_split_lambda);
        self.csm_splits.extend(camera_splits.iter().map(|c| c.far));

        let mut texture_packer = TiledTexture2DPacker::new(
            SHADOW_ATLAS_TILE_SIZE,
            SHADOW_ATLAS_WIDTH,
            SHADOW_ATLAS_HEIGHT,
        )
        .expect("Failed to create packer");

        let mut shadow_maps = vec![];

        let mut pov_idx = 1;

        for (light_id, l) in scene.lights.iter().enumerate() {
            if !l.enabled {
                continue;
            }
            let ty = match l.ty {
                crate::LightType::Point => 1,
                crate::LightType::Directional { .. } => 2,
                crate::LightType::Spotlight { .. } => 3,
                crate::LightType::Rect { .. } => 4,
            };

            let num_maps = match l.ty {
                crate::LightType::Point => 6,
                crate::LightType::Directional { .. } => self.csm_slices as u32,
                crate::LightType::Spotlight { .. } => 1,
                crate::LightType::Rect { .. } => 1,
            };

            let shadow_map_width =
                SHADOW_ATLAS_TILE_SIZE * l.shadow_configuration.unwrap().importance.get();
            let shadow_map_height = shadow_map_width;

            let mut gpu_light: GpuLightInfo = l.into();
            if let Ok(allocation) =
                texture_packer.allocate(shadow_map_width * num_maps, shadow_map_height)
            {
                let this_light_cameras = l.light_cameras();
                let povs = self.get_light_povs(l, &camera_splits, this_light_cameras);
                let shadow_map_idx = shadow_maps.len();
                let light_shadow_maps = povs
                    .iter()
                    .enumerate()
                    .map(|(i, _)| ShadowMap {
                        offset_size: [
                            allocation.x + shadow_map_width * i as u32,
                            allocation.y,
                            shadow_map_width,
                            shadow_map_height,
                        ],
                        type_num_maps_pov_lightid: [
                            ty,
                            num_maps,
                            pov_idx + i as u32,
                            light_id as u32,
                        ],
                    })
                    .collect::<Vec<_>>();
                pov_idx += light_shadow_maps.len() as u32;
                self.light_povs.extend(povs);
                shadow_maps.extend(light_shadow_maps.into_iter());
                gpu_light.ty[1] = shadow_map_idx as i32;
            }
            self.active_lights.push(gpu_light);
        }

        let ambient = vector![
            self.ambient_color.x,
            self.ambient_color.y,
            self.ambient_color.z,
            self.ambient_intensity
        ];

        let current_buffers = &self.frame_buffers[self.in_flight_frame];

        gpu.write_buffer(
            &current_buffers.light_buffer,
            0,
            bytemuck::cast_slice(&[ambient.x, ambient.y, ambient.z, ambient.w]),
        )?;
        gpu.write_buffer(
            &current_buffers.light_buffer,
            std::mem::size_of::<Vector4<f32>>() as _,
            bytemuck::cast_slice(&[self.active_lights.len() as u32]),
        )?;
        gpu.write_buffer(
            &current_buffers.light_buffer,
            std::mem::size_of::<Vector4<f32>>() as u64 + size_of::<u32>() as u64 * 4,
            bytemuck::cast_slice(&self.active_lights),
        )?;

        let current_buffers = &self.csm_buffers[self.in_flight_frame];
        self.shadow_maps = shadow_maps;

        gpu.write_buffer(
            &current_buffers.shadow_casters,
            0,
            bytemuck::cast_slice(&self.shadow_maps),
        )?;

        gpu.write_buffer(
            &current_buffers.csm_splits,
            0,
            bytemuck::cast_slice(&[self.csm_splits.len() as u32]),
        )?;
        gpu.write_buffer(
            &current_buffers.csm_splits,
            std::mem::size_of::<u32>() as u64 * 4u64,
            bytemuck::cast_slice(&self.csm_splits),
        )?;

        Ok(())
    }

    fn get_light_povs(
        &self,
        l: &Light,
        camera_splits: &Vec<Camera>,
        light_povs: Vec<Camera>,
    ) -> Vec<PointOfViewData> {
        match l.ty {
            LightType::Directional { direction, .. } => {
                let povs = camera_splits.iter().map(|split| {
                    let split_viewproj = split.projection() * split.view();
                    let inv_splitviewproj = split_viewproj.try_inverse().unwrap();
                    let ndc_cube_corners = [
                        vector![1.0, 1.0, 1.0, 1.0],
                        vector![-1.0, 1.0, 1.0, 1.0],
                        vector![1.0, 1.0, -1.0, 1.0],
                        vector![-1.0, 1.0, -1.0, 1.0],
                        vector![1.0, -1.0, 1.0, 1.0],
                        vector![-1.0, -1.0, 1.0, 1.0],
                        vector![1.0, -1.0, -1.0, 1.0],
                        vector![-1.0, -1.0, -1.0, 1.0],
                    ];
                    let frustum_corners = ndc_cube_corners
                        .iter()
                        .map(|corner| inv_splitviewproj * corner)
                        .map(|v| v / v.w);

                    let mut frustum_center = Vector3::default();
                    for corner in frustum_corners.clone() {
                        frustum_center += corner.xyz();
                    }
                    frustum_center /= 8.0;
                    let frustum_center = Point3::from(frustum_center.xyz());

                    let light_view = Matrix4::look_at_rh(
                        &frustum_center,
                        &(frustum_center + direction),
                        &vector![0.0, 1.0, 0.0],
                    );
                    let frustum_bounds = BoundingShape::bounding_box_from_points(
                        frustum_corners
                            .map(|pt| light_view * pt)
                            .map(|corner| Point3::from(corner.xyz())),
                    );
                    let (frustum_min, mut frustum_max) = frustum_bounds.box_extremes();

                    let radius = (frustum_max - frustum_min).magnitude();
                    let mut frustum_min = frustum_center + vector![-radius, -radius, -radius];
                    let mut frustum_max = frustum_center + vector![radius, radius, radius];

                    if frustum_min.z < 0.0 {
                        frustum_min.z *= self.z_mult;
                    } else {
                        frustum_min.z /= self.z_mult;
                    }

                    if frustum_max.z < 0.0 {
                        frustum_max.z *= self.z_mult;
                    } else {
                        frustum_max.z /= self.z_mult;
                    }

                    let light_projection = Matrix4::new_orthographic(
                        frustum_min.x,
                        frustum_max.x,
                        frustum_min.y,
                        frustum_max.y,
                        frustum_min.z,
                        frustum_max.z,
                    );
                    PointOfViewData {
                        eye: point![l.position.x, l.position.y, l.position.z, 0.0],
                        eye_forward: direction.to_homogeneous(),
                        view: light_view,
                        projection: light_projection,
                    }
                });
                povs.collect()
            }
            _ => light_povs
                .into_iter()
                .map(|c| PointOfViewData {
                    eye: point![c.location.x, c.location.y, c.location.z, 0.0],
                    eye_forward: c.forward.to_homogeneous(),
                    view: c.view(),
                    projection: c.projection(),
                })
                .collect(),
        }
    }
}

pub fn bind_master_material(
    master: &MasterMaterial,
    pipeline_target: PipelineTarget,
    render_pass: &mut RenderPass,
    frame_buffers: &FrameBuffers,
) {
    let permutation = master
        .get_permutation(pipeline_target)
        .expect("failed to fetch permutation {pipeline_target:?}");
    render_pass.set_vertex_shader(permutation.vertex_shader.clone());
    if let Some(fragment_shader) = &permutation.fragment_shader {
        render_pass.set_fragment_shader(fragment_shader.clone());
    }
    render_pass.bind_resources(
        0,
        &[
            Binding {
                ty: gpu::DescriptorBindingType::StorageBuffer {
                    handle: frame_buffers.camera_buffer.clone(),
                    offset: 0,
                    range: gpu::WHOLE_SIZE as _,
                },
                binding_stage: ShaderStage::ALL_GRAPHICS,
                location: 0,
            },
            Binding {
                ty: gpu::DescriptorBindingType::UniformBuffer {
                    handle: frame_buffers.light_buffer.clone(),
                    offset: 0,
                    range: 100 * size_of::<ObjectDrawInfo>(),
                },
                binding_stage: ShaderStage::ALL_GRAPHICS,
                location: 1,
            },
        ],
    );
}

fn draw_mesh_primitive(
    gpu: &dyn Gpu,
    render_pass: &mut RenderPass,
    material: &MaterialInstance,
    master: &MasterMaterial,
    primitive: &MeshPrimitive,
    model: Matrix4<f32>,
    resource_map: &ResourceMap,
    sampler_allocator: &SamplerAllocator,
    camera_index: u32,
) -> anyhow::Result<()> {
    render_pass.set_index_buffer(primitive.index_buffer.clone(), IndexType::Uint32, 0);

    let mut user_bindings = vec![];
    user_bindings.extend(
        &mut master
            .texture_inputs
            .iter()
            .enumerate()
            .map(|(i, tex_info)| {
                let texture_parameter = &material.textures[i];
                let tex = resource_map.get(texture_parameter);

                let sampler_handle = sampler_allocator.get(gpu, &tex.sampler_settings);

                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: tex.view.clone(),
                        sampler_handle,
                        layout: gpu::ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: tex_info.shader_stage,
                    location: i as _,
                }
            }),
    );
    for user_buffer in &material.parameter_buffers {
        user_bindings.push(Binding {
            ty: gpu::DescriptorBindingType::UniformBuffer {
                handle: user_buffer.clone(),
                offset: 0,
                range: gpu::WHOLE_SIZE as _,
            },
            binding_stage: master.parameter_shader_stages,
            location: master.texture_inputs.len() as u32,
        });
    }

    render_pass.set_vertex_buffers(&[
        VertexBindingInfo {
            handle: primitive.position_component.clone(),
            location: 0,
            offset: 0,
            stride: std::mem::size_of::<Vector3<f32>>() as _,
            format: ImageFormat::RgbFloat32,
            input_rate: InputRate::PerVertex,
        },
        VertexBindingInfo {
            handle: primitive.color_component.clone(),
            location: 1,
            offset: 0,
            stride: std::mem::size_of::<Vector3<f32>>() as _,
            format: ImageFormat::RgbFloat32,
            input_rate: InputRate::PerVertex,
        },
        VertexBindingInfo {
            handle: primitive.normal_component.clone(),
            location: 2,
            offset: 0,
            stride: std::mem::size_of::<Vector3<f32>>() as _,
            format: ImageFormat::RgbFloat32,
            input_rate: InputRate::PerVertex,
        },
        VertexBindingInfo {
            handle: primitive.tangent_component.clone(),
            location: 3,
            offset: 0,
            stride: std::mem::size_of::<Vector3<f32>>() as _,
            format: ImageFormat::RgbFloat32,
            input_rate: InputRate::PerVertex,
        },
        VertexBindingInfo {
            handle: primitive.uv_component.clone(),
            location: 4,
            offset: 0,
            stride: std::mem::size_of::<Vector2<f32>>() as _,
            format: ImageFormat::RgFloat32,
            input_rate: InputRate::PerVertex,
        },
    ]);
    render_pass.bind_resources(1, &user_bindings);
    render_pass.push_constants(
        0,
        0,
        bytemuck::cast_slice(&[ObjectDrawInfo {
            model,
            camera_index,
        }]),
        ShaderStage::ALL_GRAPHICS,
    );
    render_pass.draw_indexed(primitive.index_count, 1, 0, 0, 0)
}

impl RenderingPipeline for DeferredRenderingPipeline {
    fn render(
        &mut self,
        gpu: &dyn Gpu,
        camera: &Camera,
        scene: &RenderScene,
        resource_map: &ResourceMap,
        cvar_manager: &CvarManager,
    ) -> anyhow::Result<ImageViewHandle> {
        if self.update_frustum {
            self.frustum = camera.frustum();
        }

        self.update_light_data(gpu, camera, scene)?;

        let projection = camera.projection();

        let current_buffers = &self.frame_buffers[self.in_flight_frame];

        let mut per_frame_data = vec![PointOfViewData {
            eye: Point4::new(
                camera.location[0],
                camera.location[1],
                camera.location[2],
                0.0,
            ),
            eye_forward: vector![camera.forward.x, camera.forward.y, camera.forward.z, 0.0],
            view: camera.view(),
            projection,
        }];

        per_frame_data.extend_from_slice(&self.light_povs);

        gpu.write_buffer(
            &current_buffers.camera_buffer,
            0,
            bytemuck::cast_slice(&per_frame_data),
        )
        .unwrap();

        // let slices = pov.split_into_slices(4);
        // let slices = slices.iter().map(|sl| sl.near).collect::<Vec<_>>();

        let color_desc = RenderImageDescription {
            format: ImageFormat::Rgba8,
            samples: SampleCount::Sample1,
            width: self.view_size.width,
            height: self.view_size.height,
            view_type: ImageViewType::Type2D,
        };

        let color_output =
            self.image_allocator
                .get(gpu, "color_output/post_process_1", &color_desc);

        let mut graphics_command_buffer = gpu.start_command_buffer(gpu::QueueType::Graphics)?;

        self.render_shadow_atlas(
            gpu,
            camera,
            scene,
            &mut graphics_command_buffer,
            current_buffers,
            resource_map,
        )?;

        let gbuffer = self.get_gbuffer(gpu);
        let primitives = scene.intersect_frustum(&self.frustum);
        self.drawcalls_last_frame = primitives.len() as u64;
        self.main_pass(
            gpu,
            &mut graphics_command_buffer,
            &color_output,
            &gbuffer,
            resource_map,
            self.view_size,
            current_buffers,
            &primitives,
            camera,
            scene,
        )
        .context("Main pass")?;

        let final_color_output = self
            .post_process_pass(
                gpu,
                &mut graphics_command_buffer,
                color_output,
                color_desc,
                self.view_size,
                cvar_manager,
            )
            .context("Post process pass")?;

        graphics_command_buffer.submit(&CommandBufferSubmitInfo {
            wait_semaphores: &[],
            wait_stages: &[],
            signal_semaphores: &[],
            fence: None,
        })?;

        self.in_flight_frame = (1 + self.in_flight_frame) % self.max_frames_in_flight;
        Ok(final_color_output.view)
    }

    fn create_material(
        &mut self,
        _gpu: &dyn Gpu,
        material_description: MaterialDescription,
    ) -> anyhow::Result<MasterMaterial> {
        let master_description = MasterMaterialDescription {
            name: material_description.name,
            domain: material_description.domain,
            texture_inputs: material_description.texture_inputs,
            material_parameters: material_description.material_parameters,
            vertex_info: &VertexStageInfo {
                entry_point: "main",
                module: material_description.vertex_module.clone(),
            },
            fragment_info: &FragmentStageInfo {
                entry_point: "main",
                module: material_description.fragment_module,
            },
            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::CounterClockWise,
            parameters_visibility: material_description.parameter_shader_visibility,
        };

        MasterMaterial::new(&master_description)
    }

    fn on_resolution_changed(&mut self, new_resolution: Extent2D) {
        self.view_size = new_resolution;
    }
}
