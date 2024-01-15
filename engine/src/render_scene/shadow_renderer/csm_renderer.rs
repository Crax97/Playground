use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use engine_macros::glsl;
use gpu::{
    lifetime_cache_constants::NEVER_DEALLOCATE, AccessFlags, AttachmentReference, Binding,
    BufferCreateInfo, BufferHandle, BufferUsageFlags, ComponentMapping, DescriptorBindingType,
    Extent2D, FramebufferColorAttachment, FramebufferDepthAttachment, Gpu, ImageAspectFlags,
    ImageCreateInfo, ImageHandle, ImageLayout, ImageSubresourceRange, ImageUsageFlags,
    ImageViewCreateInfo, ImageViewHandle, MemoryDomain, PipelineStageFlags, Rect2D, SamplerHandle,
    ShaderModuleHandle, ShaderStage, SubpassDependency, SubpassDescription, Viewport,
};
use nalgebra::{point, Point3, Point4, Vector4};

use crate::{
    DeferredRenderingPipeline, FrameBuffers, Gbuffer, ImageAllocator, PipelineTarget,
    PointOfViewData, RenderImage, RenderImageDescription, ResourceMap, SamplerAllocator,
    TiledTexture2DPacker,
};

use super::ShadowRenderer;

pub const SHADOW_ATLAS_TILE_SIZE: u32 = 128;
pub const SHADOW_ATLAS_WIDTH: u32 = 7680;
pub const SHADOW_ATLAS_HEIGHT: u32 = 4352;

const SHADOW_EMIT_SOURCE: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/shadow_emit.frag",
    entry_point = "main"
);

pub struct CsmBuffers {
    shadow_casters: BufferHandle,
    csm_splits: BufferHandle,
}

pub struct CsmRenderer {
    buffer_allocator: ImageAllocator,
    sampler_allocator: SamplerAllocator,
    csm_buffers: Vec<CsmBuffers>,
    frame_buffers: Vec<FrameBuffers>,
    num_framebuffers: usize,
    cur_framebuffer: usize,
    viewport_size: Extent2D,

    shadow_atlas: ImageHandle,
    shadow_atlas_view: ImageViewHandle,
    shadow_atlas_sampler: SamplerHandle,
    linear_sampler: SamplerHandle,

    csm_split_count: u32,

    screen_quad_vs: ShaderModuleHandle,
    shadow_emit_handle: ShaderModuleHandle,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
pub struct ShadowCaster {
    offset_size: [u32; 4],
    // type: x, num_maps: y, pov: z, split_idx: w (directional only)
    type_num_maps_pov_splitidx: [u32; 4],
}

impl CsmRenderer {
    pub const MAX_SHADOW_CASTERS: usize = 100;
    pub fn new(
        gpu: &dyn Gpu,
        num_framebuffers: usize,
        initial_viewport_size: Extent2D,
        linear_sampler: SamplerHandle,
        screen_quad_vs: ShaderModuleHandle,
    ) -> anyhow::Result<Self> {
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
        let shadow_atlas_sampler = gpu.make_sampler(&gpu::SamplerCreateInfo {
            mag_filter: gpu::Filter::Nearest,
            min_filter: gpu::Filter::Nearest,
            address_u: gpu::SamplerAddressMode::ClampToEdge,
            address_v: gpu::SamplerAddressMode::ClampToEdge,
            address_w: gpu::SamplerAddressMode::ClampToEdge,
            mip_lod_bias: 0.0,
            compare_function: Some(gpu::CompareOp::LessEqual),
            min_lod: 0.0,
            max_lod: 1.0,
            border_color: [0.0; 4],
        })?;

        let mut csm_buffers = Vec::with_capacity(num_framebuffers);
        let mut frame_buffers = vec![];
        for i in 0..num_framebuffers {
            let shadow_casters = gpu.make_buffer(
                &BufferCreateInfo {
                    label: Some(&format!("Shadow Casters #{}", i)),
                    size: Self::MAX_SHADOW_CASTERS * std::mem::size_of::<ShadowCaster>(),
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
                    size: std::mem::size_of::<PointOfViewData>() * 1000,
                    usage: BufferUsageFlags::UNIFORM_BUFFER
                        | BufferUsageFlags::STORAGE_BUFFER
                        | BufferUsageFlags::TRANSFER_DST,
                };
                gpu.make_buffer(
                    &create_info,
                    MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
                )?
            };
            let csm_buffer = {
                let create_info = BufferCreateInfo {
                    label: Some("CSM Buffer"),
                    size: std::mem::size_of::<f32>() * 17,
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
                csm_buffer,
            });
        }

        let shadow_emit_handle = gpu.make_shader_module(&gpu::ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(SHADOW_EMIT_SOURCE),
        })?;

        Ok(Self {
            buffer_allocator: ImageAllocator::new(NEVER_DEALLOCATE),
            sampler_allocator: SamplerAllocator::new(NEVER_DEALLOCATE),
            num_framebuffers,
            csm_buffers,
            frame_buffers,
            cur_framebuffer: 0,
            viewport_size: initial_viewport_size,
            shadow_atlas,
            shadow_atlas_view,
            shadow_atlas_sampler,
            linear_sampler,
            csm_split_count: 4,
            screen_quad_vs,
            shadow_emit_handle,
        })
    }
}

impl ShadowRenderer for CsmRenderer {
    fn render_shadows(
        &mut self,
        gpu: &dyn Gpu,
        gbuffer: &Gbuffer,
        camera: &crate::Camera,
        scene: &crate::RenderScene,
        command_buffer: &mut gpu::CommandBuffer,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<()> {
        let mut texture_packer = TiledTexture2DPacker::new(
            SHADOW_ATLAS_TILE_SIZE,
            SHADOW_ATLAS_WIDTH,
            SHADOW_ATLAS_HEIGHT,
        )
        .expect("Failed to create packer");

        let mut shadow_casters = vec![];
        let mut light_cameras = vec![];
        let mut povs = vec![];

        let mut pov_idx = 0;
        scene
            .lights
            .iter()
            .filter(|l| l.enabled)
            .filter(|l| l.shadow_setup.is_some())
            .filter_map(|l| {
                let ty = match l.ty {
                    crate::LightType::Point => 1,
                    crate::LightType::Directional { .. } => 2,
                    crate::LightType::Spotlight { .. } => 3,
                    crate::LightType::Rect { .. } => 4,
                };

                let num_maps = match l.ty {
                    crate::LightType::Point => 6,
                    crate::LightType::Directional { .. } => 1,
                    crate::LightType::Spotlight { .. } => 1,
                    crate::LightType::Rect { .. } => 1,
                };

                let shadow_map_width =
                    SHADOW_ATLAS_TILE_SIZE * l.shadow_setup.unwrap().importance.get();
                let shadow_map_height = shadow_map_width;
                if let Ok(allocation) =
                    texture_packer.allocate(shadow_map_width * num_maps, shadow_map_height)
                {
                    Some((
                        l,
                        ty,
                        num_maps,
                        shadow_map_width,
                        shadow_map_height,
                        allocation,
                    ))
                } else {
                    None
                }
            })
            .for_each(
                |(light, ty, num_maps, shadow_map_width, shadow_map_height, allocation)| {
                    let shadow_caster = ShadowCaster {
                        offset_size: [
                            allocation.x,
                            allocation.y,
                            shadow_map_width,
                            shadow_map_height,
                        ],
                        type_num_maps_pov_splitidx: [ty, num_maps, pov_idx, 0],
                    };
                    let light_povs = light.point_of_views();
                    pov_idx += light_povs.len() as u32;
                    light_cameras.extend(light_povs.clone());
                    povs.extend(light_povs.into_iter().map(|p| PointOfViewData {
                        eye: point![p.location.x, p.location.y, p.location.z, 0.0],
                        eye_forward: p.forward.to_homogeneous(),
                        view: p.view(),
                        projection: p.projection(),
                    }));
                    shadow_casters.push(shadow_caster);
                },
            );

        let current_buffers = &self.csm_buffers[self.cur_framebuffer];
        let frame_buffers = &self.frame_buffers[self.cur_framebuffer];

        gpu.write_buffer(&frame_buffers.camera_buffer, 0, bytemuck::cast_slice(&povs))?;
        gpu.write_buffer(
            &current_buffers.shadow_casters,
            0,
            bytemuck::cast_slice(&[shadow_casters.len() as u32]),
        )?;
        gpu.write_buffer(
            &current_buffers.shadow_casters,
            std::mem::size_of::<u32>() as u64 * 4u64,
            bytemuck::cast_slice(&shadow_casters),
        )?;

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

            for shadow_caster in &shadow_casters {
                let num_povs = shadow_caster.type_num_maps_pov_splitidx[1];
                let first_pov = shadow_caster.type_num_maps_pov_splitidx[2];
                for n in 0..num_povs {
                    let pov = &light_cameras[first_pov as usize + n as usize];
                    let frustum = pov.frustum();
                    let primitives = scene.intersect_frustum(&frustum);
                    let pov_idx = first_pov + n;

                    shadow_atlas_pass.set_cull_mode(gpu::CullMode::Front);
                    shadow_atlas_pass.set_depth_compare_op(gpu::CompareOp::LessEqual);

                    shadow_atlas_pass.set_color_output_enabled(false);
                    shadow_atlas_pass.set_enable_depth_test(true);
                    shadow_atlas_pass.set_depth_write_enabled(true);
                    let width = shadow_caster.offset_size[2] as f32;
                    shadow_atlas_pass.set_viewport(Viewport {
                        x: shadow_caster.offset_size[0] as f32 + width * n as f32,
                        y: shadow_caster.offset_size[1] as f32,
                        width,
                        height: shadow_caster.offset_size[3] as f32,
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
        }

        {
            let mut shadow_pass = command_buffer.start_render_pass(&gpu::BeginRenderPassInfo {
                label: Some("Cascaded Shadow Map rendering"),
                color_attachments: &[FramebufferColorAttachment {
                    image_view: gbuffer.shadow_buffer.view.clone(),
                    load_op: gpu::ColorLoadOp::Clear([1.0; 4]),
                    store_op: gpu::AttachmentStoreOp::Store,
                    initial_layout: gpu::ImageLayout::Undefined,
                    final_layout: gpu::ImageLayout::ShaderReadOnly,
                }],
                depth_attachment: None,
                stencil_attachment: None,
                render_area: Rect2D {
                    offset: Default::default(),
                    extent: gbuffer.viewport_size,
                },
                subpasses: &[SubpassDescription {
                    label: Some("Shadow Buffer Emit".to_owned()),
                    input_attachments: vec![],
                    color_attachments: vec![AttachmentReference {
                        attachment: 0,
                        layout: gpu::ImageLayout::ColorAttachment,
                    }],
                    resolve_attachments: vec![],
                    depth_stencil_attachment: None,
                    preserve_attachments: vec![],
                }],
                dependencies: &[SubpassDependency {
                    src_subpass: SubpassDependency::EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                }],
            });
            shadow_pass.bind_resources(
                0,
                &[Binding {
                    ty: DescriptorBindingType::StorageBuffer {
                        handle: frame_buffers.camera_buffer.clone(),
                        offset: 0,
                        range: gpu::WHOLE_SIZE as _,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 0,
                }],
            );

            shadow_pass.bind_resources(
                1,
                &[
                    Binding {
                        ty: DescriptorBindingType::ImageView {
                            image_view_handle: gbuffer.position_component.view.clone(),
                            sampler_handle: self.linear_sampler.clone(),
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 0,
                    },
                    Binding {
                        ty: DescriptorBindingType::ImageView {
                            image_view_handle: self.shadow_atlas_view.clone(),
                            sampler_handle: self.shadow_atlas_sampler.clone(),
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 1,
                    },
                    Binding {
                        ty: DescriptorBindingType::StorageBuffer {
                            handle: current_buffers.shadow_casters.clone(),
                            offset: 0,
                            range: gpu::WHOLE_SIZE as _,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 2,
                    },
                    Binding {
                        ty: DescriptorBindingType::StorageBuffer {
                            handle: current_buffers.csm_splits.clone(),
                            offset: 0,
                            range: gpu::WHOLE_SIZE as _,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 3,
                    },
                ],
            );

            shadow_pass.set_cull_mode(gpu::CullMode::None);
            shadow_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
            shadow_pass.set_front_face(gpu::FrontFace::ClockWise);
            shadow_pass.set_enable_depth_test(false);
            shadow_pass.set_depth_write_enabled(false);
            shadow_pass.set_vertex_shader(self.screen_quad_vs.clone());
            shadow_pass.set_fragment_shader(self.shadow_emit_handle.clone());
            shadow_pass
                .draw(4, 1, 0, 0)
                .context("Shadow pass subpass")?
        }

        self.cur_framebuffer = (self.cur_framebuffer + 1) % self.num_framebuffers;
        Ok(())
    }
}
