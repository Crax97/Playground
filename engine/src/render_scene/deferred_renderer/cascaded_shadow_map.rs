use bytemuck::{Pod, Zeroable};
use gpu::{
    AccessFlags, AttachmentReference, BufferCreateInfo, BufferHandle, BufferUsageFlags,
    ComponentMapping, Extent2D, FramebufferDepthAttachment, Gpu, ImageAspectFlags, ImageCreateInfo,
    ImageHandle, ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageViewCreateInfo,
    ImageViewHandle, MemoryDomain, PipelineStageFlags, Rect2D, SubpassDependency,
    SubpassDescription, Viewport,
};
use nalgebra::{point, vector, Matrix4, Point3, Vector3};

use crate::{
    math::shape::BoundingShape, Camera, DeferredRenderingPipeline, FrameBuffers, Frustum, Light,
    LightType, PipelineTarget, PointOfViewData, ResourceMap, TiledTexture2DPacker,
};

use super::SamplerAllocator;

pub const SHADOW_ATLAS_TILE_SIZE: u32 = 128;
pub const SHADOW_ATLAS_WIDTH: u32 = SHADOW_ATLAS_TILE_SIZE * 70;
pub const SHADOW_ATLAS_HEIGHT: u32 = SHADOW_ATLAS_TILE_SIZE * 35;

pub struct CsmBuffers {
    pub shadow_casters: BufferHandle,
    pub csm_splits: BufferHandle,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
pub struct ShadowMap {
    pub offset_size: [u32; 4],
    // type: x, num_maps: y, pov: z, split_idx: w (directional only)
    pub type_num_maps_pov_lightid: [u32; 4],
}

pub struct NewShadowMapAllocation {
    pub povs: Vec<PointOfViewData>,
    pub shadow_map_index: usize,
}

pub struct CascadedShadowMap {
    pub num_cascades: u8,
    pub csm_split_lambda: f32,
    pub z_mult: f32,

    pub(crate) csm_buffers: Vec<CsmBuffers>,
    #[allow(dead_code)]
    pub(crate) shadow_atlas: ImageHandle,
    pub(crate) shadow_atlas_view: ImageViewHandle,
    pub(crate) shadow_maps: Vec<ShadowMap>,
    pub(crate) camera_splits: Vec<Camera>,
    pub(crate) csm_splits: Vec<f32>,

    texture_packer: TiledTexture2DPacker,
}

pub const MAX_CASCADES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CSMConstantData {
    count: u32,
    splits: [f32; MAX_CASCADES],
}

impl CascadedShadowMap {
    pub const MAX_SHADOW_CASTERS: usize = 100;

    pub fn new(gpu: &dyn Gpu, max_frames_in_flight: usize) -> anyhow::Result<Self> {
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

        let mut csm_buffers = Vec::with_capacity(max_frames_in_flight);
        for i in 0..max_frames_in_flight {
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
            num_cascades: 4,
            csm_split_lambda: 0.89,
            z_mult: 10.0,

            csm_buffers,
            shadow_atlas,
            shadow_atlas_view,
            shadow_maps: vec![],
            camera_splits: vec![],
            csm_splits: vec![],
            texture_packer: TiledTexture2DPacker::new(
                SHADOW_ATLAS_TILE_SIZE,
                SHADOW_ATLAS_WIDTH,
                SHADOW_ATLAS_HEIGHT,
            )
            .expect("Failed to create texture packer"),
        })
    }
    pub(crate) fn render_shadow_atlas(
        &self,
        gpu: &dyn Gpu,
        scene: &crate::RenderScene,
        command_buffer: &mut gpu::CommandBuffer,
        frame_buffers: &FrameBuffers,
        resource_map: &ResourceMap,

        light_povs: &Vec<PointOfViewData>,
        sampler_allocator: &SamplerAllocator,
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
                let pov = &light_povs[caster_pov as usize - 1];
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
                    &sampler_allocator,
                )?;
            }
        }

        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.shadow_maps.clear();
        self.texture_packer = TiledTexture2DPacker::new(
            SHADOW_ATLAS_TILE_SIZE,
            SHADOW_ATLAS_WIDTH,
            SHADOW_ATLAS_HEIGHT,
        )
        .expect("Failed to recreate texture packer");
    }

    pub(crate) fn update_cascade_splits(&mut self, scene_camera: &crate::Camera) {
        self.camera_splits =
            scene_camera.split_into_slices(self.num_cascades, self.csm_split_lambda);

        self.csm_splits = self.camera_splits.iter().map(|c| c.far).collect();
    }

    // Returns the povs added for each shadow map added
    pub(crate) fn add_light(
        &mut self,
        light: &Light,
        pov_idx: u32,
        light_id: u32,
    ) -> Option<NewShadowMapAllocation> {
        let ty = match light.ty {
            crate::LightType::Point => 1,
            crate::LightType::Directional { .. } => 2,
            crate::LightType::Spotlight { .. } => 3,
            crate::LightType::Rect { .. } => 4,
        };

        let num_maps = match light.ty {
            crate::LightType::Point => 6,
            crate::LightType::Directional { .. } => self.num_cascades as u32,
            crate::LightType::Spotlight { .. } => 1,
            crate::LightType::Rect { .. } => 1,
        };

        let shadow_map_width =
            SHADOW_ATLAS_TILE_SIZE * light.shadow_configuration.unwrap().importance.get();
        let shadow_map_height = shadow_map_width;

        if let Ok(allocation) = self
            .texture_packer
            .allocate(shadow_map_width * num_maps, shadow_map_height)
        {
            let this_light_cameras = light.light_cameras();
            let povs = self.get_light_povs(light, this_light_cameras);
            let shadow_map_index = self.shadow_maps.len();
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
                    type_num_maps_pov_lightid: [ty, num_maps, pov_idx + i as u32, light_id],
                })
                .collect::<Vec<_>>();
            self.shadow_maps.extend(light_shadow_maps.into_iter());

            Some(NewShadowMapAllocation {
                povs,
                shadow_map_index,
            })
        } else {
            None
        }
    }

    fn get_light_povs(&self, l: &Light, light_povs: Vec<Camera>) -> Vec<PointOfViewData> {
        match l.ty {
            LightType::Directional { direction, .. } => {
                let povs = self.camera_splits.iter().map(|split| {
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
                    let (frustum_min, frustum_max) = frustum_bounds.box_extremes();

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

    pub(crate) fn get_csm_constant_data(&self) -> CSMConstantData {
        let mut data = CSMConstantData {
            count: self.csm_splits.len() as _,
            splits: [0.0; MAX_CASCADES],
        };

        for (i, split) in self.csm_splits.iter().take(MAX_CASCADES).enumerate() {
            data.splits[i] = *split;
        }
        data
    }

    pub(crate) fn update_buffers(&self, gpu: &dyn Gpu, buffer_index: usize) -> anyhow::Result<()> {
        let current_buffers = &self.csm_buffers[buffer_index];
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
}
