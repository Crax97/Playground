use bytemuck::{Pod, Zeroable};
use gpu::{
    BufferCreateInfo, BufferHandle, BufferUsageFlags, ComponentMapping, DepthAttachment, Extent2D,
    Gpu, ImageAspectFlags, ImageCreateInfo, ImageHandle, ImageSubresourceRange, ImageUsageFlags,
    ImageViewCreateInfo, ImageViewHandle, MemoryDomain, Rect2D, Viewport,
};
use nalgebra::{point, vector, Matrix4, Point3};

use crate::{
    components::Transform, AssetMap, Camera, DeferredRenderingPipeline, FrameBuffers, Frustum,
    LightType, PipelineTarget, PointOfViewData, SceneLightInfo, TiledTexture2DPacker,
    TiledTexture2DSection,
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
    pub csm_split: i32,
}

pub struct CascadedShadowMap {
    pub num_cascades: u8,
    pub csm_split_lambda: f32,
    pub z_mult: f32,
    pub debug_csm_splits: bool,
    pub is_pcf_enabled: bool,
    pub stabilize_cascades: bool,

    pub(crate) csm_buffers: Vec<CsmBuffers>,
    #[allow(dead_code)]
    pub(crate) shadow_atlas: ImageHandle,
    pub(crate) shadow_atlas_view: ImageViewHandle,
    pub(crate) shadow_maps: Vec<ShadowMap>,
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
            label: Some("Shadow atlas view"),
            image: shadow_atlas,
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
                    size: Self::MAX_SHADOW_CASTERS * std::mem::size_of::<f32>()
                        + std::mem::size_of::<[u32; 4]>(),
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
            csm_split_lambda: 0.05,
            z_mult: 1.35,
            debug_csm_splits: false,
            is_pcf_enabled: true,
            stabilize_cascades: true,

            csm_buffers,
            shadow_atlas,
            shadow_atlas_view,
            shadow_maps: vec![],
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
        scene: &crate::GameScene,
        command_buffer: &mut gpu::CommandBuffer,
        frame_buffers: &FrameBuffers,
        resource_map: &AssetMap,

        light_povs: &Vec<PointOfViewData>,
        sampler_allocator: &SamplerAllocator,
    ) -> anyhow::Result<()> {
        {
            let mut shadow_atlas_pass =
                command_buffer.start_render_pass_2(&gpu::BeginRenderPassInfo2 {
                    label: Some("CSM Shadow Atlas Emit"),
                    color_attachments: &[],
                    depth_attachment: Some(DepthAttachment {
                        image_view: self.shadow_atlas_view,
                        load_op: gpu::DepthLoadOp::Clear(1.0),
                        store_op: gpu::AttachmentStoreOp::Store,
                    }),
                    stencil_attachment: None,
                    render_area: Rect2D {
                        offset: Default::default(),
                        extent: Extent2D {
                            width: SHADOW_ATLAS_WIDTH,
                            height: SHADOW_ATLAS_HEIGHT,
                        },
                    },
                });

            let lights = scene.all_enabled_lights().collect::<Vec<_>>();

            for shadow_map in &self.shadow_maps {
                let caster_pov = shadow_map.type_num_maps_pov_lightid[2];
                let pov_idx = shadow_map.type_num_maps_pov_lightid[2];
                let light = lights[shadow_map.type_num_maps_pov_lightid[3] as usize].1;
                let light = light.ty.as_light();
                // We subtract 1 because index 0 is reserved for the camera
                let pov = &light_povs[caster_pov as usize - 1];
                let frustum = if let LightType::Directional { .. } = light.ty {
                    Frustum::from_view(&pov.view)
                } else {
                    Frustum::from_view_proj(&pov.view, &pov.projection)
                };
                let primitives = scene.intersect_frustum(&frustum);
                let setup = light
                    .shadow_configuration
                    .expect("Bug: a light is set to render a shadow map, but has no shadow setup");

                shadow_atlas_pass.set_depth_bias(setup.depth_bias, setup.depth_slope);
                shadow_atlas_pass.set_enable_depth_clamp(true);
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
                    sampler_allocator,
                )?;
            }
        }

        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.shadow_maps.clear();
        self.csm_splits.clear();
        self.texture_packer = TiledTexture2DPacker::new(
            SHADOW_ATLAS_TILE_SIZE,
            SHADOW_ATLAS_WIDTH,
            SHADOW_ATLAS_HEIGHT,
        )
        .expect("Failed to recreate texture packer");
    }

    // Returns the povs added for each shadow map added
    pub(crate) fn add_light(
        &mut self,
        light: &SceneLightInfo,
        transform: &Transform,
        scene_camera: &crate::Camera,
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

        if let Some(shadow_map_configuration) = &light.shadow_configuration {
            let shadow_map_width = shadow_map_configuration.shadow_map_width;
            let shadow_map_height = shadow_map_configuration.shadow_map_height;

            if let Ok(allocation) = self
                .texture_packer
                .allocate(shadow_map_width * num_maps, shadow_map_height)
            {
                let this_light_cameras = light.light_cameras(&transform);
                let (povs, splits) = self.get_light_povs_and_splits(
                    light,
                    transform,
                    scene_camera,
                    this_light_cameras,
                    &allocation,
                );
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
                self.shadow_maps.extend(light_shadow_maps);
                let idx = self.csm_splits.len();
                let csm_split = if let Some(splits) = splits {
                    self.csm_splits.extend(splits);
                    idx as i32
                } else {
                    -1
                };

                Some(NewShadowMapAllocation {
                    povs,
                    shadow_map_index,
                    csm_split,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_light_povs_and_splits(
        &self,
        l: &SceneLightInfo,
        transform: &Transform,
        scene_camera: &crate::Camera,
        light_povs: Vec<Camera>,
        allocation: &TiledTexture2DSection,
    ) -> (Vec<PointOfViewData>, Option<Vec<f32>>) {
        let position = transform.position;
        match l.ty {
            LightType::Directional { .. } => {
                let mut cascade_splits = Vec::with_capacity(MAX_CASCADES);
                let mut splits = Vec::with_capacity(MAX_CASCADES);

                let near_clip = scene_camera.near;
                let far_clip = scene_camera.far;
                let clip_range = far_clip - near_clip;

                let min_z = near_clip;
                let max_z = near_clip + clip_range;
                let range = max_z - min_z;
                let ratio = max_z / min_z;

                let mut povs = vec![];

                for i in 0..MAX_CASCADES {
                    let p = (i + 1) as f32 / MAX_CASCADES as f32;
                    let log = min_z + ratio.powf(p);
                    let linear = min_z + range * p;
                    let d = self.csm_split_lambda * (log - linear) + linear;
                    cascade_splits.push((d - near_clip) / range);
                }

                let mut last_split_dist = 0.0;
                (0..MAX_CASCADES).for_each(|i| {
                    let split_dist = cascade_splits[i];

                    let mut frustum_corners = [
                        vector![-1.0, 1.0, -1.0],
                        vector![1.0, 1.0, -1.0],
                        vector![1.0, -1.0, -1.0],
                        vector![-1.0, -1.0, -1.0],
                        vector![-1.0, 1.0, 1.0],
                        vector![1.0, 1.0, 1.0],
                        vector![1.0, -1.0, 1.0],
                        vector![-1.0, -1.0, 1.0],
                    ];

                    let inv_camera = scene_camera.projection() * scene_camera.view();
                    let inv_camera = inv_camera
                        .try_inverse()
                        .expect("Failed to invert scene matrix");
                    (0..8).for_each(|i| {
                        let corner = frustum_corners[i];
                        let corner = inv_camera * vector![corner.x, corner.y, corner.z, 1.0];
                        frustum_corners[i] = corner.xyz() / corner.w;
                    });

                    for i in 0..4 {
                        let dist = frustum_corners[i + 4] - frustum_corners[i];
                        frustum_corners[i + 4] = frustum_corners[i] + (dist * split_dist);
                        frustum_corners[i] += dist * last_split_dist;
                    }

                    let mut center = vector![0.0, 0.0, 0.0];
                    (0..8).for_each(|i| {
                        center += frustum_corners[i];
                    });
                    center /= 8.0;

                    let mut radius: f32 = 0.0;
                    (0..8).for_each(|i| {
                        let dist = (frustum_corners[i] - center).magnitude();
                        radius = radius.max(dist);
                    });
                    let radius = (radius * 16.0).ceil() / 16.0;
                    let radius = radius * self.z_mult;

                    let max_extents = vector![radius, radius, radius];
                    let min_extents = -max_extents;

                    let center = Point3::from(center);
                    let light_dir = transform.forward();
                    let view_matrix = Matrix4::look_at_rh(
                        &(center - light_dir * -min_extents.z * self.z_mult),
                        &center,
                        &vector![0.0, 1.0, 0.0],
                    );
                    let mut light_ortho = Matrix4::new_orthographic(
                        min_extents.x,
                        max_extents.x,
                        min_extents.y,
                        max_extents.y,
                        0.0,
                        max_extents.z - min_extents.z,
                    );

                    if self.stabilize_cascades {
                        // Compute texel snapping factor
                        // https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
                        let shadow_matrix = light_ortho * view_matrix;
                        let shadow_origin = shadow_matrix * vector![0.0, 0.0, 0.0, 1.0];
                        let shadow_origin = shadow_origin * (allocation.height as f32) / 2.0;
                        let rounded_origin = shadow_origin.map(|v| v.round());
                        let round_offset = rounded_origin - shadow_origin;
                        let mut round_offset = round_offset * 2.0 / (allocation.height as f32);
                        round_offset.z = 0.0;
                        round_offset.w = 0.0;
                        light_ortho.set_column(3, &(light_ortho.column(3) + round_offset));
                    }

                    povs.push(PointOfViewData {
                        eye: point![position.x, position.y, position.z, 0.0],
                        eye_forward: light_dir.to_homogeneous(),
                        view: view_matrix,
                        projection: light_ortho,
                    });

                    splits.push((near_clip + split_dist * clip_range) * -1.0);

                    last_split_dist = cascade_splits[i];
                });

                (povs, Some(splits))
            }
            _ => (
                light_povs
                    .into_iter()
                    .map(|c| PointOfViewData {
                        eye: point![c.location.x, c.location.y, c.location.z, 0.0],
                        eye_forward: c.forward.to_homogeneous(),
                        view: c.view(),
                        projection: c.projection(),
                    })
                    .collect(),
                None,
            ),
        }
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
            bytemuck::cast_slice(&[
                self.num_cascades as u32,
                if self.debug_csm_splits { 1 } else { 0 },
                if self.is_pcf_enabled { 1 } else { 0 },
            ]),
        )?;

        gpu.write_buffer(
            &current_buffers.csm_splits,
            std::mem::size_of::<[u32; 4]>() as u64,
            bytemuck::cast_slice(&self.csm_splits),
        )?;
        Ok(())
    }

    pub(crate) fn destroy(&self, gpu: &dyn Gpu) {
        gpu.destroy_image_view(self.shadow_atlas_view);
        gpu.destroy_image(self.shadow_atlas);

        for buffers in &self.csm_buffers {
            gpu.destroy_buffer(buffers.csm_splits);
            gpu.destroy_buffer(buffers.shadow_casters);
        }
    }
}
