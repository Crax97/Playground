use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use engine_macros::glsl;
use std::{collections::HashMap, mem::size_of};

use crate::{
    material::{MasterMaterial, MasterMaterialDescription},
    math::shape::BoundingShape,
    post_process_pass::{PostProcessPass, PostProcessResources},
    Camera, CvarManager, Frustum, Light, LightType, MaterialDescription, MaterialInstance, Mesh,
    MeshPrimitive, PipelineTarget, RenderScene, RenderingPipeline, ScenePrimitive, Texture,
    TextureSamplerSettings,
};

use crate::resource_map::{ResourceHandle, ResourceMap};
use gpu::{
    AccessFlags, AttachmentReference, AttachmentStoreOp, BeginRenderPassInfo, Binding,
    BufferCreateInfo, BufferHandle, BufferUsageFlags, ColorLoadOp, CommandBuffer,
    CommandBufferSubmitInfo, Extent2D, FragmentStageInfo, FramebufferColorAttachment,
    FramebufferDepthAttachment, Gpu, ImageAspectFlags, ImageFormat, ImageHandle, ImageLayout,
    ImageMemoryBarrier, ImageSubresourceRange, ImageUsageFlags, ImageViewHandle, ImageViewType,
    IndexType, InputRate, LifetimedCache, MemoryDomain, Offset2D, PipelineBarrierInfo,
    PipelineStageFlags, Rect2D, RenderPass, SampleCount, SamplerHandle, ShaderModuleCreateInfo,
    ShaderModuleHandle, ShaderStage, SubpassDependency, SubpassDescription, VertexBindingInfo,
    VertexStageInfo, VkSwapchain,
};
use nalgebra::{point, vector, Matrix, Matrix4, Point3, Point4, Vector2, Vector3, Vector4};

use super::{CsmRenderer, ShadowRenderer};

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

const TEXTURE_MUL: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/texture_mul.frag",
    entry_point = "main"
);

pub struct DeferredRenderingPipeline {
    frame_buffers: Vec<FrameBuffers>,
    image_allocator: ImageAllocator,
    sampler_allocator: SamplerAllocator,
    screen_quad: ShaderModuleHandle,
    texture_copy: ShaderModuleHandle,
    combine_shader: ShaderModuleHandle,
    shadow_renderer: Box<dyn ShadowRenderer>,

    post_process_stack: Vec<Box<dyn PostProcessPass>>,

    in_flight_frame: usize,
    max_frames_in_flight: usize,
    active_lights: Vec<GpuLightInfo>,
    light_povs: Vec<PointOfViewData>,
    pub(crate) irradiance_map: Option<ResourceHandle<Texture>>,
    default_irradiance_map: ResourceHandle<Texture>,

    cube_mesh: ResourceHandle<Mesh>,

    pub depth_bias_constant: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope: f32,

    pub csm_slices: u8,

    pub ambient_color: Vector3<f32>,
    pub ambient_intensity: f32,

    pub update_frustum: bool,
    pub drawcalls_last_frame: u64,

    frustum: Frustum,
    gbuffer_nearest_sampler: SamplerHandle,
    screen_quad_flipped: ShaderModuleHandle,
    early_z_pass_enabled: bool,
    view_size: Extent2D,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RenderImageDescription {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub samples: SampleCount,
    pub view_type: ImageViewType,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct ImageId {
    pub label: &'static str,
    pub desc: RenderImageDescription,
}

#[derive(Clone)]
pub struct RenderImage {
    pub image: ImageHandle,
    pub view: ImageViewHandle,
}

pub struct Gbuffer {
    pub depth_component: RenderImage,
    pub position_component: RenderImage,
    pub normal_component: RenderImage,
    pub diffuse_component: RenderImage,
    pub emissive_component: RenderImage,
    pub pbr_component: RenderImage,
    pub shadow_buffer: RenderImage,

    pub viewport_size: Extent2D,
}

pub struct SamplerAllocator {
    sampler_allocator: LifetimedCache<SamplerHandle>,
}

impl SamplerAllocator {
    pub fn get(&self, gpu: &dyn Gpu, desc: &TextureSamplerSettings) -> SamplerHandle {
        self.sampler_allocator.get_clone(desc, || {
            gpu.make_sampler(&gpu::SamplerCreateInfo {
                mag_filter: desc.mag_filter,
                min_filter: desc.min_filter,
                address_u: desc.address_u,
                address_v: desc.address_v,
                address_w: desc.address_w,
                // TODO: Have a global lod bias
                mip_lod_bias: 0.0,
                compare_function: None,
                min_lod: 0.0,
                max_lod: 1.0,
                border_color: [0.0; 4],
            })
            .expect("failed to create sampler")
        })
    }

    pub fn new(lifetime: u32) -> Self {
        Self {
            sampler_allocator: LifetimedCache::new(lifetime),
        }
    }
}
pub struct ImageAllocator {
    image_allocator: LifetimedCache<RenderImage>,
}

impl ImageAllocator {
    pub fn get(
        &self,
        gpu: &dyn Gpu,
        label: &'static str,
        desc: &RenderImageDescription,
    ) -> RenderImage {
        self.image_allocator
            .get_clone(&ImageId { label, desc: *desc }, || {
                let image = gpu
                    .make_image(
                        &gpu::ImageCreateInfo {
                            label: Some(label),
                            width: desc.width,
                            height: desc.height,
                            depth: 1,
                            mips: 1,
                            layers: 1,
                            samples: desc.samples,
                            format: desc.format,
                            usage: ImageUsageFlags::SAMPLED
                                | ImageUsageFlags::INPUT_ATTACHMENT
                                | if desc.format.is_color() {
                                    ImageUsageFlags::COLOR_ATTACHMENT
                                } else {
                                    ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                                },
                        },
                        MemoryDomain::DeviceLocal,
                        None,
                    )
                    .expect("failed to create image");

                let view = gpu
                    .make_image_view(&gpu::ImageViewCreateInfo {
                        image: image.clone(),
                        view_type: gpu::ImageViewType::Type2D,
                        format: desc.format,
                        components: gpu::ComponentMapping::default(),
                        subresource_range: gpu::ImageSubresourceRange {
                            aspect_mask: if desc.format.is_color() {
                                ImageAspectFlags::COLOR
                            } else {
                                ImageAspectFlags::DEPTH
                            },
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    })
                    .expect("failed to create view");
                RenderImage { image, view }
            })
    }

    pub fn new(lifetime: u32) -> Self {
        Self {
            image_allocator: LifetimedCache::new(lifetime),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ObjectDrawInfo {
    pub model: Matrix4<f32>,
    pub camera_index: u32,
}

unsafe impl Pod for ObjectDrawInfo {}
unsafe impl Zeroable for ObjectDrawInfo {}

#[derive(Clone, Copy)]
pub struct FxaaSettings {
    pub fxaa_quality_subpix: f32,
    pub fxaa_quality_edge_threshold: f32,
    pub fxaa_quality_edge_threshold_min: f32,
}

unsafe impl Pod for FxaaSettings {}
unsafe impl Zeroable for FxaaSettings {}

impl Default for FxaaSettings {
    fn default() -> Self {
        Self {
            fxaa_quality_subpix: 0.75,
            fxaa_quality_edge_threshold: 0.166,
            fxaa_quality_edge_threshold_min: 0.0833,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PointOfViewData {
    pub eye: Point4<f32>,
    pub eye_forward: Vector4<f32>,
    pub view: nalgebra::Matrix4<f32>,
    pub projection: nalgebra::Matrix4<f32>,
}

unsafe impl Pod for PointOfViewData {}
unsafe impl Zeroable for PointOfViewData {}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuLightInfo {
    position_radius: Vector4<f32>,
    direction: Vector4<f32>,
    color: Vector4<f32>,
    extras: Vector4<f32>,
    ty: [i32; 4],
}

unsafe impl Pod for GpuLightInfo {}
unsafe impl Zeroable for GpuLightInfo {}

impl From<&Light> for GpuLightInfo {
    fn from(light: &Light) -> Self {
        let (direction, extras, ty) = match light.ty {
            LightType::Point => (Default::default(), Default::default(), 0),
            LightType::Directional { direction, .. } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                Default::default(),
                1,
            ),
            LightType::Spotlight {
                direction,
                inner_cone_degrees,
                outer_cone_degrees,
            } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                vector![
                    inner_cone_degrees.to_radians().cos(),
                    outer_cone_degrees.to_radians().cos(),
                    0.0,
                    0.0
                ],
                2,
            ),
            LightType::Rect {
                direction,
                width,
                height,
            } => (
                vector![direction.x, direction.y, direction.z, 0.0],
                vector![width, height, 0.0, 0.0],
                3,
            ),
        };
        Self {
            position_radius: vector![
                light.position.x,
                light.position.y,
                light.position.z,
                light.radius
            ],
            color: vector![light.color.x, light.color.y, light.color.z, light.intensity],
            direction,
            extras,
            ty: [ty, -1, 0, 0],
        }
    }
}

pub(crate) struct FrameBuffers {
    pub(crate) camera_buffer: BufferHandle,
    pub(crate) light_buffer: BufferHandle,
}

struct DrawCall<'a> {
    prim: &'a MeshPrimitive,
    transform: Matrix4<f32>,
    material: MaterialInstance,
}

impl DeferredRenderingPipeline {
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

        let texture_mul = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(TEXTURE_MUL),
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
        Ok(Self {
            image_allocator: ImageAllocator::new(4),
            sampler_allocator: SamplerAllocator::new(4),
            screen_quad: screen_quad.clone(),
            screen_quad_flipped,
            combine_shader,
            texture_copy,
            frame_buffers,
            post_process_stack: vec![],
            in_flight_frame: 0,
            max_frames_in_flight: VkSwapchain::MAX_FRAMES_IN_FLIGHT,
            depth_bias_constant: 2.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope: 4.0,
            ambient_color: vector![1.0, 1.0, 1.0],
            ambient_intensity: 0.3,
            active_lights: vec![],
            light_povs: vec![],
            cube_mesh,
            irradiance_map: None,
            default_irradiance_map,
            gbuffer_nearest_sampler: gbuffer_nearest_sampler.clone(),
            early_z_pass_enabled: true,
            view_size,

            update_frustum: true,
            frustum: Frustum::default(),
            drawcalls_last_frame: 0,
            csm_slices: 4,
            shadow_renderer: Box::new(CsmRenderer::new(
                gpu,
                VkSwapchain::MAX_FRAMES_IN_FLIGHT,
                gbuffer_nearest_sampler,
                screen_quad,
            )?),
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

                    render_pass.set_cull_mode(master.cull_mode);
                    render_pass.set_front_face(master.front_face);
                    draw_mesh_primitive(
                        gpu,
                        render_pass,
                        material,
                        master,
                        mesh_prim,
                        primitive.transform,
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

    fn generate_draw_calls<'r, 's>(
        frustum: &Frustum,
        resource_map: &'r ResourceMap,
        scene: &'s RenderScene,
    ) -> (HashMap<&'s MasterMaterial, Vec<DrawCall<'s>>>, u64)
    where
        'r: 's,
    {
        let mut drawcalls = 0;
        let mut draw_hashmap: HashMap<&MasterMaterial, Vec<DrawCall>> = HashMap::new();

        let frustum_intersections = scene.intersect_frustum(frustum);

        for primitive in frustum_intersections {
            let mesh = resource_map.get(&primitive.mesh);
            for (idx, mesh_prim) in mesh.primitives.iter().enumerate() {
                let material = primitive.materials[idx].clone();
                let master = resource_map.get(&material.owner);
                draw_hashmap.entry(master).or_default().push(DrawCall {
                    prim: mesh_prim,
                    transform: primitive.transform,
                    material,
                });
                drawcalls += 1;
            }
        }
        (draw_hashmap, drawcalls)
    }

    fn update_lights(&mut self, scene: &RenderScene) {
        self.active_lights.clear();
        self.light_povs.clear();

        let active_lights = scene.all_enabled_lights().collect::<Vec<_>>();

        for active_light in active_lights {
            let gpu_light: GpuLightInfo = active_light.into();
            self.active_lights.push(gpu_light);
        }
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

    // fn shadow_atlas_pass(
    //     &self,
    //     gpu: &dyn Gpu,
    //     graphics_command_buffer: &mut CommandBuffer,
    //     shadow_atlas_component: &RenderImage,
    //     per_frame_data: Vec<PerFrameData>,
    //     resource_map: &ResourceMap,
    //     draw_hashmap: &HashMap<&MasterMaterial, Vec<DrawCall>>,
    //     current_buffers: &FrameBuffers,
    // ) -> anyhow::Result<()> {
    //     {
    //         let mut shadow_atlas_command =
    //             graphics_command_buffer.start_render_pass(&gpu::BeginRenderPassInfo {
    //                 label: Some("Shadow atlas"),
    //                 color_attachments: &[],
    //                 depth_attachment: Some(FramebufferDepthAttachment {
    //                     image_view: shadow_atlas_component.view.clone(),
    //                     load_op: gpu::DepthLoadOp::Clear(1.0),
    //                     store_op: gpu::AttachmentStoreOp::Store,
    //                     initial_layout: ImageLayout::Undefined,
    //                     final_layout: ImageLayout::DepthStencilAttachment,
    //                 }),
    //                 stencil_attachment: None,
    //                 render_area: Rect2D {
    //                     offset: Offset2D::default(),
    //                     extent: Extent2D {
    //                         width: SHADOW_ATLAS_WIDTH,
    //                         height: SHADOW_ATLAS_HEIGHT,
    //                     },
    //                 },
    //                 subpasses: &[SubpassDescription {
    //                     label: None,
    //                     input_attachments: vec![],
    //                     color_attachments: vec![],
    //                     resolve_attachments: vec![],
    //                     depth_stencil_attachment: Some(AttachmentReference {
    //                         attachment: 0,
    //                         layout: ImageLayout::DepthStencilAttachment,
    //                     }),
    //                     preserve_attachments: vec![],
    //                 }],
    //                 dependencies: &[],
    //             });

    //         shadow_atlas_command.set_depth_bias(
    //             self.depth_bias_constant,
    //             self.depth_bias_clamp,
    //             self.depth_bias_slope,
    //         );

    //         shadow_atlas_command.set_cull_mode(gpu::CullMode::Front);
    //         shadow_atlas_command.set_depth_compare_op(gpu::CompareOp::LessEqual);

    //         shadow_atlas_command.set_color_output_enabled(false);
    //         shadow_atlas_command.set_enable_depth_test(true);
    //         shadow_atlas_command.set_depth_write_enabled(true);

    //         for (i, pov) in per_frame_data.iter().enumerate().skip(1) {
    //             shadow_atlas_command.set_viewport(gpu::Viewport {
    //                 x: pov.viewport_size_offset.x,
    //                 y: pov.viewport_size_offset.y,
    //                 width: pov.viewport_size_offset.z,
    //                 height: pov.viewport_size_offset.w,
    //                 min_depth: 0.0,
    //                 max_depth: 1.0,
    //             });

    //             Self::main_render_loop(
    //                 gpu,
    //                 resource_map,
    //                 PipelineTarget::DepthOnly,
    //                 draw_hashmap,
    //                 &mut shadow_atlas_command,
    //                 i as _,
    //                 &current_buffers,
    //                 &self.sampler_allocator,
    //             )?;
    //         }
    //     }

    //     graphics_command_buffer.pipeline_barrier(&gpu::PipelineBarrierInfo {
    //         src_stage_mask: PipelineStageFlags::LATE_FRAGMENT_TESTS,
    //         dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
    //         memory_barriers: &[],
    //         buffer_memory_barriers: &[],
    //         image_memory_barriers: &[ImageMemoryBarrier {
    //             src_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    //             dst_access_mask: AccessFlags::SHADER_READ,
    //             old_layout: ImageLayout::DepthStencilAttachment,
    //             new_layout: ImageLayout::ShaderReadOnly,
    //             src_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
    //             dst_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
    //             image: shadow_atlas_component.image.clone(),
    //             subresource_range: gpu::ImageSubresourceRange {
    //                 aspect_mask: ImageAspectFlags::DEPTH,
    //                 base_mip_level: 0,
    //                 level_count: 1,
    //                 base_array_layer: 0,
    //                 layer_count: 1,
    //             },
    //         }],
    //     });
    //     Ok(())
    // }

    fn gbuffer_output(
        &self,
        gpu: &dyn Gpu,
        graphics_command_buffer: &mut CommandBuffer,
        gbuffer: &Gbuffer,
        resource_map: &ResourceMap,
        render_size: Extent2D,
        current_buffers: &FrameBuffers,
        primitives: &Vec<&ScenePrimitive>,
        pov: &Camera,
        scene: &RenderScene,
    ) -> anyhow::Result<()> {
        let Gbuffer {
            depth_component,
            position_component,
            normal_component,
            diffuse_component,
            emissive_component,
            pbr_component,
            shadow_buffer: _,
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
                        attachment: 5,
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
                        attachment: 5,
                        layout: ImageLayout::DepthStencilReadOnly,
                    }),
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
                    attachment: 5,
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

            let mut gbuffer_render_pass =
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
                gbuffer_render_pass.set_cull_mode(gpu::CullMode::Back);
                gbuffer_render_pass.set_depth_compare_op(gpu::CompareOp::LessEqual);

                gbuffer_render_pass.set_color_output_enabled(false);
                gbuffer_render_pass.set_enable_depth_test(true);
                gbuffer_render_pass.set_depth_write_enabled(true);
                Self::main_render_loop(
                    gpu,
                    &primitives,
                    resource_map,
                    PipelineTarget::DepthOnly,
                    &mut gbuffer_render_pass,
                    0,
                    current_buffers,
                    &self.sampler_allocator,
                )
                .context("Early Z Pass")?;

                gbuffer_render_pass.advance_to_next_subpass();
            }

            if let Some(material) = scene.get_skybox_material() {
                let cube_mesh = resource_map.get(&self.cube_mesh);
                let skybox_master = resource_map.get(&material.owner);
                bind_master_material(
                    skybox_master,
                    PipelineTarget::ColorAndDepth,
                    &mut gbuffer_render_pass,
                    current_buffers,
                );
                Self::draw_skybox(
                    gpu,
                    &pov.location,
                    &mut gbuffer_render_pass,
                    cube_mesh,
                    material,
                    skybox_master,
                    resource_map,
                    &self.sampler_allocator,
                )?;
            }

            gbuffer_render_pass.set_front_face(gpu::FrontFace::CounterClockWise);
            gbuffer_render_pass.set_enable_depth_test(true);
            gbuffer_render_pass.set_depth_write_enabled(!self.early_z_pass_enabled);
            gbuffer_render_pass.set_color_output_enabled(true);
            gbuffer_render_pass.set_cull_mode(gpu::CullMode::Back);
            gbuffer_render_pass.set_depth_compare_op(if self.early_z_pass_enabled {
                gpu::CompareOp::Equal
            } else {
                gpu::CompareOp::LessEqual
            });
            Self::main_render_loop(
                gpu,
                &primitives,
                resource_map,
                PipelineTarget::ColorAndDepth,
                &mut gbuffer_render_pass,
                0,
                current_buffers,
                &self.sampler_allocator,
            )
            .context("Gbuffer output pass")?;
        }

        // graphics_command_buffer.pipeline_barrier(&gpu::PipelineBarrierInfo {
        //     src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        //     dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
        //     memory_barriers: &[],
        //     buffer_memory_barriers: &[],
        //     image_memory_barriers: &[ImageMemoryBarrier {
        //         src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
        //         dst_access_mask: AccessFlags::SHADER_READ,
        //         old_layout: ImageLayout::ColorAttachment,
        //         new_layout: ImageLayout::ShaderReadOnly,
        //         src_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
        //         dst_queue_family_index: gpu::QUEUE_FAMILY_IGNORED,
        //         image: color_output.image.clone(),
        //         subresource_range: gpu::ImageSubresourceRange {
        //             aspect_mask: ImageAspectFlags::COLOR,
        //             base_mip_level: 0,
        //             level_count: 1,
        //             base_array_layer: 0,
        //             layer_count: 1,
        //         },
        //     }],
        // });
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
        self.shadow_renderer.gettext()
    }

    fn get_gbuffer(&self, gpu: &dyn Gpu) -> Gbuffer {
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
        let rgba_desc = RenderImageDescription {
            format: ImageFormat::Rgba8,
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
        let shadow_buffer = self.image_allocator.get(gpu, "shadow_buffer", &rgba_desc);
        Gbuffer {
            depth_component,
            position_component,
            normal_component,
            diffuse_component,
            emissive_component,
            pbr_component,
            shadow_buffer,
            viewport_size: self.view_size,
        }
    }

    fn gbuffer_combine(
        &self,
        gpu: &dyn Gpu,
        graphics_command_buffer: &mut CommandBuffer,
        color_output: &RenderImage,
        gbuffer: &Gbuffer,
        resource_map: &ResourceMap,
        view_size: Extent2D,
    ) -> anyhow::Result<()> {
        let mut render_pass = graphics_command_buffer.start_render_pass(&BeginRenderPassInfo {
            label: Some("GBuffer combine"),
            color_attachments: &[FramebufferColorAttachment {
                image_view: color_output.view.clone(),
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
            }],
            depth_attachment: None,
            stencil_attachment: None,
            render_area: Rect2D {
                extent: self.view_size,
                ..Default::default()
            },
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
            dependencies: &[SubpassDependency {
                src_subpass: SubpassDependency::EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
                src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: AccessFlags::SHADER_READ,
            }],
        });

        let framebuffers = &self.frame_buffers[self.in_flight_frame];
        render_pass.bind_resources(
            0,
            &[
                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: gbuffer.position_component.view.clone(),
                        sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        layout: ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 0,
                },
                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: gbuffer.normal_component.view.clone(),
                        sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        layout: ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 1,
                },
                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: gbuffer.diffuse_component.view.clone(),
                        sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        layout: ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 2,
                },
                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: gbuffer.emissive_component.view.clone(),
                        sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        layout: ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 3,
                },
                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: gbuffer.pbr_component.view.clone(),
                        sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        layout: ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 4,
                },
                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: gbuffer.shadow_buffer.view.clone(),
                        sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        layout: ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 5,
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
                    location: 6,
                },
                Binding {
                    ty: gpu::DescriptorBindingType::StorageBuffer {
                        handle: framebuffers.camera_buffer.clone(),
                        offset: 0,
                        range: gpu::WHOLE_SIZE as usize,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 7,
                },
                Binding {
                    ty: gpu::DescriptorBindingType::StorageBuffer {
                        handle: framebuffers.light_buffer.clone(),
                        offset: 0,
                        range: gpu::WHOLE_SIZE as usize,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 8,
                },
            ],
        );
        render_pass.set_front_face(gpu::FrontFace::ClockWise);
        render_pass.set_cull_mode(gpu::CullMode::None);
        render_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
        render_pass.set_enable_depth_test(false);
        render_pass.set_depth_write_enabled(false);
        render_pass.set_vertex_shader(self.screen_quad.clone());
        render_pass.set_fragment_shader(self.combine_shader.clone());
        render_pass.draw(4, 1, 0, 0)
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

        self.update_lights(scene);

        let projection = camera.projection();

        let current_buffers = &self.frame_buffers[self.in_flight_frame];

        self.in_flight_frame = (1 + self.in_flight_frame) % self.max_frames_in_flight;

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

        let ambient = vector![
            self.ambient_color.x,
            self.ambient_color.y,
            self.ambient_color.z,
            self.ambient_intensity
        ];

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

        let gbuffer = self.get_gbuffer(gpu);
        self.shadow_renderer.render_shadows(
            gpu,
            &gbuffer,
            camera,
            scene,
            &mut graphics_command_buffer,
            resource_map,
        )?;
        {
            let primitives = scene.intersect_frustum(&self.frustum);
            self.drawcalls_last_frame = primitives.len() as u64;
            self.gbuffer_output(
                gpu,
                &mut graphics_command_buffer,
                &gbuffer,
                resource_map,
                self.view_size,
                current_buffers,
                &primitives,
                camera,
                scene,
            )
            .context("Main pass")?;
        }
        self.gbuffer_combine(
            gpu,
            &mut graphics_command_buffer,
            &color_output,
            &gbuffer,
            resource_map,
            self.view_size,
        )
        .context("GBuffer combine")?;

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
