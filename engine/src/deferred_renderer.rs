use bytemuck::{Pod, Zeroable};
use engine_macros::glsl;
use std::{collections::HashMap, mem::size_of};

use crate::{
    app_state, app_state_mut,
    camera::Camera,
    material::{MasterMaterial, MasterMaterialDescription},
    Backbuffer, Light, LightType, MaterialDescription, MaterialDomain, MaterialInstance, Mesh,
    MeshPrimitive, PipelineTarget, RenderingPipeline, Scene, Texture,
};

use gpu::{
    AccessFlags, AttachmentReference, AttachmentStoreOp, Binding, BindingType, BlendMode, BlendOp,
    BlendState, BufferCreateInfo, BufferHandle, BufferUsageFlags, ColorComponentFlags, ColorLoadOp,
    Extent2D, FragmentStageInfo, FramebufferColorAttachment, Gpu, Handle, ImageAspectFlags,
    ImageFormat, ImageHandle, ImageLayout, ImageMemoryBarrier, ImageUsageFlags, ImageViewHandle,
    ImageViewType, IndexType, InputRate, LifetimedCache, MemoryDomain, PipelineStageFlags,
    PushConstantRange, RenderPassAttachment, SampleCount, SamplerHandle, ShaderModuleCreateInfo,
    ShaderModuleHandle, ShaderStage, StencilLoadOp, SubpassDependency, SubpassDescription,
    VertexBindingInfo, VertexStageInfo, VkCommandBuffer, VkGpu, VkRenderPassCommand, VkSwapchain,
};
use nalgebra::{vector, Matrix4, Point3, Point4, Vector2, Vector3, Vector4};
use resource_map::{ResourceHandle, ResourceMap};

const FXAA_FS: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/fxaa_fs.frag",
    entry_point = "main"
);

const FXAA_VS: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/fxaa_vs.vert",
    entry_point = "main"
);

const SCREEN_QUAD: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/screen_quad.vert",
    entry_point = "main"
);

const GBUFFER_COMBINE: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/gbuffer_combine.frag",
    entry_point = "main"
);

const TONEMAP: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/tonemap.frag",
    entry_point = "main"
);

const TEXTURE_COPY: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/texture_copy.frag",
    entry_point = "main"
);

const SHADOW_ATLAS_TILE_SIZE: u32 = 128;
const SHADOW_ATLAS_WIDTH: u32 = 7680;
const SHADOW_ATLAS_HEIGHT: u32 = 4352;

#[repr(C)]
#[derive(Clone, Copy)]
struct FxaaShaderParams {
    rcp_frame: Vector2<f32>,
    fxaa_quality_subpix: f32,
    fxaa_quality_edge_threshold: f32,
    fxaa_quality_edge_threshold_min: f32,
    iterations: u32,
}

unsafe impl Pod for FxaaShaderParams {}
unsafe impl Zeroable for FxaaShaderParams {}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RenderImageDescription {
    width: u32,
    height: u32,
    format: ImageFormat,
    samples: SampleCount,
    view_type: ImageViewType,
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

pub struct ImageAllocator {
    image_allocator: LifetimedCache<RenderImage>,
}

impl ImageAllocator {
    fn get(&self, gpu: &VkGpu, label: &'static str, desc: &RenderImageDescription) -> RenderImage {
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
                            samples: desc.samples.into(),
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

    fn new(lifetime: u32) -> Self {
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
    pub iterations: u32,
}

unsafe impl Pod for FxaaSettings {}
unsafe impl Zeroable for FxaaSettings {}

impl Default for FxaaSettings {
    fn default() -> Self {
        Self {
            fxaa_quality_subpix: 0.75,
            fxaa_quality_edge_threshold: 0.166,
            fxaa_quality_edge_threshold_min: 0.0833,
            iterations: 12,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PerFrameData {
    pub eye: Point4<f32>,
    pub eye_forward: Vector4<f32>,
    pub view: nalgebra::Matrix4<f32>,
    pub projection: nalgebra::Matrix4<f32>,
    pub viewport_size_offset: Vector4<f32>,
}

unsafe impl Pod for PerFrameData {}
unsafe impl Zeroable for PerFrameData {}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuLightInfo {
    position_radius: Vector4<f32>,
    direction: Vector4<f32>,
    color: Vector4<f32>,
    extras: Vector4<f32>,
    ty_shadowcaster: [i32; 4],
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
            ty_shadowcaster: [ty, -1, 0, 0],
        }
    }
}

struct FrameBuffers {
    camera_buffer: BufferHandle,
    light_buffer: BufferHandle,
}

struct DrawCall<'a> {
    prim: &'a MeshPrimitive,
    transform: Matrix4<f32>,
    material: ResourceHandle<MaterialInstance>,
}

pub struct DeferredRenderingPipeline {
    frame_buffers: Vec<FrameBuffers>,
    image_allocator: ImageAllocator,
    screen_quad: ShaderModuleHandle,
    texture_copy: ShaderModuleHandle,
    gbuffer_combine: ShaderModuleHandle,
    tonemap_fs: ShaderModuleHandle,

    fxaa_settings: FxaaSettings,

    fxaa_vs: ShaderModuleHandle,
    fxaa_fs: ShaderModuleHandle,
    in_flight_frame: usize,
    max_frames_in_flight: usize,
    light_iteration: u64,
    active_lights: Vec<GpuLightInfo>,
    light_povs: Vec<PerFrameData>,
    pub(crate) irradiance_map: Option<ResourceHandle<Texture>>,
    default_irradiance_map: ResourceHandle<Texture>,

    cube_mesh: ResourceHandle<Mesh>,

    pub depth_bias_constant: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope: f32,

    pub ambient_color: Vector3<f32>,
    pub ambient_intensity: f32,

    gbuffer_nearest_sampler: SamplerHandle,
    shadow_atlas_sampler: SamplerHandle,
}

impl DeferredRenderingPipeline {
    pub fn new(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        cube_mesh: ResourceHandle<Mesh>,
    ) -> anyhow::Result<Self> {
        let mut frame_buffers = vec![];
        for _ in 0..VkSwapchain::MAX_FRAMES_IN_FLIGHT {
            let camera_buffer = {
                let create_info = BufferCreateInfo {
                    label: Some("Deferred Renderer - Camera buffer"),
                    size: std::mem::size_of::<PerFrameData>() * 100,
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

        let fxaa_vs = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(FXAA_VS),
        })?;
        let fxaa_fs = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(FXAA_FS),
        })?;

        let screen_quad = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(SCREEN_QUAD),
        })?;
        let gbuffer_combine = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(GBUFFER_COMBINE),
        })?;
        let tonemap_fs = gpu.make_shader_module(&ShaderModuleCreateInfo {
            code: bytemuck::cast_slice(TONEMAP),
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

        let shadow_atlas_sampler = gpu.make_sampler(&gpu::SamplerCreateInfo {
            mag_filter: gpu::Filter::Linear,
            min_filter: gpu::Filter::Linear,
            address_u: gpu::SamplerAddressMode::ClampToBorder,
            address_v: gpu::SamplerAddressMode::ClampToBorder,
            address_w: gpu::SamplerAddressMode::ClampToBorder,
            mip_lod_bias: 0.0,
            compare_function: Some(gpu::CompareOp::LessEqual),
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: [0.0; 4],
        })?;

        let default_irradiance_map = Texture::new_with_data(
            gpu,
            resource_map,
            1,
            1,
            bytemuck::cast_slice(&[255u8; 6 * 3]),
            Some("Default White Irradiance Map"),
            ImageFormat::Rgb8,
            gpu::ImageViewType::Cube,
        )?;
        let default_irradiance_map = resource_map.add(default_irradiance_map);

        Ok(Self {
            image_allocator: ImageAllocator::new(4),
            screen_quad,
            gbuffer_combine,
            texture_copy,
            frame_buffers,
            tonemap_fs,
            fxaa_vs,
            fxaa_fs,
            fxaa_settings: Default::default(),
            light_iteration: 0,
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
            gbuffer_nearest_sampler,
            shadow_atlas_sampler,
        })
    }

    pub fn fxaa_settings(&self) -> FxaaSettings {
        self.fxaa_settings
    }
    pub fn fxaa_settings_mut(&mut self) -> &mut FxaaSettings {
        &mut self.fxaa_settings
    }
    pub fn set_fxaa_settings_mut(&mut self, settings: FxaaSettings) {
        self.fxaa_settings = settings;
    }

    fn main_render_loop(
        resource_map: &ResourceMap,
        pipeline_target: PipelineTarget,
        draw_hashmap: &HashMap<&MasterMaterial, Vec<DrawCall>>,
        render_pass: &mut VkRenderPassCommand,
        camera_index: u32,
        frame_buffers: &FrameBuffers,
    ) {
        let mut total_primitives_rendered = 0;
        for (master, material_draw_calls) in draw_hashmap.iter() {
            {
                bind_master_material(master, pipeline_target, render_pass, frame_buffers);

                for (idx, draw_call) in material_draw_calls.iter().enumerate() {
                    let material = &draw_call.material;
                    let material = resource_map.get(material);
                    let draw_label = render_pass.begin_debug_region(
                        &format!(
                            "{} - {}, total primitives rendered {total_primitives_rendered}",
                            material.name, idx
                        ),
                        [0.0, 0.3, 0.4, 1.0],
                    );
                    draw_mesh_primitive(
                        render_pass,
                        material,
                        master,
                        &draw_call.prim,
                        draw_call.transform,
                        resource_map,
                        camera_index,
                    );
                    draw_label.end();
                    total_primitives_rendered += 1;
                }
                render_pass.insert_debug_label(
                    &format!("Total primtives drawn this frame: {total_primitives_rendered}"),
                    [0.0, 0.3, 0.4, 1.0],
                );
            }
        }
    }

    fn generate_draw_calls<'r, 's>(
        resource_map: &'r ResourceMap,
        scene: &'s Scene,
    ) -> HashMap<&'s MasterMaterial, Vec<DrawCall<'s>>>
    where
        'r: 's,
    {
        let mut draw_hashmap: HashMap<&MasterMaterial, Vec<DrawCall>> = HashMap::new();

        for primitive in scene.primitives.iter() {
            let mesh = resource_map.get(&primitive.mesh);
            for (idx, mesh_prim) in mesh.primitives.iter().enumerate() {
                let material_handle = primitive.materials[idx].clone();
                let material = resource_map.get(&material_handle);
                let master = resource_map.get(&material.owner);
                draw_hashmap.entry(master).or_default().push(DrawCall {
                    prim: mesh_prim,
                    transform: primitive.transform,
                    material: material_handle,
                });
            }
        }
        draw_hashmap
    }

    fn update_lights(&mut self, scene: &Scene) {
        self.light_iteration = scene.lights_iteration();
        self.active_lights.clear();
        self.light_povs.clear();
        let mut shadow_caster_idx = 0;

        let mut sorted_active_lights = scene.all_enabled_lights().collect::<Vec<_>>();
        sorted_active_lights.sort_by(|a, b| {
            let a = a.shadow_setup.unwrap_or_default();
            let b = b.shadow_setup.unwrap_or_default();
            b.importance.cmp(&a.importance)
        });

        let mut packer = crate::utils::TiledTexture2DPacker::new(
            SHADOW_ATLAS_TILE_SIZE,
            SHADOW_ATLAS_WIDTH,
            SHADOW_ATLAS_HEIGHT,
        )
        .expect("Could not create packer");

        for active_light in sorted_active_lights {
            let mut gpu_light: GpuLightInfo = active_light.into();
            if active_light.shadow_setup.is_some() {
                gpu_light.ty_shadowcaster[1] = shadow_caster_idx;
                let povs = active_light.shadow_view_matrices();
                let w =
                    active_light.shadow_setup.unwrap().importance.get() * SHADOW_ATLAS_TILE_SIZE;
                let h = w;
                let allocated_slot = packer.allocate(w * povs.len() as u32, h);
                let allocated_slot = if let Ok(slot) = allocated_slot {
                    slot
                } else {
                    break;
                };
                for (i, pov) in povs.into_iter().enumerate() {
                    let pfd = Some(PerFrameData {
                        eye: Point4::new(
                            active_light.position.x,
                            active_light.position.y,
                            active_light.position.z,
                            0.0,
                        ),
                        eye_forward: gpu_light.direction,
                        view: pov,
                        projection: active_light.projection_matrix(),
                        viewport_size_offset: vector![
                            allocated_slot.x as f32 + w as f32 * i as f32,
                            allocated_slot.y as f32,
                            w as f32,
                            h as f32
                        ],
                    });

                    shadow_caster_idx += 1;
                    self.light_povs.extend(pfd);
                }
            }
            self.active_lights.push(gpu_light);
        }
    }

    fn draw_skybox(
        camera_location: &Point3<f32>,
        render_pass: &mut VkRenderPassCommand,
        skybox_mesh: &Mesh,
        skybox_material: &MaterialInstance,
        skybox_master: &MasterMaterial,
        resource_map: &ResourceMap,
    ) {
        let label = render_pass.begin_debug_region(
            &format!("Skybox - using material {}", skybox_master.name),
            [0.2, 0.2, 0.0, 1.0],
        );
        const SKYBOX_SCALE: f32 = 1.0;
        let skybox_transform =
            Matrix4::new_translation(&camera_location.coords) * Matrix4::new_scaling(SKYBOX_SCALE);
        render_pass.set_enable_depth_test(false);
        render_pass.set_depth_write_enabled(false);
        render_pass.set_color_output_enabled(true);
        render_pass.set_cull_mode(gpu::CullMode::None);
        render_pass.set_depth_compare_op(gpu::CompareOp::Always);
        draw_mesh_primitive(
            render_pass,
            skybox_material,
            skybox_master,
            &skybox_mesh.primitives[0],
            skybox_transform,
            resource_map,
            0,
        );
        label.end();
    }

    pub fn set_irradiance_texture(
        &mut self,
        irradiance_map: Option<ResourceHandle<crate::Texture>>,
    ) {
        self.irradiance_map = irradiance_map;
    }
}

fn bind_master_material(
    master: &MasterMaterial,
    pipeline_target: PipelineTarget,
    render_pass: &mut VkRenderPassCommand<'_, '_>,
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
    render_pass: &mut VkRenderPassCommand,
    material: &MaterialInstance,
    master: &MasterMaterial,
    primitive: &MeshPrimitive,
    model: Matrix4<f32>,
    resource_map: &ResourceMap,
    camera_index: u32,
) {
    render_pass.set_index_buffer(primitive.index_buffer.clone(), IndexType::Uint32, 0);

    let mut user_bindings = vec![];
    user_bindings.extend(
        &mut master
            .texture_inputs
            .iter()
            .enumerate()
            .map(|(i, tex_info)| {
                let texture_parameter = &material.current_inputs[&tex_info.name];
                let tex = resource_map.get(texture_parameter);

                // TODO: these can be avoided
                let view = resource_map.get(&tex.image_view);
                let sampler = resource_map.get(&tex.sampler);
                Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: view.view.clone(),
                        sampler_handle: sampler.0.clone(),
                    },
                    binding_stage: tex_info.shader_stage,
                    location: i as _,
                }
            }),
    );
    if material.parameter_buffer.is_valid() {
        user_bindings.push(Binding {
            ty: gpu::DescriptorBindingType::UniformBuffer {
                handle: material.parameter_buffer.clone(),
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
    render_pass.draw_indexed(primitive.index_count, 1, 0, 0, 0);
}

impl RenderingPipeline for DeferredRenderingPipeline {
    fn render(
        &mut self,
        pov: &Camera,
        scene: &Scene,
        backbuffer: &Backbuffer,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<VkCommandBuffer> {
        let projection = pov.projection();

        if self.light_iteration != scene.lights_iteration() {
            self.update_lights(scene);
        }

        let current_buffers = &self.frame_buffers[self.in_flight_frame];

        self.in_flight_frame = (1 + self.in_flight_frame) % self.max_frames_in_flight;

        let mut per_frame_data = vec![PerFrameData {
            eye: Point4::new(pov.location[0], pov.location[1], pov.location[2], 0.0),
            eye_forward: vector![pov.forward.x, pov.forward.y, pov.forward.z, 0.0],
            view: pov.view(),
            projection,
            viewport_size_offset: vector![
                0.0,
                0.0,
                backbuffer.size.width as f32,
                backbuffer.size.height as f32
            ],
        }];

        per_frame_data.extend_from_slice(&self.light_povs);

        super::app_state()
            .gpu
            .write_buffer(
                &current_buffers.camera_buffer,
                0,
                bytemuck::cast_slice(&[per_frame_data.len() as u32 - 1]),
            )
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer(
                &current_buffers.camera_buffer,
                size_of::<u32>() as u64 * 4,
                bytemuck::cast_slice(&per_frame_data),
            )
            .unwrap();

        let ambient = vector![
            self.ambient_color.x,
            self.ambient_color.y,
            self.ambient_color.z,
            self.ambient_intensity
        ];

        super::app_state()
            .gpu
            .write_buffer(
                &current_buffers.light_buffer,
                0,
                bytemuck::cast_slice(&[ambient.x, ambient.y, ambient.z, ambient.w]),
            )
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer(
                &current_buffers.light_buffer,
                std::mem::size_of::<Vector4<f32>>() as _,
                bytemuck::cast_slice(&[self.active_lights.len() as u32]),
            )
            .unwrap();
        super::app_state()
            .gpu
            .write_buffer(
                &current_buffers.light_buffer,
                std::mem::size_of::<Vector4<f32>>() as u64 + size_of::<u32>() as u64 * 4,
                bytemuck::cast_slice(&self.active_lights),
            )
            .unwrap();

        app_state_mut().begin_frame()?;

        let draw_hashmap = Self::generate_draw_calls(resource_map, scene);

        let skybox_material = match scene.get_skybox_material() {
            Some(material) => Some(resource_map.get(material)),

            _ => None,
        };
        let vector_desc = RenderImageDescription {
            format: ImageFormat::RgbaFloat32,
            samples: SampleCount::Sample1,
            width: backbuffer.size.width,
            height: backbuffer.size.height,
            view_type: ImageViewType::Type2D,
        };
        let depth_desc = RenderImageDescription {
            format: ImageFormat::Depth,
            samples: SampleCount::Sample1,
            width: backbuffer.size.width,
            height: backbuffer.size.height,
            view_type: ImageViewType::Type2D,
        };
        let shadow_atlas_desc = RenderImageDescription {
            format: ImageFormat::Depth,
            samples: SampleCount::Sample1,
            width: SHADOW_ATLAS_WIDTH,
            height: SHADOW_ATLAS_HEIGHT,
            view_type: ImageViewType::Type2D,
        };
        let color_desc = RenderImageDescription {
            format: ImageFormat::Rgba8,
            samples: SampleCount::Sample1,
            width: 1920,
            height: 1080,
            view_type: ImageViewType::Type2D,
        };

        let shadow_atlas_component =
            self.image_allocator
                .get(&app_state().gpu, "shadow_atlas", &shadow_atlas_desc);
        let depth_component = self
            .image_allocator
            .get(&app_state().gpu, "depth", &depth_desc);
        let pos_component = self
            .image_allocator
            .get(&app_state().gpu, "position", &vector_desc);
        let normal_component = self
            .image_allocator
            .get(&app_state().gpu, "normal", &vector_desc);
        let diffuse_component = self
            .image_allocator
            .get(&app_state().gpu, "diffuse", &vector_desc);
        let emissive_component =
            self.image_allocator
                .get(&app_state().gpu, "emissive", &vector_desc);
        let pbr_component =
            self.image_allocator
                .get(&app_state().gpu, "pbr_component", &vector_desc);
        let color_output = self
            .image_allocator
            .get(&app_state().gpu, "color_output", &color_desc);

        let irradiance_map_texture = match &self.irradiance_map {
            Some(texture) => texture,
            None => &self.default_irradiance_map,
        };
        let irradiance_map_texture = resource_map.get(irradiance_map_texture);
        let irradiance_image_view = resource_map.get(&irradiance_map_texture.image_view);
        let mut graphics_command_buffer = app_state()
            .gpu
            .create_command_buffer(gpu::QueueType::Graphics)?;

        //            ctx.render_pass_command.set_depth_bias(
        //                self.depth_bias_constant,
        //                self.depth_bias_clamp,
        //                self.depth_bias_slope,
        //            );
        //
        //            ctx.render_pass_command.set_cull_mode(gpu::CullMode::Front);
        //            ctx.render_pass_command
        //                .set_depth_compare_op(gpu::CompareOp::LessEqual);
        //
        //            ctx.render_pass_command.set_color_output_enabled(false);
        //            ctx.render_pass_command.set_enable_depth_test(true);
        //            ctx.render_pass_command.set_depth_write_enabled(true);
        //
        //            for (i, pov) in per_frame_data.iter().enumerate().skip(1) {
        //                ctx.render_pass_command.set_viewport(gpu::Viewport {
        //                    x: pov.viewport_size_offset.x,
        //                    y: pov.viewport_size_offset.y,
        //                    width: pov.viewport_size_offset.z,
        //                    height: pov.viewport_size_offset.w,
        //                    min_depth: 0.0,
        //                    max_depth: 1.0,
        //                });
        //
        //                Self::main_render_loop(
        //                    resource_map,
        //                    PipelineTarget::DepthOnly,
        //                    &draw_hashmap,
        //                    i as _,
        //                    ctx,
        //                    &current_buffers.camera_buffer,
        //                    &current_buffers.light_buffer,
        //                );
        //            }

        {
            let mut gbuffer_render_pass =
                graphics_command_buffer.begin_render_pass(&gpu::BeginRenderPassInfo {
                    label: Some("GBuffer"),
                    color_attachments: &[
                        FramebufferColorAttachment {
                            image_view: pos_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ColorAttachment,
                        },
                        FramebufferColorAttachment {
                            image_view: normal_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.5; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ColorAttachment,
                        },
                        FramebufferColorAttachment {
                            image_view: diffuse_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ColorAttachment,
                        },
                        FramebufferColorAttachment {
                            image_view: emissive_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ColorAttachment,
                        },
                        FramebufferColorAttachment {
                            image_view: pbr_component.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ColorAttachment,
                        },
                        FramebufferColorAttachment {
                            image_view: color_output.view.clone(),
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ColorAttachment,
                        },
                    ],
                    depth_attachment: Some(gpu::FramebufferDepthAttachment {
                        image_view: depth_component.view,
                        load_op: gpu::DepthLoadOp::Clear(1.0),
                        store_op: AttachmentStoreOp::Store,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::DepthStencilAttachment,
                    }),
                    stencil_attachment: None,
                    render_area: gpu::Rect2D {
                        offset: gpu::Offset2D::default(),
                        extent: backbuffer.size,
                    },
                    subpasses: &[
                        SubpassDescription {
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
                    ],
                    dependencies: &[
                        SubpassDependency {
                            src_subpass: 0,
                            dst_subpass: 1,
                            src_stage_mask: PipelineStageFlags::LATE_FRAGMENT_TESTS,
                            dst_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                            src_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                            dst_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        },
                        SubpassDependency {
                            src_subpass: 1,
                            dst_subpass: 2,
                            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dst_stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
                            src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                            dst_access_mask: AccessFlags::SHADER_READ,
                        },
                    ],
                });

            gbuffer_render_pass.set_cull_mode(gpu::CullMode::Back);
            gbuffer_render_pass.set_depth_compare_op(gpu::CompareOp::LessEqual);

            gbuffer_render_pass.set_color_output_enabled(false);
            gbuffer_render_pass.set_enable_depth_test(true);
            gbuffer_render_pass.set_depth_write_enabled(true);
            Self::main_render_loop(
                resource_map,
                PipelineTarget::DepthOnly,
                &draw_hashmap,
                &mut gbuffer_render_pass,
                0,
                &current_buffers,
            );

            gbuffer_render_pass.advance_to_next_subpass();

            if let Some(material) = skybox_material {
                let cube_mesh = resource_map.get(&self.cube_mesh);
                let skybox_master = resource_map.get(&material.owner);
                bind_master_material(
                    skybox_master,
                    PipelineTarget::ColorAndDepth,
                    &mut gbuffer_render_pass,
                    &current_buffers,
                );
                Self::draw_skybox(
                    &pov.location,
                    &mut gbuffer_render_pass,
                    cube_mesh,
                    material,
                    skybox_master,
                    resource_map,
                );
            }

            gbuffer_render_pass.set_front_face(gpu::FrontFace::CounterClockWise);
            gbuffer_render_pass.set_enable_depth_test(true);
            gbuffer_render_pass.set_depth_write_enabled(false);
            gbuffer_render_pass.set_color_output_enabled(true);
            gbuffer_render_pass.set_cull_mode(gpu::CullMode::Back);
            gbuffer_render_pass.set_depth_compare_op(gpu::CompareOp::Equal);
            Self::main_render_loop(
                resource_map,
                PipelineTarget::ColorAndDepth,
                &draw_hashmap,
                &mut gbuffer_render_pass,
                0,
                &current_buffers,
            );

            gbuffer_render_pass.advance_to_next_subpass();

            gbuffer_render_pass.bind_resources(
                0,
                &[
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: pos_component.view.clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 0,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: normal_component.view.clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 1,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: diffuse_component.view.clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 2,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: emissive_component.view.clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 3,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: pbr_component.view.clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 4,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: shadow_atlas_component.view.clone(),
                            sampler_handle: self.shadow_atlas_sampler.clone(),
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 5,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::ImageView {
                            image_view_handle: irradiance_image_view.view.clone(),
                            sampler_handle: self.gbuffer_nearest_sampler.clone(),
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 6,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::StorageBuffer {
                            handle: current_buffers.camera_buffer.clone(),
                            offset: 0,
                            range: gpu::WHOLE_SIZE as _,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 7,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::StorageBuffer {
                            handle: current_buffers.light_buffer.clone(),
                            offset: 0,
                            range: gpu::WHOLE_SIZE as _,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: 8,
                    },
                ],
            );
            gbuffer_render_pass.set_cull_mode(gpu::CullMode::None);
            gbuffer_render_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
            gbuffer_render_pass.set_front_face(gpu::FrontFace::ClockWise);
            gbuffer_render_pass.set_enable_depth_test(false);
            gbuffer_render_pass.set_depth_write_enabled(false);
            gbuffer_render_pass.set_vertex_shader(self.screen_quad.clone());
            gbuffer_render_pass.set_fragment_shader(self.gbuffer_combine.clone());
            gbuffer_render_pass.draw(4, 1, 0, 0);
        }

        graphics_command_buffer.pipeline_barrier(&gpu::PipelineBarrierInfo {
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
                image: color_output.image,
                subresource_range: gpu::ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            }],
        });

        {
            let mut present_render_pass =
                graphics_command_buffer.begin_render_pass(&gpu::BeginRenderPassInfo {
                    label: Some("Present"),
                    color_attachments: &[FramebufferColorAttachment {
                        image_view: backbuffer.image_view.clone(),
                        load_op: ColorLoadOp::DontCare,
                        store_op: AttachmentStoreOp::Store,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::ColorAttachment,
                    }],
                    depth_attachment: None,
                    stencil_attachment: None,
                    render_area: gpu::Rect2D {
                        offset: gpu::Offset2D::default(),
                        extent: backbuffer.size,
                    },
                    subpasses: &[SubpassDescription {
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
                        image_view_handle: color_output.view,
                        sampler_handle: self.gbuffer_nearest_sampler.clone(),
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 0,
                }],
            );
            present_render_pass.set_front_face(gpu::FrontFace::ClockWise);
            present_render_pass.set_cull_mode(gpu::CullMode::None);
            present_render_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
            present_render_pass.set_enable_depth_test(false);
            present_render_pass.set_depth_write_enabled(false);
            present_render_pass.set_vertex_shader(self.screen_quad.clone());
            present_render_pass.set_fragment_shader(self.texture_copy.clone());
            present_render_pass.draw(4, 1, 0, 0);
        }

        //
        //#region context setup
        //        context.register_callback(&shadow_atlas_rendering_pass, |_: &VkGpu, ctx| {
        //            ctx.render_pass_command.set_depth_bias(
        //                self.depth_bias_constant,
        //                self.depth_bias_clamp,
        //                self.depth_bias_slope,
        //            );
        //
        //            ctx.render_pass_command.set_cull_mode(gpu::CullMode::Front);
        //            ctx.render_pass_command
        //                .set_depth_compare_op(gpu::CompareOp::LessEqual);
        //
        //            ctx.render_pass_command.set_color_output_enabled(false);
        //            ctx.render_pass_command.set_enable_depth_test(true);
        //            ctx.render_pass_command.set_depth_write_enabled(true);
        //
        //            for (i, pov) in per_frame_data.iter().enumerate().skip(1) {
        //                ctx.render_pass_command.set_viewport(gpu::Viewport {
        //                    x: pov.viewport_size_offset.x,
        //                    y: pov.viewport_size_offset.y,
        //                    width: pov.viewport_size_offset.z,
        //                    height: pov.viewport_size_offset.w,
        //                    min_depth: 0.0,
        //                    max_depth: 1.0,
        //                });
        //
        //                Self::main_render_loop(
        //                    resource_map,
        //                    PipelineTarget::DepthOnly,
        //                    &draw_hashmap,
        //                    i as _,
        //                    ctx,
        //                    &current_buffers.camera_buffer,
        //                    &current_buffers.light_buffer,
        //                );
        //            }
        //        });
        //        context.register_callback(&gbuffer_pass, |_: &VkGpu, ctx| {
        //        });
        //
        //        context.register_callback(&combine_pass, |_: &VkGpu, ctx| {
        //        });
        //        context.register_callback(&tonemap_pass, |_: &VkGpu, ctx| {
        //            ctx.render_pass_command.bind_resources(0, &ctx.bindings);
        //            ctx.render_pass_command.set_cull_mode(gpu::CullMode::None);
        //            ctx.render_pass_command.set_enable_depth_test(false);
        //            ctx.render_pass_command.set_depth_write_enabled(false);
        //            ctx.render_pass_command
        //                .set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
        //            ctx.render_pass_command
        //                .set_vertex_shader(self.screen_quad.clone());
        //            ctx.render_pass_command
        //                .set_fragment_shader(self.tonemap_fs.clone());
        //            ctx.render_pass_command.draw(4, 1, 0, 0);
        //        });
        //        context.register_callback(&fxaa_pass, |_: &VkGpu, ctx| {
        //            ctx.render_pass_command.bind_resources(0, &ctx.bindings);
        //            ctx.render_pass_command.set_cull_mode(gpu::CullMode::None);
        //            ctx.render_pass_command.set_enable_depth_test(false);
        //            ctx.render_pass_command.set_depth_write_enabled(false);
        //            ctx.render_pass_command
        //                .set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
        //            let rcp_frame = vector![backbuffer.size.width as f32, backbuffer.size.height as f32];
        //            let rcp_frame = vector![1.0 / rcp_frame.x, 1.0 / rcp_frame.y];
        //
        //            let params = FxaaShaderParams {
        //                rcp_frame,
        //                fxaa_quality_subpix: self.fxaa_settings.fxaa_quality_subpix,
        //                fxaa_quality_edge_threshold: self.fxaa_settings.fxaa_quality_edge_threshold,
        //                fxaa_quality_edge_threshold_min: self.fxaa_settings.fxaa_quality_edge_threshold_min,
        //                iterations: self.fxaa_settings.iterations,
        //            };
        //            ctx.render_pass_command
        //                .set_vertex_shader(self.fxaa_vs.clone());
        //            ctx.render_pass_command
        //                .set_fragment_shader(self.fxaa_fs.clone());
        //            ctx.render_pass_command.push_constants(
        //                0,
        //                0,
        //                bytemuck::cast_slice(&[params]),
        //                ShaderStage::ALL_GRAPHICS,
        //            );
        //            ctx.render_pass_command.draw(3, 1, 0, 0);
        //        });
        //        context.register_callback(&present_render_pass, |_: &VkGpu, ctx| {
        //            ctx.render_pass_command.bind_resources(0, &ctx.bindings);
        //            ctx.render_pass_command
        //                .set_front_face(gpu::FrontFace::ClockWise);
        //            ctx.render_pass_command.set_cull_mode(gpu::CullMode::None);
        //            ctx.render_pass_command
        //                .set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
        //            ctx.render_pass_command.set_enable_depth_test(false);
        //            ctx.render_pass_command.set_depth_write_enabled(false);
        //            ctx.render_pass_command
        //                .set_vertex_shader(self.screen_quad.clone());
        //            ctx.render_pass_command
        //                .set_fragment_shader(self.texture_copy.clone());
        //            ctx.render_pass_command.draw(4, 1, 0, 0);
        //        });
        //
        //        let irradiance_map_texture = match &self.irradiance_map {
        //            Some(texture) => texture,
        //            None => &self.default_irradiance_map,
        //        };
        //        let irradiance_map_texture = resource_map.get(irradiance_map_texture);
        //        let image_view = resource_map.get(&irradiance_map_texture.image_view);
        //        let image = &resource_map.get(&image_view.image).0;
        //
        //        context.inject_external_image(&irradiance_map, image.clone(), image_view.view.clone());
        //        context.inject_external_image(
        //            &swapchain_image,
        //            backbuffer.image.clone(),
        //            backbuffer.image_view.clone(),
        //        );
        //        context.injext_external_buffer(&camera_buffer, current_buffers.camera_buffer.clone());
        //        context.injext_external_buffer(&light_buffer, current_buffers.light_buffer.clone());
        //        //#endregion
        //        self.render_graph.run(context, &mut self.runner)?;

        Ok(graphics_command_buffer)
    }

    fn create_material(
        &mut self,
        _gpu: &VkGpu,
        material_description: MaterialDescription,
    ) -> anyhow::Result<MasterMaterial> {
        let color_attachments = &[
            // Position
            RenderPassAttachment {
                format: ImageFormat::RgbaFloat32,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Normals
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Diffuse
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Emissive
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
            // Metal/Roughness
            RenderPassAttachment {
                format: ImageFormat::Rgba8,
                samples: SampleCount::Sample1,
                load_op: ColorLoadOp::Clear([0.0; 4]),
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: StencilLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::ShaderReadOnly,
                blend_state: BlendState {
                    blend_enable: true,
                    src_color_blend_factor: BlendMode::One,
                    dst_color_blend_factor: BlendMode::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendMode::One,
                    dst_alpha_blend_factor: BlendMode::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::RGBA,
                },
            },
        ];
        let master_description = MasterMaterialDescription {
            name: material_description.name,
            domain: material_description.domain,
            global_inputs: match material_description.domain {
                MaterialDomain::Surface => &[BindingType::Storage],
                MaterialDomain::PostProcess => &[
                    BindingType::Storage,              // Camera buffer
                    BindingType::CombinedImageSampler, // Previous post process result/ Initial scene color,
                    BindingType::CombinedImageSampler, // Scene position,
                    BindingType::CombinedImageSampler, // Scene diffuse,
                    BindingType::CombinedImageSampler, // Scene normal,
                ],
            },
            texture_inputs: material_description.texture_inputs,
            material_parameters: material_description.material_parameters,
            vertex_info: &VertexStageInfo {
                entry_point: "main",
                module: material_description.vertex_module.clone(),
            },
            fragment_info: &FragmentStageInfo {
                entry_point: "main",
                module: material_description.fragment_module,
                color_attachments,
                depth_stencil_attachments: &[],
            },
            primitive_restart: false,
            polygon_mode: gpu::PolygonMode::Fill,
            cull_mode: gpu::CullMode::Back,
            front_face: gpu::FrontFace::CounterClockWise,
            push_constant_ranges: &[PushConstantRange {
                stage_flags: ShaderStage::ALL,
                offset: 0,
                size: std::mem::size_of::<ObjectDrawInfo>() as u32,
            }],
            logic_op: None,
            parameters_visibility: material_description.parameter_shader_visibility,
        };

        MasterMaterial::new(&master_description)
    }
}
