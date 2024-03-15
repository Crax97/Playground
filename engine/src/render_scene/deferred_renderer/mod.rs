mod cascaded_shadow_map;
mod gbuffer;
mod render_image;
mod sampler_allocator;

use crate::{
    post_process_pass::{PostProcessPass, PostProcessResources},
    render_scene::render_structs::*,
    Asset,
};
use cascaded_shadow_map::*;
use gbuffer::*;
use render_image::*;
use sampler_allocator::*;

use anyhow::Context;
use engine_macros::glsl;
use std::mem::size_of;

use crate::{
    material::{MasterMaterial, MasterMaterialDescription},
    Camera, CvarManager, Frustum, MaterialDescription, MaterialInstance, Mesh, MeshPrimitive,
    PipelineTarget, RenderScene, RenderingPipeline, ScenePrimitive, Texture,
};

use crate::asset_map::{AssetHandle, AssetMap};
use gpu::{
    render_pass_2::RenderPass2, AttachmentStoreOp, BeginRenderPassInfo2, Binding2,
    BufferCreateInfo, BufferUsageFlags, ColorAttachment, ColorLoadOp, CommandBuffer,
    DepthAttachment, Extent2D, FragmentStageInfo, Gpu, ImageFormat, ImageViewHandle, ImageViewType,
    IndexType, InputRate, MemoryDomain, Offset2D, Rect2D, SampleCount, SamplerHandle,
    ShaderModuleCreateInfo, ShaderModuleHandle, ShaderStage, VertexBindingInfo, VertexStageInfo,
};
use nalgebra::{vector, Matrix4, Point3, Point4, Vector2, Vector3, Vector4};

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
    image_allocator: RenderImageAllocator,
    sampler_allocator: SamplerAllocator,
    screen_quad: ShaderModuleHandle,
    texture_copy: ShaderModuleHandle,
    combine_shader: ShaderModuleHandle,

    post_process_stack: Vec<Box<dyn PostProcessPass>>,

    in_flight_frame: usize,
    max_frames_in_flight: usize,
    active_lights: Vec<GpuLightInfo>,
    light_povs: Vec<PointOfViewData>,
    pub(crate) irradiance_map: Option<AssetHandle<Texture>>,
    default_irradiance_map: Texture,

    cube_mesh: Mesh,

    pub ambient_color: Vector3<f32>,
    pub ambient_intensity: f32,

    pub update_frustum: bool,
    pub drawcalls_last_frame: u64,

    pub cascaded_shadow_map: CascadedShadowMap,

    frustum: Frustum,
    gbuffer_nearest_sampler: SamplerHandle,
    screen_quad_flipped: ShaderModuleHandle,
    early_z_pass_enabled: bool,
    view_size: Extent2D,
}

impl DeferredRenderingPipeline {
    pub fn new(gpu: &dyn Gpu, combine_shader: ShaderModuleHandle) -> anyhow::Result<Self> {
        let mut frame_buffers = vec![];
        let cube_mesh = crate::utils::create_cube_mesh(gpu)?;
        for _ in 0..gpu::constants::MAX_FRAMES_IN_FLIGHT {
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
            label: Some("Screen Quad"),
            code: bytemuck::cast_slice(SCREEN_QUAD),
        })?;

        let screen_quad_flipped = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Screen Quad Flipped"),
            code: bytemuck::cast_slice(SCREEN_QUAD_FLIPPED),
        })?;

        let texture_copy = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Texture Copy"),
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
        // let default_irradiance_map = resource_map.add(default_irradiance_map);

        let view_size = Extent2D {
            width: 1920,
            height: 1080,
        };

        let cascaded_shadow_map =
            CascadedShadowMap::new(gpu, gpu::constants::MAX_FRAMES_IN_FLIGHT)?;

        Ok(Self {
            image_allocator: RenderImageAllocator::new(4),
            sampler_allocator: SamplerAllocator::new(4),
            screen_quad,
            screen_quad_flipped,
            combine_shader,
            texture_copy,
            frame_buffers,
            post_process_stack: vec![],
            in_flight_frame: 0,
            max_frames_in_flight: gpu::constants::MAX_FRAMES_IN_FLIGHT,
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
            cascaded_shadow_map,
        })
    }

    pub fn make_2d_combine_shader(gpu: &dyn Gpu) -> anyhow::Result<ShaderModuleHandle> {
        gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Combine shader 2D"),
            code: bytemuck::cast_slice(COMBINE_SHADER_2D),
        })
    }

    pub fn make_3d_combine_shader(gpu: &dyn Gpu) -> anyhow::Result<ShaderModuleHandle> {
        gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("Combine shader 3D"),
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
        resource_map: &AssetMap,
        pipeline_target: PipelineTarget,
        render_pass: &mut RenderPass2,
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
        render_pass: &mut RenderPass2,
        skybox_mesh: &Mesh,
        skybox_material: &MaterialInstance,
        skybox_master: &MasterMaterial,
        resource_map: &AssetMap,
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

    pub fn set_irradiance_texture(&mut self, irradiance_map: Option<AssetHandle<crate::Texture>>) {
        self.irradiance_map = irradiance_map;
    }

    fn main_pass(
        &self,
        gpu: &dyn Gpu,
        graphics_command_buffer: &mut CommandBuffer,
        final_scene_image: &RenderImage,
        gbuffer: &GBuffer,
        resource_map: &AssetMap,
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
            if self.early_z_pass_enabled {
                let mut early_z =
                    graphics_command_buffer.start_render_pass_2(&BeginRenderPassInfo2 {
                        label: Some("Early Z"),
                        color_attachments: &[],
                        depth_attachment: Some(DepthAttachment {
                            image_view: gbuffer.depth_component.view,
                            load_op: gpu::DepthLoadOp::Clear(1.0),
                            store_op: AttachmentStoreOp::Store,
                        }),
                        stencil_attachment: None,
                        render_area: Rect2D {
                            extent: render_size,
                            ..Default::default()
                        },
                    });
                early_z.set_cull_mode(gpu::CullMode::Back);
                early_z.set_depth_compare_op(gpu::CompareOp::LessEqual);

                early_z.set_color_output_enabled(false);
                early_z.set_enable_depth_test(true);
                early_z.set_depth_write_enabled(true);
                Self::main_render_loop(
                    gpu,
                    primitives,
                    resource_map,
                    PipelineTarget::DepthOnly,
                    &mut early_z,
                    0,
                    current_buffers,
                    &self.sampler_allocator,
                )
                .context("Early Z Pass")?;
            }
            {
                let mut gbuffer_output =
                    graphics_command_buffer.start_render_pass_2(&gpu::BeginRenderPassInfo2 {
                        label: Some("GBuffer Output pass"),
                        color_attachments: &[
                            ColorAttachment {
                                image_view: position_component.view,
                                load_op: ColorLoadOp::Clear([0.0; 4]),
                                store_op: AttachmentStoreOp::Store,
                            },
                            ColorAttachment {
                                image_view: normal_component.view,
                                load_op: ColorLoadOp::Clear([0.5; 4]),
                                store_op: AttachmentStoreOp::Store,
                            },
                            ColorAttachment {
                                image_view: diffuse_component.view,
                                load_op: ColorLoadOp::Clear([0.0; 4]),
                                store_op: AttachmentStoreOp::Store,
                            },
                            ColorAttachment {
                                image_view: emissive_component.view,
                                load_op: ColorLoadOp::Clear([0.0; 4]),
                                store_op: AttachmentStoreOp::Store,
                            },
                            ColorAttachment {
                                image_view: pbr_component.view,
                                load_op: ColorLoadOp::Clear([0.0; 4]),
                                store_op: AttachmentStoreOp::Store,
                            },
                        ],
                        depth_attachment: Some(gpu::DepthAttachment {
                            image_view: depth_component.view,
                            load_op: if self.early_z_pass_enabled {
                                gpu::DepthLoadOp::Load
                            } else {
                                gpu::DepthLoadOp::Clear(1.0)
                            },
                            store_op: AttachmentStoreOp::Store,
                        }),
                        stencil_attachment: None,
                        render_area: gpu::Rect2D {
                            offset: gpu::Offset2D::default(),
                            extent: render_size,
                        },
                    });

                if let Some(material) = scene.get_skybox_material() {
                    let skybox_master = resource_map.get(&material.owner);
                    bind_master_material(
                        skybox_master,
                        PipelineTarget::ColorAndDepth,
                        &mut gbuffer_output,
                        current_buffers,
                    );
                    Self::draw_skybox(
                        gpu,
                        &pov.location,
                        &mut gbuffer_output,
                        &self.cube_mesh,
                        material,
                        skybox_master,
                        resource_map,
                        &self.sampler_allocator,
                    )?;
                }

                gbuffer_output.set_front_face(gpu::FrontFace::CounterClockWise);
                gbuffer_output.set_enable_depth_test(true);
                gbuffer_output.set_depth_write_enabled(!self.early_z_pass_enabled);
                gbuffer_output.set_color_output_enabled(true);
                gbuffer_output.set_cull_mode(gpu::CullMode::Back);
                gbuffer_output.set_depth_compare_op(if self.early_z_pass_enabled {
                    gpu::CompareOp::Equal
                } else {
                    gpu::CompareOp::LessEqual
                });
                Self::main_render_loop(
                    gpu,
                    primitives,
                    resource_map,
                    PipelineTarget::ColorAndDepth,
                    &mut gbuffer_output,
                    0,
                    current_buffers,
                    &self.sampler_allocator,
                )
                .context("Gbuffer output pass")?;
            }
            {
                // Combine
                let mut combine_pass =
                    graphics_command_buffer.start_render_pass_2(&gpu::BeginRenderPassInfo2 {
                        label: Some("GBuffer Combine pass"),
                        color_attachments: &[ColorAttachment {
                            image_view: final_scene_image.view,
                            load_op: ColorLoadOp::Clear([0.0; 4]),
                            store_op: AttachmentStoreOp::Store,
                        }],
                        depth_attachment: None,
                        stencil_attachment: None,
                        render_area: gpu::Rect2D {
                            extent: render_size,
                            ..Default::default()
                        },
                    });
                let csm_buffers = &self.cascaded_shadow_map.csm_buffers[self.in_flight_frame];
                gbuffer.bind_as_shader_resource(&mut combine_pass, 0);
                combine_pass.bind_resources_2(
                    1,
                    &[
                        Binding2 {
                            ty: gpu::DescriptorBindingType2::ImageView {
                                image_view_handle: self.cascaded_shadow_map.shadow_atlas_view,
                                sampler_handle: self.gbuffer_nearest_sampler,
                            },
                            write: false,
                        },
                        Binding2 {
                            ty: gpu::DescriptorBindingType2::StorageBuffer {
                                handle: csm_buffers.shadow_casters,
                                offset: 0,
                                range: gpu::WHOLE_SIZE as _,
                            },
                            write: false,
                        },
                        Binding2 {
                            ty: gpu::DescriptorBindingType2::ImageView {
                                image_view_handle: if let Some(irradiance_map) =
                                    &self.irradiance_map
                                {
                                    resource_map.get(irradiance_map)
                                } else {
                                    &self.default_irradiance_map
                                }
                                .view,
                                sampler_handle: self.gbuffer_nearest_sampler,
                            },
                            write: false,
                        },
                        Binding2 {
                            ty: gpu::DescriptorBindingType2::StorageBuffer {
                                handle: current_buffers.camera_buffer,
                                offset: 0,
                                range: gpu::WHOLE_SIZE,
                            },
                            write: false,
                        },
                        Binding2 {
                            ty: gpu::DescriptorBindingType2::StorageBuffer {
                                handle: current_buffers.light_buffer,
                                offset: 0,
                                range: gpu::WHOLE_SIZE,
                            },
                            write: false,
                        },
                        Binding2 {
                            ty: gpu::DescriptorBindingType2::StorageBuffer {
                                handle: csm_buffers.csm_splits,
                                offset: 0,
                                range: gpu::WHOLE_SIZE,
                            },
                            write: false,
                        },
                    ],
                );

                combine_pass.set_front_face(gpu::FrontFace::ClockWise);
                combine_pass.set_cull_mode(gpu::CullMode::None);
                combine_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
                combine_pass.set_enable_depth_test(false);
                combine_pass.set_depth_write_enabled(false);
                combine_pass.set_vertex_shader(self.screen_quad);
                combine_pass.set_fragment_shader(self.combine_shader);
                combine_pass.draw(4, 1, 0, 0)?;
            }
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
        self.draw_textured_quad_with_callback(
            graphics_command_buffer,
            destination,
            source,
            viewport,
            flip_render_target,
            override_shader,
            |_| (),
        )
    }
    pub fn draw_textured_quad_with_callback<F: FnMut(&mut RenderPass2)>(
        &self,
        graphics_command_buffer: &mut CommandBuffer,
        destination: &ImageViewHandle,
        source: &ImageViewHandle,
        viewport: Rect2D,
        flip_render_target: bool,
        override_shader: Option<ShaderModuleHandle>,
        mut callback: F,
    ) -> anyhow::Result<()> {
        let mut present_render_pass =
            graphics_command_buffer.start_render_pass_2(&gpu::BeginRenderPassInfo2 {
                label: Some("Copy to backbuffer"),
                color_attachments: &[ColorAttachment {
                    image_view: *destination,
                    load_op: ColorLoadOp::Clear([0.0; 4]),
                    store_op: AttachmentStoreOp::Store,
                }],
                depth_attachment: None,
                stencil_attachment: None,
                render_area: viewport,
            });
        present_render_pass.bind_resources_2(
            0,
            &[Binding2 {
                ty: gpu::DescriptorBindingType2::ImageView {
                    image_view_handle: *source,
                    sampler_handle: self.gbuffer_nearest_sampler,
                },
                write: false,
            }],
        );

        let screen_quad = if flip_render_target {
            self.screen_quad_flipped
        } else {
            self.screen_quad
        };

        present_render_pass.set_front_face(gpu::FrontFace::ClockWise);
        present_render_pass.set_cull_mode(gpu::CullMode::None);
        present_render_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
        present_render_pass.set_enable_depth_test(false);
        present_render_pass.set_depth_write_enabled(false);
        present_render_pass.set_vertex_shader(screen_quad);
        present_render_pass.set_fragment_shader(override_shader.unwrap_or(self.texture_copy));

        callback(&mut present_render_pass);

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

        let post_process_backbuffer_1 = color_output;
        let post_process_backbuffer_2 = self.image_allocator.get(gpu, "post_process2", &color_desc);
        let mut current_postprocess = 0;

        let final_color_output = {
            let final_color_output = {
                for pass in &self.post_process_stack {
                    let (pass_color_target, previous_pass_result) = if current_postprocess == 0 {
                        (
                            post_process_backbuffer_2.view,
                            &post_process_backbuffer_1.view,
                        )
                    } else {
                        (
                            post_process_backbuffer_1.view,
                            &post_process_backbuffer_2.view,
                        )
                    };

                    let mut post_process_pass =
                        graphics_command_buffer.start_render_pass_2(&BeginRenderPassInfo2 {
                            label: Some("Post Process"),
                            color_attachments: &[ColorAttachment {
                                image_view: pass_color_target,
                                load_op: ColorLoadOp::Clear([0.0; 4]),
                                store_op: AttachmentStoreOp::Store,
                            }],
                            depth_attachment: None,
                            stencil_attachment: None,
                            render_area: Rect2D {
                                offset: Offset2D::default(),
                                extent: render_size,
                            },
                        });
                    post_process_pass.set_cull_mode(gpu::CullMode::None);
                    post_process_pass.set_enable_depth_test(false);
                    post_process_pass.set_depth_write_enabled(false);
                    post_process_pass.set_primitive_topology(gpu::PrimitiveTopology::TriangleStrip);
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
                }

                if current_postprocess == 0 {
                    post_process_backbuffer_1
                } else {
                    post_process_backbuffer_2
                }
            };

            final_color_output
        };
        Ok(final_color_output)
    }

    pub fn add_post_process_pass(&mut self, pass: impl PostProcessPass) {
        self.post_process_stack.push(Box::new(pass))
    }

    pub fn get_shadow_texture(&self, _gpu: &dyn Gpu) -> ImageViewHandle {
        self.cascaded_shadow_map.shadow_atlas_view
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
            gbuffer_sampler: self.gbuffer_nearest_sampler,
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
        self.cascaded_shadow_map.clear();

        let mut pov_idx = 1;

        for (light_id, light) in scene.lights.iter().enumerate() {
            if !light.enabled {
                continue;
            }

            let mut gpu_light: GpuLightInfo = light.into();
            if let Some(NewShadowMapAllocation {
                povs,
                shadow_map_index,
                csm_split,
            }) =
                self.cascaded_shadow_map
                    .add_light(light, scene_camera, pov_idx, light_id as u32)
            {
                pov_idx += povs.len() as u32;
                self.light_povs.extend(povs.into_iter());
                gpu_light.ty_shadow_map_idx_csm_split[1] = shadow_map_index as i32;
                gpu_light.ty_shadow_map_idx_csm_split[2] = csm_split;
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

        self.cascaded_shadow_map
            .update_buffers(gpu, self.in_flight_frame)?;

        Ok(())
    }
}

fn bind_master_material(
    master: &MasterMaterial,
    pipeline_target: PipelineTarget,
    render_pass: &mut RenderPass2,
    frame_buffers: &FrameBuffers,
) {
    let permutation = master
        .get_permutation(pipeline_target)
        .expect("failed to fetch permutation {pipeline_target:?}");
    render_pass.set_vertex_shader(permutation.vertex_shader);
    if let Some(fragment_shader) = &permutation.fragment_shader {
        render_pass.set_fragment_shader(*fragment_shader);
    }
    render_pass.bind_resources_2(
        0,
        &[
            Binding2 {
                ty: gpu::DescriptorBindingType2::StorageBuffer {
                    handle: frame_buffers.camera_buffer,
                    offset: 0,
                    range: gpu::WHOLE_SIZE as _,
                },
                write: false,
            },
            Binding2 {
                ty: gpu::DescriptorBindingType2::UniformBuffer {
                    handle: frame_buffers.light_buffer,
                    offset: 0,
                    range: (100 * size_of::<ObjectDrawInfo>()) as u64,
                },
                write: false,
            },
        ],
    );
}

fn draw_mesh_primitive(
    gpu: &dyn Gpu,
    render_pass: &mut RenderPass2,
    material: &MaterialInstance,
    master: &MasterMaterial,
    primitive: &MeshPrimitive,
    model: Matrix4<f32>,
    resource_map: &AssetMap,
    sampler_allocator: &SamplerAllocator,
    camera_index: u32,
) -> anyhow::Result<()> {
    render_pass.set_index_buffer(primitive.index_buffer, IndexType::Uint32, 0);

    let mut user_bindings = vec![];
    user_bindings.extend(&mut master.texture_inputs.iter().enumerate().map(|(i, _)| {
        let texture_parameter = &material.textures[i];
        let tex = resource_map.get(texture_parameter);

        let sampler_handle = sampler_allocator.get(gpu, &tex.sampler_settings);

        Binding2 {
            ty: gpu::DescriptorBindingType2::ImageView {
                image_view_handle: tex.view,
                sampler_handle,
            },
            write: false,
        }
    }));
    for user_buffer in &material.parameter_buffers {
        user_bindings.push(Binding2 {
            ty: gpu::DescriptorBindingType2::UniformBuffer {
                handle: *user_buffer,
                offset: 0,
                range: gpu::WHOLE_SIZE as _,
            },
            write: false,
        });
    }

    render_pass.set_vertex_buffers(
        &[
            VertexBindingInfo {
                handle: primitive.position_component,
                location: 0,
                offset: 0,
                stride: std::mem::size_of::<Vector3<f32>>() as _,
                format: ImageFormat::RgbFloat32,
                input_rate: InputRate::PerVertex,
            },
            VertexBindingInfo {
                handle: primitive.color_component,
                location: 1,
                offset: 0,
                stride: std::mem::size_of::<Vector3<f32>>() as _,
                format: ImageFormat::RgbFloat32,
                input_rate: InputRate::PerVertex,
            },
            VertexBindingInfo {
                handle: primitive.normal_component,
                location: 2,
                offset: 0,
                stride: std::mem::size_of::<Vector3<f32>>() as _,
                format: ImageFormat::RgbFloat32,
                input_rate: InputRate::PerVertex,
            },
            VertexBindingInfo {
                handle: primitive.tangent_component,
                location: 3,
                offset: 0,
                stride: std::mem::size_of::<Vector3<f32>>() as _,
                format: ImageFormat::RgbFloat32,
                input_rate: InputRate::PerVertex,
            },
            VertexBindingInfo {
                handle: primitive.uv_component,
                location: 4,
                offset: 0,
                stride: std::mem::size_of::<Vector2<f32>>() as _,
                format: ImageFormat::RgFloat32,
                input_rate: InputRate::PerVertex,
            },
        ],
        &[0, 0, 0, 0, 0],
    );
    render_pass.bind_resources_2(1, &user_bindings);
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
        graphics_command_buffer: &mut CommandBuffer,
        camera: &Camera,
        scene: &RenderScene,
        resource_map: &AssetMap,
        cvar_manager: &CvarManager,
    ) -> anyhow::Result<ImageViewHandle> {
        if self.update_frustum {
            self.frustum = camera.frustum();
        }
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

        let primitives = scene.intersect_frustum(&self.frustum);

        if primitives.is_empty() {
            return Ok(color_output.view);
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

        self.cascaded_shadow_map.render_shadow_atlas(
            gpu,
            scene,
            graphics_command_buffer,
            current_buffers,
            resource_map,
            &self.light_povs,
            &self.sampler_allocator,
        )?;

        let gbuffer = self.get_gbuffer(gpu);
        self.drawcalls_last_frame = primitives.len() as u64;
        self.main_pass(
            gpu,
            graphics_command_buffer,
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
                graphics_command_buffer,
                color_output,
                color_desc,
                self.view_size,
                cvar_manager,
            )
            .context("Post process pass")?;

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
                module: material_description.vertex_module,
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

    fn destroy(&mut self, gpu: &dyn Gpu) {
        self.cascaded_shadow_map.destroy(gpu);
        self.default_irradiance_map.destroyed(gpu);
        self.cube_mesh.destroyed(gpu);
        for framebuffer in &self.frame_buffers {
            gpu.destroy_buffer(framebuffer.light_buffer);
            gpu.destroy_buffer(framebuffer.camera_buffer);
        }

        for pass in &self.post_process_stack {
            pass.destroy(gpu);
        }

        self.image_allocator.destroy(gpu);
        self.sampler_allocator.destroy(gpu);
        gpu.destroy_shader_module(self.screen_quad);
        gpu.destroy_shader_module(self.screen_quad_flipped);
        gpu.destroy_shader_module(self.texture_copy);
        gpu.destroy_shader_module(self.combine_shader);
        gpu.destroy_sampler(self.gbuffer_nearest_sampler);
    }
}
