use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use mgpu::{
    Binding, BindingSetDescription, BindingSetElement, BindingSetLayout, BindingSetLayoutInfo,
    ComputePassDescription, ComputePipelineDescription, DepthStencilState, Device, Extents2D,
    Extents3D, FragmentStageInfo, Graphics, GraphicsPipelineDescription, ImageCreationFlags,
    ImageDescription, ImageFormat, ImageUsageFlags, ImageViewDescription, MgpuResult,
    PushConstantInfo, Rect2D, RenderPassDescription, RenderPassFlags, RenderTarget,
    RenderTargetInfo, SampleCount, SamplerDescription, ShaderModuleDescription, ShaderStageFlags,
    VertexStageInfo,
};

use crate::{
    assets::{
        material,
        mesh::Mesh,
        texture::{Texture, TextureSamplerConfiguration},
    },
    include_spirv,
    sampler_allocator::SamplerAllocator,
};

const SIMPLE_VS: &[u8] = include_spirv!("../spirv/simple.vert.spv");
const GEN_CUBEMAP_FS: &[u8] = include_spirv!("../spirv/generate_cubemap.frag.spv");
const GEN_IRRADIANCE_FS: &[u8] = include_spirv!("../spirv/generate_irradiance_map.frag.spv");
const PREFILTER_ENV_FS: &[u8] = include_spirv!("../spirv/prefilter_env_map.frag.spv");
const LUT_SHADER_SOURCE: &[u8] = include_spirv!("../spirv/gen_ibl_env_lut.comp.spv");

pub struct CreateCubemapParams<'a> {
    pub label: Option<&'a str>,
    pub input_texture: &'a Texture,
    pub extents: Extents2D,
    pub mips: NonZeroU32,
    pub format: ImageFormat,
    pub samples: SampleCount,
}

pub fn read_cubemap_from_hdr(
    device: &Device,
    params: &CreateCubemapParams,
    cube_mesh: &Mesh,
    sampler_allocator: &SamplerAllocator,
) -> anyhow::Result<(Texture, Texture, Texture)> {
    #[repr(C)]
    #[derive(Pod, Zeroable, Clone, Copy)]
    struct DrawPushConstant {
        mvp: Mat4,
        v_roughness: [f32; 4],
    }

    let mut all_image_views = vec![];

    let cubemap = device.create_image(&ImageDescription {
        label: params.label,
        usage_flags: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
        creation_flags: ImageCreationFlags::CUBE_COMPATIBLE,
        extents: Extents3D {
            width: params.extents.width,
            height: params.extents.height,
            depth: 1,
        },
        dimension: mgpu::ImageDimension::D2,
        mips: params.mips,
        array_layers: 6.try_into().unwrap(),
        samples: mgpu::SampleCount::One,
        format: params.format,
        memory_domain: mgpu::MemoryDomain::Gpu,
    })?;

    let cubemap_view = device.create_image_view(&ImageViewDescription {
        label: None,
        image: cubemap,
        format: params.format,
        view_ty: mgpu::ImageViewType::Cube,
        aspect: mgpu::ImageAspect::Color,
        image_subresource: cubemap.whole_subresource(),
    })?;

    let cubemap_irradiance = device.create_image(&ImageDescription {
        label: params.label,
        usage_flags: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
        creation_flags: ImageCreationFlags::CUBE_COMPATIBLE,
        extents: Extents3D {
            width: params.extents.width,
            height: params.extents.height,
            depth: 1,
        },
        dimension: mgpu::ImageDimension::D2,
        mips: params.mips,
        array_layers: 6.try_into().unwrap(),
        samples: mgpu::SampleCount::One,
        format: params.format,
        memory_domain: mgpu::MemoryDomain::Gpu,
    })?;

    let cubemap_irradiance_view = device.create_image_view(&ImageViewDescription {
        label: None,
        image: cubemap_irradiance,
        format: params.format,
        view_ty: mgpu::ImageViewType::Cube,
        aspect: mgpu::ImageAspect::Color,
        image_subresource: cubemap_irradiance.whole_subresource(),
    })?;

    let image_slices_irradiance = (0..6)
        .map(|layer| {
            device.create_image_view(&ImageViewDescription {
                label: None,
                image: cubemap_irradiance,
                format: params.format,
                view_ty: mgpu::ImageViewType::D2,
                aspect: mgpu::ImageAspect::Color,
                image_subresource: cubemap_irradiance.layer(0, layer),
            })
        })
        .collect::<MgpuResult<Vec<_>>>()?;
    all_image_views.extend(image_slices_irradiance.iter().cloned());

    let mips = Texture::compute_num_mips(params.extents.width, params.extents.height);
    let cubemap_diffuse = device.create_image(&ImageDescription {
        label: params.label,
        usage_flags: ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::SAMPLED
            | ImageUsageFlags::TRANSFER_DST
            | ImageUsageFlags::TRANSFER_SRC,
        creation_flags: ImageCreationFlags::CUBE_COMPATIBLE,
        extents: Extents3D {
            width: params.extents.width,
            height: params.extents.height,
            depth: 1,
        },
        dimension: mgpu::ImageDimension::D2,
        mips: mips.try_into().unwrap(),
        array_layers: 6.try_into().unwrap(),
        samples: mgpu::SampleCount::One,
        format: params.format,
        memory_domain: mgpu::MemoryDomain::Gpu,
    })?;

    let cubemap_diffuse_view = device.create_image_view(&ImageViewDescription {
        label: None,
        image: cubemap_diffuse,
        format: params.format,
        view_ty: mgpu::ImageViewType::Cube,
        aspect: mgpu::ImageAspect::Color,
        image_subresource: cubemap_diffuse.whole_subresource(),
    })?;
    let image_slices_views_cubemap = (0..6)
        .map(|layer| {
            device.create_image_view(&ImageViewDescription {
                label: None,
                image: cubemap,
                format: params.format,
                view_ty: mgpu::ImageViewType::D2,
                aspect: mgpu::ImageAspect::Color,
                image_subresource: cubemap.layer(0, layer),
            })
        })
        .collect::<MgpuResult<Vec<_>>>()?;

    all_image_views.extend(image_slices_views_cubemap.iter().cloned());

    let vertex_shader = device.create_shader_module(&ShaderModuleDescription {
        label: None,
        source: bytemuck::cast_slice(SIMPLE_VS),
    })?;

    let fragment_shader = device.create_shader_module(&ShaderModuleDescription {
        label: Some("GenCubemapFS"),
        source: bytemuck::cast_slice(GEN_CUBEMAP_FS),
    })?;

    let irradiance_shader = device.create_shader_module(&ShaderModuleDescription {
        label: Some("IrradianceMapFS"),
        source: bytemuck::cast_slice(GEN_IRRADIANCE_FS),
    })?;

    let prefilter_diffuse = device.create_shader_module(&ShaderModuleDescription {
        label: Some("PrefilterEnvMap"),
        source: bytemuck::cast_slice(PREFILTER_ENV_FS),
    })?;

    let sampler = device.create_sampler(&SamplerDescription {
        label: None,
        mag_filter: mgpu::FilterMode::Linear,
        min_filter: mgpu::FilterMode::Linear,
        mipmap_mode: mgpu::MipmapMode::Linear,
        address_mode_u: mgpu::AddressMode::ClampToEdge,
        address_mode_v: mgpu::AddressMode::ClampToEdge,
        address_mode_w: mgpu::AddressMode::ClampToEdge,
        lod_bias: 0.0,
        compare_op: None,
        min_lod: 0.0,
        max_lod: f32::MAX,
        border_color: Default::default(),
        unnormalized_coordinates: false,
    })?;

    let binding_set_layout = BindingSetLayout {
        binding_set_elements: &[
            BindingSetElement {
                binding: 0,
                array_length: 1,
                ty: mgpu::BindingSetElementKind::SampledImage,
                shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
            },
            BindingSetElement {
                binding: 1,
                array_length: 1,
                ty: mgpu::BindingSetElementKind::Sampler,
                shader_stage_flags: ShaderStageFlags::ALL_GRAPHICS,
            },
        ],
    };

    let bs_cubegen = device.create_binding_set(
        &BindingSetDescription {
            label: None,
            bindings: &[
                Binding {
                    binding: 0,
                    ty: mgpu::BindingType::SampledImage {
                        view: params.input_texture.view,
                    },
                    visibility: ShaderStageFlags::ALL_GRAPHICS,
                },
                Binding {
                    binding: 1,
                    ty: mgpu::BindingType::Sampler(sampler),
                    visibility: ShaderStageFlags::ALL_GRAPHICS,
                },
            ],
        },
        &binding_set_layout,
    )?;

    let bs_prefilter = device.create_binding_set(
        &BindingSetDescription {
            label: None,
            bindings: &[
                Binding {
                    binding: 0,
                    ty: mgpu::BindingType::SampledImage { view: cubemap_view },
                    visibility: ShaderStageFlags::ALL_GRAPHICS,
                },
                Binding {
                    binding: 1,
                    ty: mgpu::BindingType::Sampler(sampler),
                    visibility: ShaderStageFlags::ALL_GRAPHICS,
                },
            ],
        },
        &binding_set_layout,
    )?;

    let pipeline_cubegen = device.create_graphics_pipeline(&GraphicsPipelineDescription {
        label: None,
        vertex_stage: &VertexStageInfo {
            shader: &vertex_shader,
            entry_point: "main",
            vertex_inputs: material::mesh_vertex_inputs(),
        },
        fragment_stage: Some(&FragmentStageInfo {
            shader: &fragment_shader,
            entry_point: "main",
            render_targets: &[RenderTargetInfo { blend: None }],
            depth_stencil_target: None,
        }),
        primitive_restart_enabled: false,
        primitive_topology: mgpu::PrimitiveTopology::TriangleList,
        polygon_mode: mgpu::PolygonMode::Filled,
        cull_mode: mgpu::CullMode::None,
        front_face: mgpu::FrontFace::ClockWise,
        multisample_state: None,
        depth_stencil_state: DepthStencilState {
            depth_test_enabled: false,
            depth_write_enabled: false,
            depth_compare_op: mgpu::CompareOp::Always,
        },
        binding_set_layouts: &[BindingSetLayoutInfo {
            set: 0,
            layout: &binding_set_layout,
        }],
        push_constant_info: Some(PushConstantInfo {
            size: std::mem::size_of::<DrawPushConstant>(),
            visibility: ShaderStageFlags::ALL_GRAPHICS,
        }),
    })?;
    let pipeline_irradiance = device.create_graphics_pipeline(&GraphicsPipelineDescription {
        label: None,
        vertex_stage: &VertexStageInfo {
            shader: &vertex_shader,
            entry_point: "main",
            vertex_inputs: material::mesh_vertex_inputs(),
        },
        fragment_stage: Some(&FragmentStageInfo {
            shader: &irradiance_shader,
            entry_point: "main",
            render_targets: &[RenderTargetInfo { blend: None }],
            depth_stencil_target: None,
        }),
        primitive_restart_enabled: false,
        primitive_topology: mgpu::PrimitiveTopology::TriangleList,
        polygon_mode: mgpu::PolygonMode::Filled,
        cull_mode: mgpu::CullMode::None,
        front_face: mgpu::FrontFace::ClockWise,
        multisample_state: None,
        depth_stencil_state: DepthStencilState {
            depth_test_enabled: false,
            depth_write_enabled: false,
            depth_compare_op: mgpu::CompareOp::Always,
        },
        binding_set_layouts: &[BindingSetLayoutInfo {
            set: 0,
            layout: &binding_set_layout,
        }],
        push_constant_info: Some(PushConstantInfo {
            size: std::mem::size_of::<DrawPushConstant>(),
            visibility: ShaderStageFlags::ALL_GRAPHICS,
        }),
    })?;
    let pipeline_prefilter = device.create_graphics_pipeline(&GraphicsPipelineDescription {
        label: None,
        vertex_stage: &VertexStageInfo {
            shader: &vertex_shader,
            entry_point: "main",
            vertex_inputs: material::mesh_vertex_inputs(),
        },
        fragment_stage: Some(&FragmentStageInfo {
            shader: &prefilter_diffuse,
            entry_point: "main",
            render_targets: &[RenderTargetInfo { blend: None }],
            depth_stencil_target: None,
        }),
        primitive_restart_enabled: false,
        primitive_topology: mgpu::PrimitiveTopology::TriangleList,
        polygon_mode: mgpu::PolygonMode::Filled,
        cull_mode: mgpu::CullMode::None,
        front_face: mgpu::FrontFace::ClockWise,
        multisample_state: None,
        depth_stencil_state: DepthStencilState {
            depth_test_enabled: false,
            depth_write_enabled: false,
            depth_compare_op: mgpu::CompareOp::Always,
        },
        binding_set_layouts: &[BindingSetLayoutInfo {
            set: 0,
            layout: &binding_set_layout,
        }],
        push_constant_info: Some(PushConstantInfo {
            size: std::mem::size_of::<DrawPushConstant>(),
            visibility: ShaderStageFlags::ALL_GRAPHICS,
        }),
    })?;

    let mut command_recorder = device.create_command_recorder::<Graphics>();

    let projection = Mat4::perspective_rh(90.0f32.to_radians(), 1.0, 0.01, 100.0);

    #[allow(clippy::needless_range_loop)]
    for i in 0..6 {
        let target = image_slices_views_cubemap[i];
        let mut render_pass = command_recorder.begin_render_pass(&RenderPassDescription {
            label: Some(&format!("Render cube face {i}")),
            flags: RenderPassFlags::default(),
            render_targets: &[RenderTarget {
                view: target,
                sample_count: SampleCount::One,
                load_op: mgpu::RenderTargetLoadOp::DontCare,
                store_op: mgpu::AttachmentStoreOp::Store,
            }],
            depth_stencil_attachment: None,
            render_area: Rect2D {
                offset: Default::default(),
                extents: target.extents_2d(),
            },
        })?;

        let view = match i {
            0 => Mat4::look_to_rh(Vec3::ZERO, Vec3::X, Vec3::Y),
            1 => Mat4::look_to_rh(Vec3::ZERO, -Vec3::X, Vec3::Y),
            2 => Mat4::look_to_rh(Vec3::ZERO, Vec3::Y, Vec3::Z),
            3 => Mat4::look_to_rh(Vec3::ZERO, -Vec3::Y, -Vec3::Z),
            4 => Mat4::look_to_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y),
            5 => Mat4::look_to_rh(Vec3::ZERO, Vec3::Z, Vec3::Y),
            _ => unreachable!(),
        };

        let mvp = projection * view;

        render_pass.set_pipeline(pipeline_cubegen);
        render_pass.set_binding_sets(&[&bs_cubegen]);
        render_pass.set_index_buffer(cube_mesh.index_buffer);
        render_pass.set_vertex_buffers([
            cube_mesh.position_component,
            cube_mesh.normal_component,
            cube_mesh.tangent_component,
            cube_mesh.color_component,
            cube_mesh.uv_component,
        ]);
        render_pass.set_push_constant(
            bytemuck::cast_slice(&[DrawPushConstant {
                mvp,
                v_roughness: [1.0; 4],
            }]),
            ShaderStageFlags::ALL_GRAPHICS,
        );
        render_pass.draw_indexed(cube_mesh.info.num_indices, 1, 0, 0, 0)?;
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..6 {
        let target = image_slices_irradiance[i];
        let mut render_pass = command_recorder.begin_render_pass(&RenderPassDescription {
            label: Some(&format!("Render irradiance slice {i}")),
            flags: RenderPassFlags::default(),
            render_targets: &[RenderTarget {
                view: target,
                sample_count: SampleCount::One,
                load_op: mgpu::RenderTargetLoadOp::DontCare,
                store_op: mgpu::AttachmentStoreOp::Store,
            }],
            depth_stencil_attachment: None,
            render_area: Rect2D {
                offset: Default::default(),
                extents: target.extents_2d(),
            },
        })?;

        let view = match i {
            0 => Mat4::look_to_rh(Vec3::ZERO, Vec3::X, Vec3::Y),
            1 => Mat4::look_to_rh(Vec3::ZERO, -Vec3::X, Vec3::Y),
            2 => Mat4::look_to_rh(Vec3::ZERO, Vec3::Y, Vec3::Z),
            3 => Mat4::look_to_rh(Vec3::ZERO, -Vec3::Y, -Vec3::Z),
            4 => Mat4::look_to_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y),
            5 => Mat4::look_to_rh(Vec3::ZERO, Vec3::Z, Vec3::Y),
            _ => unreachable!(),
        };

        let mvp = projection * view;

        render_pass.set_pipeline(pipeline_irradiance);
        render_pass.set_binding_sets(&[&bs_prefilter]);
        render_pass.set_index_buffer(cube_mesh.index_buffer);
        render_pass.set_vertex_buffers([
            cube_mesh.position_component,
            cube_mesh.normal_component,
            cube_mesh.tangent_component,
            cube_mesh.color_component,
            cube_mesh.uv_component,
        ]);
        render_pass.set_push_constant(
            bytemuck::cast_slice(&[DrawPushConstant {
                mvp,
                v_roughness: [1.0; 4],
            }]),
            ShaderStageFlags::ALL_GRAPHICS,
        );
        render_pass.draw_indexed(cube_mesh.info.num_indices, 1, 0, 0, 0)?;
    }

    for mip in 0..cubemap_diffuse.mips() {
        let image_slices_views_cubemap_diffuse = (0..6)
            .map(|layer| {
                device.create_image_view(&ImageViewDescription {
                    label: None,
                    image: cubemap_diffuse,
                    format: params.format,
                    view_ty: mgpu::ImageViewType::D2,
                    aspect: mgpu::ImageAspect::Color,
                    image_subresource: cubemap_diffuse.layer(mip, layer),
                })
            })
            .collect::<MgpuResult<Vec<_>>>()?;

        all_image_views.extend(image_slices_views_cubemap_diffuse.iter().cloned());
        #[allow(clippy::needless_range_loop)]
        for i in 0..6 {
            let target = image_slices_views_cubemap_diffuse[i];
            let mut render_pass = command_recorder.begin_render_pass(&RenderPassDescription {
                label: Some(&format!("Render cube diffuse {i}")),
                flags: RenderPassFlags::default(),
                render_targets: &[RenderTarget {
                    view: target,
                    sample_count: SampleCount::One,
                    load_op: mgpu::RenderTargetLoadOp::DontCare,
                    store_op: mgpu::AttachmentStoreOp::Store,
                }],
                depth_stencil_attachment: None,
                render_area: Rect2D {
                    offset: Default::default(),
                    extents: target.extents_2d(),
                },
            })?;

            let (view, vec) = match i {
                0 => (Mat4::look_to_rh(Vec3::ZERO, Vec3::X, Vec3::Y), Vec3::X),
                1 => (Mat4::look_to_rh(Vec3::ZERO, -Vec3::X, Vec3::Y), -Vec3::X),
                2 => (Mat4::look_to_rh(Vec3::ZERO, Vec3::Y, Vec3::Z), Vec3::Y),
                3 => (Mat4::look_to_rh(Vec3::ZERO, -Vec3::Y, -Vec3::Z), -Vec3::Y),
                4 => (Mat4::look_to_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y), -Vec3::Z),
                5 => (Mat4::look_to_rh(Vec3::ZERO, Vec3::Z, Vec3::Y), Vec3::Z),
                _ => unreachable!(),
            };

            let roughness = (mip as f32) / cubemap_diffuse.mips() as f32;

            let mvp = projection * view;

            render_pass.set_pipeline(pipeline_prefilter);
            render_pass.set_binding_sets(&[&bs_prefilter]);
            render_pass.set_index_buffer(cube_mesh.index_buffer);
            render_pass.set_vertex_buffers([
                cube_mesh.position_component,
                cube_mesh.normal_component,
                cube_mesh.tangent_component,
                cube_mesh.color_component,
                cube_mesh.uv_component,
            ]);
            render_pass.set_push_constant(
                bytemuck::cast_slice(&[DrawPushConstant {
                    mvp,
                    v_roughness: [vec.x, vec.y, vec.z, roughness],
                }]),
                ShaderStageFlags::ALL_GRAPHICS,
            );
            render_pass.draw_indexed(cube_mesh.info.num_indices, 1, 0, 0, 0)?;
        }
    }
    command_recorder.submit()?;
    device.submit()?;
    // device.generate_mip_chain(cubemap_diffuse, mgpu::FilterMode::Linear)?;

    device.destroy_graphics_pipeline(pipeline_cubegen)?;
    device.destroy_graphics_pipeline(pipeline_irradiance)?;
    device.destroy_graphics_pipeline(pipeline_prefilter)?;
    device.destroy_binding_set(bs_cubegen)?;
    device.destroy_binding_set(bs_prefilter)?;
    device.destroy_sampler(sampler)?;
    device.destroy_shader_module(fragment_shader)?;
    device.destroy_shader_module(irradiance_shader)?;
    device.destroy_shader_module(vertex_shader)?;

    for image_slice in all_image_views {
        device.destroy_image_view(image_slice)?;
    }

    Ok((
        Texture {
            image: cubemap,
            view: cubemap_view,
            sampler_configuration: TextureSamplerConfiguration::default(),
            sampler: sampler_allocator.get(device, &TextureSamplerConfiguration::default()),
        },
        Texture {
            image: cubemap_irradiance,
            view: cubemap_irradiance_view,
            sampler_configuration: TextureSamplerConfiguration::default(),
            sampler: sampler_allocator.get(device, &TextureSamplerConfiguration::default()),
        },
        Texture {
            image: cubemap_diffuse,
            view: cubemap_diffuse_view,
            sampler_configuration: TextureSamplerConfiguration::default(),
            sampler: sampler_allocator.get(device, &TextureSamplerConfiguration::default()),
        },
    ))
}

pub fn generate_ibl_lut(
    device: &Device,
    sampler_allocator: &SamplerAllocator,
) -> anyhow::Result<Texture> {
    let image = device.create_image(&ImageDescription {
        label: Some("BRDF Env LUT"),
        usage_flags: ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE,
        creation_flags: ImageCreationFlags::default(),
        extents: Extents3D {
            width: 128,
            height: 128,
            depth: 1,
        },
        dimension: mgpu::ImageDimension::D2,
        mips: 1.try_into().unwrap(),
        array_layers: 1.try_into().unwrap(),
        samples: SampleCount::One,
        format: ImageFormat::Rg16f,
        memory_domain: mgpu::MemoryDomain::Gpu,
    })?;
    let view = device.create_image_view(&ImageViewDescription {
        label: Some("BRDF Env LUT View"),
        image,
        format: image.format(),
        view_ty: mgpu::ImageViewType::D2,
        aspect: mgpu::ImageAspect::Color,
        image_subresource: image.whole_subresource(),
    })?;

    let layout = BindingSetLayout {
        binding_set_elements: &[BindingSetElement {
            binding: 0,
            array_length: 1,
            ty: mgpu::BindingSetElementKind::StorageImage {
                format: image.format(),
                access_mode: mgpu::StorageAccessMode::ReadWrite,
            },
            shader_stage_flags: ShaderStageFlags::COMPUTE,
        }],
    };

    let binding_set = device.create_binding_set(
        &BindingSetDescription {
            label: Some("LUT Gen BS"),
            bindings: &[Binding {
                binding: 0,
                ty: mgpu::BindingType::StorageImage {
                    view,
                    access_mode: mgpu::StorageAccessMode::ReadWrite,
                },
                visibility: ShaderStageFlags::COMPUTE,
            }],
        },
        &layout,
    )?;

    let lut_shader = device.create_shader_module(&ShaderModuleDescription {
        label: Some("LUT shader"),
        source: bytemuck::cast_slice(LUT_SHADER_SOURCE),
    })?;

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescription {
        label: Some("Gen LUT"),
        shader: lut_shader,
        entry_point: "main",
        binding_set_layouts: &[BindingSetLayoutInfo {
            set: 0,
            layout: &layout,
        }],
        push_constant_info: None,
    })?;

    let mut command_recorder = device.create_command_recorder::<Graphics>();
    {
        let mut pass = command_recorder.begin_compute_pass(&ComputePassDescription {
            label: Some("Generate LUT"),
        });

        let dispatch_x = image.extents().width / 16;
        let dispatch_y = image.extents().height / 8;

        pass.set_binding_sets(&[&binding_set]);
        pass.set_pipeline(pipeline);
        pass.dispatch(dispatch_x, dispatch_y, 1)?;
    }

    command_recorder.submit()?;

    device.submit()?;

    device.destroy_compute_pipeline(pipeline)?;
    device.destroy_shader_module(lut_shader)?;
    device.destroy_binding_set(binding_set)?;
    Ok(Texture {
        view,
        image,
        sampler_configuration: TextureSamplerConfiguration::default(),
        sampler: sampler_allocator.get(device, &TextureSamplerConfiguration::default()),
    })
}
