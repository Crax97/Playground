#![allow(dead_code)]
use anyhow::bail;
use engine::{Mesh, MeshPrimitiveCreateInfo, Texture};
use image::DynamicImage;
use log::{debug, info};
use nalgebra::{point, vector, Vector2, Vector3};
use resource_map::{ResourceHandle, ResourceMap};
use std::path::Path;

use half::f16;

pub struct LoadedImage {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

use gpu::{
    AccessFlags, AttachmentReference, BeginRenderPassInfo, Binding, ComponentMapping, Extent2D,
    FramebufferColorAttachment, Gpu, ImageAspectFlags, ImageCreateInfo, ImageFormat, ImageHandle,
    ImageSubresourceRange, ImageUsageFlags, ImageViewCreateInfo, ImageViewHandle, InputRate,
    MemoryDomain, Offset2D, PipelineStageFlags, Rect2D, SamplerCreateInfo, ShaderModuleCreateInfo,
    ShaderModuleHandle, ShaderStage, SubpassDescription, VertexBindingInfo, VkGpu,
};

pub fn read_file_to_vk_module<P: AsRef<Path>>(
    gpu: &VkGpu,
    path: P,
) -> anyhow::Result<ShaderModuleHandle> {
    info!("Reading path from {:?}", path.as_ref());
    let input_file = std::fs::read(path)?;
    let create_info = ShaderModuleCreateInfo { code: &input_file };
    Ok(gpu.make_shader_module(&create_info)?)
}

pub fn read_file_to_shader_module<P: AsRef<Path>>(
    gpu: &VkGpu,
    path: P,
) -> anyhow::Result<ShaderModuleHandle> {
    info!(
        "Reading path from {:?}",
        path.as_ref()
            .canonicalize()
            .expect("Failed to canonicalize path")
    );
    let input_file = std::fs::read(path)?;
    let create_info = ShaderModuleCreateInfo { code: &input_file };
    Ok(gpu.make_shader_module(&create_info)?)
}

pub fn load_image_from_path<P: AsRef<Path>>(
    path: P,
    target_format: ImageFormat,
) -> anyhow::Result<LoadedImage> {
    let image_data = std::fs::read(path)?;
    let image = image::load_from_memory(&image_data)?;
    let (width, height) = (image.width(), image.height());
    let bytes = match target_format {
        ImageFormat::Rgba8 => DynamicImage::ImageRgba8(image.to_rgba8()).into_bytes(),
        ImageFormat::Rgb8 => DynamicImage::ImageRgb8(image.to_rgb8()).into_bytes(),
        ImageFormat::RgbaFloat32 => DynamicImage::ImageRgba32F(image.to_rgba32f()).into_bytes(),
        ImageFormat::RgbaFloat16 => {
            let image = image.into_rgba8().into_vec();
            let image = image.into_iter().map(|px| {
                let r: f16 = f16::from_f32((px as f32 + 0.5) / 255.0);
                r
            });

            let pixels = image.collect::<Vec<_>>();
            let pixels = bytemuck::cast_slice::<f16, u8>(&pixels);
            pixels.to_vec()
        }
        _ => anyhow::bail!(format!("Format not supported: {target_format:?}")),
    };

    Ok(LoadedImage {
        bytes,
        width,
        height,
    })
}

// Loads 6 images from a folders and turns them into a cubemap texture
// Searches for "left", "right", "up", "down", "front", "back" in the path
pub fn load_cubemap_from_path<P: AsRef<Path>>(
    gpu: &VkGpu,
    path: P,
    extension: &str,
    target_format: ImageFormat,
    resource_map: &mut resource_map::ResourceMap,
) -> anyhow::Result<Texture> {
    let images = ["posx", "negx", "posy", "negy", "posz", "negz"];

    let mut loaded_images = vec![];

    for image in images {
        let path = path.as_ref().join(image.to_string() + extension);
        debug!("Loading cubemap image {:?}", &path);
        let dyn_image = load_image_from_path(path, target_format)?;
        loaded_images.push(dyn_image);
    }

    let width = loaded_images[0].width;
    let height = loaded_images[0].height;

    info!("Cubemap is expected to be {width}x{height}");

    let mut accumulated_bytes = vec![];

    for image in loaded_images {
        if image.width != width || image.height != height {
            bail!(format!(
                "Images aren't of the same size! Found {}x{}, expected {}x{}",
                image.width, image.height, width, height
            ));
        }
        accumulated_bytes.extend(image.bytes.into_iter());
    }

    Texture::new_with_data(
        gpu,
        resource_map,
        width,
        height,
        &accumulated_bytes,
        Some("Cubemap"),
        target_format,
        gpu::ImageViewType::Cube,
    )
}

pub fn load_hdr_to_cubemap<P: AsRef<Path>>(
    gpu: &VkGpu,
    output_size: Extent2D,
    cube_mesh: ResourceHandle<Mesh>,
    resource_map: &mut ResourceMap,
    path: P,
) -> anyhow::Result<Texture> {
    let size = output_size;
    let cube_image_format = ImageFormat::RgbaFloat16;
    let hdr_image = load_image_from_path(path, cube_image_format)?;
    let hdr_texture = gpu.make_image(
        &ImageCreateInfo {
            label: None,
            width: hdr_image.width,
            height: hdr_image.height,
            depth: 1,
            mips: 1,
            layers: 1,
            samples: gpu::SampleCount::Sample1,
            format: cube_image_format,
            usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
        },
        MemoryDomain::DeviceLocal,
        Some(&hdr_image.bytes),
    )?;
    let hdr_texture_view = gpu.make_image_view(&ImageViewCreateInfo {
        image: hdr_texture,
        view_type: gpu::ImageViewType::Type2D,
        format: cube_image_format,
        components: ComponentMapping::default(),
        subresource_range: ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    })?;

    let equilateral_fragment = read_file_to_vk_module(&gpu, "./shaders/skybox_spherical.spirv")?;
    let (backing_image, view) = cubemap_main_loop(
        gpu,
        cube_image_format,
        &hdr_texture_view,
        equilateral_fragment,
        size,
        resource_map,
        &cube_mesh,
    )?;

    Texture::wrap(gpu, backing_image, view, resource_map)
}

fn cubemap_main_loop(
    gpu: &VkGpu,
    cube_image_format: ImageFormat,
    input_texture_view: &ImageViewHandle,
    fragment_shader_to_apply: ShaderModuleHandle,
    size: Extent2D,
    resource_map: &ResourceMap,
    cube_mesh: &ResourceHandle<Mesh>,
) -> Result<(ImageHandle, ImageViewHandle), anyhow::Error> {
    let vertex_module = read_file_to_vk_module(&gpu, "./shaders/vertex_simple.spirv")?;
    let backing_image = gpu.make_image(
        &ImageCreateInfo {
            label: Some("Cubemap"),
            width: size.width,
            height: size.height,
            depth: 1,
            mips: 1,
            layers: 6,
            samples: gpu::SampleCount::Sample1,
            format: cube_image_format,
            usage: ImageUsageFlags::SAMPLED
                | ImageUsageFlags::TRANSFER_DST
                | ImageUsageFlags::COLOR_ATTACHMENT,
        },
        MemoryDomain::DeviceLocal,
        None,
    )?;
    let skybox_sampler = gpu.make_sampler(&SamplerCreateInfo {
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

    let make_image_view = |i| {
        gpu.make_image_view(&gpu::ImageViewCreateInfo {
            image: backing_image.clone(),
            view_type: gpu::ImageViewType::Type2D,
            format: cube_image_format,
            components: gpu::ComponentMapping::default(),
            subresource_range: gpu::ImageSubresourceRange {
                base_array_layer: i,
                layer_count: 1,
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
            },
        })
        .expect("Failed to create image view")
    };
    let make_pov = |forward: Vector3<f32>, up| engine::PerFrameData {
        eye: point![0.0, 0.0, 0.0, 0.0],
        eye_forward: vector![forward.x, forward.y, forward.z, 0.0],
        view: nalgebra::Matrix4::look_at_rh(
            &point![0.0, 0.0, 0.0],
            &point![forward.x, forward.y, forward.z],
            &up,
        ),
        projection: nalgebra::Matrix4::new_perspective(1.0, 90.0f32.to_radians(), 0.001, 1000.0),
        viewport_size_offset: vector![size.width as f32, size.height as f32, 0.0, 0.0],
    };
    let views = [
        make_image_view(0),
        make_image_view(1),
        make_image_view(2),
        make_image_view(3),
        make_image_view(4),
        make_image_view(5),
    ];
    let povs = [
        make_pov(vector![-1.0, 0.0, 0.0], vector![0.0, 1.0, 0.0]),
        make_pov(vector![1.0, 0.0, 0.0], vector![0.0, 1.0, 0.0]),
        make_pov(vector![0.0, -1.0, 0.0], vector![0.0, 0.0, 1.0]),
        make_pov(vector![0.0, 1.0, 0.0], vector![0.0, 0.0, -1.0]),
        make_pov(vector![0.0, 0.0, 1.0], vector![0.0, 1.0, 0.0]),
        make_pov(vector![0.0, 0.0, -1.0], vector![0.0, 1.0, 0.0]),
    ];
    gpu.transition_image_layout(
        &backing_image,
        gpu::TransitionInfo {
            layout: gpu::ImageLayout::Undefined,
            access_mask: AccessFlags::empty(),
            stage_mask: PipelineStageFlags::TOP_OF_PIPE,
        },
        gpu::TransitionInfo {
            layout: gpu::ImageLayout::ColorAttachment,
            access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
            stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        },
        ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        },
    )?;
    let mesh = resource_map.get(&cube_mesh);
    for (i, view) in views.iter().enumerate() {
        let mvp = povs[i].projection * povs[i].view;
        let views = vec![FramebufferColorAttachment {
            image_view: view.clone(),
            load_op: gpu::ColorLoadOp::DontCare,
            store_op: gpu::AttachmentStoreOp::Store,
            initial_layout: gpu::ImageLayout::ColorAttachment,
            final_layout: gpu::ImageLayout::ColorAttachment,
        }];
        let mut command_buffer = gpu.create_command_buffer(gpu::QueueType::Graphics)?;
        {
            let mut render_pass_command = command_buffer.begin_render_pass(&BeginRenderPassInfo {
                color_attachments: &views,
                depth_attachment: None,
                stencil_attachment: None,
                render_area: Rect2D {
                    offset: Offset2D::default(),
                    extent: size,
                },
                label: Some("Cubemap main loop"),
                subpasses: &[SubpassDescription {
                    label: None,
                    input_attachments: vec![],
                    color_attachments: vec![AttachmentReference {
                        attachment: 0,
                        layout: gpu::ImageLayout::ColorAttachment,
                    }],
                    resolve_attachments: vec![],
                    depth_stencil_attachment: None,
                    preserve_attachments: vec![],
                }],
                dependencies: &[],
            });
            render_pass_command.set_vertex_shader(vertex_module.clone());
            render_pass_command.set_fragment_shader(fragment_shader_to_apply.clone());

            render_pass_command.set_cull_mode(gpu::CullMode::None);
            render_pass_command.set_viewport(gpu::Viewport {
                x: 0.0,
                y: 0.0,
                width: size.width as f32,
                height: size.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            });

            render_pass_command.bind_resources(
                0,
                &[Binding {
                    ty: gpu::DescriptorBindingType::ImageView {
                        image_view_handle: input_texture_view.clone(),
                        sampler_handle: skybox_sampler.clone(),
                        layout: gpu::ImageLayout::ShaderReadOnly,
                    },
                    binding_stage: ShaderStage::FRAGMENT,
                    location: 0,
                }],
            );

            render_pass_command.set_index_buffer(
                mesh.primitives[0].index_buffer.clone(),
                gpu::IndexType::Uint32,
                0,
            );
            render_pass_command.set_vertex_buffers(&[
                VertexBindingInfo {
                    handle: mesh.primitives[0].position_component.clone(),
                    location: 0,
                    offset: 0,
                    stride: std::mem::size_of::<Vector3<f32>>() as _,
                    format: ImageFormat::RgbFloat32,
                    input_rate: InputRate::PerVertex,
                },
                VertexBindingInfo {
                    handle: mesh.primitives[0].color_component.clone(),
                    location: 1,
                    offset: 0,
                    stride: std::mem::size_of::<Vector3<f32>>() as _,
                    format: ImageFormat::RgbFloat32,
                    input_rate: InputRate::PerVertex,
                },
                VertexBindingInfo {
                    handle: mesh.primitives[0].normal_component.clone(),
                    location: 2,
                    offset: 0,
                    stride: std::mem::size_of::<Vector3<f32>>() as _,
                    format: ImageFormat::RgbFloat32,
                    input_rate: InputRate::PerVertex,
                },
                VertexBindingInfo {
                    handle: mesh.primitives[0].tangent_component.clone(),
                    location: 3,
                    offset: 0,
                    stride: std::mem::size_of::<Vector3<f32>>() as _,
                    format: ImageFormat::RgbFloat32,
                    input_rate: InputRate::PerVertex,
                },
                VertexBindingInfo {
                    handle: mesh.primitives[0].uv_component.clone(),
                    location: 4,
                    offset: 0,
                    stride: std::mem::size_of::<Vector2<f32>>() as _,
                    format: ImageFormat::RgFloat32,
                    input_rate: InputRate::PerVertex,
                },
            ]);
            render_pass_command.push_constants(
                0,
                0,
                engine::to_u8_slice(&[mvp]),
                ShaderStage::ALL_GRAPHICS,
            );
            render_pass_command.draw_indexed(mesh.primitives[0].index_count, 1, 0, 0, 0);
        }
        command_buffer.submit(&gpu::CommandBufferSubmitInfo {
            wait_semaphores: &[],
            wait_stages: &[],
            signal_semaphores: &[],
            fence: None,
        })?;
        gpu.wait_device_idle()?;
    }
    gpu.wait_device_idle()?;
    let view = gpu.make_image_view(&gpu::ImageViewCreateInfo {
        image: backing_image.clone(),
        view_type: gpu::ImageViewType::Cube,
        format: cube_image_format,
        components: gpu::ComponentMapping::default(),
        subresource_range: gpu::ImageSubresourceRange {
            base_array_layer: 0,
            layer_count: 6,
            aspect_mask: ImageAspectFlags::COLOR,
            level_count: 1,
            base_mip_level: 0,
        },
    })?;
    gpu.transition_image_layout(
        &backing_image,
        gpu::TransitionInfo {
            layout: gpu::ImageLayout::ColorAttachment,
            access_mask: AccessFlags::empty(),
            stage_mask: PipelineStageFlags::TOP_OF_PIPE,
        },
        gpu::TransitionInfo {
            layout: gpu::ImageLayout::ShaderReadOnly,
            access_mask: AccessFlags::SHADER_READ,
            stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
        },
        ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        },
    )?;
    gpu.wait_device_idle()?;
    Ok((backing_image, view))
}

pub fn generate_irradiance_map(
    gpu: &VkGpu,
    source_cubemap: &Texture,
    resource_map: &mut ResourceMap,
    cube_mesh: &ResourceHandle<Mesh>,
) -> anyhow::Result<Texture> {
    let size = Extent2D {
        width: 512,
        height: 512,
    };
    let input_texture_view = resource_map.get(&source_cubemap.image_view);
    let convolve = read_file_to_vk_module(&gpu, "./shaders/skybox_convolve_irradiance.spirv")?;
    let (backing_image, view) = cubemap_main_loop(
        gpu,
        ImageFormat::RgbaFloat16,
        &input_texture_view.view,
        convolve,
        size,
        resource_map,
        cube_mesh,
    )?;
    Texture::wrap(gpu, backing_image, view, resource_map)
}

pub fn load_cube_to_resource_map(
    gpu: &VkGpu,
    resource_map: &mut resource_map::ResourceMap,
) -> anyhow::Result<resource_map::ResourceHandle<engine::Mesh>> {
    let mesh_create_info = engine::MeshCreateInfo {
        label: Some("cube"),
        primitives: &[MeshPrimitiveCreateInfo {
            indices: vec![
                0, 1, 2, 3, 1, 0, //Bottom
                6, 5, 4, 4, 5, 7, // Front
                10, 9, 8, 8, 9, 11, // Left
                12, 13, 14, 15, 13, 12, // Right
                16, 17, 18, 19, 17, 16, // Up
                22, 21, 20, 20, 21, 23, // Down
            ],
            positions: vec![
                // Back
                vector![-1.0, -1.0, 1.0],
                vector![1.0, 1.0, 1.0],
                vector![-1.0, 1.0, 1.0],
                vector![1.0, -1.0, 1.0],
                // Front
                vector![-1.0, -1.0, -1.0],
                vector![1.0, 1.0, -1.0],
                vector![-1.0, 1.0, -1.0],
                vector![1.0, -1.0, -1.0],
                // Left
                vector![1.0, -1.0, -1.0],
                vector![1.0, 1.0, 1.0],
                vector![1.0, 1.0, -1.0],
                vector![1.0, -1.0, 1.0],
                // Right
                vector![-1.0, -1.0, -1.0],
                vector![-1.0, 1.0, 1.0],
                vector![-1.0, 1.0, -1.0],
                vector![-1.0, -1.0, 1.0],
                // Up
                vector![-1.0, 1.0, -1.0],
                vector![1.0, 1.0, 1.0],
                vector![1.0, 1.0, -1.0],
                vector![-1.0, 1.0, 1.0],
                // Down
                vector![-1.0, -1.0, -1.0],
                vector![1.0, -1.0, 1.0],
                vector![1.0, -1.0, -1.0],
                vector![-1.0, -1.0, 1.0],
            ],
            colors: vec![],
            normals: vec![
                // Back
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                // Front
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                // Left
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                // Right
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                // Up
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                // Down
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
            ],
            tangents: vec![],
            uvs: vec![
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
            ],
        }],
    };
    let cube_mesh = Mesh::new(gpu, &mesh_create_info)?;
    let cube_mesh_handle = resource_map.add(cube_mesh);
    Ok(cube_mesh_handle)
}
