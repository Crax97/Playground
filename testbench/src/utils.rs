#![allow(dead_code)]
use anyhow::bail;
use engine::{
    Camera, DeferredRenderingPipeline, MaterialInstance, Mesh, MeshPrimitiveCreateInfo,
    RenderingPipeline, Scene, Texture, TextureInput,
};
use image::DynamicImage;
use log::{debug, info};
use nalgebra::{point, vector};
use resource_map::{ResourceHandle, ResourceMap};
use std::{collections::HashMap, path::Path};

use half::f16;

pub struct LoadedImage {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

use gpu::{
    AccessFlags, Extent2D, ImageAspectFlags, ImageCreateInfo, ImageFormat, ImageSubresourceRange,
    ImageUsageFlags, MemoryDomain, PipelineStageFlags, ShaderModuleCreateInfo, VkGpu,
    VkShaderModule,
};

pub fn read_file_to_vk_module<P: AsRef<Path>>(
    gpu: &VkGpu,
    path: P,
) -> anyhow::Result<VkShaderModule> {
    info!(
        "Reading path from {:?}",
        path.as_ref()
            .canonicalize()
            .expect("Failed to canonicalize path")
    );
    let input_file = std::fs::read(path)?;
    let create_info = ShaderModuleCreateInfo { code: &input_file };
    Ok(gpu.create_shader_module(&create_info)?)
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
    cube_mesh: ResourceHandle<Mesh>,
    resource_map: &mut ResourceMap,
    path: P,
) -> anyhow::Result<Texture> {
    let hdr_image = load_image_from_path(path, ImageFormat::RgbaFloat16)?;
    let equi_texture = Texture::new_with_data(
        gpu,
        resource_map,
        hdr_image.width,
        hdr_image.height,
        &hdr_image.bytes,
        Some("Equilateral texture"),
        ImageFormat::RgbaFloat16,
        gpu::ImageViewType::Type2D,
    )?;
    let equi_texture = resource_map.add(equi_texture);

    let vertex_module = read_file_to_vk_module(&gpu, "./shaders/vertex_deferred.spirv")?;
    let equilateral_fragment = read_file_to_vk_module(&gpu, "./shaders/skybox_spherical.spirv")?;

    let mut scene_renderer = DeferredRenderingPipeline::new(&gpu, cube_mesh)?;

    let skybox_material = scene_renderer.create_material(
        &gpu,
        engine::MaterialDescription {
            name: "skybox material",
            domain: engine::MaterialDomain::Surface,
            texture_inputs: &[TextureInput {
                name: "Cubemap".to_owned(),
                format: ImageFormat::Rgba8,
            }],
            material_parameters: HashMap::new(),
            fragment_module: &equilateral_fragment,
            vertex_module: &vertex_module,
        },
    )?;
    let skybox_master = resource_map.add(skybox_material);

    let mut skybox_textures = HashMap::new();
    skybox_textures.insert("Cubemap".to_string(), equi_texture);

    let skybox_instance = MaterialInstance::create_instance(
        &gpu,
        skybox_master,
        &resource_map,
        &engine::MaterialInstanceDescription {
            name: "Skybox Generator",
            texture_inputs: skybox_textures,
        },
    )?;
    let skybox_instance = resource_map.add(skybox_instance);

    let mut scene = Scene::new();
    scene.set_skybox_material(Some(skybox_instance));

    let size = Extent2D {
        width: 2048,
        height: 2048,
    };

    let backing_image = gpu.create_image(
        &ImageCreateInfo {
            label: Some("Cubemap"),
            width: size.width,
            height: size.height,
            depth: 1,
            mips: 1,
            layers: 6,
            samples: gpu::SampleCount::Sample1,
            format: ImageFormat::RgbaFloat16,
            usage: ImageUsageFlags::SAMPLED
                | ImageUsageFlags::TRANSFER_DST
                | ImageUsageFlags::COLOR_ATTACHMENT,
        },
        MemoryDomain::DeviceLocal,
        None,
    )?;
    let make_image_view = |i| {
        gpu.create_image_view(&gpu::ImageViewCreateInfo {
            image: &backing_image,
            view_type: gpu::ImageViewType::Type2D,
            format: ImageFormat::RgbaFloat16,
            components: gpu::ComponentMapping::default(),
            subresource_range: gpu::ImageSubresourceRange {
                base_array_layer: i,
                layer_count: 1,
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
            },
        })
        .unwrap()
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
        Camera {
            location: point![0.0, 0.0, 0.0],
            forward: vector![1.0, 0.0, 0.0],
            fov: 90.0,
            width: size.width as f32,
            height: size.height as f32,
            near: 0.01,
            far: 1000.0,
        },
        Camera {
            location: point![0.0, 0.0, 0.0],
            forward: vector![-1.0, 0.0, 0.0],
            fov: 90.0,
            width: 2048 as f32,
            height: 2048 as f32,
            near: 0.01,
            far: 1000.0,
        },
        Camera {
            location: point![0.0, 0.0, 0.0],
            forward: vector![0.0, 1.0, 0.0],
            fov: 90.0,
            width: size.width as f32,
            height: size.height as f32,
            near: 0.01,
            far: 1000.0,
        },
        Camera {
            location: point![0.0, 0.0, 0.0],
            forward: vector![0.0, -1.0, 0.0],
            fov: 90.0,
            width: 2048 as f32,
            height: 2048 as f32,
            near: 0.01,
            far: 1000.0,
        },
        Camera {
            location: point![0.0, 0.0, 0.0],
            forward: vector![0.0, 0.0, 1.0],
            fov: 90.0,
            width: 2048 as f32,
            height: 2048 as f32,
            near: 0.01,
            far: 1000.0,
        },
        Camera {
            location: point![0.0, 0.0, 0.0],
            forward: vector![0.0, 0.0, -1.0],
            fov: 90.0,
            width: 2048 as f32,
            height: 2048 as f32,
            near: 0.01,
            far: 1000.0,
        },
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
        ImageAspectFlags::COLOR,
        ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        },
    )?;

    for (i, view) in views.iter().enumerate() {
        let pov = &povs[i];
        let command_buffer = scene_renderer.render(
            pov,
            &scene,
            &engine::Backbuffer {
                size,
                format: ImageFormat::RgbaFloat16,
                image: &backing_image,
                image_view: view,
            },
            resource_map,
        )?;

        command_buffer.submit(&gpu::CommandBufferSubmitInfo {
            wait_semaphores: &[],
            wait_stages: &[],
            signal_semaphores: &[],
            fence: None,
        })?;
    }

    gpu.wait_device_idle()?;
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
            stage_mask: PipelineStageFlags::ALL_GRAPHICS,
        },
        ImageAspectFlags::COLOR,
        ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        },
    )?;
    let view = gpu.create_image_view(&gpu::ImageViewCreateInfo {
        image: &backing_image,
        view_type: gpu::ImageViewType::Cube,
        format: ImageFormat::RgbaFloat16,
        components: gpu::ComponentMapping::default(),
        subresource_range: gpu::ImageSubresourceRange {
            base_array_layer: 0,
            layer_count: 6,
            aspect_mask: ImageAspectFlags::COLOR,
            level_count: 1,
            base_mip_level: 0,
        },
    })?;

    Texture::wrap(gpu, backing_image, view, resource_map)
}

pub fn generate_irradiance_map(gpu: &VkGpu, source_cubemap: &Texture) -> anyhow::Result<Texture> {
    todo!()
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
