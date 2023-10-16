#![allow(dead_code)]
use anyhow::bail;
use engine::{Mesh, MeshPrimitiveCreateInfo, Texture};
use image::DynamicImage;
use log::{debug, info};
use nalgebra::vector;
use std::path::Path;

use half::f16;

pub struct LoadedImage {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

use gpu::{ImageFormat, ShaderModuleCreateInfo, VkGpu, VkShaderModule};

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
