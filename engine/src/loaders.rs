use std::{ffi::OsStr, io::BufReader};

use gpu::{ImageFormat, ImageViewType, VkGpu};

// A texture loader that works directly with the file system
use crate::{ResourceLoader, Texture};

pub struct FileSystemTextureLoader;

impl ResourceLoader for FileSystemTextureLoader {
    type LoadedResource = Texture;

    fn load(&self, gpu: &VkGpu, path: &std::path::Path) -> anyhow::Result<Self::LoadedResource> {
        let cpu_image = image::load(
            BufReader::new(std::fs::File::open(path)?),
            image::ImageFormat::from_path(path)?,
        )?;

        let format = match cpu_image.color() {
            image::ColorType::Rgb8 => ImageFormat::Rgb8,
            image::ColorType::Rgba8 => ImageFormat::Rgba8,
            image::ColorType::Rgb32F => ImageFormat::RgbFloat32,
            image::ColorType::Rgba32F => ImageFormat::RgbaFloat32,
            f => return Err(anyhow::format_err!("Unsupported texture format: {:?}", f)),
        };

        Ok(Texture::new_with_data(
            gpu,
            cpu_image.width(),
            cpu_image.height(),
            cpu_image.as_bytes(),
            path.file_name()
                .unwrap_or(&OsStr::new("Loaded texture"))
                .to_str(),
            format,
            ImageViewType::Type2D,
        )?)
    }
}
