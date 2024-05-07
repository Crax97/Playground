use std::num::NonZeroU32;

use bitflags::bitflags;
use mgpu::{
    AddressMode, Device, Extents2D, Extents3D, FilterMode, Image, ImageDescription, ImageDimension,
    ImageFormat, ImageUsageFlags, ImageView, ImageViewDescription, ImageWriteParams, MipmapMode,
    Sampler,
};
use serde::{Deserialize, Serialize};

use crate::sampler_allocator::SamplerAllocator;

#[derive(
    Default, Hash, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct SamplerConfiguration {
    pub minmag_filter: FilterMode,
    pub mipmap_mode: MipmapMode,
}

pub struct Texture {
    pub(crate) image: Image,
    pub(crate) view: ImageView,
    pub(crate) sampler_configuration: SamplerConfiguration,
    pub(crate) sampler: Sampler,
}
pub struct TextureDescription<'a> {
    pub label: Option<&'a str>,
    pub data: &'a [&'a [u8]],
    pub ty: TextureType,
    pub format: ImageFormat,
    pub usage_flags: TextureUsageFlags,
    pub num_mips: NonZeroU32,
    pub auto_generate_mips: bool,
    pub sampler_configuration: SamplerConfiguration,
}

pub enum TextureType {
    D1(usize),
    D2(Extents2D),
    D3(Extents3D),
    Cubemap(Extents2D),
}

bitflags! {
    #[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
    pub struct TextureUsageFlags : u8 {
        const RENDER_ATTACHMENT = 0x01;
        const STORAGE = 0x02;
    }
}

impl<'a> TextureDescription<'a> {
    fn image_usage_flags(&self) -> ImageUsageFlags {
        let mut usages = ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST;
        if self
            .usage_flags
            .contains(TextureUsageFlags::RENDER_ATTACHMENT)
        {
            usages |= match self.format.aspect() {
                mgpu::ImageAspect::Color => ImageUsageFlags::COLOR_ATTACHMENT,
                mgpu::ImageAspect::Depth => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            };
        }

        if self.usage_flags.contains(TextureUsageFlags::STORAGE) {
            usages |= ImageUsageFlags::STORAGE;
        }

        usages
    }
}

impl Texture {
    pub fn new(
        device: &Device,
        description: &TextureDescription,
        sampler_allocator: &mut SamplerAllocator,
    ) -> anyhow::Result<Texture> {
        let (extents, dimension, array_layers) = match description.ty {
            TextureType::D1(s) => (
                Extents3D {
                    width: s as u32,
                    height: 1,
                    depth: 1,
                },
                ImageDimension::D1,
                1,
            ),
            TextureType::D2(e) => (
                Extents3D {
                    width: e.width,
                    height: e.height,
                    depth: 1,
                },
                ImageDimension::D2,
                1,
            ),
            TextureType::D3(e) => (e, ImageDimension::D3, 1),
            TextureType::Cubemap(e) => todo!(),
        };

        let image = device.create_image(&ImageDescription {
            label: Some(&format!(
                "Texture Image for {}",
                description.label.unwrap_or("Unknown")
            )),
            usage_flags: description.image_usage_flags(),
            extents,
            dimension,
            mips: description.num_mips,
            array_layers: NonZeroU32::new(array_layers).unwrap(),
            samples: mgpu::SampleCount::One,
            format: description.format,
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;

        let view = device.create_image_view(&ImageViewDescription {
            label: Some(&format!(
                "Texture Image View for {}",
                description.label.unwrap_or("Unknown")
            )),
            image,
            format: description.format,
            dimension,
            aspect: description.format.aspect(),
            image_subresource: image.whole_subresource(),
        })?;

        for (mip, data) in description.data.iter().enumerate() {
            let region = image.mip_region(mip as u32);
            #[cfg(debug_assertions)]
            {
                let expected_data_size =
                    region.extents.area() as usize * description.format.byte_size();
                debug_assert!(data.len() >= expected_data_size);
            }
            device.write_image_data(image, &ImageWriteParams { data, region })?;
        }

        if description.data.len() == 1 && description.auto_generate_mips {
            device.generate_mip_chain(image, mgpu::FilterMode::Linear)?;
        }

        let sampler = sampler_allocator.get(device, &description.sampler_configuration);
        Ok(Texture {
            image,
            view,
            sampler_configuration: description.sampler_configuration,
            sampler,
        })
    }
}
