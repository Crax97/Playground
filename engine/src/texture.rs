use crate::asset_map::Asset;
use gpu::{
    AccessFlags, ComponentMapping, Filter, Gpu, ImageAspectFlags, ImageCreateInfo, ImageFormat,
    ImageHandle, ImageSubresourceRange, ImageUsageFlags, ImageViewHandle, ImageViewType,
    MemoryDomain, PipelineStageFlags, SamplerAddressMode,
};

/*
This struct describes how sampling a texture is performed, and can be cheapily modified/cloned:
the actual samplers are cached by the renderer.
*/
#[derive(Copy, Clone, Hash, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct TextureSamplerSettings {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub address_u: SamplerAddressMode,
    pub address_v: SamplerAddressMode,

    // Ignored when Texture Type is 2D
    pub address_w: SamplerAddressMode,
}

impl Default for TextureSamplerSettings {
    fn default() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_u: SamplerAddressMode::Repeat,
            address_v: SamplerAddressMode::Repeat,
            address_w: SamplerAddressMode::Repeat,
        }
    }
}

pub struct Texture {
    pub image: ImageHandle,
    pub view: ImageViewHandle,
    pub sampler_settings: TextureSamplerSettings,
}

impl Texture {
    fn new_impl(
        gpu: &dyn Gpu,
        width: u32,
        height: u32,
        data: Option<&[u8]>,
        label: Option<&str>,
        format: ImageFormat,
        view_type: ImageViewType,
    ) -> anyhow::Result<(ImageHandle, ImageViewHandle)> {
        let layers = match view_type {
            ImageViewType::Type2D => 1,
            ImageViewType::Cube => 6,
            _ => unimplemented!(),
        };
        let image = gpu.make_image(
            &ImageCreateInfo {
                label,
                width,
                height,
                depth: 1,

                mips: 1,
                layers,
                samples: gpu::SampleCount::Sample1,
                format,
                usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            },
            MemoryDomain::DeviceLocal,
            data,
        )?;

        gpu.transition_image_layout(
            &image,
            gpu::TransitionInfo {
                layout: gpu::ImageLayout::Undefined,
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
                layer_count: layers,
            },
        )?;
        let rgba_view = gpu.make_image_view(&gpu::ImageViewCreateInfo {
            label: Some(&(label.unwrap_or("Texture image").to_owned() + " - RGBA View")),
            image,
            view_type,
            format,
            components: ComponentMapping::default(),
            subresource_range: ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layers,
            },
        })?;

        Ok((image, rgba_view))
    }

    pub fn wrap(image: ImageHandle, view: ImageViewHandle) -> anyhow::Result<Self> {
        Ok(Self {
            image,
            view,
            sampler_settings: Default::default(),
        })
    }
    pub fn new_empty(
        gpu: &dyn Gpu,
        width: u32,
        height: u32,
        label: Option<&str>,
        format: ImageFormat,
        view_type: ImageViewType,
    ) -> anyhow::Result<Self> {
        let (image, view) = Self::new_impl(gpu, width, height, None, label, format, view_type)?;

        Ok(Self {
            image,
            view,
            sampler_settings: Default::default(),
        })
    }
    pub fn new_with_data(
        gpu: &dyn Gpu,
        width: u32,
        height: u32,
        data: &[u8],
        label: Option<&str>,
        format: ImageFormat,
        view_type: ImageViewType,
    ) -> anyhow::Result<Self> {
        let (image, view) =
            Self::new_impl(gpu, width, height, Some(data), label, format, view_type)?;

        Ok(Self {
            image,
            view,
            sampler_settings: Default::default(),
        })
    }
}

impl Asset for Texture {
    fn get_description(&self) -> &str {
        "Texture"
    }

    fn destroyed(&mut self, gpu: &dyn Gpu) {
        gpu.destroy_image_view(self.view);
        gpu.destroy_image(self.image);
    }
}
