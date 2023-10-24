use gpu::{
    AccessFlags, ComponentMapping, Filter, Gpu, ImageAspectFlags, ImageCreateInfo, ImageFormat,
    ImageHandle, ImageSubresourceRange, ImageUsageFlags, ImageViewHandle, ImageViewType,
    MemoryDomain, PipelineStageFlags, SamplerAddressMode, SamplerCreateInfo, SamplerHandle, VkGpu,
    VkImage, VkImageView, VkSampler,
};
use resource_map::{Resource, ResourceHandle, ResourceMap};

pub struct ImageResource(pub ImageHandle);
impl Resource for ImageResource {
    fn get_description(&self) -> &str {
        "GPU Image"
    }
}

pub struct TextureImageView {
    pub view: ImageViewHandle,
    pub image: ResourceHandle<ImageResource>,
}
impl Resource for TextureImageView {
    fn get_description(&self) -> &str {
        "GPU Image View"
    }
}

pub struct SamplerResource(pub SamplerHandle);
impl Resource for SamplerResource {
    fn get_description(&self) -> &str {
        "GPU Sampler"
    }
}
pub struct Texture {
    pub image_view: ResourceHandle<TextureImageView>,
    pub sampler: ResourceHandle<SamplerResource>,
}

impl Texture {
    fn new_impl(
        gpu: &VkGpu,
        width: u32,
        height: u32,
        data: Option<&[u8]>,
        label: Option<&str>,
        format: ImageFormat,
        view_type: ImageViewType,
    ) -> anyhow::Result<(ImageHandle, ImageViewHandle, SamplerHandle)> {
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
            image: image.clone(),
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

        let sampler = gpu.make_sampler(&SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_u: SamplerAddressMode::Repeat,
            address_v: SamplerAddressMode::Repeat,
            address_w: SamplerAddressMode::Repeat,
            mip_lod_bias: 0.0,
            compare_function: None,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: [0.0; 4],
        })?;
        Ok((image, rgba_view, sampler))
    }

    pub fn wrap(
        gpu: &VkGpu,
        image: ImageHandle,
        view: ImageViewHandle,
        resource_map: &mut ResourceMap,
    ) -> anyhow::Result<Self> {
        let sampler = gpu.make_sampler(&SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_u: SamplerAddressMode::Repeat,
            address_v: SamplerAddressMode::Repeat,
            address_w: SamplerAddressMode::Repeat,
            mip_lod_bias: 0.0,
            compare_function: None,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: [0.0; 4],
        })?;
        let image = resource_map.add(ImageResource(image));
        let image_view = TextureImageView { image, view };
        let image_view = resource_map.add(image_view);
        let sampler = resource_map.add(SamplerResource(sampler));
        Ok(Self {
            image_view,
            sampler,
        })
    }
    pub fn new_empty(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        width: u32,
        height: u32,
        label: Option<&str>,
        format: ImageFormat,
        view_type: ImageViewType,
    ) -> anyhow::Result<Self> {
        let (image, view, sampler) =
            Self::new_impl(gpu, width, height, None, label, format, view_type)?;
        let image = resource_map.add(ImageResource(image));
        let image_view = TextureImageView { image, view };
        let image_view = resource_map.add(image_view);
        let sampler = resource_map.add(SamplerResource(sampler));
        Ok(Self {
            image_view,
            sampler,
        })
    }
    pub fn new_with_data(
        gpu: &VkGpu,
        resource_map: &mut ResourceMap,
        width: u32,
        height: u32,
        data: &[u8],
        label: Option<&str>,
        format: ImageFormat,
        view_type: ImageViewType,
    ) -> anyhow::Result<Self> {
        let (image, view, sampler) =
            Self::new_impl(gpu, width, height, Some(data), label, format, view_type)?;

        let image = resource_map.add(ImageResource(image));
        let image_view = TextureImageView { image, view };
        let image_view = resource_map.add(image_view);
        let sampler = resource_map.add(SamplerResource(sampler));

        Ok(Self {
            image_view,
            sampler,
        })
    }
}

impl Resource for Texture {
    fn get_description(&self) -> &str {
        "Texture"
    }
}
