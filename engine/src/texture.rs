use ash::{
    prelude::VkResult,
    vk::{
        self, BorderColor, CompareOp, ComponentMapping, Filter, Format, ImageUsageFlags,
        ImageViewType, SamplerAddressMode, SamplerCreateFlags, SamplerCreateInfo,
        SamplerMipmapMode, StructureType,
    },
};
use gpu::{
    Gpu, GpuImage, GpuImageView, GpuSampler, ImageAspectFlags, ImageCreateInfo,
    ImageSubresourceRange, MemoryDomain,
};
use resource_map::{Resource, ResourceHandle, ResourceMap};

pub struct ImageResource(pub GpuImage);
impl Resource for ImageResource {
    fn get_description(&self) -> &str {
        "GPU Image"
    }
}

pub struct TextureImageView {
    pub view: GpuImageView,
    pub image: ResourceHandle<ImageResource>,
}
impl Resource for TextureImageView {
    fn get_description(&self) -> &str {
        "GPU Image View"
    }
}

pub struct SamplerResource(pub GpuSampler);
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
        gpu: &Gpu,
        width: u32,
        height: u32,
        data: Option<&[u8]>,
        label: Option<&str>,
    ) -> VkResult<(GpuImage, GpuImageView, GpuSampler)> {
        let image = gpu.create_image(
            &ImageCreateInfo {
                label,
                width,
                height,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            },
            MemoryDomain::DeviceLocal,
            data,
        )?;

        let rgba_view = gpu.create_image_view(&gpu::ImageViewCreateInfo {
            image: &image,
            view_type: ImageViewType::TYPE_2D,
            format: Format::R8G8B8A8_UNORM,
            components: ComponentMapping::default(),
            subresource_range: ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        })?;

        let sampler = gpu.create_sampler(&SamplerCreateInfo {
            s_type: StructureType::SAMPLER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: SamplerCreateFlags::empty(),
            mag_filter: Filter::LINEAR,
            min_filter: Filter::LINEAR,
            mipmap_mode: SamplerMipmapMode::LINEAR,
            address_mode_u: SamplerAddressMode::REPEAT,
            address_mode_v: SamplerAddressMode::REPEAT,
            address_mode_w: SamplerAddressMode::REPEAT,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: gpu
                .physical_device_properties()
                .limits
                .max_sampler_anisotropy,
            compare_enable: vk::FALSE,
            compare_op: CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: BorderColor::default(),
            unnormalized_coordinates: vk::FALSE,
        })?;
        Ok((image, rgba_view, sampler))
    }

    pub fn new_empty(
        gpu: &Gpu,
        resource_map: &mut ResourceMap,
        width: u32,
        height: u32,
        label: Option<&str>,
    ) -> VkResult<Self> {
        let (image, view, sampler) = Self::new_impl(gpu, width, height, None, label)?;
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
        gpu: &Gpu,
        resource_map: &mut ResourceMap,
        width: u32,
        height: u32,
        data: &[u8],
        label: Option<&str>,
    ) -> VkResult<Self> {
        let (image, view, sampler) = Self::new_impl(gpu, width, height, Some(data), label)?;

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
