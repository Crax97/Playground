use ash::{
    prelude::VkResult,
    vk::{
        self, BorderColor, CompareOp, ComponentMapping, Filter, Format, ImageAspectFlags,
        ImageSubresourceRange, ImageUsageFlags, ImageViewType, SamplerAddressMode,
        SamplerCreateFlags, SamplerCreateInfo, SamplerMipmapMode, StructureType,
    },
};
use gpu::{Gpu, GpuImage, GpuImageView, GpuSampler, ImageCreateInfo, MemoryDomain};
use resource_map::Resource;

pub struct Texture {
    pub image: GpuImage,
    pub rgba_view: GpuImageView,
    pub sampler: GpuSampler,
}

impl Texture {
    pub fn new_empty(gpu: &Gpu, width: u32, height: u32, label: Option<&str>) -> VkResult<Self> {
        let image = gpu.create_image(
            &ImageCreateInfo {
                label,
                width,
                height,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            },
            MemoryDomain::DeviceLocal,
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
        Ok(Self {
            image,
            rgba_view,
            sampler,
        })
    }
    pub fn new_with_data(
        gpu: &Gpu,
        width: u32,
        height: u32,
        data: &[u8],
        label: Option<&str>,
    ) -> VkResult<Self> {
        let mut texture = Self::new_empty(gpu, width, height, label)?;
        texture.copy_data_immediate(gpu, data)?;
        Ok(texture)
    }

    fn copy_data_immediate(&mut self, gpu: &Gpu, data: &[u8]) -> VkResult<()> {
        gpu.write_image_data(&self.image, data)
    }
}

impl Resource for Texture {
    fn get_description(&self) -> &str {
        "todo"
    }
}
