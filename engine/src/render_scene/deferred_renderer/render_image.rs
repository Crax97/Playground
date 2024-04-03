use gpu::{
    Gpu, ImageAspectFlags, ImageFormat, ImageHandle, ImageUsageFlags, ImageViewHandle,
    ImageViewType, LifetimedCache, MemoryDomain, SampleCount,
};

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RenderImageDescription {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub samples: SampleCount,
    pub view_type: ImageViewType,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RenderImageId {
    pub label: &'static str,
    pub desc: RenderImageDescription,
}

#[derive(Clone)]
pub struct RenderImage {
    pub image: ImageHandle,
    pub view: ImageViewHandle,
}

pub struct RenderImageAllocator {
    image_allocator: LifetimedCache<RenderImage>,
}

impl RenderImageAllocator {
    pub fn get(
        &self,
        gpu: &dyn Gpu,
        label: &'static str,
        desc: &RenderImageDescription,
    ) -> RenderImage {
        self.image_allocator
            .get_clone(&RenderImageId { label, desc: *desc }, || {
                let image = gpu.make_image(
                    &gpu::ImageCreateInfo {
                        label: Some(label),
                        width: desc.width,
                        height: desc.height,
                        depth: 1,
                        mips: 1,
                        layers: 1,
                        samples: desc.samples,
                        format: desc.format,
                        usage: ImageUsageFlags::SAMPLED
                            | ImageUsageFlags::INPUT_ATTACHMENT
                            | if desc.format.is_color() {
                                ImageUsageFlags::COLOR_ATTACHMENT
                            } else {
                                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                            },
                    },
                    MemoryDomain::DeviceLocal,
                    None,
                )?;

                let view = gpu.make_image_view(&gpu::ImageViewCreateInfo {
                    label: Some(&(label.to_owned() + " - view")),
                    image,
                    view_type: gpu::ImageViewType::Type2D,
                    format: desc.format,
                    components: gpu::ComponentMapping::default(),
                    subresource_range: gpu::ImageSubresourceRange {
                        aspect_mask: if desc.format.is_color() {
                            ImageAspectFlags::COLOR
                        } else {
                            ImageAspectFlags::DEPTH
                        },
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                })?;
                Ok(RenderImage { image, view })
            })
            .expect("Failed to create RenderImage")
    }

    pub fn new(lifetime: u32) -> Self {
        Self {
            image_allocator: LifetimedCache::new(lifetime),
        }
    }

    pub(crate) fn destroy(&self, gpu: &dyn Gpu) {
        self.image_allocator.for_each(|v| {
            gpu.destroy_image_view(v.view);
            gpu.destroy_image(v.image);
        })
    }
}
