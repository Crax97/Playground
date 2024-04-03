use gpu::{Gpu, LifetimedCache, SamplerCreateInfo, SamplerHandle};

use crate::TextureSamplerSettings;

pub struct SamplerAllocator {
    sampler_allocator: LifetimedCache<SamplerHandle>,
}

impl SamplerAllocator {
    pub fn get(&self, gpu: &dyn Gpu, desc: &TextureSamplerSettings) -> SamplerHandle {
        self.sampler_allocator
            .get_clone(desc, || {
                let sam = gpu.make_sampler(&SamplerCreateInfo {
                    mag_filter: desc.mag_filter,
                    min_filter: desc.min_filter,
                    address_u: desc.address_u,
                    address_v: desc.address_v,
                    address_w: desc.address_w,
                    // TODO: Have a global lod bias
                    mip_lod_bias: 0.0,
                    compare_function: None,
                    min_lod: 0.0,
                    max_lod: 1.0,
                    border_color: [0.0; 4],
                })?;
                Ok(sam)
            })
            .expect("Failed to create sampler")
    }

    pub fn new(lifetime: u32) -> Self {
        Self {
            sampler_allocator: LifetimedCache::new(lifetime),
        }
    }

    pub(crate) fn destroy(&self, gpu: &dyn Gpu) {
        self.sampler_allocator.for_each(|v| {
            gpu.destroy_sampler(*v);
        })
    }
}
