use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use mgpu::{util::hash_type, Device, Sampler, SamplerDescription};

use crate::assets::texture::TextureSamplerConfiguration;

#[derive(Default, Clone)]
pub struct SamplerAllocator {
    samplers: Arc<Mutex<HashMap<u64, Sampler>>>,
}

impl SamplerAllocator {
    pub fn get(&self, device: &Device, sampler_configuration: &TextureSamplerConfiguration) -> Sampler {
        let mut samplers = self.samplers.lock().unwrap();
        let hash = hash_type(sampler_configuration);
        let sampler = *samplers.entry(hash).or_insert_with(|| {
            let description = SamplerDescription {
                label: None,
                mag_filter: sampler_configuration.minmag_filter,
                min_filter: sampler_configuration.minmag_filter,
                mipmap_mode: sampler_configuration.mipmap_mode,
                address_mode_u: mgpu::AddressMode::ClampToEdge,
                address_mode_v: mgpu::AddressMode::ClampToEdge,
                address_mode_w: mgpu::AddressMode::ClampToEdge,
                lod_bias: 0.0,
                compare_op: None,
                min_lod: 0.0,
                max_lod: f32::MAX,
                border_color: mgpu::BorderColor::Black,
                unnormalized_coordinates: false,
            };
            device
                .create_sampler(&description)
                .expect("Failed to create sampler")
        });

        sampler
    }
}
