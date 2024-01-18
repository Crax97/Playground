use bytemuck::{Pod, Zeroable};
use gpu::BufferHandle;

pub const SHADOW_ATLAS_TILE_SIZE: u32 = 128;
pub const SHADOW_ATLAS_WIDTH: u32 = SHADOW_ATLAS_TILE_SIZE * 70;
pub const SHADOW_ATLAS_HEIGHT: u32 = SHADOW_ATLAS_TILE_SIZE * 35;

pub struct CsmBuffers {
    pub shadow_casters: BufferHandle,
    pub csm_splits: BufferHandle,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
pub struct ShadowMap {
    pub offset_size: [u32; 4],
    // type: x, num_maps: y, pov: z, split_idx: w (directional only)
    pub type_num_maps_pov_lightid: [u32; 4],
}
