use ash::{
    prelude::VkResult,
    vk::{BufferCreateFlags, BufferUsageFlags, SharingMode, StructureType},
};
use nalgebra::{Vector2, Vector3};

use crate::gpu::{BufferCreateInfo, Gpu, GpuBuffer, MemoryDomain, ResourceHandle};

pub struct MeshCreateInfo<'a> {
    pub indices: &'a [u32],
    pub positions: &'a [Vector3<f32>],
    pub normals: &'a [Vector3<f32>],
    pub tangents: &'a [Vector3<f32>],
    pub uvs: &'a [Vector2<f32>],
}

pub struct Mesh {
    pub index_buffer: ResourceHandle<GpuBuffer>,
    pub position_component: ResourceHandle<GpuBuffer>,
    pub normal_component: ResourceHandle<GpuBuffer>,
    pub tangent_component: ResourceHandle<GpuBuffer>,
    pub uv_component: ResourceHandle<GpuBuffer>,
}

impl Mesh {
    pub fn new(gpu: &Gpu, create_info: &MeshCreateInfo) -> VkResult<Self> {
        let index_buffer = gpu.create_buffer(
            &BufferCreateInfo {
                size: std::mem::size_of::<u32>() * create_info.indices.len(),
                usage: BufferUsageFlags::INDEX_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        );
        todo!()
    }
}
