use ash::{prelude::VkResult, vk::BufferUsageFlags};
use nalgebra::{Vector2, Vector3};

use gpu::{BufferCreateInfo, Gpu, GpuBuffer, MemoryDomain};
use resource_map::Resource;

pub struct MeshCreateInfo<'a> {
    pub label: Option<&'a str>,
    pub indices: &'a [u32],
    pub positions: &'a [Vector3<f32>],
    pub colors: &'a [Vector3<f32>],
    pub normals: &'a [Vector3<f32>],
    pub tangents: &'a [Vector3<f32>],
    pub uvs: &'a [Vector2<f32>],
}

pub struct Mesh {
    pub index_buffer: GpuBuffer,
    pub position_component: GpuBuffer,
    pub color_component: GpuBuffer,
    pub normal_component: GpuBuffer,
    pub tangent_component: GpuBuffer,
    pub uv_component: GpuBuffer,

    pub index_count: u32,
}

impl Mesh {
    pub fn new(gpu: &Gpu, create_info: &MeshCreateInfo) -> VkResult<Self> {
        let label = create_info.label.map(|s| s.to_owned());
        let index_buffer = gpu.create_buffer(
            &BufferCreateInfo {
                label: label.clone().map(|lab| lab + ": Index buffer").as_deref(),
                size: std::mem::size_of::<u32>() * create_info.indices.len().max(1),
                usage: BufferUsageFlags::INDEX_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        )?;
        gpu.write_buffer_data(&index_buffer, create_info.indices)?;
        let position_component = gpu.create_buffer(
            &BufferCreateInfo {
                label: label.clone().map(|lab| lab + ": Index buffer").as_deref(),
                size: std::mem::size_of::<Vector3<f32>>() * create_info.positions.len().max(1),
                usage: BufferUsageFlags::VERTEX_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        )?;
        gpu.write_buffer_data(&position_component, create_info.positions)?;
        let color_component = gpu.create_buffer(
            &BufferCreateInfo {
                label: label
                    .clone()
                    .map(|lab| lab + ": Position buffer")
                    .as_deref(),
                size: std::mem::size_of::<Vector3<f32>>() * create_info.positions.len().max(1),
                usage: BufferUsageFlags::VERTEX_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        )?;
        gpu.write_buffer_data(&color_component, create_info.colors)?;
        let normal_component = gpu.create_buffer(
            &BufferCreateInfo {
                label: label.clone().map(|lab| lab + ": Color buffer").as_deref(),
                size: std::mem::size_of::<Vector3<f32>>() * create_info.normals.len().max(1),
                usage: BufferUsageFlags::VERTEX_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        )?;
        gpu.write_buffer_data(&normal_component, create_info.normals)?;
        let tangent_component = gpu.create_buffer(
            &BufferCreateInfo {
                label: label.clone().map(|lab| lab + ": Normal buffer").as_deref(),
                size: std::mem::size_of::<Vector3<f32>>() * create_info.tangents.len().max(1),
                usage: BufferUsageFlags::VERTEX_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        )?;
        gpu.write_buffer_data(&tangent_component, create_info.tangents)?;
        let uv_component = gpu.create_buffer(
            &BufferCreateInfo {
                label: label.clone().map(|lab| lab + ": Tangent buffer").as_deref(),
                size: std::mem::size_of::<Vector2<f32>>() * create_info.uvs.len().max(1),
                usage: BufferUsageFlags::VERTEX_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        )?;
        gpu.write_buffer_data(&uv_component, create_info.uvs)?;
        Ok(Self {
            index_buffer,
            position_component,
            color_component,
            normal_component,
            tangent_component,
            uv_component,
            index_count: create_info.indices.len() as _,
        })
    }
}

impl Resource for Mesh {
    fn get_description(&self) -> &str {
        "Mesh"
    }
}
