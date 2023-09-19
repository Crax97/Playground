use nalgebra::{Vector2, Vector3};

use gpu::{BufferCreateInfo, BufferUsageFlags, Gpu, GpuBuffer, MemoryDomain};
use resource_map::Resource;

pub struct MeshPrimitiveCreateInfo {
    pub indices: Vec<u32>,
    pub positions: Vec<Vector3<f32>>,
    pub colors: Vec<Vector3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub tangents: Vec<Vector3<f32>>,
    pub uvs: Vec<Vector2<f32>>,
}

pub struct MeshCreateInfo<'a> {
    pub label: Option<&'a str>,
    pub primitives: &'a [MeshPrimitiveCreateInfo],
}

pub struct MeshPrimitive {
    pub index_buffer: GpuBuffer,
    pub position_component: GpuBuffer,
    pub color_component: GpuBuffer,
    pub normal_component: GpuBuffer,
    pub tangent_component: GpuBuffer,
    pub uv_component: GpuBuffer,

    pub index_count: u32,
}

pub struct Mesh {
    pub primitives: Vec<MeshPrimitive>,
}

impl Mesh {
    pub fn new(gpu: &Gpu, mesh_create_info: &MeshCreateInfo) -> anyhow::Result<Self> {
        let primitives: Vec<anyhow::Result<MeshPrimitive>> = mesh_create_info
            .primitives
            .iter()
            .enumerate()
            .map(|(idx, create_info)| {
                let label = mesh_create_info
                    .label
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| "GPU Mesh".to_owned())
                    + &format!(" - primitive {idx}");
                let index_buffer = gpu.create_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Index buffer")),
                        size: std::mem::size_of::<u32>() * create_info.indices.len().max(1),
                        usage: BufferUsageFlags::INDEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer_data(&index_buffer, &create_info.indices)?;
                let position_component = gpu.create_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Position buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.positions.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer_data(&position_component, &create_info.positions)?;
                let color_component = gpu.create_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Color buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.positions.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer_data(&color_component, &create_info.colors)?;
                let normal_component = gpu.create_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Normal buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.normals.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer_data(&normal_component, &create_info.normals)?;
                let tangent_component = gpu.create_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Tangent buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.tangents.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer_data(&tangent_component, &create_info.tangents)?;
                let uv_component = gpu.create_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label + ": TexCoord[0] buffer")),
                        size: std::mem::size_of::<Vector2<f32>>() * create_info.uvs.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer_data(&uv_component, &create_info.uvs)?;
                Ok(MeshPrimitive {
                    index_buffer,
                    position_component,
                    color_component,
                    normal_component,
                    tangent_component,
                    uv_component,
                    index_count: create_info.indices.len() as _,
                })
            })
            .collect();
        let mut generated_primitives = vec![];
        generated_primitives.reserve(primitives.len());
        for prim in primitives {
            match prim {
                Ok(prim) => generated_primitives.push(prim),
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(Self {
            primitives: generated_primitives,
        })
    }
}

impl Resource for Mesh {
    fn get_description(&self) -> &str {
        "Mesh"
    }
}
