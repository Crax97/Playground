use nalgebra::{Vector2, Vector3};

use crate::{math::shape::Shape, resource_map::Resource};
use gpu::{BufferCreateInfo, BufferHandle, BufferUsageFlags, Gpu, MemoryDomain};

use crate::utils::to_u8_slice;

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

#[derive(Clone)]
pub struct MeshPrimitive {
    pub index_buffer: BufferHandle,
    pub position_component: BufferHandle,
    pub color_component: BufferHandle,
    pub normal_component: BufferHandle,
    pub tangent_component: BufferHandle,
    pub uv_component: BufferHandle,

    pub index_count: u32,

    pub bounding_shape: Shape,
}

#[derive(Clone)]
pub struct Mesh {
    pub primitives: Vec<MeshPrimitive>,
}

impl Mesh {
    pub fn new(gpu: &dyn Gpu, mesh_create_info: &MeshCreateInfo) -> anyhow::Result<Self> {
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
                let index_buffer = gpu.make_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Index buffer")),
                        size: std::mem::size_of::<u32>() * create_info.indices.len().max(1),
                        usage: BufferUsageFlags::INDEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&create_info.indices))?;
                let position_component = gpu.make_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Position buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.positions.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer(&position_component, 0, to_u8_slice(&create_info.positions))?;
                let color_component = gpu.make_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Color buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.positions.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer(&color_component, 0, to_u8_slice(&create_info.colors))?;
                let normal_component = gpu.make_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Normal buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.normals.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer(&normal_component, 0, to_u8_slice(&create_info.normals))?;
                let tangent_component = gpu.make_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label.clone() + ": Tangent buffer")),
                        size: std::mem::size_of::<Vector3<f32>>()
                            * create_info.tangents.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer(&tangent_component, 0, to_u8_slice(&create_info.tangents))?;
                let uv_component = gpu.make_buffer(
                    &BufferCreateInfo {
                        label: Some(&(label + ": TexCoord[0] buffer")),
                        size: std::mem::size_of::<Vector2<f32>>() * create_info.uvs.len().max(1),
                        usage: BufferUsageFlags::VERTEX_BUFFER,
                    },
                    MemoryDomain::DeviceLocal,
                )?;
                gpu.write_buffer(&uv_component, 0, to_u8_slice(&create_info.uvs))?;
                let bounding_radius = create_info
                    .positions
                    .iter()
                    .max_by(|point_a, point_b| point_a.magnitude().total_cmp(&point_b.magnitude()))
                    .map(|v| v.magnitude())
                    .unwrap_or_default();
                Ok(MeshPrimitive {
                    index_buffer,
                    position_component,
                    color_component,
                    normal_component,
                    tangent_component,
                    uv_component,
                    index_count: create_info.indices.len() as _,
                    bounding_shape: Shape::Sphere {
                        radius: bounding_radius,
                        origin: Default::default(),
                    },
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
