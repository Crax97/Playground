use glam::{Vec2, Vec3};
use mgpu::{Buffer, BufferDescription, BufferUsageFlags, BufferWriteParams, Device, MgpuResult};

pub struct Mesh {
    pub(crate) index_buffer: Buffer,
    pub(crate) position_component: Buffer,
    pub(crate) normal_component: Buffer,
    pub(crate) tangent_component: Buffer,
    pub(crate) uv_component: Buffer,
    pub(crate) color_component: Buffer,
    pub(crate) info: MeshInfo,
}

pub struct MeshInfo {
    pub num_indices: usize,
}

pub struct MeshDescription<'a> {
    pub label: Option<&'a str>,
    pub indices: &'a [u32],
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub tangents: &'a [Vec3],
    pub uvs: &'a [Vec2],
    pub colors: &'a [Vec3],
}

impl Mesh {
    pub fn new(device: &Device, description: &MeshDescription) -> anyhow::Result<Mesh> {
        debug_assert!(!description.indices.is_empty());
        debug_assert!(!description.positions.is_empty());
        debug_assert!(!description.normals.is_empty());
        debug_assert!(!description.tangents.is_empty());
        debug_assert!(!description.uvs.is_empty());
        debug_assert!(!description.colors.is_empty());

        let index_buffer = device.create_buffer(&BufferDescription {
            label: Some(&format!(
                "Index buffer for mesh {}",
                description.label.unwrap_or("Unknonw")
            )),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::INDEX_BUFFER,
            size: std::mem::size_of_val(description.indices),
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;
        let position_component = device.create_buffer(&BufferDescription {
            label: Some(&format!(
                "Position buffer for mesh {}",
                description.label.unwrap_or("Unknonw")
            )),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
            size: std::mem::size_of_val(description.positions),
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;
        let normal_component = device.create_buffer(&BufferDescription {
            label: Some(&format!(
                "normal buffer for mesh {}",
                description.label.unwrap_or("Unknonw")
            )),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
            size: std::mem::size_of_val(description.normals),
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;
        let tangent_component = device.create_buffer(&BufferDescription {
            label: Some(&format!(
                "tangent buffer for mesh {}",
                description.label.unwrap_or("Unknonw")
            )),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
            size: std::mem::size_of_val(description.tangents),
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;
        let uv_component = device.create_buffer(&BufferDescription {
            label: Some(&format!(
                "uv buffer for mesh {}",
                description.label.unwrap_or("Unknonw")
            )),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
            size: std::mem::size_of_val(description.uvs),
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;

        let color_component = device.create_buffer(&BufferDescription {
            label: Some(&format!(
                "color buffer for mesh {}",
                description.label.unwrap_or("Unknonw")
            )),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
            size: std::mem::size_of_val(description.colors),
            memory_domain: mgpu::MemoryDomain::Gpu,
        })?;

        device.write_buffer(
            index_buffer,
            &BufferWriteParams {
                data: bytemuck::cast_slice(description.indices),
                offset: 0,
                size: index_buffer.size(),
            },
        )?;
        device.write_buffer(
            position_component,
            &BufferWriteParams {
                data: bytemuck::cast_slice(description.positions),
                offset: 0,
                size: position_component.size(),
            },
        )?;
        device.write_buffer(
            normal_component,
            &BufferWriteParams {
                data: bytemuck::cast_slice(description.normals),
                offset: 0,
                size: normal_component.size(),
            },
        )?;
        device.write_buffer(
            tangent_component,
            &BufferWriteParams {
                data: bytemuck::cast_slice(description.tangents),
                offset: 0,
                size: tangent_component.size(),
            },
        )?;
        device.write_buffer(
            uv_component,
            &BufferWriteParams {
                data: bytemuck::cast_slice(description.uvs),
                offset: 0,
                size: uv_component.size(),
            },
        )?;
        device.write_buffer(
            color_component,
            &BufferWriteParams {
                data: bytemuck::cast_slice(description.colors),
                offset: 0,
                size: color_component.size(),
            },
        )?;

        Ok(Mesh {
            index_buffer,
            position_component,
            normal_component,
            tangent_component,
            uv_component,
            color_component,
            info: MeshInfo {
                num_indices: description.indices.len(),
            },
        })
    }
}
