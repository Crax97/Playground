use std::ops::Deref;
use std::sync::Arc;
use anyhow::anyhow;
use gltf::Document;
use nalgebra::{point, Point3, vector, Vector2, Vector3};

use crate::{asset_map::Asset, math::shape::BoundingShape, AssetLoader, AssetMap, AssetHandle};
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

    pub bounding_shape: BoundingShape,
}

#[derive(Clone)]
pub struct Mesh {
    pub primitives: Vec<MeshPrimitive>,
}

pub struct FileSystemMeshLoader {
    gpu: Arc<dyn Gpu>
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
                    .unwrap_or_else(|| "GPU Mesh".to_owned());
                let label = format!("{label} - primitive {idx}");
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
                Ok(MeshPrimitive {
                    index_buffer,
                    position_component,
                    color_component,
                    normal_component,
                    tangent_component,
                    uv_component,
                    index_count: create_info.indices.len() as _,
                    bounding_shape: BoundingShape::bounding_box_from_points(
                        create_info.positions.iter().map(|v| point![v.x, v.y, v.z]),
                    ),
                })
            })
            .collect();
        let mut generated_primitives = Vec::with_capacity(primitives.len());
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

impl Asset for Mesh {
    fn get_description(&self) -> &str {
        "Mesh"
    }

    fn destroyed(&self, gpu: &dyn Gpu) {
        for prim in &self.primitives {
            gpu.destroy_buffer(prim.position_component);
            gpu.destroy_buffer(prim.normal_component);
            gpu.destroy_buffer(prim.tangent_component);
            gpu.destroy_buffer(prim.uv_component);
            gpu.destroy_buffer(prim.color_component);
            gpu.destroy_buffer(prim.index_buffer);
        }
    }
}

impl AssetLoader for FileSystemMeshLoader {
    type LoadedAsset = Mesh;

    fn load(&self, path: &std::path::Path) -> anyhow::Result<Self::LoadedAsset> {
        let extension = path
            .extension()
            .ok_or(anyhow!("Could not get extension"))?
            .to_str()
            .ok_or(anyhow!("Could not conver OsStr to str"))?;
        match extension {
            "gltf" => {
                Self::load_gltf_file(self.gpu.deref(), path)
            }
            _ => unreachable!("Extension not recognized"),
        }
    }

    fn accepts_extension(&self, extension: &str) -> bool {
        matches!(extension, "gltf")
    }
}

impl FileSystemMeshLoader {
    pub fn new(gpu: Arc<dyn Gpu>) -> Self {
        Self {
            gpu
        }
    }
    fn load_gltf_file(gpu: &dyn Gpu, path: &std::path::Path) -> anyhow::Result<Mesh> {
        let (document, buffers, _) = gltf::import(path)?;
        let mut meshes = Self::load_meshes(gpu, &document, &buffers)?;
        let mesh = meshes.remove(0);
        Ok(mesh.0)
    }
    fn load_meshes(
        gpu: &dyn Gpu,
        document: &Document,
        buffers: &[gltf::buffer::Data],
    ) -> anyhow::Result<Vec<(Mesh, Point3<f32>, Point3<f32>)>> {
        let mut meshes = vec![];
        for mesh in document.meshes() {
            let mut min = point![0.0f32, 0.0, 0.0];
            let mut max = point![0.0f32, 0.0, 0.0];
            let mut primitive_create_infos = vec![];

            for prim in mesh.primitives().take(1) {
                let mut indices = vec![];
                let mut positions = vec![];
                let mut colors = vec![];
                let mut normals = vec![];
                let mut tangents = vec![];
                let mut uvs = vec![];
                let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
                if let Some(iter) = reader.read_indices() {
                    for idx in iter.into_u32() {
                        indices.push(idx);
                    }
                }
                if let Some(iter) = reader.read_positions() {
                    for vert in iter {
                        positions.push(vector![vert[0], vert[1], vert[2]]);
                        min.x = min.x.min(vert[0]);
                        min.y = min.y.min(vert[1]);
                        min.z = min.z.min(vert[2]);
                        max.x = max.x.max(vert[0]);
                        max.y = max.y.max(vert[1]);
                        max.z = max.z.max(vert[2]);
                    }
                }
                if let Some(iter) = reader.read_colors(0) {
                    for vert in iter.into_rgb_f32() {
                        colors.push(vector![vert[0], vert[1], vert[2]]);
                    }
                }
                if let Some(iter) = reader.read_normals() {
                    for vec in iter {
                        normals.push(vector![vec[0], vec[1], vec[2]]);
                    }
                }
                if let Some(iter) = reader.read_tangents() {
                    for vec in iter {
                        tangents.push(vector![vec[0], vec[1], vec[2]]);
                    }
                }
                if let Some(iter) = reader.read_tex_coords(0) {
                    for vec in iter.into_f32() {
                        uvs.push(vector![vec[0], vec[1]]);
                    }
                }
                primitive_create_infos.push(MeshPrimitiveCreateInfo {
                    positions,
                    indices,
                    colors,
                    normals,
                    tangents,
                    uvs,
                });
            }

            let label = format!("Mesh #{}", mesh.index());
            let create_info = MeshCreateInfo {
                label: Some(mesh.name().unwrap_or(&label)),
                primitives: &primitive_create_infos,
            };
            let gpu_mesh = Mesh::new(gpu, &create_info)?;
            meshes.push((gpu_mesh, min, max));
        }
        Ok(meshes)
    }
}
