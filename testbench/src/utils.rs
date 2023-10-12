use engine::{Mesh, MeshPrimitiveCreateInfo};
use log::info;
use nalgebra::vector;
use std::path::Path;

use gpu::{ShaderModuleCreateInfo, VkGpu, VkShaderModule};

pub fn read_file_to_vk_module<P: AsRef<Path>>(
    gpu: &VkGpu,
    path: P,
) -> anyhow::Result<VkShaderModule> {
    info!(
        "Reading path from {:?}",
        path.as_ref()
            .canonicalize()
            .expect("Failed to canonicalize path")
    );
    let input_file = std::fs::read(path)?;
    let create_info = ShaderModuleCreateInfo { code: &input_file };
    Ok(gpu.create_shader_module(&create_info)?)
}

pub fn load_cube_to_resource_map(
    gpu: &VkGpu,
    resource_map: &mut resource_map::ResourceMap,
) -> anyhow::Result<resource_map::ResourceHandle<engine::Mesh>> {
    let mesh_create_info = engine::MeshCreateInfo {
        label: Some("cube"),
        primitives: &[MeshPrimitiveCreateInfo {
            indices: vec![
                0, 1, 2, 3, 1, 0, //Bottom
                6, 5, 4, 4, 5, 7, // Front
                10, 9, 8, 8, 9, 11, // Left
                12, 13, 14, 15, 13, 12, // Right
                16, 17, 18, 19, 17, 16, // Up
                22, 21, 20, 20, 21, 23, // Down
            ],
            positions: vec![
                // Back
                vector![-1.0, -1.0, 1.0],
                vector![1.0, 1.0, 1.0],
                vector![-1.0, 1.0, 1.0],
                vector![1.0, -1.0, 1.0],
                // Front
                vector![-1.0, -1.0, -1.0],
                vector![1.0, 1.0, -1.0],
                vector![-1.0, 1.0, -1.0],
                vector![1.0, -1.0, -1.0],
                // Left
                vector![1.0, -1.0, -1.0],
                vector![1.0, 1.0, 1.0],
                vector![1.0, 1.0, -1.0],
                vector![1.0, -1.0, 1.0],
                // Right
                vector![-1.0, -1.0, -1.0],
                vector![-1.0, 1.0, 1.0],
                vector![-1.0, 1.0, -1.0],
                vector![-1.0, -1.0, 1.0],
                // Up
                vector![-1.0, 1.0, -1.0],
                vector![1.0, 1.0, 1.0],
                vector![1.0, 1.0, -1.0],
                vector![-1.0, 1.0, 1.0],
                // Down
                vector![-1.0, -1.0, -1.0],
                vector![1.0, -1.0, 1.0],
                vector![1.0, -1.0, -1.0],
                vector![-1.0, -1.0, 1.0],
            ],
            colors: vec![],
            normals: vec![
                // Back
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                // Front
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                // Left
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                // Right
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                // Up
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                // Down
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
            ],
            tangents: vec![],
            uvs: vec![
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
            ],
        }],
    };
    let cube_mesh = Mesh::new(gpu, &mesh_create_info)?;
    let cube_mesh_handle = resource_map.add(cube_mesh);
    Ok(cube_mesh_handle)
}
