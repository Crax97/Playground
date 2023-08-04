use std::path::Path;

use ash::vk::ShaderModuleCreateFlags;
use log::info;

use gpu::{Gpu, GpuShaderModule, ShaderModuleCreateInfo};

pub fn read_file_to_vk_module<P: AsRef<Path>>(
    gpu: &Gpu,
    path: P,
) -> anyhow::Result<GpuShaderModule> {
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
