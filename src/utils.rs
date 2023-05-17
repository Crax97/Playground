use std::{path::Path, ptr::null};

use ash::{
    vk::{ShaderModuleCreateFlags, ShaderModuleCreateInfo, StructureType},
    Device,
};
use log::info;

use crate::gpu::GpuShaderModule;

pub fn read_file_to_vk_module<P: AsRef<Path>>(
    device: &Device,
    path: P,
) -> anyhow::Result<GpuShaderModule> {
    info!(
        "Reading path from {:?}",
        path.as_ref()
            .canonicalize()
            .expect("Failed to canonicalize path")
    );
    let input_file = std::fs::read(path)?;
    let create_info = ShaderModuleCreateInfo {
        s_type: StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: null(),
        flags: ShaderModuleCreateFlags::empty(),
        code_size: input_file.len(),
        p_code: input_file.as_ptr() as *const u32,
    };
    let module = unsafe { device.create_shader_module(&create_info, None) }?;
    Ok(GpuShaderModule::create(device.clone(), &create_info)?)
}
