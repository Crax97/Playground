use std::mem::{size_of, size_of_val};
use ash::vk::BufferUsageFlags;
use gpu::{BindingElement, BindingType, BufferCreateInfo, GlobalBinding, Gpu, GpuConfiguration, MemoryDomain, GraphicsPipeline, GraphicsPipelineDescription, ShaderStage};

fn main() -> anyhow::Result<()> {
    
    
    let gpu = Gpu::new(GpuConfiguration {
        app_name: "compute sample",
        engine_name: "compute engine",
        pipeline_cache_path: None,
        enable_debug_utilities: true,
        window: None,
    })?;
    
    let inputs = [3u32, 2];


    let output_buffer = gpu.create_buffer(&BufferCreateInfo {
        label: Some("test buffer"),
        size: size_of::<u32>(),
        usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::STORAGE_BUFFER,
    }, MemoryDomain::HostVisible)?;
    
    let input_buffer = gpu.create_buffer(&BufferCreateInfo {
        label: Some("test buffer"),
        size: size_of_val(&inputs),
        usage: BufferUsageFlags::UNIFORM_BUFFER,
    }, MemoryDomain::HostVisible)?;
    input_buffer.write_data(0, &inputs);
    
    /*
        let buffer = create_buffer();
    */
    
    
    Ok(())
}