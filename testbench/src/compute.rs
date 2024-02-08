use engine_macros::*;
use gpu::{
    make_gpu, Binding, BufferCreateInfo, BufferUsageFlags, GpuConfiguration, MemoryDomain,
    QueueType, ShaderModuleCreateInfo, ShaderStage,
};
use std::mem::{size_of, size_of_val};

const COMPUTE_SUM: &[u32] = glsl!(
    entry_point = "main",
    kind = compute,
    source = "
#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform Numbers { uint a; uint b; };
layout(set = 0, binding = 1) buffer Output { uint o; };

void main() {
    o = a + b;
}
"
);

fn main() -> anyhow::Result<()> {
    if cfg!(debug_assertions) {
        // Enable all logging in debug configuration
        env_logger::builder()
            .filter(None, log::LevelFilter::Trace)
            .init();
    } else {
        env_logger::init();
    }

    let gpu = make_gpu(GpuConfiguration {
        app_name: "compute sample",
        pipeline_cache_path: None,
        enable_debug_utilities: true,
        window: None,
    })?;

    let compute_module = gpu.make_shader_module(&ShaderModuleCreateInfo {
        label: Some("Compute sum"),
        code: bytemuck::cast_slice(COMPUTE_SUM),
    })?;

    let output_buffer = gpu.make_buffer(
        &BufferCreateInfo {
            label: Some("test buffer"),
            size: size_of::<u32>(),
            usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::STORAGE_BUFFER,
        },
        MemoryDomain::HostVisible,
    )?;

    let inputs = [5, 2];
    let input_buffer = gpu.make_buffer(
        &BufferCreateInfo {
            label: Some("test buffer"),
            size: size_of_val(&inputs),
            usage: BufferUsageFlags::UNIFORM_BUFFER,
        },
        MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
    )?;
    gpu.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&inputs))?;

    let mut command_buffer = gpu.start_command_buffer(QueueType::Graphics)?;
    {
        let mut compute_pass = command_buffer.start_compute_pass();
        compute_pass.set_compute_shader(compute_module);
        compute_pass.bind_resources(
            0,
            &[
                Binding {
                    location: 0,
                    ty: gpu::DescriptorBindingType::UniformBuffer {
                        handle: input_buffer,
                        offset: 0,
                        range: gpu::WHOLE_SIZE as _,
                    },
                    binding_stage: ShaderStage::COMPUTE,
                },
                Binding {
                    location: 1,
                    ty: gpu::DescriptorBindingType::StorageBuffer {
                        handle: output_buffer,
                        offset: 0,
                        range: gpu::WHOLE_SIZE as _,
                    },
                    binding_stage: ShaderStage::COMPUTE,
                },
            ],
        );

        compute_pass.dispatch(1, 1, 1);
    }

    let output: u32 =
        bytemuck::cast_slice(&gpu.read_buffer(&output_buffer, 0, std::mem::size_of::<u32>())?)[0];
    let buffer_data = gpu.read_buffer(&input_buffer, 0, std::mem::size_of::<[u32; 2]>())?;
    let inputs: &[u32] = bytemuck::cast_slice(&buffer_data);
    println!("Output is: {output}, inputs are {inputs:?}");
    Ok(())
}
