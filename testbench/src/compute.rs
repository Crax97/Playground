use ash::vk::{BufferUsageFlags, ShaderModuleCreateFlags};
use engine_macros::*;
use gpu::{BindingElement, BindingType, BufferCreateInfo, CommandBuffer, CommandBufferSubmitInfo, ComputePipeline, ComputePipelineDescription, GlobalBinding, Gpu, GpuConfiguration, GpuShaderModule, GraphicsPipeline, GraphicsPipelineDescription, MemoryDomain, QueueType, ShaderModuleCreateInfo, ShaderStage};
use std::mem::{size_of, size_of_val};

const COMPUTE_SUM: &[u32] = glsl!(
    entry_point = "main",
    kind = compute,
    source = "
#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform Numbers { uint nums[2]; };
layout(set = 0, binding = 1) buffer Output { uint o; };

void main() {
    o = nums[0] + nums[1];
}
"
);

fn main() -> anyhow::Result<()> {
    let gpu = Gpu::new(GpuConfiguration {
        app_name: "compute sample",
        engine_name: "compute engine",
        pipeline_cache_path: None,
        enable_debug_utilities: true,
        window: None,
    })?;

    let module = gpu.create_shader_module(&ShaderModuleCreateInfo {
        flags: ShaderModuleCreateFlags::empty(),
        code: &bytemuck::cast_slice(COMPUTE_SUM),
    })?;

    let command_pipeline = ComputePipeline::new(
        &gpu,
        &ComputePipelineDescription {
            module: &module,
            entry_point: "main",
            bindings: &[GlobalBinding {
                set_index: 0,
                elements: &[
                    BindingElement {
                        binding_type: BindingType::Uniform,
                        index: 0,
                        stage: ShaderStage::Compute,
                    },
                    BindingElement {
                        binding_type: BindingType::Storage,
                        index: 1,
                        stage: ShaderStage::Compute,
                    },
                ],
            }],
            push_constant_ranges: &[],
        },
    )?;

    let inputs = [3u32, 2];
    let output_buffer = gpu.create_buffer(
        &BufferCreateInfo {
            label: Some("test buffer"),
            size: size_of::<u32>(),
            usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::STORAGE_BUFFER,
        },
        MemoryDomain::HostVisible,
    )?;

    let input_buffer = gpu.create_buffer(
        &BufferCreateInfo {
            label: Some("test buffer"),
            size: size_of_val(&inputs),
            usage: BufferUsageFlags::UNIFORM_BUFFER,
        },
        MemoryDomain::HostVisible,
    )?;
    input_buffer.write_data(0, &inputs);

    let mut command_buffer = CommandBuffer::new(&gpu, QueueType::Graphics)?;

    {}

    command_buffer.submit(&CommandBufferSubmitInfo {
        wait_semaphores: &[],
        wait_stages: &[],
        signal_semaphores: &[],
        fence: None,
    })?;

    Ok(())
}
