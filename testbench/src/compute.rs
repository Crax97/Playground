use ash::vk;
use ash::vk::PipelineStageFlags;
use engine_macros::*;
use gpu::DescriptorType::{StorageBuffer, UniformBuffer};
use gpu::{
    BindingElement, BindingType, BufferCreateInfo, BufferRange, BufferUsageFlags, CommandBuffer,
    CommandBufferSubmitInfo, ComputePipeline, ComputePipelineDescription, DescriptorInfo,
    DescriptorSetInfo, GPUFence, GlobalBinding, Gpu, GpuConfiguration, GraphicsPipeline,
    GraphicsPipelineDescription, MemoryDomain, QueueType, ShaderModuleCreateInfo, ShaderStage,
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
    let gpu = Gpu::new(GpuConfiguration {
        app_name: "compute sample",
        engine_name: "compute engine",
        pipeline_cache_path: None,
        enable_debug_utilities: true,
        window: None,
    })?;

    let module = gpu.create_shader_module(&ShaderModuleCreateInfo {
        code: bytemuck::cast_slice(COMPUTE_SUM),
    })?;

    let wait_fence = GPUFence::new(
        &gpu,
        &gpu::FenceCreateInfo {
            flags: gpu::FenceCreateFlags::empty(),
        },
    )?;

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

    let output_buffer = gpu.create_buffer(
        &BufferCreateInfo {
            label: Some("test buffer"),
            size: size_of::<u32>(),
            usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::STORAGE_BUFFER,
        },
        MemoryDomain::HostVisible,
    )?;

    let inputs = [5, 2];
    let input_buffer = gpu.create_buffer(
        &BufferCreateInfo {
            label: Some("test buffer"),
            size: size_of_val(&inputs),
            usage: BufferUsageFlags::UNIFORM_BUFFER,
        },
        MemoryDomain::HostVisible | MemoryDomain::HostCoherent,
    )?;
    input_buffer.write_data::<u32>(0, &inputs);

    let descriptor_set = gpu.create_descriptor_set(&DescriptorSetInfo {
        descriptors: &[
            DescriptorInfo {
                binding: 0,
                element_type: UniformBuffer(BufferRange {
                    handle: &input_buffer,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }),
                binding_stage: ShaderStage::Compute,
            },
            DescriptorInfo {
                binding: 1,
                element_type: StorageBuffer(BufferRange {
                    handle: &output_buffer,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }),
                binding_stage: ShaderStage::Compute,
            },
        ],
    })?;

    let mut command_buffer = CommandBuffer::new(&gpu, QueueType::Graphics)?;
    {
        let mut compute_pass = command_buffer.begin_compute_pass();
        compute_pass.bind_pipeline(&command_pipeline);
        compute_pass.bind_descriptor_sets(&command_pipeline, 0, &[&descriptor_set]);
        compute_pass.dispatch(1, 1, 1);
    }

    command_buffer.submit(&CommandBufferSubmitInfo {
        wait_semaphores: &[],
        wait_stages: &[PipelineStageFlags::COMPUTE_SHADER],
        signal_semaphores: &[],
        fence: Some(&wait_fence),
    })?;

    gpu.wait_for_fences(&[&wait_fence], true, 100000)
        .expect("Fence not triggered!");

    let output = output_buffer.read::<u32>(0);
    let inputs = input_buffer.read::<[u32; 2]>(0);
    println!("Output is: {output}, inputs are {inputs:?}");
    Ok(())
}
