use engine_macros::glsl;
use gpu::*;

const SHADER_SOURCE_VS: &[u32] = glsl!(
    source = "
#version 460

struct Foo {
    float a;
    uint b;
    int c;
};

struct Global {
    float time;
    uint ticks;
    float delta_time;

    mat4 view;
    mat4 projection;

    Foo foo[4];
};

struct Object {
    mat4 model;
};

layout(set = 0, binding = 0, std140) uniform GlobalData {
    Global globals;
};

layout(set = 1, binding = 0) uniform sampler2D some_texture;


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec3 out_position;

layout(push_constant) uniform ObjectData {
        Object object;
};

void main() {
    mat4 mvp = object.model * globals.view * globals.projection;
    vec4 screen_position = vec4(in_position, 1.0) * mvp;
    vec3 sample_dir = texture(some_texture, in_uv).xyz;
    out_position = sin(globals.time) * sample_dir + screen_position.xyz;
}
",
    kind = vertex,
    entry_point = "main"
);

fn main() -> anyhow::Result<()> {
    let gpu = VkGpu::new(GpuConfiguration {
        app_name: "shader reflection sample",
        pipeline_cache_path: None,
        enable_debug_utilities: true,
        window: None,
    })?;

    let shader_module = gpu.make_shader_module(&ShaderModuleCreateInfo {
        code: bytemuck::cast_slice(SHADER_SOURCE_VS),
    })?;

    println!("Dumping infos about shader");

    let shader_info = gpu.get_shader_info(&shader_module);

    for var in &shader_info.inputs {
        println!(
            "layout(location = {}) in {:?} {}",
            var.location, var.format, var.name
        );
    }

    for var in &shader_info.outputs {
        println!(
            "layout(location = {}) out {:?} {}",
            var.location, var.format, var.name
        );
    }

    for (set_idx, set) in shader_info.descriptor_layouts.iter().enumerate() {
        for (binding_idx, binding) in set.bindings.iter().enumerate() {
            println!(
                "layout(set = {}, binding = {}) {:?} {}",
                set_idx, binding_idx, binding.ty, binding.name
            );
        }
    }

    println!("Uniform variables");

    for range in &shader_info.push_constant_ranges {
        println!(
            "layout(push_constant_range) uniform {} {{ /* size: {} */ }}",
            range.name, range.size
        );
    }

    print_uniform_info_recursive(&shader_info.uniform_variables, 0);

    Ok(())
}

fn print_uniform_info_recursive(
    uniform_variables: &std::collections::HashMap<String, UniformVariableDescription>,

    depth: usize,
) {
    for (name, info) in uniform_variables {
        let sep = "\t".repeat(depth);
        println!("{sep}{}: offset {} size {}", name, info.offset, info.size);

        print_uniform_info_recursive(&info.inner_members, depth + 1);
    }
}
