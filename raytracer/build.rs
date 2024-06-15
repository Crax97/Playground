use engine_build_utils::{ShaderCompiler, SourceDirectoryOptions};

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=shaders");

    ShaderCompiler::new()
        .add_source_directory("shaders/", "spirv/", SourceDirectoryOptions::default())
        .compile()
}
