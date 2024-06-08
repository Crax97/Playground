use engine_build_utils::{ShaderCompiler, SourceDirectoryOptions};

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=shaders");

    ShaderCompiler::new()
        .add_source_directory(
            "shaders/materials",
            "spirv/base_pass",
            SourceDirectoryOptions::default(),
        )
        .add_source_directory(
            "shaders/materials",
            "spirv/depth_only",
            SourceDirectoryOptions::default().define("DEPTH_ONLY_PASS"),
        )
        .compile()
}
