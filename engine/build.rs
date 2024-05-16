use engine_build_utils::{ShaderCompiler, SourceDirectoryOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-changed=examples/assets");
    println!("cargo::rerun-if-changed=src/shaders");

    ShaderCompiler::new()
        .add_source_directory(
            "examples/shaders/materials",
            "examples/spirv/",
            SourceDirectoryOptions::default(),
        )
        .add_source_directory(
            "src/shaders/",
            "src/spirv/",
            SourceDirectoryOptions::default(),
        )
        .compile()?;

    Ok(())
}
