use engine_build_utils::ShaderCompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-changed=examples/assets");
    println!("cargo::rerun-if-changed=src/shaders");

    ShaderCompiler::new("examples/spirv/")
        .add_source_directory("examples/shaders/materials")
        .compile()?;

    ShaderCompiler::new("src/spirv/")
        .add_source_directory("src/shaders/")
        .compile()?;
    Ok(())
}
