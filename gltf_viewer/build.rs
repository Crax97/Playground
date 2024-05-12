use engine_build_utils::ShaderCompiler;

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=gltf_viewer/assets");

    ShaderCompiler::new("spirv/")
        .add_source_directory("shaders/materials")
        .compile()
}
