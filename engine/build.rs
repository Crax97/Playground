use engine_build_utils::ShaderCompiler;

fn main() {
    println!("cargo::rerun-if-changed=examples/assets");

    ShaderCompiler::new("examples/spirv/")
        .add_source_directory("examples/shaders/materials")
        .compile()
        .unwrap();
}
