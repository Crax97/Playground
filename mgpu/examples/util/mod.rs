use shaderc::ShaderKind;

pub fn compile_glsl(shader_source: &str, shader_kind: ShaderKind) -> Vec<u32> {
    let compiler = shaderc::Compiler::new().unwrap();
    let compiled = compiler
        .compile_into_spirv(shader_source, shader_kind, "none", "main", None)
        .unwrap();

    compiled.as_binary().to_vec()
}

pub fn read_image_data(path: &str) -> Vec<u8> {
    let image_content = std::fs::read(path).unwrap();
    image::load_from_memory(&image_content)
        .unwrap()
        .to_rgb8()
        .to_vec()
}
