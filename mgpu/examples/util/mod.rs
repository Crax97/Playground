use shaderc::ShaderKind;

#[allow(dead_code)]
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Default)]
pub struct Position {
    pub pos: [f32; 3],
}

#[allow(dead_code)]
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Default)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
}

#[allow(dead_code)]
pub fn pos(x: f32, y: f32, z: f32) -> Position {
    Position { pos: [x, y, z] }
}

#[allow(dead_code)]
pub fn vertex(x: f32, y: f32, z: f32, uv_x: f32, uv_y: f32) -> Vertex {
    Vertex {
        pos: [x, y, z],
        uv: [uv_x, uv_y],
    }
}

#[allow(dead_code)]
pub fn compile_glsl(shader_source: &str, shader_kind: ShaderKind) -> Vec<u32> {
    let compiler = shaderc::Compiler::new().unwrap();
    let compiled = compiler
        .compile_into_spirv(shader_source, shader_kind, "none", "main", None)
        .unwrap();

    compiled.as_binary().to_vec()
}

#[allow(dead_code)]
pub fn read_image_data(path: &str) -> Vec<u8> {
    let image_content = std::fs::read(path).unwrap();
    image::load_from_memory(&image_content)
        .unwrap()
        .to_rgba8()
        .to_vec()
}
