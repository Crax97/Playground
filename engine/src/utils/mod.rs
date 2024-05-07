pub mod arena;
pub mod erased_arena;
pub mod immutable_string;
pub mod sampler_allocator;
pub mod shader_cache;
mod tick;

use shaderc::ShaderKind;
pub use tick::Tick;

pub fn ensure_vec_length<T: Default + Clone>(vec: &mut Vec<T>, index: usize) {
    if vec.len() <= index {
        let diff = index - vec.len() + 1;
        vec.extend(std::iter::repeat(T::default()).take(diff))
    }
}

pub fn compile_glsl(shader_source: &str, shader_kind: ShaderKind) -> anyhow::Result<Vec<u32>> {
    let compiler = shaderc::Compiler::new().expect("Could not create compiler");
    let compiled = compiler.compile_into_spirv(shader_source, shader_kind, "none", "main", None)?;

    Ok(compiled.as_binary().to_vec())
}

macro_rules! assert_size_does_not_exceed {
    ($t:ty, $s:expr) => {
        const _: [u8; $s] = [0; std::mem::size_of::<$t>() + $s - std::mem::size_of::<$t>()];
    };
}
pub(crate) use assert_size_does_not_exceed;
