pub mod arena;
pub mod cubemap_utils;
pub mod erased_arena;
pub mod fps_limiter;
pub mod immutable_string;
pub mod sampler_allocator;
pub mod shader_cache;
pub mod shader_parameter_writer;
mod tick;

use std::{fs, path::Path};

use shaderc::{CompileOptions, ResolvedInclude, ShaderKind};
pub use tick::Tick;

const SHADERS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/");

pub fn ensure_vec_length<T: Default + Clone>(vec: &mut Vec<T>, index: usize) {
    if vec.len() <= index {
        let diff = index - vec.len() + 1;
        vec.extend(std::iter::repeat(T::default()).take(diff))
    }
}

pub fn compile_glsl(shader_source: &str, shader_kind: ShaderKind) -> anyhow::Result<Vec<u32>> {
    let compiler = shaderc::Compiler::new().expect("Could not create compiler");
    let mut options = CompileOptions::new().unwrap();
    options.set_include_callback(|path, _, _, _| {
        let path = Path::new(SHADERS_PATH).join(path);
        let content = fs::read_to_string(&path).unwrap();

        Ok(ResolvedInclude {
            resolved_name: path.file_name().unwrap().to_string_lossy().into_owned(),
            content,
        })
    });
    let compiled =
        compiler.compile_into_spirv(shader_source, shader_kind, "none", "main", Some(&options))?;

    Ok(compiled.as_binary().to_vec())
}

macro_rules! assert_size_does_not_exceed {
    ($t:ty, $s:expr) => {
        const _: [u8; $s] = [0; std::mem::size_of::<$t>() + $s - std::mem::size_of::<$t>()];
    };
}
pub(crate) use assert_size_does_not_exceed;

#[macro_export]
macro_rules! include_bytes_align_as {
    ($align_ty:ty, $path:literal) => {{
        #[repr(C)]
        pub struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }

        const ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        &ALIGNED.bytes
    }};
}

#[macro_export]
macro_rules! include_spirv {
    ($path:literal) => {
        $crate::include_bytes_align_as!(u32, $path)
    };
}
