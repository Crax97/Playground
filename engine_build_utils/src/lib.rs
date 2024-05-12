use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use shaderc::ResolvedInclude;

pub use shaderc;

const ENGINE_SHADERS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../engine/src/shaders/");

pub struct ShaderCompiler<'a> {
    compiler: shaderc::Compiler,
    options: shaderc::CompileOptions<'a>,

    input_directories: Vec<PathBuf>,
    output_directory: PathBuf,
}

impl<'a> ShaderCompiler<'a> {
    pub fn new(output_directory: impl Into<PathBuf>) -> Self {
        let compiler = shaderc::Compiler::new().expect("Could not create compiler");
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_include_callback(|path, _, _, _| {
            let path = Path::new(ENGINE_SHADERS_PATH).join(path);
            let content = fs::read_to_string(&path)
                .inspect_err(|_| eprintln!("In file {path:?}"))
                .unwrap();

            Ok(ResolvedInclude {
                resolved_name: path.file_name().unwrap().to_string_lossy().into_owned(),
                content,
            })
        });

        Self {
            compiler,
            options,
            input_directories: vec![],
            output_directory: output_directory.into(),
        }
    }

    pub fn add_source_directory(mut self, directory: impl AsRef<Path>) -> Self {
        self.input_directories
            .push(directory.as_ref().to_path_buf());
        self
    }

    pub fn compile(self) -> anyhow::Result<()> {
        let _ = fs::remove_dir(&self.output_directory);
        let _ = fs::create_dir(&self.output_directory);
        for directory in self.input_directories {
            let entries = std::fs::read_dir(&directory)
                .context(format!("Could not open directory {:?}", directory))?;
            for entry in entries {
                let entry = entry?;

                let path = if entry.file_type()?.is_file() {
                    entry.path()
                } else {
                    continue;
                };

                let name = entry.file_name();
                let name = name.to_string_lossy();

                let content = std::fs::read_to_string(&path)
                    .context(format!("While reading include file {:?}", path))?;
                let extension = path
                    .extension()
                    .expect("No file extension for file")
                    .to_string_lossy();
                if extension == "incl" {
                    continue;
                }

                let shader_kind = match extension.as_ref() {
                    "vert" => shaderc::ShaderKind::Vertex,
                    "frag" => shaderc::ShaderKind::Fragment,
                    "comp" => shaderc::ShaderKind::Compute,
                    _ => anyhow::bail!("Unrecognized extension {}", extension),
                };
                let spirv = self.compiler.compile_into_spirv(
                    &content,
                    shader_kind,
                    &name,
                    "main",
                    Some(&self.options),
                )?;

                let output_filename = format!("{}.spv", name);
                let output_path = self.output_directory.join(output_filename);
                std::fs::write(&output_path, bytemuck::cast_slice(spirv.as_binary()))
                    .context(format!("Could not find output folder {:?}", output_path))?;
            }
        }
        Ok(())
    }
}
