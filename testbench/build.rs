fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=src/shaders/*");
    recompile_all_shaders();
}

fn recompile_all_shaders() {
    use shaderc::*;

    let compiler = shaderc::Compiler::new().expect("Failed to create compiler");

    let out_dir = "..";

    let out_dir = std::path::Path::new(&out_dir);
    let input_shader_path = std::path::Path::new("src/shaders");
    let out_shader_path = out_dir.join("shaders");
    println!("Compiling shaders in {:?}", out_shader_path);

    let _ = std::fs::create_dir(&out_shader_path);
    let shader_folder =
        std::fs::read_dir(input_shader_path).expect("failed to find input shaders folder");
    let mut options = CompileOptions::new().expect("Failed to create compiler options");
    options.set_include_callback(|incl, _, _, _| {
        let file_path = input_shader_path.join(incl);
        let absolute = file_path.canonicalize();
        let absolute = match absolute {
            Ok(s) => s,
            Err(e) => {
                return Err(e.to_string());
            }
        };
        let content = std::fs::read_to_string(&absolute);
        let content = match content {
            Ok(s) => s,
            Err(e) => {
                return Err(e.to_string());
            }
        };

        Ok(ResolvedInclude {
            resolved_name: incl.to_string(),
            content,
        })
    });
    for file in shader_folder.flatten() {
        if let Ok(file) = file {
            let fty = file.file_type().expect("Could not get filetype");
            if fty.is_dir() {
                continue;
            }
            let path = file.path();
            let path = std::path::Path::new(&path);
            let name = path
                .file_stem()
                .expect("Failed to get file name")
                .to_string_lossy();
            let new_name = name + ".spirv";
            let new_name = out_shader_path
                .join(std::path::Path::new(&new_name.to_string()))
                .to_owned();

            println!("Compiling {:?}", &path);
            let extension = path.extension().expect("Shader has no extension!");
            if extension == "glsl" {
                continue;
            }
            let extension = match extension.to_str().unwrap() {
                "vert" => ShaderKind::Vertex,
                "frag" => ShaderKind::Fragment,
                "compute" => ShaderKind::Compute,
                _ => panic!("Invalid shader extension!"),
            };

            let source = std::fs::read_to_string(path).expect("Failed to read input file");

            let compiled =
                compiler.compile_into_spirv(&source, extension, &name, "main", Some(&options));

            match compiled {
                Ok(new) => std::fs::write(new_name, new.as_binary_u8()).unwrap(),
                Err(e) => panic!("Failed to compile shader {}! Error: {}", name, e),
            }
        } else {
            println!("Failed to open shader folder!");
        }
    }
}
