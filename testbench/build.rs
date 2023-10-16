fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=src/shaders/*");
    recompile_all_shaders();
}

fn workspace_path() -> Result<String, anyhow::Error> {
    let output = std::process::Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .output()
        .expect("Failed to cargo locate-project")
        .stdout;

    let cargo_toml_path = String::from_utf8(output)?;
    let mut cargo_path = std::path::PathBuf::from(&cargo_toml_path);
    cargo_path.pop();

    Ok(cargo_path.to_string_lossy().to_string())
}

fn recompile_all_shaders() {
    use shaderc::*;

    let compiler = shaderc::Compiler::new().expect("Failed to create compiler");

    let out_dir = "..";

    let workspace_path = workspace_path().unwrap();
    let out_dir = std::path::Path::new(&out_dir);
    let input_shader_path = std::path::Path::new("src/shaders");
    let out_shader_path = out_dir.join("shaders");
    println!("Compiling shaders in {:?}", out_shader_path);

    let _ = std::fs::create_dir(&out_shader_path);
    let shader_folder =
        std::fs::read_dir(input_shader_path).expect("failed to find input shaders folder");
    let mut options = CompileOptions::new().expect("Failed to create compiler options");
    options.set_include_callback(|incl, _, source, _| {
        let file_name = std::path::Path::new(incl);
        let crate_paths = ["testbench", "engine"];
        for path in crate_paths {
            let file_path = std::path::Path::new(path)
                .join("src")
                .join("shaders")
                .join(file_name);
            let file_path = std::path::Path::new(&workspace_path).join(file_path);
            println!("Checking path {file_path:?}");
            let file_in_path = file_path.canonicalize();
            let file_in_path = match file_in_path {
                Ok(s) => s,
                Err(_) => continue,
            };
            let file_in_path = std::fs::read_to_string(file_in_path);

            if let Ok(s) = file_in_path {
                return Ok(ResolvedInclude {
                    resolved_name: incl.to_string(),
                    content: s,
                });
            }
        }

        Err(format!("Failed to resolve include {incl} in {source}"))
    });
    for file in shader_folder.flatten() {
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
        let new_name = name.clone() + ".spirv";
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
    }
}
