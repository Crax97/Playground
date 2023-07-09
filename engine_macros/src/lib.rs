extern crate proc_macro;
use std::path::PathBuf;

use proc_macro::TokenStream;
use quote::quote;
use shaderc::{ResolvedInclude, ShaderKind};
use syn::{parse::Parse, parse_macro_input, ExprLit, Lit, Token};

#[derive(Debug)]
enum GlslSource {
    Path(String),
    Source(String),
}

#[derive(Debug)]
enum GlslShaderKind {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Debug)]
struct GlslInfo {
    content: GlslSource,
    kind: GlslShaderKind,
    entry_point: String,
}

mod kw {
    use syn::parse::Parse;

    syn::custom_keyword!(path);
    syn::custom_keyword!(source);
    syn::custom_keyword!(entry_point);
    syn::custom_keyword!(kind);
    syn::custom_keyword!(vertex);
    syn::custom_keyword!(fragment);
    syn::custom_keyword!(compute);

    #[derive(Debug)]
    pub enum GlslKeyword {
        Path,
        Source,
        EntryPoint,
        Kind,
    }

    impl Parse for GlslKeyword {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            if input.peek(path) {
                input.parse::<path>()?;
                Ok(GlslKeyword::Path)
            } else if input.peek(source) {
                input.parse::<source>()?;
                Ok(GlslKeyword::Source)
            } else if input.peek(entry_point) {
                input.parse::<entry_point>()?;
                Ok(GlslKeyword::EntryPoint)
            } else if input.peek(kind) {
                input.parse::<kind>()?;
                Ok(GlslKeyword::Kind)
            } else {
                Err(input.error("expected path | source | entry_point | kind"))
            }
        }
    }
}

impl GlslInfo {
    fn parse_glsl_path(input: &syn::parse::ParseBuffer<'_>) -> syn::Result<GlslSource> {
        consume_eq(input)?;
        let path = input.parse::<ExprLit>()?;
        let path = match path.lit {
            Lit::Str(s) => s.value(),
            _ => {
                return Err(input.error("path can only be folowed by a literal string"));
            }
        };

        Ok(GlslSource::Path(path))
    }

    fn parse_glsl_source(input: &syn::parse::ParseBuffer<'_>) -> syn::Result<GlslSource> {
        consume_eq(input)?;

        if input.peek(Lit) {
            let source = input.parse::<Lit>()?;
            match source {
                Lit::Str(s) => return Ok(GlslSource::Source(s.value())),
                _ => {
                    return Err(input.error("path can only be folowed by a literal string"));
                }
            };
        }
        Err(input.error("source can only be folowed by a literal string"))
    }

    fn parse_entrypoint(input: &syn::parse::ParseBuffer<'_>) -> syn::Result<String> {
        consume_eq(input)?;

        if input.peek(Lit) {
            let source = input.parse::<Lit>()?;
            match source {
                Lit::Str(s) => return Ok(s.value()),
                _ => {
                    return Err(input.error("path can only be folowed by a literal string"));
                }
            };
        }
        Err(input.error("source can only be folowed by a literal string"))
    }

    fn parse_shader_kind(input: &syn::parse::ParseBuffer<'_>) -> syn::Result<GlslShaderKind> {
        consume_eq(input)?;

        if input.peek(kw::compute) {
            let _ = input.parse::<kw::compute>()?;
            Ok(GlslShaderKind::Compute)
        } else if input.peek(kw::vertex) {
            let _ = input.parse::<kw::vertex>()?;
            Ok(GlslShaderKind::Vertex)
        } else if input.peek(kw::fragment) {
            let _ = input.parse::<kw::fragment>()?;
            Ok(GlslShaderKind::Fragment)
        } else {
            Err(input.error("valid shader types are compute | vertex | fragment"))
        }
    }
}

fn consume_eq(input: &syn::parse::ParseBuffer<'_>) -> Result<(), syn::Error> {
    Ok(if input.peek(Token![=]) {
        input.parse::<Token![=]>()?;
    } else {
        return Err(input.error(format!("Expected equals sign")));
    })
}

impl Parse for GlslInfo {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut go_on = true;

        let mut source = None;
        let mut kind = None;
        let mut entry_point = None;

        while go_on {
            let keyword = kw::GlslKeyword::parse(&input)?;

            match keyword {
                kw::GlslKeyword::Path => {
                    source = Some(Self::parse_glsl_path(&input)?);
                }
                kw::GlslKeyword::Source => {
                    source = Some(Self::parse_glsl_source(&input)?);
                }
                kw::GlslKeyword::EntryPoint => {
                    entry_point = Some(Self::parse_entrypoint(&input)?);
                }
                kw::GlslKeyword::Kind => {
                    kind = Some(Self::parse_shader_kind(&input)?);
                }
            }

            if input.parse::<Token![,]>().is_err() {
                go_on = false;
            }
        }
        match (source, kind) {
            (Some(source), Some(kind)) => Ok(Self {
                content: source,
                kind,
                entry_point: entry_point.unwrap_or("main".to_owned()),
            }),
            _ => Err(input.error("missing 'kind' or 'path|source' attributes")),
        }
    }
}

fn compile_shader(info: GlslInfo) -> anyhow::Result<Vec<u32>> {
    let crate_path = crate_path()?;
    let shader_content = match info.content {
        GlslSource::Path(p) => {
            let path = get_source_file_path(p, &crate_path)?;
            std::fs::read_to_string(path.clone()).map_err(|err| {
                eprintln!("Failed to open path {path:?}: {err}");
                err
            })?
        }
        GlslSource::Source(s) => s,
    };

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_include_callback(|incl, _, _, _| {
        let file_path = std::path::Path::new(incl);
        let file_path = crate_path.join(&file_path);
        let absolute = file_path.canonicalize();
        let absolute = match absolute {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to canonicalize path {file_path:?}");
                return Err(e.to_string());
            }
        };
        let content = std::fs::read_to_string(absolute.clone());
        let content = match content {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to find include file {absolute:?}");
                return Err(e.to_string());
            }
        };

        Ok(ResolvedInclude {
            resolved_name: incl.to_string(),
            content,
        })
    });

    let spirv = compiler.compile_into_spirv(
        &shader_content,
        match info.kind {
            GlslShaderKind::Vertex => ShaderKind::Vertex,
            GlslShaderKind::Fragment => ShaderKind::Fragment,
            GlslShaderKind::Compute => ShaderKind::Compute,
        },
        "inline_glsl_shader",
        &info.entry_point,
        Some(&options),
    )?;
    Ok(spirv.as_binary().to_vec())
}

fn get_source_file_path(path: String, crate_path: &PathBuf) -> Result<PathBuf, anyhow::Error> {
    let path = std::path::Path::new(&path);
    let path = crate_path.join(std::path::Path::new(&path));
    let path = path.canonicalize().map_err(|e| {
        eprintln!("Failed to canonicalize path {path:?}");
        e
    })?;
    Ok(path)
}

fn crate_path() -> Result<PathBuf, anyhow::Error> {
    let crate_path = std::env::var("CARGO_MANIFEST_DIR")?;
    let crate_path = std::path::Path::new(&crate_path);
    Ok(crate_path.to_owned())
}

#[proc_macro]
pub fn glsl(input: TokenStream) -> TokenStream {
    let info = parse_macro_input!(input as GlslInfo);

    let spirv_bytecode = match compile_shader(info) {
        Ok(bc) => bc,
        Err(e) => panic!("{}", e.to_string()),
    };

    make_spirv_bytecode_slice(spirv_bytecode)
}

fn make_spirv_bytecode_slice(spirv_bytecode: Vec<u32>) -> TokenStream {
    let data_iter = spirv_bytecode.iter();
    let tokens = quote! {
        &[#(#data_iter,)*]
    };
    TokenStream::from(tokens)
}
