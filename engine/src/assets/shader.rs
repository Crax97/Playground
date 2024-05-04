use crate::immutable_string::ImmutableString;

#[derive(Hash)]
pub enum Shader {
    Source { content: String },
    Static(ImmutableString),
}
