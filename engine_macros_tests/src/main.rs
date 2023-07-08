use engine_macros::glsl;

#[allow(dead_code)]
const TEST: &[u32] = glsl!(
    entry_point = "main",
    kind = vertex,
    source = "
#version 460

void main() {
    gl_Position = vec4(vec3(1.0), 1.0);
}
"
);

#[allow(dead_code)]
const TEST2: &[u32] = glsl!(
    kind = fragment,
    source = "
#version 460

layout(location = 0) out vec4 color;

void main() {
    color = vec4(vec3(1.0), 1.0);
}
"
);

#[allow(dead_code)]
const TEST3: &[u32] = glsl!(kind = vertex, path = "src/test.vert");

fn main() {}
