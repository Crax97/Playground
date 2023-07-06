#version 450

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D source;

// Reinhard Tonemapping
void main() {
    vec4 col = texture(source, uv);
    color = col / (col + vec4(1.0));
}