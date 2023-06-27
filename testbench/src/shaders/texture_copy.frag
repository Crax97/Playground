#version 450

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D source;


void main() {
    vec2 nuv = uv;
    nuv.y = 1.0 - nuv.y;
    color = texture(source, nuv);
}