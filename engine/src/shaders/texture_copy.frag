#version 450

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D source;


void main() {
    vec2 nuv = vec2(uv.x, 1.0 - uv.y);
    color = texture(source, nuv);
}
