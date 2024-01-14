#version 450

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D texA;
layout(set = 0, binding = 1) uniform sampler2D texB;


void main() {
    vec2 nuv = vec2(uv.x, 1.0 - uv.y);
    color = vec4(texture(texA, nuv).rgb * texture(texB, nuv).rgb, 1.0);
}
