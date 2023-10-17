#version 460

#include "definitions.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec3 in_tangent;
layout(location = 4) in vec2 in_uv;

layout(push_constant) uniform PerObjectData {
    mat4 mvp;
} pod;

layout(location = 0) out FragmentOut frag_out;

void main() {
    gl_Position = pod.mvp * vec4(in_position, 1.0);
    frag_out.color = in_color;
    frag_out.uv = in_uv;
    frag_out.position = vec3(0.0);
    frag_out.vert_position = in_position;
    frag_out.normal = in_normal;
    frag_out.model = mat4(1.0);
    frag_out.tangent = in_tangent;
}