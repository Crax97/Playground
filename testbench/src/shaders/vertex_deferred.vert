#version 460

#include "definitions.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec3 in_tangent;
layout(location = 4) in vec2 in_uv;

layout(set = 0, binding = 0) readonly buffer PerFrameDataBlock {
    uint shadow_count;
    PerFrameData pfd[];
} per_frame_data;

layout(push_constant) uniform PerObjectData {
    mat4 model;
    uint camera_index;
} pod;

layout(location = 0) out FragmentOut frag_out;

void main() {
    mat4 mv = per_frame_data.pfd[pod.camera_index].proj * per_frame_data.pfd[pod.camera_index].view;
    vec4 world_pos = pod.model  * vec4(in_position, 1.0);
    gl_Position = mv * world_pos;
    frag_out.color = in_color;
    frag_out.uv = in_uv;
    frag_out.position = world_pos.xyz;
    frag_out.normal = in_normal;
    frag_out.model = pod.model;
    frag_out.tangent = in_tangent;

    vec3 T = normalize(vec3(pod.model * vec4(in_tangent, 0.0)));
    vec3 N = normalize(vec3(pod.model * vec4(in_normal, 0.0)));
    vec3 B = normalize(cross(N, T));
    mat3 TBN = transpose(mat3(T, B, N));
    frag_out.TBN = TBN;
}