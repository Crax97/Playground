#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec2 inUv;

layout(set = 0, binding = 0) uniform PerFrameData {
    vec4 eye;
    mat4 view;
    mat4 proj;
} pfd;

layout(push_constant) uniform PerObjectData {
    mat4 model;
} pod;

struct FragmentOut {
    vec3 Position;
    vec3 Normal;
    vec3 Tangent;
    mat4 model;
    mat3 TBN;
    vec2 uv;
    vec3 color;
};

layout(location = 0) out FragmentOut fragOut;

void main() {

    mat4 view_correct = mat4(
      1, 0,  0, 0,
      0, -1, 0, 0,
      0, 0,  1, 0,
      0, 0,  0, 1
    );

    mat4 mv = pfd.proj * view_correct * pfd.view;
    vec4 world_pos = pod.model * vec4(inPosition, 1.0);
    world_pos = world_pos.xzyw;
    gl_Position = mv * world_pos;
    fragOut.color = inColor;
    fragOut.uv = inUv;
    fragOut.Position = world_pos.xyz;
    fragOut.Normal = inNormal;
    fragOut.model = pod.model;
    fragOut.Tangent = inTangent;

    vec3 T = normalize(vec3(pod.model * vec4(inTangent, 0.0)));
    vec3 N = normalize(vec3(pod.model * vec4(inNormal, 0.0)));
    vec3 B = normalize(cross(N, T));
    mat3 TBN = transpose(mat3(T, B, N));
    fragOut.TBN = TBN;
}