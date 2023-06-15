#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec2 inUv;

layout(set = 0, binding = 0) uniform PerFrameData {
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
    vec2 uv;
    vec3 color;
};

layout(location = 0) out FragmentOut fragOut;

void main() {
    mat4 mv = pfd.proj * pfd.view;
    vec4 WorldPos = pod.model * vec4(inPosition, 1.0);
    gl_Position = mv * WorldPos;
    fragOut.color = inColor;
    fragOut.uv = inUv;
    fragOut.Position = WorldPos.xyz;
    fragOut.Normal = inNormal;
    fragOut.Tangent = inTangent;

}