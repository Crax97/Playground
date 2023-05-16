#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec2 inUv;

layout(binding = 0) uniform PerObjectData {
    mat4 model;
    mat4 view;
    mat4 proj;
} per_object_data;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUv;

void main() {
    mat4 mvp = per_object_data.proj * per_object_data.view * per_object_data.model;
    gl_Position = mvp * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragUv = inUv;
}