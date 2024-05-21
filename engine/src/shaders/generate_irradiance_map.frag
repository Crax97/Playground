#version 460

#include "commons.incl"
#include "hammersley.incl"
#include "importance_sampling.incl"

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform textureCube env_map;
layout(set = 0, binding = 1) uniform sampler env_sampler;

layout(push_constant, std140) uniform PrefilterParams {
    mat4 mvp;
    vec4 params;
};

// As described on https://learnopengl.com/PBR/IBL/Diffuse-irradiance
vec3 gen_irradiance_map(vec3 normal) {
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    vec3 color = vec3(0.0);
    float total_weight = 0.0;

    const float STEP = 0.025;
    for (float phi = 0.0; phi < 2.0 * PI; phi += STEP) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += STEP) {
            vec3 T = vec3(sin(theta) * cos(phi), sin(theta)*sin(phi), cos(theta));
            vec3 sampleDir = T.x * right + T.y * up + T.z * normal;
            color += texture(samplerCube(env_map, env_sampler), sampleDir).rgb * cos(theta) * sin(theta);
            total_weight += 1.0;
        }
    }

    return PI * color / total_weight;
}

void main() {
    color = vec4(gen_irradiance_map(normalize(position)), 1.0);
}