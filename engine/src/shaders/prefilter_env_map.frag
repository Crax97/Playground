#version 460

#include "commons.incl"
#include "hammersley.incl"
#include "importance_sampling.incl"

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec3 color;

layout(set = 0, binding = 0) uniform textureCube env_map;
layout(set = 0, binding = 1) uniform sampler env_sampler;

layout(push_constant, std140) uniform PrefilterParams {
    mat4 mvp;
    vec4 params;
};

// As described in Karis's Real Shading in Unreal Engine 4 paper
vec3 prefilter_env_map(vec3 normal) {
    vec3 V = normal;
    vec3 N = normal;

    vec3 prefiltered_color = vec3(0.0);
    float total_weight = 0.0;

    const uint num_samples = 4096;
    for (uint i = 0; i < num_samples; i ++) {
        vec2 xi = hammersley(i, num_samples);
        vec3 H = importance_sample_ggx(xi, params.w, N);
        vec3 L = reflect(-V.xyz, H);

        float n_o_l = saturate(dot(N, L));
        if (n_o_l > 0.0) {
            prefiltered_color += texture(samplerCube(env_map, env_sampler), H).rgb * n_o_l;
            total_weight += n_o_l;
        }
    }


    return prefiltered_color / total_weight;
}

void main() {
    color = prefilter_env_map(normalize(position));
}