#version 460

#pragma GL_GOOGLE_include_directive : require

#include "definitions.glsl"
#include "random.glsl"
#include "light_definitions.glsl"

layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput posSampler;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput normSampler;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput difSampler;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput emissSampler;
layout(input_attachment_index = 4, set = 0, binding = 4) uniform subpassInput pbrSampler;
layout(set = 0, binding = 5) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 6) uniform samplerCube irradianceMap;

layout(set = 0, binding = 7) readonly buffer  PerFrameDataBlock {
    uint shadow_count;
    PointOfView camera;
    PointOfView shadows[];
} per_frame_data;

layout(set = 0, binding = 8, std140) readonly buffer LightData {
    vec4 ambient_light_color;
    uint light_count;
    LightInfo lights[];
} light_data;

struct FragmentInfo {
    vec3 diffuse;
    vec4 emissive;
    vec3 position;
    vec3 normal;
    float roughness;
    float metalness;
    float shadow_scale;
};

FragmentInfo get_fragment_info(vec2 in_uv) {
    FragmentInfo info;
    info.diffuse = subpassLoad(difSampler).rgb;
    info.emissive = subpassLoad(emissSampler);
    info.position = subpassLoad(posSampler).xyz;
    info.normal = subpassLoad(normSampler).xyz;
    info.normal = info.normal * 2.0 - 1.0;

    vec4 pbr_sample = subpassLoad(pbrSampler);
    info.metalness = pbr_sample.x;
    info.roughness = pbr_sample.y;

    info.shadow_scale = pbr_sample.w;
    return info;
}

void main() {
    FragmentInfo fragInfo = get_fragment_info(uv);
    color = vec4(fragInfo.diffuse, 1.0);
}
