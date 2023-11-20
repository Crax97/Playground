layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput posSampler;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput normSampler;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput difSampler;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput emissSampler;
layout(input_attachment_index = 4, set = 0, binding = 4) uniform subpassInput pbrSampler;

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
