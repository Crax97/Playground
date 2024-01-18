layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput pos_input;
layout (input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput norm_input;
layout (input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput diff_input;
layout (input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput emiss_input;
layout (input_attachment_index = 4, set = 0, binding = 4) uniform subpassInput pbr_input;

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
    info.diffuse = subpassLoad(diff_input).rgb;
    info.emissive = subpassLoad(emiss_input);
    info.position = subpassLoad(pos_input).xyz;
    info.normal = subpassLoad(norm_input).xyz;
    info.normal = info.normal * 2.0 - 1.0;

    vec4 pbr_sample = subpassLoad(pbr_input);
    info.metalness = pbr_sample.x;
    info.roughness = pbr_sample.y;

    in_uv.y = 1.0 - in_uv.y;
    info.shadow_scale = pbr_sample.w;
    return info;
}
