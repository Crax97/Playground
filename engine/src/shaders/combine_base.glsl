layout (set = 0, binding = 0) uniform sampler2D pos_input;
layout (set = 0, binding = 1) uniform sampler2D norm_input;
layout (set = 0, binding = 2) uniform sampler2D diff_input;
layout (set = 0, binding = 3) uniform sampler2D emiss_input;
layout (set = 0, binding = 4) uniform sampler2D pbr_input;

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
    info.diffuse = texture(diff_input, in_uv).rgb;
    info.emissive = texture(emiss_input, in_uv);
    info.position = texture(pos_input, in_uv).xyz;
    info.normal = texture(norm_input, in_uv).xyz;
    info.normal = info.normal * 2.0 - 1.0;

    vec4 pbr_sample = texture(pbr_input, in_uv);
    info.metalness = pbr_sample.x;
    info.roughness = pbr_sample.y;

    in_uv.y = 1.0 - in_uv.y;
    info.shadow_scale = pbr_sample.w;
    return info;
}
