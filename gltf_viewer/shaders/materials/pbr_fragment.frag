#version 460

#include "templates/material_fragment_base.incl"

DEFINE_USER_TEXTURE(texture2D, base_color, 0);
DEFINE_USER_TEXTURE(texture2D, normal, 1);
DEFINE_USER_TEXTURE(texture2D, occlusion, 2);
DEFINE_USER_TEXTURE(texture2D, emissive, 3);
DEFINE_USER_TEXTURE(texture2D, metallic_roughness, 4);

USER_PARAMS PbrBlock {
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
};

MaterialAttributes fragment() {
    vec4 diffuse_color = SAMPLE_USER_TEXTURE_2D(base_color, uv); 
    vec4 normal_sample = SAMPLE_USER_TEXTURE_2D(normal, uv) * 2.0 - 1.0; 
    vec4 occlusion_sample = SAMPLE_USER_TEXTURE_2D(occlusion, uv); 
    vec4 emissive_sample = SAMPLE_USER_TEXTURE_2D(emissive, uv); 
    vec4 metallic_roughness_sample = SAMPLE_USER_TEXTURE_2D(metallic_roughness, uv); 

    MaterialAttributes result;
    result.diffuse_color = diffuse_color;
    result.emissive = emissive_sample.xyz;
    result.emissive_strength = 1.0;

    result.normal = get_tbn() * normal_sample.xyz;
    result.metallic = metallic_roughness_sample.b * metallic_factor;
    result.roughness = metallic_roughness_sample.g * roughness_factor;
    result.ambient_occlusion = occlusion_sample.x;
    result.lit = 1.0;

    return result;
}
