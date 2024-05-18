#version 460

#include "templates/material_fragment_base.incl"

DEFINE_USER_TEXTURE(textureCube, base_color, 0);

USER_PARAMS PbrBlock {
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
};

MaterialAttributes fragment() {
    vec4 diffuse_color = SAMPLE_USER_TEXTURE_CUBE(base_color, vertex_position); 

    MaterialAttributes result;
    result.diffuse_color = vec4(0.0);
    result.emissive_strength = 1.0;
    result.emissive = diffuse_color.rgb;

    result.roughness = 1.0;

    return result;
}
