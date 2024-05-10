#version 460

#include "templates/material_fragment_base.incl"

DEFINE_USER_TEXTURE(texture2D, tex, 0);

MaterialAttributes fragment() {
    vec4 diffuse_color = SAMPLE_USER_TEXTURE_2D(tex, uv); 

    MaterialAttributes result;
    result.diffuse_color = diffuse_color;
    return result;
}
