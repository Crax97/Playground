#version 460

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec4 rgba_gamma;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 color;


// 0-1 sRGB gamma  from  0-1 linear
// vec3 srgb_gamma_from_linear(vec3 rgb) {
//     bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
//     vec3 lower = rgb * vec3(12.92);
//     vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
//     return mix(higher, lower, vec3(cutoff));
// }

// // 0-1 sRGBA gamma  from  0-1 linear
// vec4 srgba_gamma_from_linear(vec4 rgba) {
//     return vec4(srgb_gamma_from_linear(rgba.rgb), rgba.a);
// }

void main() {
// #if SRGB_TEXTURES
//     vec4 texture_in_gamma = srgba_gamma_from_linear(texture2D(u_sampler, v_tc));
// #else
    vec4 texture_in_gamma = texture(tex, uv);
// #endif

    // We multiply the colors in gamma space, because that's the only way to get text to look right.
    color = rgba_gamma * texture_in_gamma;
}