#version 460

#include "definitions.glsl"

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 sample_spherical_map(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

layout(set = 0, binding = 0) uniform sampler2D baseColorSampler;

layout(location = 0) out vec4 outEmissive;

layout(location = 0) in FragmentOut fragOut;

void main() {
    vec3 sample_dir = normalize(fragOut.vert_position);
    outEmissive = texture(baseColorSampler, sample_spherical_map(sample_dir));
}
