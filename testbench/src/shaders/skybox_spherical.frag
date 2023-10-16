#version 460

#include "definitions.glsl"

layout(set = 0, binding = 0) readonly buffer PerFrameDataBlock {
    uint shadow_count;
    PointOfView pfd[];
} per_frame_data;

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 sample_spherical_map(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outDiffuse;
layout(location = 3) out vec4 outEmissive;
layout(location = 4) out vec4 outPbr;

layout(location = 0) in FragmentOut fragOut;

void main() {
    outPosition = vec4(fragOut.position, 1.0);
    outNormal = vec4(fragOut.normal, 1.0);
    //outDiffuse = texture(baseColorSampler, fragOut.uv);
    outDiffuse = vec4(0.0);
    outEmissive = texture(baseColorSampler, sample_spherical_map(normalize(fragOut.vert_position)));
    outPbr = vec4(0.0);
}
