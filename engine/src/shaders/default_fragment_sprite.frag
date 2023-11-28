#version 460

#include "definitions.glsl"

struct SpriteData {
    uvec4 offset_size;
};

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(set = 1, binding = 1, std140) uniform SpriteDataBlock {
    SpriteData data;
};

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outDiffuse;
layout(location = 3) out vec4 outEmissive;
layout(location = 4) out vec4 outPbr;

layout(location = 0) in FragmentOut fragOut;

void main() {
    vec2 tex_size = textureSize(texSampler, 0);
    vec2 one_over_tex_size = 1.0 / tex_size;
    vec2 offset = one_over_tex_size * data.offset_size.xy;
    vec2 size = one_over_tex_size * data.offset_size.zw;
    vec2 uv = offset + size * fragOut.uv;
    vec4 s = texture(texSampler, uv);
    if (s.a < 0.1) {
        discard;
    }
    outPosition = vec4(fragOut.position, 0.0);
    outNormal = vec4(fragOut.normal, 0.0);
    outDiffuse = s;
}