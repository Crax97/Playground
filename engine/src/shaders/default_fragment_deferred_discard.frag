#version 460

#include "definitions.glsl"

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outDiffuse;
layout(location = 3) out vec4 outEmissive;
layout(location = 4) out vec4 outPbr;

layout(location = 0) in FragmentOut fragOut;

void main() {
    vec4 s = texture(texSampler, fragOut.uv);
    if (s.a < 0.1) {
        discard;
    }
    outPosition = vec4(fragOut.position, 0.0);
    outNormal = vec4(fragOut.normal, 0.0);
    outDiffuse = s;
}