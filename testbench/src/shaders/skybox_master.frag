#version 460

#include "definitions.glsl"

layout(set = 0, binding = 0) readonly buffer PerFrameDataBlock {
    uint shadow_count;
    PointOfView pfd[];
} per_frame_data;

layout(set = 1, binding = 0) uniform samplerCube baseColorSampler;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outDiffuse;
layout(location = 3) out vec4 outEmissive;
layout(location = 4) out vec4 outPbr;

layout(location = 0) in FragmentOut fragOut;

void main() {
    outPosition = vec4(0.0);
    outNormal = vec4(0.0);
    outDiffuse = vec4(0.0);
    outEmissive = texture(baseColorSampler, normalize(fragOut.vert_position));
    outPbr = vec4(0.0);
}
