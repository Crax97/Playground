#version 460

#include "definitions.glsl"

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) readonly buffer PerFrameDataBlock {
    PointOfView shadows[];
} per_frame_data;


struct ShadowCaster {
    uvec4 offset_size;
    uvec4 type_nummaps_povs_splitidx;
};

layout(set = 1, binding = 0) uniform sampler2D position_component;
layout(set = 1, binding = 1) uniform sampler2DShadow shadow_atlas;
layout(set = 1, binding = 2) readonly buffer ShadowCasters {
    uint shadow_caster_count;
    ShadowCaster casters[];
} shadow_casters;

layout(set = 1, binding = 3) readonly buffer CsmSplits {
    uint split_count;
    float splits[];
};

void main() {
    color = vec4(1.0);
}




