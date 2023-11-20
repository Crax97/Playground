#version 460

#pragma GL_GOOGLE_include_directive : require

#include "definitions.glsl"
#include "random.glsl"
#include "light_definitions.glsl"
#include "combine_base.glsl"

layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;

layout(set = 0, binding = 5) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 6) uniform samplerCube irradianceMap;

layout(set = 0, binding = 7) readonly buffer  PerFrameDataBlock {
    uint shadow_count;
    PointOfView camera;
    PointOfView shadows[];
} per_frame_data;

layout(set = 0, binding = 8, std140) readonly buffer LightData {
    vec4 ambient_light_color;
    uint light_count;
    LightInfo lights[];
} light_data;

void main() {
    FragmentInfo fragInfo = get_fragment_info(uv);
    color = vec4(fragInfo.diffuse, 1.0);
}
