#version 450

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outDiffuse;
layout(location = 3) out vec4 outEmissive;
layout(location = 4) out vec4 outPbr;

struct FragmentOut {
    vec3 Position;
    vec3 Normal;
    vec3 Tangent;
    mat4 model;
    mat3 TBN;
    vec2 uv;
    vec3 color;
};

layout(location = 0) in FragmentOut fragOut;

void main() {
    outPosition = vec4(fragOut.Position, 0.0);
    outNormal = vec4(fragOut.Normal, 0.0);
    outDiffuse = texture(texSampler, fragOut.uv);
}