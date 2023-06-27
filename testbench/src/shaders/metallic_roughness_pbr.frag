#version 460

struct PbrProperties {
    vec4 baseColor;
    
    // x: metallic, y: roughness
    vec4 metallicRoughness;
    vec3 emissiveFactor;
};

layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;
layout(set = 1, binding = 1) uniform sampler2D normalSampler;
layout(set = 1, binding = 2) uniform sampler2D occlusionSampler;
layout(set = 1, binding = 3) uniform sampler2D emissiveSampler;
layout(set = 1, binding = 4) uniform sampler2D metallicRoughnessSampler;
layout(set = 1, binding = 5, std140) uniform PbrPropertiesBlock { 
    PbrProperties pbrProperties;
}; 

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
    outNormal = vec4( fragOut.TBN * texture(normalSampler, fragOut.uv).xyz, 1.0);
    outDiffuse = texture(baseColorSampler, fragOut.uv) * pbrProperties.baseColor;
    outEmissive = texture(emissiveSampler, fragOut.uv) * vec4(pbrProperties.emissiveFactor, 1.0);
    outPbr = texture(metallicRoughnessSampler, fragOut.uv) * pbrProperties.metallicRoughness;
}