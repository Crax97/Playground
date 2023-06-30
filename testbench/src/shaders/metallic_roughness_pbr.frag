#version 460

#include "definitions.glsl"

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

layout(location = 0) in FragmentOut fragOut;

void main() {
    outPosition = vec4(fragOut.position, 1.0);

    vec3 T = normalize(vec3(fragOut.model * vec4(fragOut.tangent, 0.0))); // * vec3(-1, -1, 1);
    vec3 N = normalize(vec3(fragOut.model * vec4(fragOut.normal, 0.0))) ; //* vec3(-1, -1, 1);
    vec3 B = normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);
    vec3 sample_normal = texture(normalSampler, fragOut.uv).xyz;
    sample_normal = sample_normal * 2.0 - 1.0;

    sample_normal = normalize(TBN * sample_normal);
    sample_normal = (sample_normal + 1.0) * 0.5;
    
    // outNormal = vec4((N + 1.0) * 0.5, 1.0);
    outNormal = vec4(sample_normal, 1.0);
    outDiffuse = texture(baseColorSampler, fragOut.uv) * pbrProperties.baseColor;
    outEmissive = texture(emissiveSampler, fragOut.uv) * vec4(pbrProperties.emissiveFactor, 1.0);
    outPbr = texture(metallicRoughnessSampler, fragOut.uv) * pbrProperties.metallicRoughness;
}