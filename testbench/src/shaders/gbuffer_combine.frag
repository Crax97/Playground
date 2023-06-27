#version 460

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D posSampler;
layout(set = 0, binding = 1) uniform sampler2D normSampler;
layout(set = 0, binding = 2) uniform sampler2D difSampler;
layout(set = 0, binding = 3) uniform sampler2D emissSampler;
layout(set = 0, binding = 4) uniform sampler2D pbrSampler;

struct LightInfo {
    vec3 position;
    vec3 color;
};

struct FragmentInfo {
    vec3 position;
    vec3 normal;
};

FragmentInfo getFragmentInfo(vec2 inUv) {
    FragmentInfo info;
    info.position = texture(posSampler, inUv).xyz;
    info.normal = texture(normSampler, inUv).xyz;
    return info;
}

vec3 calculateLightInfluence(LightInfo lightInfo, FragmentInfo fragInfo) {
    vec3 dirToPixel = -normalize(fragInfo.position - lightInfo.position);
    float a = clamp(dot(dirToPixel, fragInfo.normal), 0.0, 1.0);
    return a * lightInfo.color;
}

vec3 rgb(int r, int g, int b) {
    return vec3(
        255.0 / float(r),
        255.0 / float(g),
        255.0 / float(b)
    );
}

void main() {
    FragmentInfo fragInfo = getFragmentInfo(uv);
    
    LightInfo testLightInfo;
    testLightInfo.position = vec3(100.0, 50.0, 0.0);
    testLightInfo.color = rgb(247, 143, 111);
        
    vec3 lightA = calculateLightInfluence(testLightInfo, fragInfo);
      
    color = vec4(texture(difSampler, uv).rgb * lightA, 1.0);
}