#version 460

layout(set = 0, binding = 0) uniform sampler gbuffer_sampler;
layout(set = 0, binding = 1) uniform texture2D diffuse;
layout(set = 0, binding = 2) uniform texture2D emissive_ao;
layout(set = 0, binding = 3) uniform texture2D normal;
layout(set = 0, binding = 4) uniform texture2D metallic_roughness;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 color;

struct GBufferData {
    vec4 color;
};

GBufferData extract_gbuffer() {
    GBufferData data;
    data.color = texture(sampler2D(diffuse, gbuffer_sampler), uv); 

    return data;
}


void main() {
    GBufferData data = extract_gbuffer();
    color = data.color;
}

