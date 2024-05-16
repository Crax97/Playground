#version 460

#include "commons.incl"

#define F0 0.04

#ifndef USE_GLTF_G
    #define USE_GLTF_G 1
#endif

layout(set = 0, binding = 0) uniform sampler gbuffer_sampler;
layout(set = 0, binding = 1) uniform texture2D diffuse;
layout(set = 0, binding = 2) uniform texture2D emissive_ao;
layout(set = 0, binding = 3) uniform texture2D position;
layout(set = 0, binding = 4) uniform texture2D normal;
layout(set = 0, binding = 5) uniform texture2D metallic_roughness;

layout(set = 1, binding = 0, std140) uniform GlobalFrameData {
    mat4 projection;
    mat4 view;
    float frame_time;
};

layout(set = 1, binding = 1, std140) uniform SceneParameters {
    vec3 eye_location;
    vec3 ambient_color;
    float ambient_intensity;
};

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 color;

struct GBufferData {
    vec4 color;
    vec3 world_position;
    vec3 normal;
    float ao;
    float metallic;
    float roughness;
    vec3 emissive;
};

GBufferData extract_gbuffer() {
    GBufferData data;
    data.color = texture(sampler2D(diffuse, gbuffer_sampler), uv); 
    data.world_position = texture(sampler2D(position, gbuffer_sampler), uv).xyz; 
    data.normal = texture(sampler2D(normal, gbuffer_sampler), uv).xyz; 

    vec4 mr_sample = texture(sampler2D(metallic_roughness, gbuffer_sampler), uv);
    vec4 emissive_ao_sample = texture(sampler2D(emissive_ao, gbuffer_sampler), uv);
    data.metallic = mr_sample.x;
    data.roughness = mr_sample.y;
    data.emissive = emissive_ao_sample.xyz;
    data.ao = emissive_ao_sample.w;


    return data;
}

float chi(float x) {
    return x > 0 ? 1.0 : 0.0;
}

float trowbridge_reitz_ggx(float aa, vec3 half_view_vector, vec3 surf_normal) {
    float dot_n_h = dot(surf_normal, half_view_vector);
    // gltf uses top = aa * chi(dot_n_h)
    // epic uses top = aa
    float top = aa;
    float bottom = (dot_n_h * dot_n_h * (aa - 1.0) + 1.0);
    return top / (PI * bottom * bottom);
}

#if USE_GLTF_G
float visibility_smith_mask_shadowing(float a, vec3 half_view_vector, vec3 light, vec3 view_vector, vec3 normal) {
    float aa = a * a;
    float dot_normal_light = dot(normal, light);
    float abs_dot_normal_light = abs(dot_normal_light) ;
    float dot_normal_view = abs(dot(normal, view_vector));
    float abs_dot_normal_view = abs(dot_normal_view);
    float top_1 = 2 * abs_dot_normal_light * chi(dot(half_view_vector, light));
    float top_2 = 2 * abs_dot_normal_view * chi(dot(half_view_vector, view_vector));
    float bottom_1 = abs_dot_normal_light + sqrt(aa + (1-aa) * dot_normal_light * dot_normal_light);
    float bottom_2 = abs_dot_normal_view + sqrt(aa + (1-aa) * dot_normal_view * dot_normal_view);

    float g = (top_1 / bottom_1) * (top_2 / bottom_2);
    return g / (4 * abs_dot_normal_light * abs_dot_normal_view);
}
#else
float visibility_smith_mask_shadowing(float a, vec3 half_view_vector, vec3 light, vec3 view_vector, vec3 normal) {
    float k = (a + 1) * (a + 1) / 8;
    return (dot(normal, light) / (dot(normal, light) * (1 - k) + k)) * (dot(normal, view_vector) / (dot(normal, view_vector) * (1 - k) + k));
}
#endif

vec3 fresnel(vec3 view_direction, vec3 half_view_vector, vec3 f0) {
    return f0 + (1 - f0) * pow(1 - abs(dot(view_direction, half_view_vector)), 5);
}

// Implements the glTF brdf described in https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
vec3 light_contribute(vec3 eye_direction, vec3 light_direction, vec3 half_view_vector, GBufferData data) {

    vec3 c_diff = mix(data.color.rgb, vec3(0.0), data.metallic);
    vec3 f0 = mix(vec3(0.04), data.color.rgb, data.metallic);
    float aa = data.roughness * data.roughness;

    vec3 F = fresnel(eye_direction, half_view_vector, f0);
    vec3 v_diffuse = (1.0 - F) * (1.0 / PI) * c_diff;
    vec3 v_specular = F * trowbridge_reitz_ggx(aa, half_view_vector, data.normal.xyz)
    * visibility_smith_mask_shadowing(data.roughness, half_view_vector, light_direction, eye_direction, data.normal.xyz);
    return v_diffuse + v_specular;
}

void main() {
    GBufferData data = extract_gbuffer();
    vec3 light = normalize(vec3(-1.0, -1.0, -1.0));
    vec3 eye_direction = normalize(eye_location - data.world_position); 
    vec3 half_view_vector = normalize(light + eye_direction);

    vec3 light_0 = light_contribute(eye_direction, light, half_view_vector, data);
    color.rgb =  light_0 * data.ao; 
    color.a = 1.0;
}


