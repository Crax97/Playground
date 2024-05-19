#version 460

#include "commons.incl"

#define F0 0.04

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

struct LightInfo {
    vec4 pos_radius;
    // [strength, _]
    vec4 info;
    uvec4 type;
};

layout(set = 1, binding = 1, std140) readonly buffer SceneParameters {
    vec3 eye_location;
    vec3 eye_forward;
    vec3 ambient_color;
    float ambient_intensity;
    
    uint light_count;
    LightInfo lights[];
    
};
 layout(set = 2, binding = 0) uniform textureCube diffuse_env_map;
 layout(set = 2, binding = 1) uniform sampler diffuse_env_map_sampler;

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
    float lit;
};

GBufferData extract_gbuffer() {
    GBufferData data;
    data.color = texture(sampler2D(diffuse, gbuffer_sampler), uv); 
    data.world_position = texture(sampler2D(position, gbuffer_sampler), uv).xyz; 
    vec4 normal_sam = texture(sampler2D(normal, gbuffer_sampler), uv);
    data.normal = normal_sam.xyz; 

    vec4 mr_sample = texture(sampler2D(metallic_roughness, gbuffer_sampler), uv);
    vec4 emissive_ao_sample = texture(sampler2D(emissive_ao, gbuffer_sampler), uv);
    data.metallic = mr_sample.x;
    data.roughness = mr_sample.y;
    data.lit = mr_sample.w;
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

float visibility_smith_mask_shadowing(float aa, float n_dot_l, float n_dot_v) {
    float GGX1 = n_dot_v * sqrt(aa + (1.0-aa) * n_dot_l * n_dot_l);
    float GGX2 = n_dot_l * sqrt(aa + (1.0-aa) * n_dot_v * n_dot_v);
    float GGX = (GGX1 + GGX2);

    if (GGX > 0.0) {
        return 0.5 / GGX;
    }
    
    return 0.0;
}


vec3 fresnel(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Implements the glTF brdf described in https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
vec3 light_contribute(vec3 eye_direction, vec3 light_direction, vec3 half_view_vector, vec3 diffuse_ambient, GBufferData data) {

    vec3 c_diff = mix(data.color.rgb, vec3(0.0), data.metallic);
    vec3 f0 = mix(vec3(0.04), data.color.rgb, data.metallic);
    float aa = data.roughness * data.roughness;
    
    float n_dot_h = saturate(dot(data.normal, half_view_vector));
    float n_dot_v = saturate(dot(data.normal, eye_direction));
    float n_dot_l = saturate(dot(data.normal, light_direction));

    vec3 F = fresnel(n_dot_h, f0, data.roughness);
    vec3 v_diffuse = (1.0 - F) * c_diff * diffuse_ambient * data.ao / PI;
    vec3 v_specular = F * trowbridge_reitz_ggx(aa, half_view_vector, data.normal.xyz)
    * visibility_smith_mask_shadowing(aa, n_dot_l, n_dot_v);
    
    return v_diffuse + v_specular;
}

void main() {
    GBufferData data = extract_gbuffer();


    vec3 light_0 = vec3(0.0);
    if (data.lit > 0.0) {
        for (uint i = 0; i < light_count; i ++) {
            vec3 light_pos = lights[i].pos_radius.xyz;
            float light_radius = lights[i].pos_radius.w;
            vec4 diffuse_ambient = textureLod(samplerCube(diffuse_env_map, diffuse_env_map_sampler), data.normal, (1.0 - data.roughness) * 9.0);
            vec3 light = normalize(data.world_position - light_pos);
            vec3 eye_direction = normalize(eye_location - data.world_position); 
            vec3 half_view_vector = normalize(light + eye_direction);
            
            float dist = distance(light_pos, data.world_position);
            float light_falloff = pow(saturate(1.0 - pow(dist / light_radius, 4)), 2) / (dist * dist + 1.0);

            light_0 += light_contribute(eye_direction, light, half_view_vector, diffuse_ambient.xyz, data) * light_falloff;
        }
    }
    color.rgb =  light_0  + data.emissive; 
    color.a = 1.0;
}


