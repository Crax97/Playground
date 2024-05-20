#version 460

#include "commons.incl"
#include "pbr.incl"

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
    vec4 color_strength;
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
layout(set = 2, binding = 2) uniform textureCube env_map;
layout(set = 2, binding = 3) uniform sampler env_map_sampler;
layout(set = 2, binding = 4) uniform texture2D brdf_lut;
layout(set = 2, binding = 5) uniform sampler brdf_lut_sampler;
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



// Implements the glTF brdf described in https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
vec3 light_contribute(vec3 eye_direction, vec3 light_direction, vec3 half_view_vector, vec3 f0, GBufferData data) {

    vec3 c_diff = mix(data.color.rgb, vec3(0.0), data.metallic);
    float aa = data.roughness * data.roughness;
    
    float n_dot_h = saturate(dot(data.normal, half_view_vector));
    float n_dot_v = saturate(dot(data.normal, eye_direction));
    float n_dot_l = saturate(dot(data.normal, light_direction));
    float vis_factor = 1.0 / 4.0 * n_dot_l * n_dot_v + 0.0001;

    float k = (data.roughness + 1.0) * (data.roughness + 1.0) / 8;
    vec3 F = fresnel(n_dot_h, f0, data.roughness);
    vec3 D = (vec3(1.0) - F) * data.metallic;
    vec3 v_diffuse = D * c_diff / PI; // * diffuse_ambient * data.ao;
    vec3 v_specular = F * trowbridge_reitz_ggx(aa, n_dot_h)
    * ggx_smith(k, n_dot_l, n_dot_v) * vis_factor;
    
    return (v_diffuse + v_specular) * n_dot_l;
}

void main() {
    GBufferData data = extract_gbuffer();

    vec3 view_dir = normalize(eye_location - data.world_position); 
    vec3 f0 = mix(vec3(F0), data.color.rgb, data.metallic);
    vec3 F = fresnel(saturate(dot(data.normal, view_dir)), f0, data.roughness);
    vec3 d_fact = (1.0 - F) * data.metallic;
    vec3 R = reflect(-view_dir, data.normal);

    vec3 diffuse_sample_dir = vec3(1, -1, 1) * data.normal;
    vec3 diffuse_irradiance = textureLod(samplerCube(diffuse_env_map, diffuse_env_map_sampler), diffuse_sample_dir,  data.roughness * 9.0).rgb;
    vec3 prefiltered = texture(samplerCube(env_map, env_map_sampler), R).rgb;
    vec2 env_brdf = texture(sampler2D(brdf_lut, brdf_lut_sampler), vec2(saturate(dot(data.normal, view_dir)), data.roughness)).xy;
    vec3 ibl_spec = prefiltered * (F * env_brdf.x + env_brdf.y);
    vec3 diffuse_ambient = (d_fact * diffuse_irradiance * data.color.rgb + ibl_spec) * data.ao;

    vec3 light_0 = vec3(0.0);
    if (data.lit > 0.0) {
        for (uint i = 0; i < light_count; i ++) {
            vec3 light_pos = lights[i].pos_radius.xyz;
            float light_radius = lights[i].pos_radius.w;
            vec3 light_color = lights[i].color_strength.xyz;
            float light_strength = lights[i].color_strength.w;

            vec3 light_dir = normalize(light_pos - data.world_position);
            vec3 half_view_vector = normalize(light_dir + view_dir);
            
            float dist = distance(light_pos, data.world_position);
            float light_falloff = pow(saturate(1.0 - pow(dist / light_radius, 4)), 2) / (dist * dist + 1.0);

            light_0 += light_contribute(view_dir, light_dir,  half_view_vector, f0, data)
                * light_color * light_falloff;
        }
    }
    color.rgb =  light_0 + data.color.rgb * diffuse_ambient.rgb * ambient_intensity * data.ao + data.emissive; 
    
    // Perform tonemapping + gamma correction
    color.rgb = color.rgb / (color.rgb + vec3(1.0));
    color.rgb = pow(color.rgb, vec3(1.0/2.2));  
    color.a = 1.0;
}


