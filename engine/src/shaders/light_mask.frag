#version 460

#include "definitions.glsl"
#include "light_definitions.glsl"

layout (input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput pos_input;
layout (input_attachment_index = 1, set = 1, binding = 1) uniform subpassInput norm_input;
layout (input_attachment_index = 2, set = 1, binding = 2) uniform subpassInput diff_input;
layout (input_attachment_index = 3, set = 1, binding = 3) uniform subpassInput emiss_input;
layout (input_attachment_index = 4, set = 1, binding = 4) uniform subpassInput pbr_input;

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) readonly buffer  PerFrameDataBlock {
    PointOfView camera;
} per_frame_data;
layout(set = 0, binding = 1, std140) readonly buffer LightData {

    vec4 ambient_light_color;
    uint light_count;
    LightInfo lights[];
} light_data;

vec3 get_unnormalized_light_direction(LightInfo info, vec3 position) {
    if (info.type_shadow_map.x == DIRECTIONAL_LIGHT) {
        return info.direction.xyz;
    } else if (info.type_shadow_map.x == SPOT_LIGHT) {
        return info.direction.xyz;
    } else {
        return position - info.position_radius.xyz ;
    }
}

float get_light_mask(float n_dot_l, LightInfo light, vec3 position) {
    if (light.type_shadow_map.x == DIRECTIONAL_LIGHT) {
        // Directional lights are not attenuated
        return 1.0;
    }
    vec3 light_dir = light.position_radius.xyz - position;
    float light_distance = length(light_dir);
    light_dir /= light_distance;
    float attenuation = clamp(1.0 - pow(light_distance / light.position_radius.w, 4.0), 0.0, 1.0) / max(light_distance * light_distance, 0.01);
    if (light.type_shadow_map.x == SPOT_LIGHT) {
        float inner_angle_cutoff = light.extras.x;
        float outer_angle_cutoff = light.extras.y;

        vec3 frag_direction = light_dir;
        float cos_theta = dot(-light.direction.xyz, frag_direction);
        float eps = inner_angle_cutoff - outer_angle_cutoff;
        float spotlight_cutoff = float(cos_theta > 0.0) * clamp((cos_theta - outer_angle_cutoff) / eps, 0.0, 1.0);
        attenuation *= spotlight_cutoff;
    }
    return attenuation;
}

float ggx_smith(float v_dot_n, float v_dot_l, float a)
{
    float r = a + 1.0;
    float k = (r * r) / 8.0;

    float gv = v_dot_n / (v_dot_n * (1.0 - k) + k);
    float gl = v_dot_l / (v_dot_l * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnel_schlick(float cos_theta, vec3 F0, vec3 F90)
{
    return F0 + (F90 - F0) * pow(1.0 - cos_theta, 5.0);
}

float d_trowbridge_reitz_ggx(float n_dot_h, float rough)
{
    float a = rough * rough;
    float n_dot_h_2 = n_dot_h * n_dot_h;
    float a_2 = a * a;
    float a_2_sub = a - 1.0;

    float d = n_dot_h_2 * a_2_sub + 1.0;
    return a_2 / (PI * d * d);
}

struct LightFragmentInfo {
    vec3 position;
    vec3 normal;
    vec3 diffuse;
    float metalness;
    float roughness;
};

vec3 cook_torrance(vec3 view_direction, LightFragmentInfo frag_info, float l_dot_n, vec3 h) {
    float v_dot_n = max(dot(view_direction, frag_info.normal), 0.0);
    float n_dot_h = max(dot(frag_info.normal, h), 0.0);
    float h_dot_v = max(dot(h, view_direction), 0.0);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, frag_info.diffuse, frag_info.metalness);

    // Reflective component
    float d = d_trowbridge_reitz_ggx(n_dot_h, frag_info.roughness);
    float g = ggx_smith(v_dot_n, l_dot_n, frag_info.roughness);
    vec3  f = fresnel_schlick(h_dot_v, F0, vec3(1.0));
    vec3 dfg = d * g * f;

    const float eps = 0.0001;

    float denom = max(4.0 * (l_dot_n * v_dot_n), eps);
    vec3 s_cook_torrance = dfg / denom;

    // Refracftion component
    vec3 lambert = frag_info.diffuse / PI;
    vec3 ks = f;
    vec3 kd = mix(vec3(1.0) - f, vec3(0.0), frag_info.metalness);
    vec3 o = (kd * lambert + s_cook_torrance) * l_dot_n;
    return vec3(o);
}

void main() {
    vec3 pos = subpassLoad(pos_input).rgb;
    vec3 normal = subpassLoad(norm_input).rgb;
    vec3 diffuse = subpassLoad(diff_input).rgb;
    vec3 emissive = subpassLoad(emiss_input).rgb;
    vec4 pbr = subpassLoad(pbr_input);

        
    LightFragmentInfo info;
    info.position = pos;
    info.normal = normal;
    info.diffuse = diffuse;
    info.metalness = pbr.x;
    info.roughness = pbr.y;

    vec3 overall_color = vec3(0.0);
    vec3 view = normalize(per_frame_data.camera.eye.xyz - pos);

    for (uint i = 0; i < light_data.light_count; i ++) {
        LightInfo light_info = light_data.lights[i];

        vec3 light_dir_unnorm = get_unnormalized_light_direction(light_info, pos);
        float light_dist = length(light_dir_unnorm);
        vec3 light_dir = -light_dir_unnorm / light_dist;
        float l_dot_n = max(dot(light_dir, normal), 0.0);
        float light_mask = get_light_mask(l_dot_n, light_info, pos);
        vec3 masked_light_color = light_mask * light_info.color_intensity.xyz * light_info.color_intensity.w;
        vec3 h = normalize(view + light_dir);
        vec3 light_color = cook_torrance(view, info, l_dot_n, h) * masked_light_color;
        overall_color += light_color;
    }

    color = vec4(overall_color, 1.0);
}
