#version 460

#pragma GL_GOOGLE_include_directive : require

#include "definitions.glsl"
#include "light_definitions.glsl"

layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D posSampler;
layout(set = 0, binding = 1) uniform sampler2D normSampler;
layout(set = 0, binding = 2) uniform sampler2D difSampler;
layout(set = 0, binding = 3) uniform sampler2D emissSampler;
layout(set = 0, binding = 4) uniform sampler2D pbrSampler;
layout(set = 0, binding = 5) uniform sampler2DShadow shadowMap;

layout(set = 0, binding = 6) readonly buffer  PerFrameDataBlock {
    uint shadow_count;
    PointOfView camera;
    PointOfView shadows[];
} per_frame_data;

layout(set = 0, binding = 7, std140) readonly buffer LightData {
    vec4 ambient_light_color;
    uint light_count;
    LightInfo lights[];
} light_data;

struct FragmentInfo {
    vec3 diffuse;
    vec4 emissive;
    vec3 position;
    vec3 normal;
    float roughness;
    float metalness;
};

struct CubeSample {
		vec2 uv;
		uint face_index;
};

CubeSample sample_cube(vec3 v)
{
	vec3 v_abs = abs(v);
	float ma;
	vec2 uv;
	uint face_index;

	if(v_abs.z >= v_abs.x && v_abs.z >= v_abs.y)
	{
		face_index = v.z < 0.0 ? 5 : 4;
		ma = 0.5 / v_abs.z;
		uv = vec2(v.z < 0.0 ? -v.x : v.x, -v.y);
	}
	else if(v_abs.y >= v_abs.x)
	{
		face_index = v.y < 0.0 ? 3 : 2;
		ma = 0.5 / v_abs.y;
		uv = vec2(v.x, v.y < 0.0 ? -v.z : v.z);
	}
	else
	{
		face_index = v.x < 0.0 ? 1 : 0;
		ma = 0.5 / v_abs.x;
		uv = vec2(v.x < 0.0 ? v.z : -v.z, -v.y);
	}
	uv = uv * ma + 0.5;
	CubeSample sam;
	sam.uv = uv;
	sam.face_index = face_index;
	return sam;
}

vec3 get_unnormalized_light_direction(LightInfo info, FragmentInfo frag_info) {
    if (info.type_shadowcaster.x == DIRECTIONAL_LIGHT) {
        return info.direction.xyz;
    } else if (info.type_shadowcaster.x == SPOT_LIGHT) {
        return info.direction.xyz;
    } else {
        return frag_info.position - info.position_radius.xyz ;
    }
}

float get_light_mask(float n_dot_l, LightInfo light, FragmentInfo frag_info) {
    float attenuation = 1.0;
    vec3 light_dir = light.position_radius.xyz - frag_info.position;
    float light_distance = length(light_dir);
    light_dir /= light_distance;
    float i = 1.0;
    float light_distance_normalized = clamp(light_distance / light.position_radius.w, 0.0, 1.0);
    attenuation = 1.0 / (light_distance_normalized * light_distance_normalized + 0.01);
    if (light.type_shadowcaster.x == POINT_LIGHT) {
        i *= attenuation;
    } else if (light.type_shadowcaster.x == SPOT_LIGHT) {
        vec3 frag_direction = light_dir;
        float cos_theta = dot(-light.direction.xyz, frag_direction);
        float inner_angle_cutoff = light.extras.x;
        float outer_angle_cutoff = light.extras.y;
        float eps = inner_angle_cutoff - outer_angle_cutoff;
        float cutoff = float(cos_theta > 0.0) * clamp((cos_theta - outer_angle_cutoff) / eps, 0.0, 1.0);
        i *= cutoff;
        i *= attenuation;
    }
    return i;
}

FragmentInfo get_fragment_info(vec2 in_uv) {
    FragmentInfo info;
    info.diffuse = texture(difSampler, in_uv).rgb;
    info.emissive = texture(emissSampler, in_uv);
    info.position = texture(posSampler, in_uv).xyz;
    info.normal = texture(normSampler, in_uv).xyz;
    info.normal = info.normal * 2.0 - 1.0;

    vec4 pbr_sample = texture(pbrSampler, in_uv);
    info.metalness = pbr_sample.x;
    info.roughness = pbr_sample.y;

    return info;
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

vec3 cook_torrance(vec3 view_direction, FragmentInfo frag_info, float l_dot_n, vec3 h) {
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

float shadow_map_sample(vec2 uv, float z, vec2 offset, vec2 size) {
    uv *= size;
    uv += offset;

	vec3 loc = vec3(uv, z);
	loc.x = clamp(loc.x, offset.x, offset.x + size.x);
	loc.y = clamp(loc.y, offset.y, offset.y + size.y);
	return texture(shadowMap, loc);
}

float shadow_influence(uint shadow_index, FragmentInfo frag_info, float light_dist) {
    vec2 tex_size = textureSize(shadowMap, 0);
    PointOfView shadow = per_frame_data.shadows[shadow_index];

    mat4 light_vp = shadow.proj * shadow.view;
    vec4 frag_pos_light_unnorm = light_vp * vec4(frag_info.position, 1.0);
    vec4 frag_pos_light = frag_pos_light_unnorm / frag_pos_light_unnorm.w;
    frag_pos_light.xy = frag_pos_light.xy * 0.5 + 0.5;

    if (frag_pos_light.x < 0.0 || frag_pos_light.x > 1.0 ||
    frag_pos_light.y < 0.0 || frag_pos_light.y > 1.0) {
        return 0.0;
    }

    vec2 scaled_light_offset = shadow.viewport_size_offset.xy / tex_size;
    vec2 scaled_light_size = shadow.viewport_size_offset.zw / tex_size;

	return shadow_map_sample(frag_pos_light.xy, frag_pos_light.z, scaled_light_offset, scaled_light_size);
}

float calculate_shadow_influence(FragmentInfo frag_info, LightInfo light_info, vec3 light_dir, float light_dist) {
    float shadow = 0.0;

	int base_shadow_index = light_info.type_shadowcaster.y;
    if (light_info.type_shadowcaster.x == POINT_LIGHT) {
		CubeSample sam = sample_cube(-light_dir);
        return shadow_influence(base_shadow_index + sam.face_index, frag_info, light_dist);
    } else {
        return shadow_influence(base_shadow_index, frag_info, light_dist);
	}
}

vec3 lit_fragment(FragmentInfo frag_info) {

    vec3 view = normalize(per_frame_data.camera.eye.xyz - frag_info.position);

    vec3 ambient_color = light_data.ambient_light_color.xyz * light_data.ambient_light_color.w ;

    vec3 fragment_light = vec3(0.0);

    float overall_mask = 0.0;
    for (uint i = 0; i < light_data.light_count; i ++) {

        LightInfo light_info = light_data.lights[i];
        vec3 light_dir_unnorm = get_unnormalized_light_direction(light_info, frag_info);
        float light_dist = length(light_dir_unnorm);
        vec3 light_dir = -light_dir_unnorm / light_dist;
        float l_dot_n = max(dot(light_dir, frag_info.normal), 0.0);
        vec3 h = normalize(view + light_dir);
        float light_mask = get_light_mask(l_dot_n, light_info, frag_info);
        vec3 masked_light_color = light_mask * light_info.color_intensity.xyz * light_info.color_intensity.w;
        overall_mask += light_mask;
        vec3 light_color = cook_torrance(view, frag_info, l_dot_n, h) * masked_light_color;
        if (light_info.type_shadowcaster.y != -1) {
            light_color *= calculate_shadow_influence(frag_info, light_info, light_dir, light_dist);
        }

        fragment_light += light_color;
    }

    return frag_info.diffuse * (ambient_color + fragment_light);
}

vec3 rgb(int r, int g, int b) {
    return vec3(
        255.0 / float(r),
        255.0 / float(g),
        255.0 / float(b)
    );
}

void main() {
    FragmentInfo fragInfo = get_fragment_info(uv);
    vec3 light_a = lit_fragment(fragInfo);
    color = vec4(light_a, 1.0) + fragInfo.emissive;
}
