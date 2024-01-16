#version 460

#pragma GL_GOOGLE_include_directive : require

#include "definitions.glsl"
#include "random.glsl"
#include "light_definitions.glsl"
#include "combine_base.glsl"

layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;

layout(set = 0, binding = 5) uniform sampler2D shadowBuffer;
// layout(set = 0, binding = 6) uniform sampler2D lightMask;
layout(set = 0, binding = 7) uniform samplerCube irradianceMap;

layout(set = 0, binding = 8) readonly buffer  PerFrameDataBlock {
    PointOfView camera;
    PointOfView shadows[];
} per_frame_data;

layout(set = 0, binding = 9, std140) readonly buffer LightData {
    vec4 ambient_light_color;
    uint light_count;
    LightInfo lights[];
} light_data;

// layout(set = 0, binding = 8, std140) readonly buffer CsmData {
//     uint csm_count;
//     float csm_splits[];
// } csm_data;

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
vec2 poisson_disk[16] = vec2[](
        vec2(0.0, 0.0),
		vec2( -0.94201624, -0.39906216 ),
		vec2( 0.94558609, -0.76890725 ),
		vec2( -0.094184101, -0.92938870 ),
		vec2( 0.34495938, 0.29387760 ),
		vec2( -0.91588581, 0.45771432 ),
		vec2( -0.81544232, -0.87912464 ),
		vec2( -0.38277543, 0.27676845 ),
		vec2( 0.97484398, 0.75648379 ),
		vec2( 0.44323325, -0.97511554 ),
		vec2( 0.53742981, -0.47373420 ),
		vec2( -0.26496911, -0.41893023 ),
		vec2( 0.79197514, 0.19090188 ),
		vec2( -0.24188840, 0.99706507 ),
		vec2( -0.81409955, 0.91437590 ),
		vec2( 0.19984126, 0.78641367 )
);

vec3 get_unnormalized_light_direction(LightInfo info, vec3 position) {
    if (info.type_shadowcaster.x == DIRECTIONAL_LIGHT) {
        return info.direction.xyz;
    } else if (info.type_shadowcaster.x == SPOT_LIGHT) {
        return info.direction.xyz;
    } else {
        return position - info.position_radius.xyz;
    }
}

float get_light_mask(float n_dot_l, LightInfo light, vec3 position) {
    if (light.type_shadowcaster.x == DIRECTIONAL_LIGHT) {
        // Directional lights are not attenuated
        return 1.0;
    }
    vec3 light_dir = light.position_radius.xyz - position;
    float light_distance = length(light_dir);
    light_dir /= light_distance;
    float attenuation = clamp(1.0 - pow(light_distance / light.position_radius.w, 4.0), 0.0, 1.0) / max(light_distance * light_distance, 0.01);
    if (light.type_shadowcaster.x == SPOT_LIGHT) {
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

vec2 rotate(vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, s, -s, c);
	return m * v;
}

// float shadow_map_sample(vec2 uv, float z, vec2 offset, vec2 size, vec2 tex_size) {
//     uv *= size;
//     uv += offset;

//     vec2 pixel_size = 1.0 / tex_size;

//     if (uv.x < offset.x + pixel_size.x || uv.x > offset.x + size.x - pixel_size.x || 
//         uv.y < offset.y + pixel_size.y || uv.y > offset.y + size.y - pixel_size.y) {
//         return 0.0;
//     }
// 	vec3 loc = vec3(uv, z);
// 	loc.x = clamp(loc.x, offset.x + pixel_size.x, offset.x + size.x - pixel_size.x);
// 	loc.y = clamp(loc.y, offset.y + pixel_size.y, offset.y + size.y - pixel_size.y);
// 	return texture(shadowMap, loc);
// }

// float shadow_influence(uint shadow_index, FragmentInfo frag_info, int light_type, float light_dist) {
//     vec2 tex_size = textureSize(shadowMap, 0);
//     PointOfView shadow = per_frame_data.shadows[shadow_index];

//     mat4 light_vp = shadow.proj * shadow.view;
//     vec4 frag_pos_light_unnorm = light_vp * vec4(frag_info.position, 1.0);
//     vec4 frag_pos_light = frag_pos_light_unnorm / frag_pos_light_unnorm.w;
//     frag_pos_light.xy = frag_pos_light.xy * 0.5 + 0.5;

//     int layer = 0;
//     if (light_type == DIRECTIONAL_LIGHT) {
//         for (int i = 0; i < csm_data.csm_count; i ++) {
//             if (abs(frag_pos_light.z) < csm_data.csm_splits[i]) {
//                 layer = i;
//                 break;
//             }
//         }
//     }

//     vec2 scaled_light_size = shadow.viewport_size_offset.zw / tex_size;
//     vec2 scaled_light_offset = (shadow.viewport_size_offset.xy - vec2(0.5)) / tex_size + vec2(scaled_light_size.x * layer, 0.0);

// 	float sam = 0.0;
// 	for (int i = 0; i < 16; i ++) {
// 		vec2 offset = poisson_disk[i] / tex_size;
// 		offset *= light_dist;
// 		sam += shadow_map_sample(frag_pos_light.xy + offset,
// 				frag_pos_light.z, scaled_light_offset, scaled_light_size, tex_size);
// 	}
// 	return sam / 16.0;
// }

// float calculate_shadow_influence(FragmentInfo frag_info, LightInfo light_info, vec3 light_dir, float light_dist) {
//     float shadow = 0.0;

// 	int base_shadow_index = light_info.type_shadowcaster.y;
//     uint offset = 0;
//     if (light_info.type_shadowcaster.x == POINT_LIGHT) {
// 		CubeSample sam = sample_cube(-light_dir);
//         offset = sam.face_index;
//     }
//     return shadow_influence(base_shadow_index + offset, frag_info, light_info.type_shadowcaster.x, light_dist);
// }

vec3 lit_fragment(FragmentInfo frag_info, vec2 uv) {
    vec3 overall_color = vec3(0.0);
    vec3 view = normalize(per_frame_data.camera.eye.xyz - frag_info.position);
    vec3 ambient_color = light_data.ambient_light_color.xyz * light_data.ambient_light_color.w ;
    vec3 fragment_in_shadow = texture(shadowBuffer, uv).rgb;

    for (uint i = 0; i < light_data.light_count; i ++) {
        LightInfo light_info = light_data.lights[i];

        vec3 light_dir_unnorm = get_unnormalized_light_direction(light_info, frag_info.position);
        float light_dist = length(light_dir_unnorm);
        vec3 light_dir = -light_dir_unnorm / light_dist;
        float l_dot_n = max(dot(light_dir, frag_info.normal), 0.0);
        float light_mask = get_light_mask(l_dot_n, light_info, frag_info.position);
        vec3 masked_light_color = light_mask * light_info.color_intensity.xyz * light_info.color_intensity.w;
        vec3 h = normalize(view + light_dir);
        vec3 light_color = cook_torrance(view, frag_info, l_dot_n, h) * masked_light_color;
        overall_color += light_color;
    }
    vec3 irradiance_sample = texture(irradianceMap, normalize(frag_info.normal)).rgb;
    vec3 fragment_lit = 1.0 - fragment_in_shadow;
    vec3 illumination = ambient_color + overall_color * fragment_lit;
    return frag_info.diffuse * irradiance_sample * illumination;
    

}

vec3 rgb(int r, int g, int b) {
    return vec3(
        255.0 / float(r),
        255.0 / float(g),
        255.0 / float(b)
    );
}

void main() {
    vec2 nuv = uv;
    nuv.y = 1.0 - nuv.y;
    FragmentInfo fragInfo = get_fragment_info(nuv);
    vec3 light_a = lit_fragment(fragInfo, nuv);
    color = fragInfo.shadow_scale * vec4(light_a, 1.0) + fragInfo.emissive;
}
