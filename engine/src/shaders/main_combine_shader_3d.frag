#version 460

#extension GL_GOOGLE_include_directive : require


#include "definitions.glsl"
#include "random.glsl"
#include "light_definitions.glsl"
#include "combine_base.glsl"
#include "debug.glsl"

struct ShadowMap {
    uvec4 offset_size;
    uvec4 type_nummaps_povs_splitidx;
};

layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;

layout(set = 1, binding = 0) uniform sampler2D shadow_atlas;
layout(set = 1, binding = 1) readonly buffer ShadowMaps {
    ShadowMap casters[];
} shadow_maps;
layout(set = 1, binding = 2) uniform samplerCube irradianceMap;

layout(set = 1, binding = 3) readonly buffer  PerFrameDataBlock {
    PointOfView camera;
    PointOfView light_povs[];
} per_frame_data;

layout(set = 1, binding = 4, std140) readonly buffer LightData {
    vec4 ambient_light_color;
    uint light_count;
    LightInfo lights[];
} light_data;

layout(set = 1, binding = 5, std140) readonly buffer CsmData {
    uvec4 csm_settings;
    vec4 splits[];
} csm_data;

struct CubeSample {
    vec2 uv;
    uint face_index;
};

float get_csm_split(uint index) {
    uint base = index / 4;
    uint offset = index % 4;
    vec4 split_vec = csm_data.splits[base];
    return split_vec[offset];
}

uint get_csm_count() {
    return csm_data.csm_settings[0];
}

bool is_csm_debug_enabled() {
    return csm_data.csm_settings[1] != 0;
}

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

vec3 get_unnormalized_light_direction(int light_type, vec3 light_direction, vec3 light_position, vec3 position) {
    if (light_type == DIRECTIONAL_LIGHT) {
        return light_direction;
    } else if (light_type == SPOT_LIGHT) {
        return light_direction;
    } else {
        return position - light_position;
    }
}

float get_light_mask(float n_dot_l, int light_type, vec3 light_position,  LightInfo light, vec3 position) {
    if (light.type_shadow_map_csmsplit_idx.x == DIRECTIONAL_LIGHT) {
        // Directional lights are not attenuated
        return 1.0;
    }
    vec3 light_dir = light.position_radius.xyz - position;
    float light_distance = length(light_dir);
    light_dir /= light_distance;
    float attenuation = clamp(1.0 - pow(light_distance / light.position_radius.w, 4.0), 0.0, 1.0) / max(light_distance * light_distance, 0.01);
    if (light.type_shadow_map_csmsplit_idx.x == SPOT_LIGHT) {
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
    if (info.type_shadow_map_csmsplit_idx.x == DIRECTIONAL_LIGHT) {
        return info.direction.xyz;
    } else if (info.type_shadow_map_csmsplit_idx.x == SPOT_LIGHT) {
        return info.direction.xyz;
    } else {
        return position - info.position_radius.xyz;
    }
}

float get_light_mask(float n_dot_l, LightInfo light, vec3 position) {
    if (light.type_shadow_map_csmsplit_idx.x == DIRECTIONAL_LIGHT) {
        // Directional lights are not attenuated
        return 1.0;
    }
    vec3 light_dir = light.position_radius.xyz - position;
    float light_distance = length(light_dir);
    light_dir /= light_distance;
    float attenuation = clamp(1.0 - pow(light_distance / light.position_radius.w, 4.0), 0.0, 1.0) / max(light_distance * light_distance, 0.01);
    if (light.type_shadow_map_csmsplit_idx.x == SPOT_LIGHT) {
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

float find_search_width(float light_size, float receiver_distance, vec3 view_position) {
    const float NEAR = 0.1;
    return light_size * (receiver_distance - NEAR) / view_position.z;
}

vec3 compute_sample_location(vec3 light_pos, ShadowMap shadow_map) {
    vec2 texture_size = textureSize(shadow_atlas, 0);
    vec2 texel_size = 1.0 / texture_size;

    vec2 shadow_map_size = shadow_map.offset_size.zw * texel_size;
    vec2 shadow_map_offset = shadow_map.offset_size.xy * texel_size;
    light_pos.y = 1.0 - light_pos.y;
    light_pos.xy = shadow_map_offset + light_pos.xy * shadow_map_size;
    light_pos.x = clamp(light_pos.x, shadow_map_offset.x, shadow_map_offset.x + shadow_map_size.x);
    light_pos.y = clamp(light_pos.y, shadow_map_offset.y, shadow_map_offset.y + shadow_map_size.y);
    
    return light_pos;
}

float find_blocker_distance(LightInfo light_info, ShadowMap shadow_map, vec3 light_pos, vec3 eye_position) {
    int blockers = 0;
    float blocker_distances_sum = 0.0;
    float search_width = find_search_width(light_info.position_radius.w, light_pos.z, eye_position);
    const int NUM_BLOCKER_SAMPLES = 4;
    for (int i = 0; i < NUM_BLOCKER_SAMPLES; i ++) {
        vec3 sample_loc = compute_sample_location(light_pos + vec3(random_direction_disk(light_pos.xy) / search_width, 0.0), shadow_map);
        float compare_z = sample_loc.z;
        float stored_z = texture(shadow_atlas, sample_loc.xy).r;
        if (compare_z > stored_z) {
            blockers +=1;
            blocker_distances_sum += stored_z;
        } 
    }
    if (blockers > 0) {
        return blocker_distances_sum / blockers;
    } else {
        return -1.0;
    }
}

// returns 1.0 if pixel is in shadow, 0 otherwise
float sample_shadow_atlas(vec3 light_pos, ShadowMap shadow_map) {
    light_pos = compute_sample_location(light_pos, shadow_map);
    float compare_z = light_pos.z; 
    float stored_z = texture(shadow_atlas, light_pos.xy).r;
    return compare_z > stored_z ? 1.0 : 0.0;
}

uint select_directiona_light_offset(vec4 pos_in_view, int base){

    float depth = pos_in_view.z;
    uint csm_count = get_csm_count();
    uint offset = 0;
    for (uint i = 0; i < csm_count - 1; i++)
    {
		if (depth < get_csm_split(base + i)) {
			offset = i + 1;
		}
    }

    return offset;
}

ShadowMap select_directiona_light_layer(int first_shadow_map, vec2 uv, vec3 pixel_pos_world, int base_split) {
    vec4 pos_in_view = per_frame_data.camera.view * vec4(pixel_pos_world, 1.0);

    printf("Base %d 1 %f 2 %f 3 %f 4 %f", base_split, get_csm_split(base_split),get_csm_split(base_split + 1),get_csm_split(base_split + 2),   get_csm_split(base_split + 3) );
    uint offset = select_directiona_light_offset(pos_in_view, base_split);

    return shadow_maps.casters[first_shadow_map + offset];
}

ShadowMap select_point_light_face(int first_shadow_map, vec3 direction) {
    CubeSample sam = sample_cube(-direction);
    return shadow_maps.casters[first_shadow_map + sam.face_index];
}

float pcf(FragmentInfo frag_info, vec3 light_proj, ShadowMap caster, float penumbra_size) {
    const int NUM_SAMPLES = 16;
    float accum = 0.0;
    for (int i = 0; i < NUM_SAMPLES; i ++) {
        light_proj.xy += rotate(poisson_disk[i], random(frag_info.position)) * penumbra_size;
        accum += sample_shadow_atlas(light_proj, caster);
    }
    return accum / NUM_SAMPLES;
}

float is_in_shadow(FragmentInfo frag_info, LightInfo light_info, vec3 light_proj, ShadowMap caster, vec3 eye_position) {
    const int do_pcf = 1;

    if (do_pcf == 1) {
        float blocker_distance = find_blocker_distance(light_info, caster, light_proj, eye_position);
        float penumbra_size = (light_proj.z - blocker_distance) / blocker_distance;
        penumbra_size = penumbra_size * light_info.position_radius.w * 0.0001 / light_proj.z;
        return 1.0 - pcf(frag_info, light_proj, caster, penumbra_size);
    } else {
        return 1.0 - sample_shadow_atlas(light_proj, caster);
    }
}

float is_fragment_lit(vec2 uv, FragmentInfo frag_info, LightInfo light_info, vec3 pixel_pos) {
    int light_type = light_info.type_shadow_map_csmsplit_idx[0];
    int light_shadow_map = light_info.type_shadow_map_csmsplit_idx[1];
    int base_split = light_info.type_shadow_map_csmsplit_idx[2];
    if (light_shadow_map == -1) {
        return 1.0;
    }

    ShadowMap caster;
    if (light_type == DIRECTIONAL_LIGHT) {
        caster = select_directiona_light_layer(light_shadow_map, uv, pixel_pos, base_split);
    } else if (light_type == POINT_LIGHT) {
        caster = select_point_light_face(light_shadow_map, light_info.direction.xyz);
    } else {
        caster = shadow_maps.casters[light_shadow_map];
    }

    uint pov_idx = caster.type_nummaps_povs_splitidx[2];
    PointOfView pov = per_frame_data.light_povs[pov_idx - 1];
    mat4 pov_vp = pov.proj * pov.view;
    vec4 from_light = pov_vp * vec4(pixel_pos, 1.0); 
    vec3 light_proj = from_light.xyz / from_light.w;
    light_proj.xy = light_proj.xy * 0.5 + 0.5;

    if (light_proj.x < 0.0 || light_proj.x > 1.0
        || light_proj.y < 0.0 || light_proj.y > 1.0 
        || light_proj.z < 0.0 || light_proj.z > 1.0 
    ) {
        return 1.0;
    }

    float in_shadow = is_in_shadow(frag_info, light_info, light_proj, caster, per_frame_data.camera.eye.xyz);
    return in_shadow;
}

vec3 lit_fragment(FragmentInfo frag_info, vec2 uv) {
    vec3 overall_color = vec3(0.0);
    vec3 view = normalize(per_frame_data.camera.eye.xyz - frag_info.position);
    vec3 ambient_color = light_data.ambient_light_color.xyz * light_data.ambient_light_color.w ;

    for (uint i = 0; i < light_data.light_count; i ++) {
        LightInfo light_info = light_data.lights[i];

        float is_lit = is_fragment_lit(uv, frag_info, light_info, frag_info.position);
        vec3 light_dir_unnorm = get_unnormalized_light_direction(light_info, frag_info.position);
        float light_dist = length(light_dir_unnorm);
        vec3 light_dir = -light_dir_unnorm / light_dist;
        float l_dot_n = max(dot(light_dir, frag_info.normal), 0.0);
        float light_mask = get_light_mask(l_dot_n, light_info, frag_info.position);
        vec3 masked_light_color = is_lit * light_mask * light_info.color_intensity.xyz * light_info.color_intensity.w;
        vec3 h = normalize(view + light_dir);
        vec3 light_color = cook_torrance(view, frag_info, l_dot_n, h) * masked_light_color;
        overall_color += light_color;
    }
    vec3 irradiance_sample = texture(irradianceMap, normalize(frag_info.normal)).rgb;
    vec3 illumination = ambient_color + overall_color;
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
    color = (fragInfo.shadow_scale * vec4(light_a, 1.0) + fragInfo.emissive);

    if (is_csm_debug_enabled()) {
        vec4 pos_in_view = per_frame_data.camera.view * vec4(fragInfo.position, 1.0);
        uint offset = select_directiona_light_offset(pos_in_view, 0);

        vec3 colors[6] = vec3[] (
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.5, 0.5, 1.0),
            vec3(0.5, 1.0, 0.5),
            vec3(0.1, 0.3, 0.4)
        );

        vec4 col = vec4(colors[offset], 1.0);
        color *= col;
    }
}
