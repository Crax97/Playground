#version 460

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_debug_printf : enable

#include "definitions.glsl"
#include "light_definitions.glsl"


layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) readonly buffer PerFrameDataBlock {
    PointOfView shadows[];
} per_frame_data;


struct ShadowCaster {
    uvec4 offset_size;
    uvec4 type_nummaps_povs_splitidx;
};

layout(set = 1, binding = 0) uniform sampler2D position_component;
layout(set = 1, binding = 1) uniform sampler2D shadow_atlas;
layout(set = 1, binding = 2) readonly buffer ShadowCasters {
    uint shadow_caster_count;
    ShadowCaster casters[];
} shadow_casters;

layout(set = 1, binding = 3) readonly buffer CsmSplits {
    uint split_count;
    float splits[];
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

// returns 1.0 if pixel is in shadow, 0 otherwise
float sample_shadow_atlas(vec3 light_pos, ShadowCaster shadow_caster) {
    const float depth_bias = 0.0;
    vec2 texture_size = textureSize(shadow_atlas, 0);
    vec2 texel_size = 1.0 / texture_size;

    vec2 shadow_map_size = shadow_caster.offset_size.zw * texel_size;
    vec2 shadow_map_offset = shadow_caster.offset_size.xy * texel_size;
    light_pos.xy = shadow_map_offset + light_pos.xy * shadow_map_size;

    float compare_z = light_pos.z - depth_bias; 
    float stored_z = texture(shadow_atlas, light_pos.xy).r;
    return compare_z > stored_z ? 1.0 : 0.0;
}


void main() {
    float shadow_count = 0.0;
    vec2 nuv = vec2(uv.x, 1.0 - uv.y);
    vec3 pixel_pos = texture(position_component, nuv).xyz;
    int num_lights = 0;
    for (int c = 0; c < shadow_casters.shadow_caster_count; c ++) {
        ShadowCaster caster = shadow_casters.casters[c];
        uint pov_idx = caster.type_nummaps_povs_splitidx[2];
        PointOfView pov = per_frame_data.shadows[pov_idx];
        mat4 pov_vp = pov.proj * pov.view;
        vec4 from_light = pov_vp * vec4(pixel_pos, 1.0); 
        vec3 light_proj = from_light.xyz / from_light.w;
        light_proj.xy = light_proj.xy * 0.5 + 0.5;

        if (light_proj.x < 0.0 || light_proj.x > 1.0
            || light_proj.y < 0.0 || light_proj.y > 1.0 
            || light_proj.z < 0.0 || light_proj.z > 1.0
        ) {
            continue;
        }

        num_lights += 1;
        float in_shadow = sample_shadow_atlas(light_proj, caster);
        shadow_count += in_shadow;
    }
    shadow_count /= float(num_lights);
    
    float this_pixel_in_shadow = shadow_count;
    color = vec4(vec3(this_pixel_in_shadow), 1.0);
}




