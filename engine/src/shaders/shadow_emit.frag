#version 460

#include "definitions.glsl"

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

// returns 1.0 if pixel is in shadow, 0 otherwise
float sample_shadow_atlas(vec3 light_pos, ShadowCaster shadow_caster) {
    vec2 texture_size = textureSize(shadow_atlas, 0);
    vec2 texel_size = 1.0 / texture_size;

    vec2 shadow_map_size = shadow_caster.offset_size.zw * texel_size;
    vec2 shadow_map_offset = shadow_caster.offset_size.xy * texel_size;
    light_pos.xy = shadow_map_offset + light_pos.xy * shadow_map_size;

    float compare_z = light_pos.z; 
    float stored_z = texture(shadow_atlas, light_pos.xy).r;
    return compare_z > stored_z ? 1.0 : 0.0;
}

void main() {
    float shadow_count = 0.0;
    vec3 pixel_pos = texture(position_component, uv).xyz;
    int num_lights = 0;
    for (int c = 0; c < shadow_casters.shadow_caster_count; c ++) {
        ShadowCaster caster = shadow_casters.casters[c];
        uint pov_idx = caster.type_nummaps_povs_splitidx[2];
        PointOfView pov = per_frame_data.shadows[pov_idx];
        mat4 pov_vp = pov.proj * pov.view;
        vec4 from_light = pov_vp * vec4(pixel_pos, 1.0); 
        vec3 light_proj = from_light.xyz / from_light.w;
        light_proj = light_proj * 0.5 + 0.5;

        if (light_proj.x < 0.0 || light_proj.x > 1.0 || light_proj.y < 0.0 || light_proj.y > 1.0 
            || light_proj.z < 0.0 || light_proj.z > 1.0) {
            continue;
        }

        num_lights += 1;
        float in_shadow = sample_shadow_atlas(light_proj, caster);
        shadow_count += in_shadow;
    }
    shadow_count /= float(num_lights);
    
    float this_pixel_lit = 1.0 - shadow_count;
    color = vec4(vec3(this_pixel_lit), 1.0);
}




