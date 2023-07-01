struct LightInfo {
    vec4 position_radius;
    vec4 direction;
    vec4 color_intensity;
    vec4 extras;
    uint type;
};

const uint POINT_LIGHT = 0;
const uint DIRECTIONAL_LIGHT = 1;
const uint SPOT_LIGHT = 2;
const uint RECT_LIGHT = 3;

vec3 get_light_intensity(float n_dot_l, float light_distance, LightInfo light) {
    float dist = clamp(light_distance / light.position_radius.w, 0.0, 1.0);
    dist = max(dist * dist, 0.01);
    vec3 i = light.color_intensity.rgb * light.color_intensity.a;
    float luminous_power = 0.0;
    if (light.type == POINT_LIGHT) {
        return i / dist;
    } else if (light.type == SPOT_LIGHT) {
        vec3 top = 2.0 * (1.0 - cos(light.extras.y / 2.0)) * i;
        return top / dist;
    } else {
        // for now
        return i / dist;
    }
}