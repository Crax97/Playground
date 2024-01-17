struct LightInfo {
    vec4 position_radius;
    vec4 direction;
    vec4 color_intensity;
    vec4 extras;
    ivec4 type_shadow_map; // -1 means no shadow map, since 0 is reserved for the camera
};

const int POINT_LIGHT = 0;
const int DIRECTIONAL_LIGHT = 1;
const int SPOT_LIGHT = 2;
const int RECT_LIGHT = 3;
