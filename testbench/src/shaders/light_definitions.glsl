struct LightInfo {
    vec4 position_radius;
    vec4 direction;
    vec4 color;
    vec4 extras;
    uint type;
};

const uint POINT_LIGHT = 0;
const uint DIRECTIONAL_LIGHT = 1;
const uint SPOT_LIGHT = 2;
const uint RECT_LIGHT = 3;