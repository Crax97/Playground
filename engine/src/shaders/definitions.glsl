struct FragmentOut {
    vec3 position;
    vec3 vert_position;
    vec3 normal;
    vec3 tangent;
    mat4 model;
    mat3 TBN;
    vec2 uv;
    vec3 color;
};

struct PointOfView {
    vec4 eye;
    vec4 eye_forward;
    mat4 view;
    mat4 proj;
    vec4 viewport_size_offset;
};

const float PI = 3.14159265359;
