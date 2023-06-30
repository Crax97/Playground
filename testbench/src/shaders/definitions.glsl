struct FragmentOut {
    vec3 position;
    vec3 normal;
    vec3 tangent;
    mat4 model;
    mat3 TBN;
    vec2 uv;
    vec3 color;
};

struct PerFrameData {
    vec4 eye;
    mat4 view;
    mat4 proj;
};

const float PI = 3.14159265359;
