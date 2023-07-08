#version 460 core

#include "./incl.ini"

struct VsOutput {
    vec3 pos;
};

layout(location=1) out VsOutput o;

void main() {
    gl_Position = vec4(vec3(1.0), 0.0);
    o.pos = amazing_constant;
}