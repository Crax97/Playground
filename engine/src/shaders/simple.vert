#version 460

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_normal;
layout(location = 2) in vec3 vertex_tangent;
layout(location = 3) in vec3 vertex_color;
layout(location = 4) in vec2 vertex_uv;

layout(location = 0) out vec3 position;
layout(location = 1) out vec2 uv;

layout(push_constant) uniform RenderData {
    mat4 mvp;
};

void main() {
    vec4 vertex = mvp * vec4(vertex_position, 1.0);
    position = vertex_position.xyz;
    uv = vertex_uv;

    gl_Position = vertex;
}