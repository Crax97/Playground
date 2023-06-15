#version 450

layout(location=0) out vec2 uv;

void main() {

    vec2[] vertices = vec2[4](vec2(-1.0, 1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0));
    vec2[] uvs = vec2[4](vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 0.0));

    gl_Position = vec4(vertices[gl_VertexIndex], 0.0, 1.0);
    uv = uvs[gl_VertexIndex];
}