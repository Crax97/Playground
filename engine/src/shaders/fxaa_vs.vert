#version 460

layout(binding = 0) uniform sampler2D tex;

layout(location = 0) out vec2 uv;

void main() {
    vec2[] vertices = vec2[3](vec2(-1.0, 1.0), vec2(3.0, 1.0), vec2(-1.0, -3.0));
    vec2[] uvs = vec2[3](vec2(0.0, 1.0), vec2(2.0, 1.0), vec2(0.0, -1.0));

    uv = uvs[gl_VertexIndex];
    vec4 pos = vec4(vertices[gl_VertexIndex], 0.0, 1.0);
    gl_Position = pos;
}
