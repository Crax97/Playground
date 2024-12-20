#version 460

layout(location = 0) out vec2 uv;

void main() {
    vec2 uvs[6] = vec2[](
        vec2(0.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0),
        vec2(0.0, 0.0),
        vec2(1.0, 1.0),
        vec2(1.0, 0.0)
    );

    gl_Position = vec4(uvs[gl_VertexIndex] * 2.0 - 1.0, 0.0, 1.0);
    uv = uvs[gl_VertexIndex];
}