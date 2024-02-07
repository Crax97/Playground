#version 460

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec4 rgba_gamma;
layout(location = 1) out vec2 out_uv;

layout(push_constant) uniform ScreenData {
    vec2 screen_size;
} sd;

void main() {
    gl_Position = vec4(
        2.0 * in_position.x / sd.screen_size.x - 1.0,
        1.0 - 2.0 * in_position.y / sd.screen_size.y,
        0.0,
        1.0
    );
    rgba_gamma = in_color;
    out_uv = in_uv;
}

