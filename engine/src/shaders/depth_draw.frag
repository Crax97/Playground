#version 460

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D source;

layout(std140, push_constant) uniform DepthSettings {
    vec4 near_far;
} depth_settings;

float linearize_depth(float d,float zNear,float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

void main() {
    float depth = texture(source, uv);
    depth = linearize_depth(depth, depth_settings.x, depth_settings.y);
    color = texture(source, uv);
}
