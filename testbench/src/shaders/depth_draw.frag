#version 460

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D source;

float linearize_depth(float original_depth) {
    float near = 0.1;
    float far = 100.0;
    return (2.0 * near) / (far + near - original_depth * (far - near));
}


void main() {
    float f = texture(source, uv).r; 
    f = linearize_depth(f);
    color = vec4(f, f, f, 1.0);
}
