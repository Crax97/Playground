#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 color;

layout(set = 0, binding =  0) uniform texture2D source;
layout(set = 0, binding =  1) uniform sampler source_sampler;
const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}
void main() {
    vec2 sphere_uv = SampleSphericalMap(normalize(position));
    color = texture(sampler2D(source, source_sampler), sphere_uv);
}