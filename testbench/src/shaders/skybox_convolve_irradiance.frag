#version 460

#include "definitions.glsl"

layout(set = 0, binding = 0) uniform samplerCube baseColorSampler;

layout(location = 0) out vec4 outEmissive;

layout(location = 0) in FragmentOut fragOut;

void main() {
    vec3 irradiance = vec3(0.0);  

    vec3 up    = vec3(0.0, -1.0, 0.0);
    vec3 right = normalize(cross(up, fragOut.vert_position));
    up         = normalize(cross(fragOut.vert_position, right));

    float sample_delta = 0.025;
    float num_samples = 0.0; 
    for(float phi = 0.0; phi < 2.0 * PI; phi += sample_delta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sample_delta)
        {
            // spherical to cartesian (in tangent space)
            vec3 tangent_sample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 sample_vec = tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * fragOut.vert_position; 

            irradiance += texture(baseColorSampler, sample_vec).rgb * cos(theta) * sin(theta);
            num_samples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / float(num_samples));
    outEmissive = vec4(irradiance, 1.0);
}
