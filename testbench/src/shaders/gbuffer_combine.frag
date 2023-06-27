#version 460

const float PI = 3.14159265359;

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D posSampler;
layout(set = 0, binding = 1) uniform sampler2D normSampler;
layout(set = 0, binding = 2) uniform sampler2D difSampler;
layout(set = 0, binding = 3) uniform sampler2D emissSampler;
layout(set = 0, binding = 4) uniform sampler2D pbrSampler;

layout(push_constant) uniform PerFrameData {
    vec4 eye_pos;
} pfd;

struct LightInfo {
    vec3 position;
    vec3 color;
};

struct FragmentInfo {
    vec3 diffuse;
    vec4 emissive;
    vec3 position;
    vec3 normal;
    float roughness;
    float metalness;
};

FragmentInfo get_fragment_info(vec2 in_uv) {
    FragmentInfo info;
    info.diffuse = texture(difSampler, uv).rgb;
    info.emissive = texture(emissSampler, uv);
    info.position = texture(posSampler, in_uv).xyz;
    info.normal = texture(normSampler, in_uv).xyz;
    info.normal = info.normal * 2.0 - 1.0;
    
    vec4 pbr_sample = texture(pbrSampler, in_uv);
    info.metalness = pbr_sample.x;
    info.roughness = pbr_sample.y;

    return info;
}

float ggx_shclick_beckmann(vec3 n, vec3 view, float k) 
{
    float dotNV = max(dot(n, view), 0.0);
    return dotNV / ((dotNV - k) * (1.0 - k) + k);
}

float ggx_combined(vec3 n, vec3 view, vec3 light_dir, float k)
{
    return ggx_shclick_beckmann(n, view, k) * ggx_shclick_beckmann(n, light_dir, k);
}

vec3 fresnel_schlick(vec3 h, vec3 view, vec3 F0)
{
    float h_dot_v = max(dot(h, view), 0.0);
    vec3 inv_F0 = 1.0 - F0;
    return F0 + inv_F0 * pow(clamp(1.0 - h_dot_v, 0.0, 1.0), 5.0);
}

float d_trowbridge_reitz_ggx(vec3 n, vec3 h, float rough)
{
    float a = rough * rough;
    float n_dot_h = max(dot(n, h), 0.0);
    float n_dot_h_2 = n_dot_h * n_dot_h;
    float a_2 = a * a;
    float a_2_sub = a - 1.0;
    
    float d = n_dot_h_2 * a_2_sub + 1.0;
    return a_2 / (PI * d * d);
}

vec3 cook_torrance(vec3 view_direction, FragmentInfo frag_info, LightInfo light_info) {
    vec3 light_dir = normalize(light_info.position - frag_info.position);
    vec3 h = normalize(view_direction + light_dir);
    return vec3(dot(h, frag_info.normal));
    
    float k_d = pow((frag_info.roughness + 1.0), 2.0) / 8.0;
    float d = d_trowbridge_reitz_ggx(frag_info.normal, h, frag_info.roughness);
    float g = ggx_combined(frag_info.normal, view_direction, light_dir, k_d);
    vec3 f = fresnel_schlick(h, view_direction, vec3(0.04));
    vec3 dfg = d * g * f;
    
    float v_dot_n = max(dot(view_direction, frag_info.normal), 0.0);
    float l_dot_n = max(dot(light_dir, frag_info.normal), 0.0);
    
    float denom = max((4.0 * l_dot_n * v_dot_n), 0.0001);
    vec3 s_cook_torrance = dfg / denom;
    vec3 lambert = frag_info.diffuse / PI;
    vec3 kd = 1.0 - f;
    kd *= 1.0 - frag_info.metalness;
    vec3 o = (kd * lambert + s_cook_torrance) * l_dot_n;
    return vec3(o);
}


vec3 calculate_light_influence(FragmentInfo frag_info, LightInfo light_info) {
    vec3 light_dir = normalize(frag_info.position - light_info.position);
    vec3 view = normalize(pfd.eye_pos.xyz - frag_info.position);
    return vec3(dot(view, frag_info.normal));

    vec3 ck = cook_torrance(view, frag_info, light_info) * frag_info.diffuse;
    return ck;
    // float a = max(dot(light_dir, fragInfo.normal), 0.0);
    // return a * lightInfo.color;
}

vec3 rgb(int r, int g, int b) {
    return vec3(
        255.0 / float(r),
        255.0 / float(g),
        255.0 / float(b)
    );
}

void main() {
    FragmentInfo fragInfo = get_fragment_info(uv);
    
    LightInfo testLightInfo;
    testLightInfo.position = vec3(100.0, -50.0, 0.0);
    testLightInfo.color = rgb(255, 255, 255);
        
    vec3 light_a = calculate_light_influence(fragInfo, testLightInfo);
    color = vec4(light_a, 1.0) + fragInfo.emissive;
}