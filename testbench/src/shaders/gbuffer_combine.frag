#version 460

struct LightInfo {
    vec4 position_radius;
    vec3 color;
    int type;
    vec4 extras;
};
const float PI = 3.14159265359;

layout(location=0) in vec2 uv;

layout(location=0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D posSampler;
layout(set = 0, binding = 1) uniform sampler2D normSampler;
layout(set = 0, binding = 2) uniform sampler2D difSampler;
layout(set = 0, binding = 3) uniform sampler2D emissSampler;
layout(set = 0, binding = 4) uniform sampler2D pbrSampler;

layout(set = 0, binding = 5, std140) readonly buffer LightData {
    uint light_count;
    LightInfo lights[];
} light_data;

layout(push_constant) uniform PerFrameData {
    vec4 eye_pos;
} pfd;


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
    info.diffuse = texture(difSampler, in_uv).rgb;
    info.emissive = texture(emissSampler, in_uv);
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

float ggx_smith(vec3 n, vec3 view, vec3 light_dir, float k)
{
    return ggx_shclick_beckmann(n, view, k) * ggx_shclick_beckmann(n, light_dir, k);
}

vec3 fresnel_schlick(float cos_theta, vec3 F0, vec3 F90)
{
    // return vec3(cos_theta);
    return F0 + (F90 - F0) * pow(1.0 - cos_theta, 5.0);
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
    vec3 light_radiance = light_info.color;
    
    vec3 light_dir = normalize(light_info.position_radius.xyz - frag_info.position);
    vec3 h = normalize(view_direction + light_dir);
    // return vec3(dot(view_direction, h));
    
    // Reflective component
    float cos_theta = max(dot(h, view_direction), 0.0);
    float k_d = pow((frag_info.roughness + 1.0), 2.0) / 8.0;
    float d = d_trowbridge_reitz_ggx(frag_info.normal, h, frag_info.roughness);
    float g = ggx_smith(frag_info.normal, view_direction, light_dir, k_d);
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, frag_info.diffuse, frag_info.metalness);
    vec3 f = fresnel_schlick(cos_theta, F0, vec3(1.0));
    
    float v_dot_n = max(dot(view_direction, frag_info.normal), 0.0);
    float l_dot_n = max(dot(light_dir, frag_info.normal), 0.0);

    vec3 dfg = d * g * f;
    float denom = 4.0 * max((l_dot_n * v_dot_n), 0.0) + 0.0001;
    vec3 s_cook_torrance = dfg / denom;
    
    // Refracftion component
    vec3 lambert = frag_info.diffuse / PI;
    vec3 ks = f;
    vec3 kd = 1.0 - ks;
    kd *= 1.0 - frag_info.metalness;
    vec3 o = (kd * lambert + s_cook_torrance) * light_radiance * l_dot_n;
    return vec3(o);
}


vec3 calculate_light_influence(FragmentInfo frag_info, LightInfo light_info) {
    vec3 view = normalize(pfd.eye_pos.xyz - frag_info.position);
    vec3 ck = cook_torrance(view, frag_info, light_info);
    return ck + 0.2 * frag_info.diffuse;
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
    testLightInfo.position_radius.xyz = vec3(-10.0, -5.0, 5.0);
    testLightInfo.position_radius.w = 1000;
    testLightInfo.color = rgb(255, 255, 255);
        
    vec3 light_a = calculate_light_influence(fragInfo, testLightInfo);
    color = vec4(light_a, 1.0) + fragInfo.emissive;
}