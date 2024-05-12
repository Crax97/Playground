#version 460

#include "templates/material_vertex_base.incl"

VertexOutput vertex() {
    mat4 vp = projection * view;
    mat4 model = get_model_matrix();

    vec4 world_vertex = model * vec4(vertex_position, 1.0);
    vec4 clip_position = vp * world_vertex;

    VertexOutput result;
    result.clip_position = clip_position;
    result.world_position = world_vertex.xyz;
    result.uv = vertex_uv;
    return result;
}