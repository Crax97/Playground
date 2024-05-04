use std::collections::HashMap;

use mgpu::{
    util::hash_type, BindingSet, BindingSetLayoutInfo, Buffer, CompareOp, CullMode,
    DepthStencilState, DepthStencilTargetInfo, Device, FragmentStageInfo, FrontFace,
    GraphicsPipeline, GraphicsPipelineDescription, ImageFormat, PolygonMode, PrimitiveTopology,
    RenderPass, RenderTargetInfo, VertexAttributeFormat, VertexInputDescription,
    VertexInputFrequency, VertexStageInfo,
};

use crate::{
    asset_map::{AssetHandle, AssetMap},
    assets::material::Material,
    immutable_string::ImmutableString,
    scene_renderer::SceneRenderer,
};

use super::shader_cache::ShaderCache;

pub(super) struct MaterialManager {
    material_datas: HashMap<ImmutableString, MaterialData>,
    shader_cache: ShaderCache,
}

#[derive(Clone)]
pub(super) struct MaterialData {
    pipeline: GraphicsPipeline,
    user_data_binding_set: BindingSet,
    user_data_buffer: Option<Buffer>,
}

impl MaterialManager {
    pub(crate) fn get_material_data(
        &mut self,
        device: &Device,
        material_handle: &AssetHandle<Material>,
        asset_map: &mut AssetMap,
    ) -> anyhow::Result<MaterialData> {
        let material = asset_map
            .get(material_handle)
            .expect("Failed to resolve material");
        let data = if let Some(material_data) = self.material_datas.get_mut(todo!()) {
            material_data.clone()
        } else {
            let vertex_shader = self
                .shader_cache
                .get_shader_module(device, &material.vertex_shader)?;
            let fragment_shader = self
                .shader_cache
                .get_shader_module(device, &material.fragment_shader)?;

            let vertex_shader_layout = device.get_shader_module_layout(vertex_shader)?;
            let fragment_shader_layout = device.get_shader_module_layout(vertex_shader)?;

            let graphics_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDescription {
                label: Some("Material Pipeline"),
                vertex_stage: &VertexStageInfo {
                    shader: &vertex_shader,
                    entry_point: "main",
                    vertex_inputs: &[
                        VertexInputDescription {
                            location: 0,
                            stride: std::mem::size_of::<[f32; 3]>(),
                            offset: 0,
                            format: VertexAttributeFormat::Float3,
                            frequency: VertexInputFrequency::PerVertex,
                        },
                        VertexInputDescription {
                            location: 1,
                            stride: std::mem::size_of::<[f32; 3]>(),
                            offset: 0,
                            format: VertexAttributeFormat::Float3,
                            frequency: VertexInputFrequency::PerVertex,
                        },
                        VertexInputDescription {
                            location: 2,
                            stride: std::mem::size_of::<[f32; 3]>(),
                            offset: 0,
                            format: VertexAttributeFormat::Float3,
                            frequency: VertexInputFrequency::PerVertex,
                        },
                        VertexInputDescription {
                            location: 3,
                            stride: std::mem::size_of::<[f32; 3]>(),
                            offset: 0,
                            format: VertexAttributeFormat::Float3,
                            frequency: VertexInputFrequency::PerVertex,
                        },
                        VertexInputDescription {
                            location: 3,
                            stride: std::mem::size_of::<[f32; 2]>(),
                            offset: 0,
                            format: VertexAttributeFormat::Float2,
                            frequency: VertexInputFrequency::PerVertex,
                        },
                    ],
                },
                fragment_stage: Some(&FragmentStageInfo {
                    shader: &fragment_shader,
                    entry_point: "main",
                    render_targets: &[RenderTargetInfo {
                        format: ImageFormat::Rgba8,
                        blend: None,
                    }],
                    depth_stencil_target: Some(&DepthStencilTargetInfo {
                        format: ImageFormat::Depth32,
                    }),
                }),
                primitive_restart_enabled: false,
                primitive_topology: PrimitiveTopology::TriangleList,
                polygon_mode: PolygonMode::Filled,
                cull_mode: CullMode::Back,
                front_face: FrontFace::ClockWise,
                multisample_state: None,
                depth_stencil_state: DepthStencilState {
                    depth_test_enabled: true,
                    depth_write_enabled: true,
                    depth_compare_op: CompareOp::Less,
                },
                binding_set_layouts: &[BindingSetLayoutInfo {
                    set: 0,
                    layout: SceneRenderer::scene_data_binding_set_layout(),
                }],
            });
            todo!()
        };
        todo!()
    }
}
