use ash::vk::{
    AttachmentLoadOp, AttachmentStoreOp, BlendFactor, BlendOp, ColorComponentFlags, ImageLayout,
    PushConstantRange, SampleCountFlags,
};
use gpu::{
    BindingElement, BlendState, CullMode, DepthStencilAttachment, DepthStencilState,
    FragmentStageInfo, FrontFace, GlobalBinding, Gpu, GpuShaderModule, LogicOp, Pipeline,
    PipelineDescription, PolygonMode, PrimitiveTopology, RenderPass, RenderPassAttachment, ToVk,
    VertexBindingDescription, VertexStageInfo,
};

use crate::{RenderGraph, RenderPassInfo};

#[rustfmt::skip]
pub(crate) mod constants {
    use nalgebra::Matrix4;
    pub(crate) const Z_INVERT_MATRIX: Matrix4<f32> = 

        Matrix4::<f32>::new(
        -1.0, 0.0, 0.0, 0.0, 
        0.0, -1.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 1.0,
    );
}

pub struct ModuleInfo<'a> {
    pub module: &'a GpuShaderModule,
    pub entry_point: &'a str,
}

pub enum RenderStage<'a> {
    Graphics {
        vertex: ModuleInfo<'a>,
        fragment: ModuleInfo<'a>,
    },
    Compute {
        shader: ModuleInfo<'a>,
    },
}

// This struct contains all the repetitive stuff that has mostly does not change in pipelines, so that
// it can be ..default()ed when needed

pub struct FragmentState<'a> {
    pub input_topology: PrimitiveTopology,
    pub primitive_restart: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub depth_stencil_state: DepthStencilState,
    pub logic_op: Option<LogicOp>,
    pub push_constant_ranges: &'a [PushConstantRange],
}

impl<'a> Default for FragmentState<'a> {
    fn default() -> Self {
        Self {
            input_topology: Default::default(),
            primitive_restart: false,
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::None,
            front_face: FrontFace::ClockWise,
            depth_stencil_state: DepthStencilState {
                depth_test_enable: false,
                ..Default::default()
            },
            logic_op: None,
            push_constant_ranges: Default::default(),
        }
    }
}

pub struct RenderGraphPipelineDescription<'a> {
    pub vertex_inputs: &'a [VertexBindingDescription<'a>],
    pub stage: RenderStage<'a>,
    pub fragment_state: FragmentState<'a>,
}

pub(crate) fn create_pipeline_for_graph_renderpass(
    graph: &RenderGraph,
    pass_info: &RenderPassInfo,
    vk_renderpass: &RenderPass,
    gpu: &Gpu,
    description: &RenderGraphPipelineDescription,
) -> anyhow::Result<Pipeline> {
    let mut set_zero_bindings = vec![];
    for (idx, read) in pass_info.reads.iter().enumerate() {
        let resource = graph.get_resource_info(read)?;
        set_zero_bindings.push(BindingElement {
            binding_type: match resource.ty {
                crate::AllocationType::Image(_) | crate::AllocationType::ExternalImage(_) => {
                    gpu::BindingType::CombinedImageSampler
                }
            },
            index: idx as _,
            stage: gpu::ShaderStage::VertexFragment,
        });
    }

    let (mut color_attachments, mut depth_stencil_attachments) = (vec![], vec![]);

    for (_, write) in pass_info.writes.iter().enumerate() {
        let resource = graph.get_resource_info(write)?;

        match resource.ty {
            crate::AllocationType::Image(desc) | crate::AllocationType::ExternalImage(desc) => {
                let format = desc.format.to_vk();
                let samples = match desc.samples {
                    1 => SampleCountFlags::TYPE_1,
                    2 => SampleCountFlags::TYPE_2,
                    4 => SampleCountFlags::TYPE_4,
                    8 => SampleCountFlags::TYPE_8,
                    16 => SampleCountFlags::TYPE_16,
                    32 => SampleCountFlags::TYPE_32,
                    64 => SampleCountFlags::TYPE_64,
                    _ => panic!("Invalid sample count! {}", desc.samples),
                };
                match &desc.format {
                    gpu::ImageFormat::Rgba8 | gpu::ImageFormat::RgbaFloat => color_attachments
                        .push(RenderPassAttachment {
                            format,
                            samples,
                            load_op: AttachmentLoadOp::DONT_CARE,
                            store_op: AttachmentStoreOp::STORE,
                            stencil_load_op: AttachmentLoadOp::DONT_CARE,
                            stencil_store_op: AttachmentStoreOp::DONT_CARE,
                            initial_layout: ImageLayout::UNDEFINED,
                            final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            blend_state: if let Some(state) = pass_info.blend_state {
                                state
                            } else {
                                BlendState::default()
                            },
                        }),
                    gpu::ImageFormat::Depth => {
                        depth_stencil_attachments.push(DepthStencilAttachment {})
                    }
                }
            }
        }
    }

    let description = PipelineDescription {
        global_bindings: &[GlobalBinding {
            set_index: 0,
            elements: &set_zero_bindings,
        }],
        vertex_inputs: &description.vertex_inputs,
        vertex_stage: if let RenderStage::Graphics { vertex, .. } = &description.stage {
            Some(VertexStageInfo {
                entry_point: vertex.entry_point,
                module: vertex.module,
            })
        } else {
            None
        },
        fragment_stage: if let RenderStage::Graphics {
            vertex: _,
            fragment,
        } = &description.stage
        {
            Some(FragmentStageInfo {
                entry_point: fragment.entry_point,
                module: fragment.module,
                color_attachments: &color_attachments,
                depth_stencil_attachments: &depth_stencil_attachments,
            })
        } else {
            None
        },

        input_topology: description.fragment_state.input_topology,
        primitive_restart: description.fragment_state.primitive_restart,
        polygon_mode: description.fragment_state.polygon_mode,
        cull_mode: description.fragment_state.cull_mode,
        front_face: description.fragment_state.front_face,
        depth_stencil_state: description.fragment_state.depth_stencil_state,
        logic_op: description.fragment_state.logic_op,
        push_constant_ranges: &description.fragment_state.push_constant_ranges,
    };

    Ok(Pipeline::new(gpu, vk_renderpass, &description)?)
}
