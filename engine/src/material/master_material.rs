use std::{collections::HashMap, hash::Hash, mem::size_of, num::NonZeroU32};

use gpu::{
    BindingElement, BindingType, CompareOp, CullMode, DepthStencilState, FragmentStageInfo,
    FrontFace, GlobalBinding, Gpu, GraphicsPipeline, GraphicsPipelineDescription, ImageFormat,
    LogicOp, PolygonMode, PushConstantRange, StencilOpState, VertexAttributeDescription,
    VertexBindingDescription, VertexStageInfo,
};
use nalgebra::{Vector2, Vector3};
use resource_map::Resource;

use crate::{MaterialDomain, MaterialParameterOffsetSize, PipelineTarget, TextureInput};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScalarType {
    Float,
    Uint,
    Int,
}

#[derive(Clone, Debug)]
pub struct MaterialParameter {
    pub name: String,
    pub ty: ScalarType,
    pub element_count: NonZeroU32,
}

pub struct MasterMaterialDescription<'a> {
    pub name: &'a str,
    pub domain: MaterialDomain,
    pub global_inputs: &'a [BindingType],
    pub texture_inputs: &'a [TextureInput],
    pub material_parameters: HashMap<String, MaterialParameterOffsetSize>,
    pub vertex_info: &'a VertexStageInfo<'a>,
    pub fragment_info: &'a FragmentStageInfo<'a>,
    pub primitive_restart: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub logic_op: Option<LogicOp>,
    pub push_constant_ranges: &'a [PushConstantRange],
}

#[derive(Eq, PartialEq)]
pub struct MasterMaterial {
    pub(crate) name: String,
    pub(crate) pipelines: HashMap<PipelineTarget, GraphicsPipeline>,
    pub(crate) texture_inputs: Vec<TextureInput>,
    pub(crate) material_parameters: HashMap<String, MaterialParameterOffsetSize>,
    pub(crate) parameter_block_size: usize,
}

impl Hash for MasterMaterial {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Resource for MasterMaterial {
    fn get_description(&self) -> &str {
        &self.name
    }
}

impl MasterMaterial {
    pub fn new(gpu: &Gpu, description: &MasterMaterialDescription) -> anyhow::Result<Self> {
        let pipelines = Self::create_pipelines(gpu, description)?;
        let parameter_block_size = size_of::<f32>() * 4 * description.material_parameters.len();
        Ok(MasterMaterial {
            name: description.name.to_owned(),
            pipelines,
            texture_inputs: description.texture_inputs.to_vec(),
            material_parameters: description.material_parameters.clone(),
            parameter_block_size,
        })
    }

    fn create_pipelines(
        gpu: &Gpu,
        description: &MasterMaterialDescription<'_>,
    ) -> anyhow::Result<HashMap<PipelineTarget, GraphicsPipeline>> {
        let global_elements: Vec<_> = description
            .global_inputs
            .iter()
            .enumerate()
            .map(|(i, d)| BindingElement {
                binding_type: *d,
                index: i as _,
                stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
            })
            .collect();
        let mut user_elements: Vec<_> = description
            .texture_inputs
            .iter()
            .enumerate()
            .map(|(i, _)| BindingElement {
                binding_type: BindingType::CombinedImageSampler,
                index: i as _,
                stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
            })
            .collect();
        if !description.material_parameters.is_empty() {
            user_elements.push(BindingElement {
                binding_type: BindingType::Uniform,
                index: user_elements.len() as _,
                stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
            })
        }

        match description.domain {
            MaterialDomain::Surface => {
                Self::create_surface_pipelines(gpu, description, global_elements, user_elements)
            }
            MaterialDomain::PostProcess => {
                Self::create_post_process_pipeline(gpu, description, global_elements, user_elements)
            }
        }
    }

    fn get_inputs_for_material_domain(
        domain: &MaterialDomain,
    ) -> &'static [gpu::VertexBindingDescription<'static>] {
        static SURFACE_INPUTS: &[VertexBindingDescription] = &[
            VertexBindingDescription {
                binding: 0,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 0,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 1,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 1,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 2,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 2,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 3,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector3<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 3,
                    format: ImageFormat::RgbFloat32,
                    offset: 0,
                }],
            },
            VertexBindingDescription {
                binding: 4,
                input_rate: gpu::InputRate::PerVertex,
                stride: size_of::<Vector2<f32>>() as u32,
                attributes: &[VertexAttributeDescription {
                    location: 4,
                    format: ImageFormat::RgFloat32,
                    offset: 0,
                }],
            },
        ];
        static PP_INPUTS: &[VertexBindingDescription] = &[
            // No inputs: the vertex shaders outputs vertices directly
        ];
        match domain {
            MaterialDomain::Surface => SURFACE_INPUTS,
            MaterialDomain::PostProcess => PP_INPUTS,
        }
    }

    pub(crate) fn get_pipeline(&self, target: PipelineTarget) -> Option<&GraphicsPipeline> {
        self.pipelines.get(&target)
    }

    fn create_surface_pipelines(
        gpu: &Gpu,
        description: &MasterMaterialDescription,
        global_elements: Vec<BindingElement>,
        user_elements: Vec<BindingElement>,
    ) -> anyhow::Result<HashMap<PipelineTarget, GraphicsPipeline>> {
        let mut pipelines = HashMap::new();
        for target in [PipelineTarget::ColorAndDepth, PipelineTarget::DepthOnly] {
            let pipeline = gpu.create_graphics_pipeline(&GraphicsPipelineDescription {
                global_bindings: &[
                    GlobalBinding {
                        set_index: 0,
                        elements: &global_elements,
                    },
                    GlobalBinding {
                        set_index: 1,
                        elements: &user_elements,
                    },
                ],
                vertex_inputs: Self::get_inputs_for_material_domain(&description.domain),
                vertex_stage: Some(*description.vertex_info),
                fragment_stage: match target {
                    PipelineTarget::ColorAndDepth | PipelineTarget::PostProcess => {
                        Some(*description.fragment_info)
                    }
                    PipelineTarget::DepthOnly => None,
                },
                input_topology: gpu::PrimitiveTopology::TriangleList,
                primitive_restart: description.primitive_restart,
                polygon_mode: description.polygon_mode,
                cull_mode: description.cull_mode,
                front_face: description.front_face,
                depth_stencil_state: match target {
                    PipelineTarget::ColorAndDepth | PipelineTarget::PostProcess => {
                        DepthStencilState {
                            depth_test_enable: true,
                            depth_write_enable: false,
                            depth_compare_op: CompareOp::Equal,
                            stencil_test_enable: false,
                            front: StencilOpState::default(),
                            back: StencilOpState::default(),
                            min_depth_bounds: 0.0,
                            max_depth_bounds: 1.0,
                        }
                    }
                    PipelineTarget::DepthOnly => DepthStencilState {
                        depth_test_enable: true,
                        depth_write_enable: true,
                        depth_compare_op: CompareOp::Less,
                        stencil_test_enable: false,
                        front: StencilOpState::default(),
                        back: StencilOpState::default(),
                        min_depth_bounds: 0.0,
                        max_depth_bounds: 1.0,
                    },
                },
                logic_op: description.logic_op,
                push_constant_ranges: description.push_constant_ranges,
            })?;
            pipelines.insert(target, pipeline);
        }

        Ok(pipelines)
    }
    fn create_post_process_pipeline(
        gpu: &Gpu,
        description: &MasterMaterialDescription,
        global_elements: Vec<BindingElement>,
        user_elements: Vec<BindingElement>,
    ) -> anyhow::Result<HashMap<PipelineTarget, GraphicsPipeline>> {
        let mut pipelines = HashMap::new();
        let pipeline = gpu.create_graphics_pipeline(&GraphicsPipelineDescription {
            global_bindings: &[
                GlobalBinding {
                    set_index: 0,
                    elements: &global_elements,
                },
                GlobalBinding {
                    set_index: 1,
                    elements: &user_elements,
                },
            ],
            vertex_inputs: Self::get_inputs_for_material_domain(&description.domain),
            vertex_stage: Some(*description.vertex_info),
            fragment_stage: Some(*description.fragment_info),
            input_topology: gpu::PrimitiveTopology::TriangleList,
            primitive_restart: description.primitive_restart,
            polygon_mode: description.polygon_mode,
            cull_mode: description.cull_mode,
            front_face: description.front_face,
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: false,
                depth_compare_op: CompareOp::Equal,
                stencil_test_enable: false,
                front: StencilOpState::default(),
                back: StencilOpState::default(),
                min_depth_bounds: 0.0,
                max_depth_bounds: 1.0,
            },
            logic_op: description.logic_op,
            push_constant_ranges: description.push_constant_ranges,
        })?;
        pipelines.insert(PipelineTarget::PostProcess, pipeline);

        Ok(pipelines)
    }
}
