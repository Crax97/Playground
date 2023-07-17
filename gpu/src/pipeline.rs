use std::ptr::addr_of;
use std::{ffi::CString, sync::Arc};

use ash::vk::{CullModeFlags, Format, PipelineRenderingCreateInfoKHR};
use ash::{
    prelude::VkResult,
    vk::{
        self, AttachmentDescription, AttachmentDescriptionFlags, AttachmentReference,
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags,
        DescriptorSetLayoutCreateInfo, DescriptorType, DynamicState, GraphicsPipelineCreateInfo,
        PipelineBindPoint, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateFlags,
        PipelineColorBlendStateCreateInfo, PipelineCreateFlags,
        PipelineDepthStencilStateCreateFlags, PipelineDepthStencilStateCreateInfo,
        PipelineDynamicStateCreateFlags, PipelineDynamicStateCreateInfo,
        PipelineInputAssemblyStateCreateFlags, PipelineInputAssemblyStateCreateInfo,
        PipelineLayout, PipelineLayoutCreateFlags, PipelineLayoutCreateInfo,
        PipelineMultisampleStateCreateFlags, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateFlags, PipelineRasterizationStateCreateInfo,
        PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo,
        PipelineTessellationStateCreateFlags, PipelineTessellationStateCreateInfo,
        PipelineVertexInputStateCreateFlags, PipelineVertexInputStateCreateInfo,
        PipelineViewportStateCreateFlags, PipelineViewportStateCreateInfo, PushConstantRange,
        RenderPassCreateFlags, RenderPassCreateInfo, SampleCountFlags, ShaderStageFlags,
        StructureType, SubpassDescriptionFlags, VertexInputAttributeDescription,
        VertexInputBindingDescription,
    },
};

use crate::{ImageFormat, ToVk};

use super::{Gpu, GpuShaderModule, GpuState, ShaderStage};

fn vk_bool(b: bool) -> u32 {
    if b {
        vk::TRUE
    } else {
        vk::FALSE
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BindingType {
    Uniform,
    Storage,
    Sampler,
    CombinedImageSampler,
}

#[derive(Clone, Copy)]
pub struct BindingElement {
    pub binding_type: BindingType,
    pub index: u32,
    pub stage: ShaderStage,
}

#[derive(Clone)]
pub struct GlobalBinding<'a> {
    pub set_index: u32,
    pub elements: &'a [BindingElement],
}

impl From<&BindingElement> for DescriptorSetLayoutBinding {
    fn from(b: &BindingElement) -> Self {
        Self {
            binding: b.index,
            descriptor_type: match b.binding_type {
                BindingType::Uniform => DescriptorType::UNIFORM_BUFFER,
                BindingType::Storage => DescriptorType::STORAGE_BUFFER,
                BindingType::Sampler => DescriptorType::SAMPLER,
                BindingType::CombinedImageSampler => DescriptorType::COMBINED_IMAGE_SAMPLER,
            },
            descriptor_count: 1,
            stage_flags: match b.stage {
                ShaderStage::Vertex => ShaderStageFlags::VERTEX,
                ShaderStage::Fragment => ShaderStageFlags::FRAGMENT,
                ShaderStage::Compute => ShaderStageFlags::COMPUTE,
                ShaderStage::VertexFragment => {
                    ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT
                }
                ShaderStage::All => ShaderStageFlags::ALL_GRAPHICS,
            },
            p_immutable_samplers: std::ptr::null(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputRate {
    PerVertex,
    PerInstance,
}

#[derive(Clone, Copy, Debug, Hash)]
pub struct VertexAttributeDescription {
    pub location: u32,
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, Hash)]
pub struct VertexBindingDescription<'a> {
    pub binding: u32,
    pub input_rate: InputRate,
    pub stride: u32,
    pub attributes: &'a [VertexAttributeDescription],
}

#[derive(Clone, Copy)]
pub struct VertexStageInfo<'a> {
    pub entry_point: &'a str,
    pub module: &'a GpuShaderModule,
}

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct BlendState {
    pub blend_enable: bool,
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
    pub color_write_mask: vk::ColorComponentFlags,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderPassAttachment {
    pub format: vk::Format,
    pub samples: SampleCountFlags,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub stencil_load_op: vk::AttachmentLoadOp,
    pub stencil_store_op: vk::AttachmentStoreOp,
    pub initial_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
    pub blend_state: BlendState,
}

#[derive(Clone, Copy, Debug)]
pub struct DepthStencilAttachment {}
#[derive(Clone, Copy)]
pub struct FragmentStageInfo<'a> {
    pub entry_point: &'a str,
    pub module: &'a GpuShaderModule,
    pub color_attachments: &'a [RenderPassAttachment],
    pub depth_stencil_attachments: &'a [DepthStencilAttachment],
}

#[derive(Clone, Copy, Debug, Default, Hash)]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line(f32),
    Point,
}

#[derive(Clone, Copy, Debug, Default, Hash)]
pub enum CullMode {
    #[default]
    Back,
    Front,
    None,
    FrontAndBack,
}
impl ToVk for CullMode {
    type Inner = vk::CullModeFlags;

    fn to_vk(&self) -> Self::Inner {
        match self {
            CullMode::Back => CullModeFlags::BACK,
            CullMode::Front => CullModeFlags::FRONT,
            CullMode::None => CullModeFlags::NONE,
            CullMode::FrontAndBack => CullModeFlags::FRONT_AND_BACK,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Hash)]
pub enum FrontFace {
    #[default]
    CounterClockWise,
    ClockWise,
}

impl ToVk for FrontFace {
    type Inner = vk::FrontFace;

    fn to_vk(&self) -> Self::Inner {
        match self {
            FrontFace::CounterClockWise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::ClockWise => vk::FrontFace::CLOCKWISE,
        }
    }
}
#[derive(Copy, Clone, Default)]
pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub stencil_test_enable: bool,
    pub front: vk::StencilOpState,
    pub back: vk::StencilOpState,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

#[derive(Copy, Clone, Default, Hash)]
pub enum LogicOp {
    #[default]
    Clear,
    And,
    AndReverse,
    Copy,
    AndInverted,
    NoOp,
    Xor,
    Or,
    Nor,
    Equivalent,
    Invert,
    OrReverse,
    CopyInverted,
    OrInverted,
    Nand,
    Set,
}

impl From<LogicOp> for vk::LogicOp {
    fn from(op: LogicOp) -> Self {
        match op {
            LogicOp::Clear => vk::LogicOp::CLEAR,
            LogicOp::And => vk::LogicOp::AND,
            LogicOp::AndReverse => vk::LogicOp::AND_REVERSE,
            LogicOp::Copy => vk::LogicOp::COPY,
            LogicOp::AndInverted => vk::LogicOp::AND_INVERTED,
            LogicOp::NoOp => vk::LogicOp::NO_OP,
            LogicOp::Xor => vk::LogicOp::XOR,
            LogicOp::Or => vk::LogicOp::OR,
            LogicOp::Nor => vk::LogicOp::NOR,
            LogicOp::Equivalent => vk::LogicOp::EQUIVALENT,
            LogicOp::Invert => vk::LogicOp::INVERT,
            LogicOp::OrReverse => vk::LogicOp::OR_REVERSE,
            LogicOp::CopyInverted => vk::LogicOp::COPY_INVERTED,
            LogicOp::OrInverted => vk::LogicOp::OR_INVERTED,
            LogicOp::Nand => vk::LogicOp::NAND,
            LogicOp::Set => vk::LogicOp::SET,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SubpassDescription<'a> {
    pub flags: SubpassDescriptionFlags,
    pub pipeline_bind_point: PipelineBindPoint,
    pub input_attachments: &'a [AttachmentReference],
    pub color_attachments: &'a [AttachmentReference],
    pub resolve_attachments: &'a [AttachmentReference],
    pub depth_stencil_attachment: &'a [AttachmentReference],
    pub preserve_attachments: &'a [u32],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderPassDescription<'a> {
    pub attachments: &'a [RenderPassAttachment],
    pub subpasses: &'a [SubpassDescription<'a>],
    pub dependencies: &'a [vk::SubpassDependency],
}
impl<'a> RenderPassDescription<'a> {
    fn get_output_attachments(&self) -> Vec<AttachmentDescription> {
        let mut attachment_descriptions = vec![];

        for attachment in self.attachments.iter() {
            attachment_descriptions.push(AttachmentDescription {
                flags: AttachmentDescriptionFlags::empty(),
                format: attachment.format,
                samples: attachment.samples,
                load_op: attachment.load_op,
                store_op: attachment.store_op,
                stencil_load_op: attachment.stencil_load_op,
                stencil_store_op: attachment.stencil_store_op,
                initial_layout: attachment.initial_layout,
                final_layout: attachment.final_layout,
            });
        }
        attachment_descriptions
    }

    fn get_subpasses(&self) -> Vec<vk::SubpassDescription> {
        self.subpasses
            .iter()
            .map(|s| vk::SubpassDescription {
                flags: s.flags,
                pipeline_bind_point: s.pipeline_bind_point,
                input_attachment_count: s.input_attachments.len() as _,
                p_input_attachments: p_or_null(s.input_attachments),
                color_attachment_count: s.color_attachments.len() as _,
                p_color_attachments: p_or_null(s.color_attachments),
                p_resolve_attachments: p_or_null(s.resolve_attachments),
                p_depth_stencil_attachment: p_or_null(s.depth_stencil_attachment),
                preserve_attachment_count: s.preserve_attachments.len() as _,
                p_preserve_attachments: p_or_null(s.preserve_attachments),
            })
            .collect()
    }
}

fn p_or_null<T>(slice: &[T]) -> *const T {
    if !slice.is_empty() {
        slice.as_ptr()
    } else {
        std::ptr::null()
    }
}

pub struct RenderPass {
    pub(super) inner: vk::RenderPass,
    state: Arc<GpuState>,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.state
                .logical_device
                .destroy_render_pass(self.inner, None);
        }
    }
}
impl RenderPass {
    pub fn new(gpu: &Gpu, pass_description: &RenderPassDescription) -> VkResult<Self> {
        let output_attachments = pass_description.get_output_attachments();
        let subpasses = pass_description.get_subpasses();
        let pass_info = RenderPassCreateInfo {
            s_type: StructureType::RENDER_PASS_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: RenderPassCreateFlags::empty(),
            attachment_count: output_attachments.len() as _,
            p_attachments: output_attachments.as_ptr(),
            subpass_count: subpasses.len() as _,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: pass_description.dependencies.len() as _,
            p_dependencies: pass_description.dependencies.as_ptr(),
        };
        let render_pass = unsafe {
            gpu.vk_logical_device()
                .create_render_pass(&pass_info, None)?
        };

        Ok(Self {
            inner: render_pass,
            state: gpu.state.clone(),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub struct PipelineDescription<'a> {
    pub global_bindings: &'a [GlobalBinding<'a>],
    pub vertex_inputs: &'a [VertexBindingDescription<'a>],
    pub vertex_stage: Option<VertexStageInfo<'a>>,
    pub fragment_stage: Option<FragmentStageInfo<'a>>,
    pub input_topology: PrimitiveTopology,
    pub primitive_restart: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub depth_stencil_state: DepthStencilState,
    pub logic_op: Option<LogicOp>,
    pub push_constant_ranges: &'a [PushConstantRange],
}

impl<'a> PipelineDescription<'a> {
    fn create_descriptor_set_layouts(&self, gpu: &Gpu) -> VkResult<Vec<DescriptorSetLayout>> {
        let mut layouts: Vec<DescriptorSetLayout> = vec![];
        for element in self.global_bindings.iter() {
            let bindings: Vec<DescriptorSetLayoutBinding> =
                element.elements.iter().map(|b| b.into()).collect();

            let create_info = DescriptorSetLayoutCreateInfo {
                s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: DescriptorSetLayoutCreateFlags::empty(),
                binding_count: bindings.len() as _,
                p_bindings: bindings.as_ptr(),
            };
            unsafe {
                let layout = gpu
                    .state
                    .logical_device
                    .create_descriptor_set_layout(&create_info, None)?;
                layouts.push(layout);
            }
        }
        Ok(layouts)
    }

    fn get_output_attachments(&self) -> Vec<PipelineColorBlendAttachmentState> {
        let mut pipeline_color_blend_attachment_states = vec![];

        if let Some(fs) = self.fragment_stage {
            for attachment in fs.color_attachments.iter() {
                pipeline_color_blend_attachment_states.push(PipelineColorBlendAttachmentState {
                    blend_enable: vk_bool(attachment.blend_state.blend_enable),
                    src_color_blend_factor: attachment.blend_state.src_color_blend_factor,
                    dst_color_blend_factor: attachment.blend_state.dst_color_blend_factor,
                    color_blend_op: attachment.blend_state.color_blend_op,
                    src_alpha_blend_factor: attachment.blend_state.src_alpha_blend_factor,
                    dst_alpha_blend_factor: attachment.blend_state.dst_alpha_blend_factor,
                    alpha_blend_op: attachment.blend_state.alpha_blend_op,
                    color_write_mask: attachment.blend_state.color_write_mask,
                })
            }
        }
        pipeline_color_blend_attachment_states
    }

    fn get_input_bindings_and_attributes(
        &self,
    ) -> (
        Vec<VertexInputBindingDescription>,
        Vec<VertexInputAttributeDescription>,
    ) {
        let mut input_bindings = vec![];
        let mut attribute_bindings = vec![];

        for binding in self.vertex_inputs {
            input_bindings.push(VertexInputBindingDescription {
                binding: binding.binding,
                stride: binding.stride,
                input_rate: match binding.input_rate {
                    InputRate::PerVertex => vk::VertexInputRate::VERTEX,
                    InputRate::PerInstance => vk::VertexInputRate::INSTANCE,
                },
            });

            for attribute in binding.attributes {
                attribute_bindings.push(VertexInputAttributeDescription {
                    location: attribute.location,
                    binding: binding.binding,
                    format: attribute.format,
                    offset: attribute.offset,
                });
            }
        }

        (input_bindings, attribute_bindings)
    }
}

pub struct Pipeline {
    pub(super) pipeline: vk::Pipeline,
    pub(super) pipeline_layout: PipelineLayout,

    shared_state: Arc<GpuState>,
}

impl Eq for Pipeline {}

impl PartialEq for Pipeline {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
    }
}

impl std::hash::Hash for Pipeline {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pipeline.hash(state);
    }
}

impl Pipeline {
    pub fn new(gpu: &Gpu, pipeline_description: &PipelineDescription) -> VkResult<Self> {
        let descriptor_set_layouts = pipeline_description.create_descriptor_set_layouts(gpu)?;
        let color_blend_attachments = pipeline_description.get_output_attachments();
        let mut stages = vec![];

        let vs_entry = if let Some(vs) = pipeline_description.vertex_stage {
            CString::new(vs.entry_point).unwrap()
        } else {
            CString::new("").unwrap()
        };
        let fs_entry = if let Some(fs) = pipeline_description.vertex_stage {
            CString::new(fs.entry_point).unwrap()
        } else {
            CString::new("").unwrap()
        };

        if let Some(vs) = pipeline_description.vertex_stage {
            let module = vs.module.inner;
            stages.push(PipelineShaderStageCreateInfo {
                s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineShaderStageCreateFlags::empty(),
                stage: ShaderStageFlags::VERTEX,
                module,
                p_name: vs_entry.as_ptr(),
                p_specialization_info: std::ptr::null(),
            })
        }
        if let Some(fs) = pipeline_description.fragment_stage {
            let module = fs.module.inner;
            stages.push(PipelineShaderStageCreateInfo {
                s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineShaderStageCreateFlags::empty(),
                stage: ShaderStageFlags::FRAGMENT,
                module,
                p_name: fs_entry.as_ptr(),
                p_specialization_info: std::ptr::null(),
            })
        }

        let (input_binding_descriptions, input_attribute_descriptions) =
            pipeline_description.get_input_bindings_and_attributes();

        let pipeline_layout = unsafe {
            let layout_infos = PipelineLayoutCreateInfo {
                s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineLayoutCreateFlags::empty(),
                set_layout_count: descriptor_set_layouts.len() as _,
                p_set_layouts: descriptor_set_layouts.as_ptr(),
                push_constant_range_count: pipeline_description.push_constant_ranges.len() as _,
                p_push_constant_ranges: pipeline_description.push_constant_ranges.as_ptr(),
            };
            gpu.vk_logical_device()
                .create_pipeline_layout(&layout_infos, None)?
        };

        let pipeline = unsafe {
            let input_stage = PipelineVertexInputStateCreateInfo {
                s_type: StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineVertexInputStateCreateFlags::empty(),
                vertex_binding_description_count: input_binding_descriptions.len() as _,
                p_vertex_binding_descriptions: input_binding_descriptions.as_ptr(),
                vertex_attribute_description_count: input_attribute_descriptions.len() as _,
                p_vertex_attribute_descriptions: input_attribute_descriptions.as_ptr(),
            };

            let assembly_state = PipelineInputAssemblyStateCreateInfo {
                s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineInputAssemblyStateCreateFlags::empty(),
                topology: match pipeline_description.input_topology {
                    PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
                    PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
                },
                primitive_restart_enable: if pipeline_description.primitive_restart {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
            };

            let tessellation_state: PipelineTessellationStateCreateInfo =
                PipelineTessellationStateCreateInfo {
                    s_type: StructureType::PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: PipelineTessellationStateCreateFlags::empty(),
                    patch_control_points: 0,
                };

            let viewport_state = PipelineViewportStateCreateInfo {
                s_type: StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineViewportStateCreateFlags::empty(),
                viewport_count: 1,
                p_viewports: std::ptr::null(),
                scissor_count: 1,
                p_scissors: std::ptr::null(),
            };

            let line_width = match pipeline_description.polygon_mode {
                PolygonMode::Line(w) => w,
                _ => 1.0,
            };

            let raster_state = PipelineRasterizationStateCreateInfo {
                s_type: StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineRasterizationStateCreateFlags::empty(),
                depth_clamp_enable: vk::FALSE,
                rasterizer_discard_enable: vk::FALSE,
                polygon_mode: match pipeline_description.polygon_mode {
                    PolygonMode::Fill => vk::PolygonMode::FILL,
                    PolygonMode::Line(_) => vk::PolygonMode::LINE,
                    PolygonMode::Point => vk::PolygonMode::POINT,
                },
                cull_mode: match pipeline_description.cull_mode {
                    CullMode::Back => vk::CullModeFlags::BACK,
                    CullMode::Front => vk::CullModeFlags::FRONT,
                    CullMode::None => vk::CullModeFlags::NONE,
                    CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
                },
                front_face: match pipeline_description.front_face {
                    FrontFace::CounterClockWise => vk::FrontFace::COUNTER_CLOCKWISE,
                    FrontFace::ClockWise => vk::FrontFace::CLOCKWISE,
                },
                depth_bias_enable: vk::FALSE,
                depth_bias_constant_factor: 0.0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_factor: 0.0,
                line_width,
            };

            let multisample_state = PipelineMultisampleStateCreateInfo {
                s_type: StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineMultisampleStateCreateFlags::empty(),
                rasterization_samples: SampleCountFlags::TYPE_1,
                sample_shading_enable: vk::FALSE,
                min_sample_shading: 1.0,
                p_sample_mask: std::ptr::null(),
                alpha_to_coverage_enable: vk::FALSE,
                alpha_to_one_enable: vk::FALSE,
            };

            let stencil_state = PipelineDepthStencilStateCreateInfo {
                s_type: StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineDepthStencilStateCreateFlags::empty(),
                depth_test_enable: vk_bool(
                    pipeline_description.depth_stencil_state.depth_test_enable,
                ),
                depth_write_enable: vk_bool(
                    pipeline_description.depth_stencil_state.depth_write_enable,
                ),
                depth_compare_op: pipeline_description.depth_stencil_state.depth_compare_op,
                depth_bounds_test_enable: vk::FALSE,
                stencil_test_enable: vk_bool(
                    pipeline_description.depth_stencil_state.stencil_test_enable,
                ),
                front: pipeline_description.depth_stencil_state.front,
                back: pipeline_description.depth_stencil_state.back,
                min_depth_bounds: pipeline_description.depth_stencil_state.min_depth_bounds,
                max_depth_bounds: pipeline_description.depth_stencil_state.max_depth_bounds,
            };

            let (logic_op_enable, logic_op) = match pipeline_description.logic_op {
                Some(op) => (vk_bool(true), op.into()),
                None => (vk_bool(false), vk::LogicOp::NO_OP),
            };

            let color_blend = PipelineColorBlendStateCreateInfo {
                s_type: StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineColorBlendStateCreateFlags::empty(),
                logic_op_enable,
                logic_op,
                attachment_count: color_blend_attachments.len() as _,
                p_attachments: color_blend_attachments.as_ptr(),
                blend_constants: [0.0, 0.0, 0.0, 0.0],
            };

            let dynamic_state_flags = [
                DynamicState::VIEWPORT,
                DynamicState::SCISSOR,
                DynamicState::FRONT_FACE,
                DynamicState::CULL_MODE,
            ];

            let dynamic_state = PipelineDynamicStateCreateInfo {
                s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineDynamicStateCreateFlags::empty(),
                dynamic_state_count: dynamic_state_flags.len() as _,
                p_dynamic_states: dynamic_state_flags.as_ptr(),
            };

            let color_attachment = pipeline_description
                .fragment_stage
                .map(|frag| (frag.color_attachments.iter().map(|c| c.format).collect()))
                .unwrap_or(vec![]);

            let rendering_ext_info = PipelineRenderingCreateInfoKHR {
                s_type: StructureType::PIPELINE_RENDERING_CREATE_INFO_KHR,
                p_next: std::ptr::null(),
                view_mask: 0,
                color_attachment_count: color_attachment.len() as _,
                p_color_attachment_formats: color_attachment.as_ptr(),
                depth_attachment_format: if pipeline_description
                    .depth_stencil_state
                    .depth_test_enable
                {
                    ImageFormat::Depth.to_vk()
                } else {
                    Format::UNDEFINED
                },
                stencil_attachment_format: Format::UNDEFINED,
            };

            let create_infos = [GraphicsPipelineCreateInfo {
                s_type: StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
                p_next: addr_of!(rendering_ext_info).cast(),
                flags: PipelineCreateFlags::ALLOW_DERIVATIVES,
                stage_count: stages.len() as _,
                p_stages: stages.as_ptr(),
                p_vertex_input_state: &input_stage as *const PipelineVertexInputStateCreateInfo,
                p_input_assembly_state: &assembly_state
                    as *const PipelineInputAssemblyStateCreateInfo,
                p_tessellation_state: &tessellation_state
                    as *const PipelineTessellationStateCreateInfo,
                p_viewport_state: &viewport_state as *const PipelineViewportStateCreateInfo,
                p_rasterization_state: &raster_state as *const PipelineRasterizationStateCreateInfo,
                p_multisample_state: &multisample_state
                    as *const PipelineMultisampleStateCreateInfo,
                p_depth_stencil_state: &stencil_state as *const PipelineDepthStencilStateCreateInfo,
                p_color_blend_state: &color_blend as *const PipelineColorBlendStateCreateInfo,
                p_dynamic_state: &dynamic_state as *const PipelineDynamicStateCreateInfo,
                layout: pipeline_layout,
                render_pass: vk::RenderPass::null(),
                subpass: 0,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: 0,
            }];

            let pipelines = gpu.state.logical_device.create_graphics_pipelines(
                gpu.state.pipeline_cache,
                &create_infos,
                None,
            );
            match pipelines {
                Ok(pipelines) => pipelines[0],
                Err((_, e)) => {
                    return Err(e);
                }
            }
        };

        Ok(Self {
            pipeline,
            pipeline_layout,
            shared_state: gpu.state.clone(),
        })
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.shared_state
                .logical_device
                .destroy_pipeline(self.pipeline, None);
            self.shared_state
                .logical_device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
