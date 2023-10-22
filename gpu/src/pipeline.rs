use std::ptr::addr_of;
use std::{ffi::CString, sync::Arc};

use ash::vk::{DependencyFlags, Format, PipelineRenderingCreateInfoKHR};
use ash::{
    prelude::VkResult,
    vk::{
        self, AttachmentDescription, AttachmentDescriptionFlags, DescriptorSetLayout,
        DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo,
        DescriptorType, DynamicState, GraphicsPipelineCreateInfo,
        PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateFlags,
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
        PipelineViewportStateCreateFlags, PipelineViewportStateCreateInfo, RenderPassCreateFlags,
        RenderPassCreateInfo, SampleCountFlags, ShaderStageFlags, StructureType,
        SubpassDescriptionFlags, VertexInputAttributeDescription, VertexInputBindingDescription,
    },
};

use crate::*;

fn vk_bool(b: bool) -> u32 {
    if b {
        vk::TRUE
    } else {
        vk::FALSE
    }
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
            stage_flags: b.stage.to_vk(),
            p_immutable_samplers: std::ptr::null(),
        }
    }
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

impl<'a> RenderPassDescription<'a> {
    fn get_output_attachments(&self) -> Vec<AttachmentDescription> {
        let mut attachment_descriptions = vec![];

        for attachment in self.attachments.iter() {
            attachment_descriptions.push(AttachmentDescription {
                flags: AttachmentDescriptionFlags::empty(),
                format: attachment.format.to_vk(),
                samples: attachment.samples.to_vk(),
                load_op: attachment.load_op.to_vk(),
                store_op: attachment.store_op.to_vk(),
                stencil_load_op: attachment.stencil_load_op.to_vk(),
                stencil_store_op: attachment.stencil_store_op.to_vk(),
                initial_layout: attachment.initial_layout.to_vk(),
                final_layout: attachment.final_layout.to_vk(),
            });
        }
        attachment_descriptions
    }

    fn get_subpasses(&self) -> Vec<vk::SubpassDescription> {
        self.subpasses
            .iter()
            .map(|s| {
                let input_attachments = s
                    .input_attachments
                    .iter()
                    .map(|a| a.to_vk())
                    .collect::<Vec<_>>();
                let color_attachments = s
                    .color_attachments
                    .iter()
                    .map(|a| a.to_vk())
                    .collect::<Vec<_>>();
                let resolve_attachments = s
                    .resolve_attachments
                    .iter()
                    .map(|a| a.to_vk())
                    .collect::<Vec<_>>();
                let depth_stencil_attachment = s
                    .depth_stencil_attachment
                    .iter()
                    .map(|a| a.to_vk())
                    .collect::<Vec<_>>();
                vk::SubpassDescription {
                    flags: SubpassDescriptionFlags::empty(),
                    pipeline_bind_point: s.pipeline_bind_point.to_vk(),
                    input_attachment_count: s.input_attachments.len() as _,
                    p_input_attachments: p_or_null(&input_attachments),
                    color_attachment_count: s.color_attachments.len() as _,
                    p_color_attachments: p_or_null(&color_attachments),
                    p_resolve_attachments: p_or_null(&resolve_attachments),
                    p_depth_stencil_attachment: p_or_null(&depth_stencil_attachment),
                    preserve_attachment_count: s.preserve_attachments.len() as _,
                    p_preserve_attachments: p_or_null(s.preserve_attachments),
                }
            })
            .collect()
    }

    pub fn get_subpass_dependencies(&self) -> Vec<vk::SubpassDependency> {
        self.dependencies
            .iter()
            .map(|d| vk::SubpassDependency {
                src_subpass: if d.src_subpass == SubpassDependency::EXTERNAL {
                    vk::SUBPASS_EXTERNAL
                } else {
                    d.src_subpass
                },
                dst_subpass: if d.dst_subpass == SubpassDependency::EXTERNAL {
                    vk::SUBPASS_EXTERNAL
                } else {
                    d.dst_subpass
                },
                src_stage_mask: d.src_stage_mask.to_vk(),
                dst_stage_mask: d.dst_stage_mask.to_vk(),
                src_access_mask: d.src_access_mask.to_vk(),
                dst_access_mask: d.dst_access_mask.to_vk(),
                dependency_flags: DependencyFlags::empty(),
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

pub struct VkRenderPass {
    pub(super) inner: vk::RenderPass,
    state: Arc<GpuThreadSharedState>,
}

impl Drop for VkRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.state
                .logical_device
                .destroy_render_pass(self.inner, None);
        }
    }
}
impl VkRenderPass {
    pub(crate) fn new(gpu: &VkGpu, pass_description: &RenderPassDescription) -> VkResult<Self> {
        let output_attachments = pass_description.get_output_attachments();
        let subpasses = pass_description.get_subpasses();
        let subpass_dependencies = pass_description.get_subpass_dependencies();
        let pass_info = RenderPassCreateInfo {
            s_type: StructureType::RENDER_PASS_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: RenderPassCreateFlags::empty(),
            attachment_count: output_attachments.len() as _,
            p_attachments: output_attachments.as_ptr(),
            subpass_count: subpasses.len() as _,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: pass_description.dependencies.len() as _,
            p_dependencies: subpass_dependencies.as_ptr(),
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

pub trait VkPipelineInfo {
    fn bind_point() -> PipelineBindPoint;
    fn vk_pipeline(&self) -> vk::Pipeline;
    fn vk_pipeline_layout(&self) -> vk::PipelineLayout;
}

fn create_descriptor_set_layouts(
    bindings: &[GlobalBinding],
    gpu: &VkGpu,
) -> VkResult<Vec<DescriptorSetLayout>> {
    let mut layouts: Vec<DescriptorSetLayout> = vec![];
    for element in bindings.iter() {
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

impl<'a> GraphicsPipelineDescription<'a> {
    fn get_output_attachments(&self) -> Vec<PipelineColorBlendAttachmentState> {
        let mut pipeline_color_blend_attachment_states = vec![];

        if let Some(fs) = self.fragment_stage {
            for attachment in fs.color_attachments.iter() {
                pipeline_color_blend_attachment_states.push(PipelineColorBlendAttachmentState {
                    blend_enable: vk_bool(attachment.blend_state.blend_enable),
                    src_color_blend_factor: attachment.blend_state.src_color_blend_factor.to_vk(),
                    dst_color_blend_factor: attachment.blend_state.dst_color_blend_factor.to_vk(),
                    color_blend_op: attachment.blend_state.color_blend_op.to_vk(),
                    src_alpha_blend_factor: attachment.blend_state.src_alpha_blend_factor.to_vk(),
                    dst_alpha_blend_factor: attachment.blend_state.dst_alpha_blend_factor.to_vk(),
                    alpha_blend_op: attachment.blend_state.alpha_blend_op.to_vk(),
                    color_write_mask: attachment.blend_state.color_write_mask.to_vk(),
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
                    format: attribute.format.to_vk(),
                    offset: attribute.offset,
                });
            }
        }

        (input_bindings, attribute_bindings)
    }
}

#[deprecated(note = "This will be removed in favour of the higher-level api")]
pub struct VkGraphicsPipeline {
    pub(super) pipeline: vk::Pipeline,
    pub(super) pipeline_layout: PipelineLayout,

    shared_state: Arc<GpuThreadSharedState>,
}

impl Eq for VkGraphicsPipeline {}

impl PartialEq for VkGraphicsPipeline {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
    }
}

impl std::hash::Hash for VkGraphicsPipeline {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pipeline.hash(state);
    }
}

impl VkGraphicsPipeline {
    pub(crate) fn new(
        gpu: &VkGpu,
        pipeline_description: &GraphicsPipelineDescription,
    ) -> VkResult<Self> {
        let descriptor_set_layouts =
            create_descriptor_set_layouts(&pipeline_description.global_bindings, gpu)?;
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
            let module = gpu.resolve_resource::<VkShaderModule>(&vs.module).inner;
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
            let module = gpu.resolve_resource::<VkShaderModule>(&fs.module).inner;
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
            let vk_constant_ranges = pipeline_description
                .push_constant_ranges
                .iter()
                .map(|r| r.to_vk())
                .collect::<Vec<_>>();
            let layout_infos = PipelineLayoutCreateInfo {
                s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineLayoutCreateFlags::empty(),
                set_layout_count: descriptor_set_layouts.len() as _,
                p_set_layouts: descriptor_set_layouts.as_ptr(),
                push_constant_range_count: pipeline_description.push_constant_ranges.len() as _,
                p_push_constant_ranges: vk_constant_ranges.as_ptr(),
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
                polygon_mode: pipeline_description.polygon_mode.to_vk(),
                cull_mode: pipeline_description.cull_mode.to_vk(),
                front_face: pipeline_description.front_face.to_vk(),
                depth_bias_enable: vk::TRUE,
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
                depth_compare_op: pipeline_description
                    .depth_stencil_state
                    .depth_compare_op
                    .to_vk(),
                depth_bounds_test_enable: vk::FALSE,
                stencil_test_enable: vk_bool(
                    pipeline_description.depth_stencil_state.stencil_test_enable,
                ),
                front: pipeline_description.depth_stencil_state.front.to_vk(),
                back: pipeline_description.depth_stencil_state.back.to_vk(),
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
                DynamicState::DEPTH_BIAS,
                DynamicState::DEPTH_BIAS_ENABLE,
                DynamicState::DEPTH_TEST_ENABLE,
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
                .map(|frag| {
                    frag.color_attachments
                        .iter()
                        .map(|c| c.format.to_vk())
                        .collect()
                })
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
                gpu.state.vk_pipeline_cache,
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

impl Drop for VkGraphicsPipeline {
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

#[deprecated(note = "This will be removed in favour of the higher-level api")]
pub struct VkComputePipeline {
    pub(super) pipeline: vk::Pipeline,
    pub(super) pipeline_layout: PipelineLayout,

    shared_state: Arc<GpuThreadSharedState>,
}
impl VkComputePipeline {
    pub(crate) fn new(gpu: &VkGpu, description: &ComputePipelineDescription) -> VkResult<Self> {
        let descriptor_set_layouts = create_descriptor_set_layouts(&description.bindings, gpu)?;

        let entry_point = CString::new(description.entry_point).unwrap();
        let compute_shader_stage_crate_info = vk::PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            stage: ShaderStageFlags::COMPUTE,
            module: gpu
                .resolve_resource::<VkShaderModule>(&description.module)
                .inner,
            p_name: entry_point.as_ptr(),
            p_specialization_info: std::ptr::null(),
            flags: PipelineShaderStageCreateFlags::empty(),
        };
        let pipeline_layout = unsafe {
            let vk_constant_ranges = description
                .push_constant_ranges
                .iter()
                .map(|r| r.to_vk())
                .collect::<Vec<_>>();
            let layout_infos = PipelineLayoutCreateInfo {
                s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: PipelineLayoutCreateFlags::empty(),
                set_layout_count: descriptor_set_layouts.len() as _,
                p_set_layouts: descriptor_set_layouts.as_ptr(),
                push_constant_range_count: description.push_constant_ranges.len() as _,
                p_push_constant_ranges: vk_constant_ranges.as_ptr(),
            };
            gpu.vk_logical_device()
                .create_pipeline_layout(&layout_infos, None)?
        };
        let compute_create_info = vk::ComputePipelineCreateInfo {
            s_type: StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: PipelineCreateFlags::empty(),
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
            stage: compute_shader_stage_crate_info,
            layout: pipeline_layout,
        };

        let pipeline = unsafe {
            gpu.vk_logical_device()
                .create_compute_pipelines(
                    gpu.state.vk_pipeline_cache,
                    &[compute_create_info],
                    get_allocation_callbacks(),
                )
                .expect("Failed to generate pipelines")[0]
        };

        Ok(Self {
            pipeline,
            pipeline_layout,
            shared_state: gpu.state.clone(),
        })
    }
}

impl Drop for VkComputePipeline {
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

impl VkPipelineInfo for VkComputePipeline {
    fn bind_point() -> PipelineBindPoint {
        PipelineBindPoint::Compute
    }

    fn vk_pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    fn vk_pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl VkPipelineInfo for VkGraphicsPipeline {
    fn bind_point() -> PipelineBindPoint {
        PipelineBindPoint::Graphics
    }

    fn vk_pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    fn vk_pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}
