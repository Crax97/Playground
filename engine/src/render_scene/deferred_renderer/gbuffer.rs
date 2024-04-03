use gpu::{
    render_pass_2::RenderPass2, Binding, Binding2, Extent2D, ImageLayout, SamplerHandle,
    ShaderStage,
};

use super::RenderImage;

pub struct GBuffer {
    pub depth_component: RenderImage,
    pub position_component: RenderImage,
    pub normal_component: RenderImage,
    pub diffuse_component: RenderImage,
    pub emissive_component: RenderImage,
    pub pbr_component: RenderImage,

    pub gbuffer_sampler: SamplerHandle,

    pub viewport_size: Extent2D,
}

impl GBuffer {
    #[allow(dead_code)]
    pub fn bind_as_input_attachments(
        &self,
        render_pass: &mut RenderPass2,
        set: u32,
        base_slot: u32,
    ) {
        render_pass
            .bind_resources(
                set,
                &[
                    Binding {
                        ty: gpu::DescriptorBindingType::InputAttachment {
                            image_view_handle: self.position_component.view,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: base_slot,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::InputAttachment {
                            image_view_handle: self.normal_component.view,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: base_slot + 1,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::InputAttachment {
                            image_view_handle: self.diffuse_component.view,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: base_slot + 2,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::InputAttachment {
                            image_view_handle: self.emissive_component.view,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: base_slot + 3,
                    },
                    Binding {
                        ty: gpu::DescriptorBindingType::InputAttachment {
                            image_view_handle: self.pbr_component.view,
                            layout: ImageLayout::ShaderReadOnly,
                        },
                        binding_stage: ShaderStage::FRAGMENT,
                        location: base_slot + 4,
                    },
                ],
            )
            .expect("Failed to bind gbuffer");
    }

    #[allow(dead_code)]
    pub fn bind_as_shader_resource(&self, render_pass: &mut RenderPass2, set: u32) {
        render_pass
            .bind_resources_2(
                set,
                &[
                    Binding2 {
                        ty: gpu::DescriptorBindingType2::ImageView {
                            image_view_handle: self.position_component.view,
                            sampler_handle: self.gbuffer_sampler,
                        },
                        write: false,
                    },
                    Binding2 {
                        ty: gpu::DescriptorBindingType2::ImageView {
                            image_view_handle: self.normal_component.view,
                            sampler_handle: self.gbuffer_sampler,
                        },
                        write: false,
                    },
                    Binding2 {
                        ty: gpu::DescriptorBindingType2::ImageView {
                            image_view_handle: self.diffuse_component.view,
                            sampler_handle: self.gbuffer_sampler,
                        },
                        write: false,
                    },
                    Binding2 {
                        ty: gpu::DescriptorBindingType2::ImageView {
                            image_view_handle: self.emissive_component.view,
                            sampler_handle: self.gbuffer_sampler,
                        },
                        write: false,
                    },
                    Binding2 {
                        ty: gpu::DescriptorBindingType2::ImageView {
                            image_view_handle: self.pbr_component.view,
                            sampler_handle: self.gbuffer_sampler,
                        },
                        write: false,
                    },
                ],
            )
            .expect("Failed to bind gbuffer");
    }
}
