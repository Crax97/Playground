use bytemuck::{Pod, Zeroable};
use gpu::{
    BufferCreateInfo, BufferHandle, BufferRange, BufferUsageFlags, DescriptorInfo,
    DescriptorSetInfo, DescriptorType, Gpu, Handle, ImageLayout, MemoryDomain, VkBuffer,
    VkDescriptorSet, VkGpu,
};
use resource_map::{Resource, ResourceHandle, ResourceMap};
use std::collections::HashMap;

use crate::{texture::Texture, utils::to_u8_slice};

use super::master_material::MasterMaterial;

#[derive(Clone)]
pub struct MaterialInstanceDescription<'a> {
    pub name: &'a str,
    pub texture_inputs: HashMap<String, ResourceHandle<Texture>>,
}

pub struct MaterialInstance {
    pub(crate) name: String,
    pub(crate) owner: ResourceHandle<MasterMaterial>,
    pub(crate) parameter_buffer: BufferHandle,
    #[allow(dead_code)]
    pub(crate) current_inputs: HashMap<String, ResourceHandle<Texture>>,
    pub(crate) parameter_block_size: usize,
}

impl Resource for MaterialInstance {
    fn get_description(&self) -> &str {
        "Material Instance"
    }
}

impl MaterialInstance {
    pub fn create_instance(
        gpu: &VkGpu,
        owner: ResourceHandle<MasterMaterial>,
        resource_map: &ResourceMap,
        description: &MaterialInstanceDescription,
    ) -> anyhow::Result<MaterialInstance> {
        let master_owner = resource_map.get(&owner);

        let parameter_buffer = if !master_owner.material_parameters.is_empty() {
            gpu.make_buffer(
                &BufferCreateInfo {
                    label: Some(&format!("{} - Parameter buffer", description.name)),
                    size: master_owner.parameter_block_size,
                    usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
                },
                MemoryDomain::DeviceLocal,
            )?
        } else {
            BufferHandle::null()
        };
        Ok(MaterialInstance {
            name: description.name.to_owned(),
            owner,
            parameter_buffer,
            current_inputs: description.texture_inputs.clone(),
            parameter_block_size: master_owner.parameter_block_size,
        })
    }

    pub fn write_parameters<T: Sized + Copy>(&self, gpu: &VkGpu, block: T) -> anyhow::Result<()> {
        assert!(
            std::mem::size_of::<T>() <= self.parameter_block_size
                && !self.parameter_buffer.is_null()
        );
        gpu.write_buffer(&self.parameter_buffer, 0, to_u8_slice(&[block]))?;
        Ok(())
    }

    fn create_user_descriptor_set(
        gpu: &VkGpu,
        resource_map: &ResourceMap,
        master: &MasterMaterial,
        description: &MaterialInstanceDescription<'_>,
        param_buffer: &BufferHandle,
    ) -> anyhow::Result<VkDescriptorSet> {
        let mut descriptors: Vec<_> = master
            .texture_inputs
            .iter()
            .enumerate()
            .map(|(i, tex)| {
                let tex = resource_map.get(&description.texture_inputs[&tex.name]);
                DescriptorInfo {
                    binding: i as _,
                    element_type: DescriptorType::CombinedImageSampler(gpu::SamplerState {
                        sampler: resource_map.get(&tex.sampler).0.clone(),
                        image_view: resource_map.get(&tex.image_view).view.clone(),
                        image_layout: ImageLayout::ShaderReadOnly,
                    }),
                    binding_stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
                }
            })
            .collect();

        if !param_buffer.is_null() {
            descriptors.push(DescriptorInfo {
                binding: descriptors.len() as _,
                binding_stage: gpu::ShaderStage::VERTEX | gpu::ShaderStage::FRAGMENT,
                element_type: DescriptorType::UniformBuffer(BufferRange {
                    handle: param_buffer.clone(),
                    offset: 0,
                    size: gpu::WHOLE_SIZE,
                }),
            });
        }

        let descriptor = gpu.create_descriptor_set(&DescriptorSetInfo {
            descriptors: &descriptors,
        })?;
        Ok(descriptor)
    }
}
