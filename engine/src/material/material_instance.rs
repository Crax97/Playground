use crate::resource_map::{ResourceHandle, ResourceMap};
use gpu::{BufferCreateInfo, BufferHandle, BufferUsageFlags, Gpu, Handle, MemoryDomain, VkGpu};

use crate::{texture::Texture, utils::to_u8_slice};

use super::master_material::MasterMaterial;

#[derive(Clone, Default)]
pub struct MaterialInstanceDescription<'a> {
    pub name: &'a str,
    // These are optional, and if present will be bound to descriptor set 1
    pub textures: Vec<ResourceHandle<Texture>>,
    // These are optional, and if present will be bound to descriptor set 2
    // After the textures
    pub parameter_buffers: Vec<BufferHandle>,
}

#[derive(Clone, Debug)]
pub struct MaterialInstance {
    pub(crate) owner: ResourceHandle<MasterMaterial>,
    pub(crate) textures: Vec<ResourceHandle<Texture>>,
    pub(crate) parameter_buffers: Vec<BufferHandle>,
}

impl MaterialInstance {
    pub fn create_instance(
        gpu: &VkGpu,
        owner: ResourceHandle<MasterMaterial>,
        resource_map: &ResourceMap,
        description: &MaterialInstanceDescription,
    ) -> anyhow::Result<MaterialInstance> {
        let master_owner = resource_map.get(&owner);

        Ok(MaterialInstance {
            owner,
            parameter_buffers: description.parameter_buffers.clone(),
            textures: description.textures.clone(),
        })
    }

    pub fn write_parameters<T: Sized + Copy>(
        &self,
        gpu: &VkGpu,
        block: T,
        buffer: usize,
    ) -> anyhow::Result<()> {
        gpu.write_buffer(&self.parameter_buffers[buffer], 0, to_u8_slice(&[block]))?;
        Ok(())
    }

    pub fn create_material_parameter_buffer(
        label: &str,
        gpu: &dyn Gpu,
        size: usize,
    ) -> anyhow::Result<BufferHandle> {
        gpu.make_buffer(
            &BufferCreateInfo {
                label: Some(label),
                size,
                usage: BufferUsageFlags::UNIFORM_BUFFER
                    | BufferUsageFlags::STORAGE_BUFFER
                    | BufferUsageFlags::TRANSFER_DST
                    | BufferUsageFlags::TRANSFER_SRC,
            },
            MemoryDomain::HostVisible,
        )
    }
}
