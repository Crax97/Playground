use std::collections::HashSet;

use crate::asset_map::AssetHandle;
use gpu::{BufferCreateInfo, BufferHandle, BufferUsageFlags, Gpu, MemoryDomain};

use crate::{texture::Texture, utils::to_u8_slice};

use super::master_material::MasterMaterial;

#[derive(Clone, Default)]
pub struct MaterialInstanceDescription<'a> {
    pub name: &'a str,
    // These are optional, and if present will be bound to descriptor set 1
    pub textures: Vec<AssetHandle<Texture>>,
    // These are optional, and if present will be bound to descriptor set 2
    // After the textures
    pub parameter_buffers: Vec<BufferHandle>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct MaterialInstance {
    pub(crate) owner: AssetHandle<MasterMaterial>,
    pub(crate) textures: Vec<AssetHandle<Texture>>,
    pub(crate) parameter_buffers: Vec<BufferHandle>,
}

impl MaterialInstance {
    pub fn create_instance(
        owner: AssetHandle<MasterMaterial>,
        description: &MaterialInstanceDescription,
    ) -> anyhow::Result<MaterialInstance> {
        Ok(MaterialInstance {
            owner,
            parameter_buffers: description.parameter_buffers.clone(),
            textures: description.textures.clone(),
        })
    }

    pub fn write_parameters<T: Sized + Copy>(
        &self,
        gpu: &dyn Gpu,
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

    pub(crate) fn destroy(&self, gpu: &dyn Gpu) {
        let mut buffers = HashSet::new();

        self.parameter_buffers.iter().for_each(|buf| {
            buffers.insert(*buf);
        });

        for buffer in buffers {
            gpu.destroy_buffer(buffer);
        }
    }
}
