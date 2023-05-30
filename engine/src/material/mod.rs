mod material;

use ash::prelude::VkResult;
use gpu::{Gpu, GpuBuffer, GpuShaderModule, RenderPass};
pub use material::*;
use resource_map::{ResourceHandle, ResourceMap};

use crate::texture::Texture;

pub struct MaterialDescription<'a> {
    pub domain: MaterialDomain,
    pub uniform_buffers: Vec<GpuBuffer>,
    pub input_textures: Vec<ResourceHandle<Texture>>,
    pub fragment_module: &'a GpuShaderModule,
    pub vertex_module: &'a GpuShaderModule,
}

/*
    A Material Context is is a structure used to determine details about a material, e.g which renderpasses should be
    used to render a primitive using the material.
*/
pub trait MaterialContext {
    fn create_material(
        &self,
        gpu: &Gpu,
        resource_map: &ResourceMap,
        material_description: MaterialDescription,
    ) -> VkResult<Material>;
    fn get_material_render_pass(&self, domain: MaterialDomain) -> &RenderPass;
}
