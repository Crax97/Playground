mod material;

use gpu::RenderPass;
pub use material::*;

/*
    A Material Context is is a structure used to determine details about a material, e.g which renderpasses should be
    used to render a primitive using the material.
*/
pub trait MaterialContext {
    fn get_material_render_pass(&self, domain: MaterialDomain) -> &RenderPass;
}
