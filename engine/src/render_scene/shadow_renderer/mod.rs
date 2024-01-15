mod csm_renderer;

pub use csm_renderer::CsmRenderer;

use gpu::{CommandBuffer, Extent2D, Gpu, ImageViewHandle};

use crate::{Camera, Gbuffer, RenderScene, ResourceMap};

pub trait ShadowRenderer {
    fn get_shadow_buffer(&self, gpu: &dyn Gpu) -> ImageViewHandle;

    // The returning shadow must be a linear gray texture (no depth)
    // It's then going to be multiplied with the scene texture
    fn render_shadows(
        &mut self,
        gpu: &dyn Gpu,
        camera: &Camera,
        scene: &RenderScene,
        gbuffer: &Gbuffer,
        command_buffer: &mut CommandBuffer,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<()>;
}
