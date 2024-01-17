mod csm_renderer;

use gpu::{CommandBuffer, Extent2D, Gpu, ImageViewHandle};

use crate::{Camera, RenderScene, ResourceMap, SceneTextures};

pub trait ShadowRenderer {
    // The rendered shadow must be a linear gray texture (no depth)
    // It's then going to be multiplied with the scene texture
    fn render_shadows(
        &mut self,
        gpu: &dyn Gpu,
        gbuffer: &SceneTextures,
        camera: &Camera,
        scene: &RenderScene,
        command_buffer: &mut CommandBuffer,
        resource_map: &ResourceMap,
    ) -> anyhow::Result<()>;

    fn gettext(&self) -> ImageViewHandle;
}
