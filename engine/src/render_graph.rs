/*


*/

use gpu::{CommandBuffer, Gpu, GpuImage, ImageFormat};

pub trait RenderGraphRunner {}

impl RenderGraphRunner for Gpu {}

#[derive(Hash, Copy, Clone)]
pub struct ResourceId {}

#[derive(Hash)]
pub struct ImageDescription {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub samples: u32,
}

pub struct RenderGraph {}

#[derive(Debug, Clone, Copy)]
pub enum CompileError {}

pub type GraphResult<T> = Result<T, CompileError>;

#[derive(Default)]
pub struct RenderPass {
    writes: Vec<ResourceId>,
    reads: Vec<ResourceId>,
}

impl RenderPass {
    pub fn write(&mut self, handle: ResourceId) {
        todo!()
    }
    pub fn read(&mut self, handle: ResourceId) {
        todo!()
    }
    pub fn set_callback<F: FnMut(&Gpu, &CommandBuffer)>(&mut self, callback: F) {
        todo!()
    }
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {}
    }

    pub fn allocate_image(&mut self, label: &str, description: &ImageDescription) -> ResourceId {
        todo!()
    }

    pub fn import_image(&mut self, label: &str, description: &GpuImage) -> ResourceId {
        todo!()
    }

    pub fn begin_render_pass(&mut self, label: &str) -> &mut RenderPass {
        todo!();
    }
    pub fn compile(&mut self) -> GraphResult<()> {
        todo!()
    }
    pub fn exec<G: RenderGraphRunner>(&self, gpu: &G) {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::{ImageDescription, RenderGraph, RenderGraphRunner};

    #[derive(Default)]
    pub struct GpuDebugger {
        num_passes_survided: u32,
    }

    impl RenderGraphRunner for GpuDebugger {}

    #[test]
    pub fn prune_empty() {
        let gpu = GpuDebugger::default();
        let mut render_graph = RenderGraph::new();

        let image_desc = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
        };

        let color_component = render_graph.allocate_image("Color component", &image_desc);
        let position_component = render_graph.allocate_image("Position component", &image_desc);
        let tangent_component = render_graph.allocate_image("Tangent component", &image_desc);
        let normal_component = render_graph.allocate_image("Normal component", &image_desc);

        let gbuffer = render_graph.begin_render_pass("gbuffer");
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        gbuffer.set_callback(|gpu, render_pass| {
            // for each primitive draw in render_pass
        });

        render_graph.compile().unwrap();
        render_graph.exec(&gpu);
        assert_eq!(gpu.num_passes_survided, 0);
    }
}
