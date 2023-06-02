/*


*/

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

use gpu::{CommandBuffer, Gpu, ImageFormat};

pub trait RenderGraphRunner {}

impl RenderGraphRunner for Gpu {}

#[derive(Hash, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceId {
    raw: u64,
}
impl ResourceId {
    fn make(label: &str) -> ResourceId {
        let mut hasher = DefaultHasher::new();
        label.hash(&mut hasher);
        Self {
            raw: hasher.finish(),
        }
    }
}

#[derive(Hash, Copy, Clone)]
pub struct ImageDescription {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub samples: u32,
}

#[derive(Hash)]
pub enum AllocationType {
    Image(ImageDescription),
}

#[derive(Hash)]
pub struct AllocationInfo {
    id: ResourceId,
    ty: AllocationType,
    persistent: bool,
}

pub struct RenderGraph<'g> {
    passes: Vec<RenderPass>,
    allocations: Vec<AllocationInfo>,
    callbacks: HashMap<u64, Box<dyn Fn(&Gpu, &mut CommandBuffer) + 'g>>,
}

#[derive(Debug, Clone, Copy)]
pub enum CompileError {}

pub type GraphResult<T> = Result<T, CompileError>;

#[derive(Default, Hash)]
pub struct RenderPass {
    label: String,
    writes: Vec<ResourceId>,
    reads: Vec<ResourceId>,
}

impl RenderPass {
    pub fn write(&mut self, handle: ResourceId) {
        self.writes.push(handle);
    }
    pub fn read(&mut self, handle: ResourceId) {
        self.reads.push(handle);
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderPassHandle {
    id: u64,
}

impl RenderPassHandle {}

impl<'g> RenderGraph<'g> {
    pub fn new() -> Self {
        Self {
            passes: vec![],
            callbacks: HashMap::default(),
            allocations: vec![],
        }
    }

    pub fn allocate_image(&mut self, label: &str, description: &ImageDescription) -> ResourceId {
        let id = ResourceId::make(label);
        let allocation = AllocationInfo {
            id: id.clone(),
            ty: AllocationType::Image(description.clone()),
            persistent: false,
        };
        self.allocations.push(allocation);
        id
    }

    pub fn begin_render_pass(&self, label: &str) -> RenderPass {
        RenderPass {
            label: label.to_owned(),
            writes: vec![],
            reads: vec![],
        }
    }

    pub fn commit_render_pass(&mut self, pass: RenderPass) -> RenderPassHandle {
        let mut hasher = DefaultHasher::new();
        pass.hash(&mut hasher);
        let id = hasher.finish();
        self.passes.push(pass);
        RenderPassHandle { id }
    }

    pub fn register_callback<F: Fn(&Gpu, &mut CommandBuffer) + 'g>(
        &mut self,
        handle: &RenderPassHandle,
        callback: F,
    ) {
        self.callbacks.insert(handle.id, Box::new(callback));
    }

    pub fn persist_resource(&mut self, id: &ResourceId) {
        for allocation in &mut self.allocations {
            if id == &allocation.id {
                allocation.persistent = true;
            }
        }
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

        let mut gbuffer = render_graph.begin_render_pass("gbuffer");
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);

        render_graph.compile().unwrap();
        render_graph.register_callback(&gbuffer, |gpu, render_pass| {
            // for each primitive draw in render_pass
        });
        render_graph.exec(&gpu);
        assert_eq!(gpu.num_passes_survided, 0);
    }

    #[test]
    pub fn survive_1() {
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

        let mut gbuffer = render_graph.begin_render_pass("gbuffer");
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);
        render_graph.register_callback(&gbuffer, |gpu, render_pass| {
            // for each primitive draw in render_pass
        });

        // We need the color component: this will let the 'gbuffer' render pass live
        render_graph.persist_resource(&color_component);

        render_graph.compile().unwrap();
        render_graph.exec(&gpu);
        assert_eq!(gpu.num_passes_survided, 1);
    }

    #[test]
    pub fn survive_2() {
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
        let output_image = render_graph.allocate_image("Output image", &image_desc);

        let mut gbuffer = render_graph.begin_render_pass("gbuffer");
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);

        let mut compose_gbuffer = render_graph.begin_render_pass("compose_gbuffer");
        compose_gbuffer.read(color_component);
        compose_gbuffer.read(position_component);
        compose_gbuffer.read(tangent_component);
        compose_gbuffer.read(normal_component);
        compose_gbuffer.write(output_image);
        let compose_gbuffer = render_graph.commit_render_pass(compose_gbuffer);

        render_graph.register_callback(&gbuffer, |gpu, render_pass| {
            // for each primitive draw in render_pass
        });

        render_graph.register_callback(&compose_gbuffer, |gpu, render_pass| {
            // bind pipeline shit for compose gbuffer
        });

        // We need the color component: this will let the 'gbuffer' render pass live
        render_graph.persist_resource(&output_image);

        render_graph.compile().unwrap();
        render_graph.exec(&gpu);
        assert_eq!(gpu.num_passes_survided, 2);
    }
}
