/*


*/

use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    hash::{Hash, Hasher},
};

use gpu::{CommandBuffer, Gpu, ImageFormat};
use resource_map::Resource;

pub trait RenderGraphRunner {
    fn run_graph(&mut self, graph: &CompiledRenderGraph);
}

#[derive(Hash, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
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

#[derive(Hash, Copy, Clone)]
pub enum AllocationType {
    Image(ImageDescription),
}

#[derive(Hash, Clone, Copy)]
pub struct ResourceInfo {
    ty: AllocationType,
}

pub struct RenderGraph {
    passes: Vec<RenderPass>,
    allocations: HashMap<ResourceId, ResourceInfo>,
    persistent_resources: HashSet<ResourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompileError {
    ResourceAlreadyDefined(ResourceId, String),
    RenderPassAlreadyDefined(String),
    CyclicGraph,
}

pub type GraphResult<T> = Result<T, CompileError>;

#[derive(Default, Debug, Clone)]
pub struct RenderPass {
    label: String,
    writes: HashSet<ResourceId>,
    reads: HashSet<ResourceId>,
}

impl Hash for RenderPass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        for write in &self.writes {
            write.hash(state);
        }
        for read in &self.reads {
            read.hash(state);
        }
    }
}

impl RenderPass {
    pub fn write(&mut self, handle: ResourceId) {
        self.writes.insert(handle);
    }
    pub fn writes(&mut self, handles: &[ResourceId]) {
        let handles: HashSet<ResourceId> = HashSet::from_iter(handles.iter().cloned());
        self.writes.extend(&handles);
    }
    pub fn read(&mut self, handle: ResourceId) {
        self.reads.insert(handle);
    }
    pub fn reads(&mut self, handles: &[ResourceId]) {
        let handles: HashSet<ResourceId> = HashSet::from_iter(handles.iter().cloned());
        self.reads.extend(&handles)
    }

    fn id(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderPassHandle {
    id: u64,
}

impl RenderPassHandle {}

#[derive(Clone, Debug)]
pub enum GraphOperation {
    Allocate(HashSet<ResourceId>),
    Destroy(HashSet<ResourceId>),
    TransitionRead(HashSet<ResourceId>),
    TransitionWrite(HashSet<ResourceId>),
    ExecuteRenderPass(usize),
}

#[derive(Default)]
pub struct CompiledRenderGraph<'g> {
    passes: Vec<RenderPass>,
    resources_used: HashMap<ResourceId, ResourceInfo>,
    callbacks: HashMap<u64, Box<dyn Fn(&Gpu, &mut CommandBuffer) + 'g>>,
    graph_operations: Vec<GraphOperation>,
}
impl<'g> CompiledRenderGraph<'g> {
    pub fn register_callback<F: Fn(&Gpu, &mut CommandBuffer) + 'g>(
        &mut self,
        handle: &RenderPassHandle,
        callback: F,
    ) {
        self.callbacks.insert(handle.id, Box::new(callback));
    }
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: vec![],
            allocations: HashMap::default(),
            persistent_resources: HashSet::default(),
        }
    }

    pub fn allocate_image(
        &mut self,
        label: &str,
        description: &ImageDescription,
    ) -> GraphResult<ResourceId> {
        let id = ResourceId::make(label);

        if self.allocations.contains_key(&id) {
            return Err(CompileError::ResourceAlreadyDefined(id, label.to_owned()));
        }

        let allocation = ResourceInfo {
            ty: AllocationType::Image(description.clone()),
        };
        self.allocations.insert(id, allocation);
        Ok(id)
    }

    pub fn begin_render_pass(&self, label: &str) -> GraphResult<RenderPass> {
        if self.render_pass_is_defined_already(&label) {
            return Err(CompileError::RenderPassAlreadyDefined(label.to_owned()));
        }
        Ok(RenderPass {
            label: label.to_owned(),
            writes: Default::default(),
            reads: Default::default(),
        })
    }

    pub fn commit_render_pass(&mut self, pass: RenderPass) -> RenderPassHandle {
        let mut hasher = DefaultHasher::new();
        pass.hash(&mut hasher);
        let id = hasher.finish();
        self.passes.push(pass);
        RenderPassHandle { id }
    }

    pub fn persist_resource(&mut self, id: &ResourceId) {
        self.persistent_resources.insert(*id);
    }

    pub fn compile(&mut self) -> GraphResult<CompiledRenderGraph> {
        let mut compiled = CompiledRenderGraph::default();

        self.prune_passes(&mut compiled);
        self.ensure_graph_acyclic(&compiled)?;
        let merge_candidates = self.find_merge_candidates(&mut compiled);

        self.find_optimal_execution_order(&mut compiled, merge_candidates);

        Ok(compiled)
    }

    fn prune_passes(&mut self, compiled: &mut CompiledRenderGraph) {
        let mut working_set: HashSet<_> = self
            .persistent_resources
            .iter()
            .map(|r| r.clone())
            .collect();
        let mut used_resources = working_set.clone();
        while !working_set.is_empty() {
            let mut remove = None;

            for (index, pass) in self.passes.iter().enumerate().rev() {
                let mut found = false;
                for target in &used_resources {
                    if pass.writes.contains(target) {
                        compiled.passes.push(pass.clone());
                        remove = Some(index);
                        working_set = pass.reads.clone();
                        used_resources.extend(&pass.reads.clone());
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if let Some(index) = remove {
                self.passes.remove(index);
            }
        }

        for used_resource in used_resources {
            compiled
                .resources_used
                .insert(used_resource, self.allocations[&used_resource]);
        }

        compiled.passes.reverse();
    }

    fn render_pass_is_defined_already(&self, label: &str) -> bool {
        self.passes.iter().find(|p| p.label == label).is_some()
    }

    fn ensure_graph_acyclic(&self, compiled: &CompiledRenderGraph) -> GraphResult<()> {
        let mut written_resources: HashSet<ResourceId> = Default::default();
        for pass in &compiled.passes {
            for res in &pass.writes {
                if written_resources.contains(res) {
                    return Err(CompileError::CyclicGraph);
                }
                written_resources.insert(*res);
            }
        }
        Ok(())
    }

    fn find_merge_candidates(&self, compiled: &mut CompiledRenderGraph) -> Vec<Vec<usize>> {
        let mut passes: Vec<_> = compiled.passes.iter().enumerate().collect();

        let mut merge_candidates = vec![];

        while let Some((pass_i, pass)) = passes.pop() {
            let matching_passes: Vec<_> = passes
                .iter()
                .enumerate()
                .filter(|(_, (_, p))| p.reads.intersection(&pass.reads).next().is_some())
                .map(|(i, _)| i)
                .collect();

            if matching_passes.len() > 0 {
                let mut merge_candidate = vec![pass_i];
                for passes_idx in matching_passes {
                    let (pass_idx, _) = passes.remove(passes_idx);
                    merge_candidate.push(pass_idx);
                }

                merge_candidates.push(merge_candidate);
            }
        }

        merge_candidates
    }

    fn find_optimal_execution_order(
        &self,
        compiled: &mut CompiledRenderGraph,
        _merge_candidates: Vec<Vec<usize>>,
    ) {
        // TODO: Upgrade to merge candidates
        let mut allocated_resources: HashSet<ResourceId> = Default::default();
        for (i, pass) in compiled.passes.iter().enumerate() {
            let mut allocate_now: HashSet<ResourceId> = Default::default();
            for writes in &pass.writes {
                if !allocated_resources.contains(writes) {
                    allocated_resources.insert(*writes);
                    allocate_now.insert(*writes);
                }
            }

            if allocate_now.len() > 0 {
                compiled
                    .graph_operations
                    .push(GraphOperation::Allocate(allocate_now));
            }
            compiled
                .graph_operations
                .push(GraphOperation::TransitionRead(pass.reads.clone()));
            compiled
                .graph_operations
                .push(GraphOperation::TransitionWrite(pass.writes.clone()));
            compiled
                .graph_operations
                .push(GraphOperation::ExecuteRenderPass(i));
        }
        compiled
            .graph_operations
            .push(GraphOperation::Destroy(allocated_resources));
    }
}

#[derive(Default)]
pub struct RenderGraphPrinter {}

impl RenderGraphRunner for RenderGraphPrinter {
    fn run_graph(&mut self, graph: &CompiledRenderGraph) {
        println!(
            "Graph contains {} render passes, dumping pass info",
            graph.passes.len()
        );
        for pass in &graph.passes {
            println!(
                "\tName: {}, reads {}, writes {}",
                pass.label,
                pass.reads
                    .iter()
                    .fold(String::new(), |s, p| s + &format!("{p:?},")),
                pass.writes
                    .iter()
                    .fold(String::new(), |s, p| s + &format!("{p:?},")),
            );
        }
        println!("Suggested execution order");

        for op in &graph.graph_operations {
            println!("\t{:?}", op);
        }
    }
}
#[cfg(test)]
mod test {
    use crate::{CompileError, ResourceId};

    use super::{ImageDescription, RenderGraph, RenderGraphPrinter, RenderGraphRunner};

    #[test]
    pub fn prune_empty() {
        let gpu = RenderGraphPrinter::default();
        let mut render_graph = RenderGraph::new();

        let image_desc = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
        };

        let color_component = render_graph
            .allocate_image("Color component", &image_desc)
            .unwrap();
        let position_component = render_graph
            .allocate_image("Position component", &image_desc)
            .unwrap();
        let tangent_component = render_graph
            .allocate_image("Tangent component", &image_desc)
            .unwrap();
        let normal_component = render_graph
            .allocate_image("Normal component", &image_desc)
            .unwrap();

        let mut gbuffer = render_graph.begin_render_pass("gbuffer").unwrap();
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);

        let mut render_graph = render_graph.compile().unwrap();
        render_graph.register_callback(&gbuffer, |gpu, render_pass| {
            // for each primitive draw in render_pass
        });
        assert_eq!(render_graph.passes.len(), 0);
    }

    #[test]
    pub fn ensure_keys_are_unique() {
        let mut render_graph = RenderGraph::new();
        let image_desc = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
        };
        let color_component_1 = render_graph
            .allocate_image("Color component", &image_desc)
            .unwrap();
        let color_component_2 = render_graph.allocate_image("Color component", &image_desc);

        let is_defined = color_component_2.is_err_and(|id| {
            id == CompileError::ResourceAlreadyDefined(
                color_component_1,
                "Color component".to_owned(),
            )
        });
        assert!(is_defined)
    }
    #[test]
    pub fn ensure_passes_are_unique() {
        let mut render_graph = RenderGraph::new();
        let p1 = render_graph.begin_render_pass("pass").unwrap();
        render_graph.commit_render_pass(p1);
        let p2 = render_graph.begin_render_pass("pass");
        assert!(p2.is_err_and(|e| e == CompileError::RenderPassAlreadyDefined("pass".to_owned())));
    }

    #[test]
    pub fn survive_1() {
        let gpu = RenderGraphPrinter::default();
        let mut render_graph = RenderGraph::new();

        let image_desc = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
        };

        let color_component = render_graph
            .allocate_image("Color component", &image_desc)
            .unwrap();
        let position_component = render_graph
            .allocate_image("Position component", &image_desc)
            .unwrap();
        let tangent_component = render_graph
            .allocate_image("Tangent component", &image_desc)
            .unwrap();
        let normal_component = render_graph
            .allocate_image("Normal component", &image_desc)
            .unwrap();
        let output_image = render_graph
            .allocate_image("Output image", &image_desc)
            .unwrap();

        let mut gbuffer = render_graph.begin_render_pass("gbuffer").unwrap();
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);

        let mut compose_gbuffer = render_graph.begin_render_pass("compose_gbuffer").unwrap();
        compose_gbuffer.read(color_component);
        compose_gbuffer.read(position_component);
        compose_gbuffer.read(tangent_component);
        compose_gbuffer.read(normal_component);
        compose_gbuffer.write(output_image);
        let compose_gbuffer = render_graph.commit_render_pass(compose_gbuffer);

        // We need the color component: this will let the 'gbuffer' render pass live
        render_graph.persist_resource(&output_image);

        assert_eq!(render_graph.passes[0].label, "gbuffer");
        assert_eq!(render_graph.passes[1].label, "compose_gbuffer");
        let mut render_graph = render_graph.compile().unwrap();
        render_graph.register_callback(&gbuffer, |gpu, render_pass| {
            // for each primitive draw in render_pass
        });
        assert_eq!(render_graph.passes.len(), 2);
        assert_eq!(render_graph.passes[0].label, "gbuffer");
        assert_eq!(render_graph.passes[1].label, "compose_gbuffer");
    }

    #[test]
    pub fn survive_2() {
        let gpu = RenderGraphPrinter::default();
        let mut render_graph = RenderGraph::new();

        let image_desc = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
        };

        let color_component = render_graph
            .allocate_image("Color component", &image_desc)
            .unwrap();
        let position_component = render_graph
            .allocate_image("Position component", &image_desc)
            .unwrap();
        let tangent_component = render_graph
            .allocate_image("Tangent component", &image_desc)
            .unwrap();
        let normal_component = render_graph
            .allocate_image("Normal component", &image_desc)
            .unwrap();
        let output_image = render_graph
            .allocate_image("Output image", &image_desc)
            .unwrap();
        let unused = render_graph
            .allocate_image("Unused resource", &image_desc)
            .unwrap();

        let mut gbuffer = render_graph.begin_render_pass("gbuffer").unwrap();
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);

        let mut compose_gbuffer = render_graph.begin_render_pass("compose_gbuffer").unwrap();
        compose_gbuffer.read(color_component);
        compose_gbuffer.read(position_component);
        compose_gbuffer.read(tangent_component);
        compose_gbuffer.read(normal_component);
        compose_gbuffer.write(output_image);
        let compose_gbuffer = render_graph.commit_render_pass(compose_gbuffer);

        // adding an empty pass that outputs to an unused buffer
        let mut unused_pass = render_graph.begin_render_pass("unused").unwrap();
        unused_pass.read(color_component);
        unused_pass.read(position_component);
        unused_pass.read(tangent_component);
        unused_pass.read(normal_component);
        unused_pass.write(unused);
        let unused_pass = render_graph.commit_render_pass(unused_pass);

        render_graph.persist_resource(&output_image);

        assert_eq!(render_graph.passes[0].label, "gbuffer");
        assert_eq!(render_graph.passes[1].label, "compose_gbuffer");
        let mut render_graph = render_graph.compile().unwrap();
        render_graph.register_callback(&gbuffer, |gpu, render_pass| {
            // for each primitive draw in render_pass
        });

        render_graph.register_callback(&compose_gbuffer, |gpu, render_pass| {
            // bind pipeline shit for compose gbuffer
        });

        // We need the color component: this will let the 'gbuffer' render pass live

        assert_eq!(render_graph.passes.len(), 2);
        assert_eq!(render_graph.passes[0].label, "gbuffer");
        assert_eq!(render_graph.passes[1].label, "compose_gbuffer");
    }

    fn alloc(name: &str, rg: &mut RenderGraph) -> ResourceId {
        let description = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
        };

        rg.allocate_image(name, &description).unwrap()
    }

    #[test]
    // A cycle happens when a render pass writes to a resource that
    // has been written in an early pass. To avoid so, we should introduce something
    // like resource aliasing
    /// TODO: Study resource aliasing
    pub fn detect_cycles() {
        {
            let mut render_graph = RenderGraph::new();

            let r1 = alloc("r1", &mut render_graph);
            let r2 = alloc("r2", &mut render_graph);
            let r3 = alloc("r3", &mut render_graph);

            let mut p1 = render_graph.begin_render_pass("p1").unwrap();
            p1.writes(&[r1, r2]);
            render_graph.commit_render_pass(p1);

            let mut p2 = render_graph.begin_render_pass("p2").unwrap();
            p2.reads(&[r1, r2]);
            p2.writes(&[r3]);
            render_graph.commit_render_pass(p2);

            let mut p3 = render_graph.begin_render_pass("p3").unwrap();
            p3.reads(&[r3]);
            p3.writes(&[r1, r2]);
            render_graph.commit_render_pass(p3);

            render_graph.persist_resource(&r3);

            let error = render_graph.compile();

            assert!(error.is_err_and(|e| e == CompileError::CyclicGraph));
        }
        {
            let mut render_graph = RenderGraph::new();

            let r1 = alloc("r1", &mut render_graph);
            let r2 = alloc("r2", &mut render_graph);

            let mut p1 = render_graph.begin_render_pass("p1").unwrap();
            p1.writes(&[r1, r2]);
            render_graph.commit_render_pass(p1);

            let mut p2 = render_graph.begin_render_pass("p2").unwrap();
            p2.reads(&[r1, r2]);
            p2.writes(&[r1]);
            render_graph.commit_render_pass(p2);

            render_graph.persist_resource(&r1);

            let error = render_graph.compile();

            assert!(error.is_err_and(|e| e == CompileError::CyclicGraph));
        }
    }

    #[test]
    pub fn big_graph() {
        let mut gpu = RenderGraphPrinter::default();
        let mut render_graph = RenderGraph::new();

        let r1 = alloc("r1", &mut render_graph);
        let r2 = alloc("r2", &mut render_graph);
        let r3 = alloc("r3", &mut render_graph);
        let r4 = alloc("r4", &mut render_graph);
        let rb = alloc("rb", &mut render_graph);

        let mut p1 = render_graph.begin_render_pass("p1").unwrap();
        p1.writes(&[r1, r2, r3, r4]);
        render_graph.commit_render_pass(p1);

        let mut p2 = render_graph.begin_render_pass("p2").unwrap();
        let r5 = alloc("r5", &mut render_graph);
        p2.reads(&[r1, r3]);
        p2.writes(&[r5]);
        render_graph.commit_render_pass(p2);

        let mut p3 = render_graph.begin_render_pass("p3").unwrap();
        let r6 = alloc("r6", &mut render_graph);
        let r7 = alloc("r7", &mut render_graph);
        let r8 = alloc("r8", &mut render_graph);
        p3.reads(&[r2, r4]);
        p3.writes(&[r6, r7, r8]);
        render_graph.commit_render_pass(p3);

        // pruned
        let mut u1 = render_graph.begin_render_pass("u1").unwrap();
        let ru1 = alloc("ru1", &mut render_graph);
        let ru2 = alloc("ru2", &mut render_graph);
        u1.reads(&[r7, r8]);
        u1.writes(&[ru1, ru2]);

        let mut p4 = render_graph.begin_render_pass("p4").unwrap();
        let r9 = alloc("r9", &mut render_graph);
        let r10 = alloc("r10", &mut render_graph);
        p4.reads(&[r7, r8]);
        p4.writes(&[r9, r10]);
        render_graph.commit_render_pass(p4);

        let mut pb = render_graph.begin_render_pass("pb").unwrap();
        pb.reads(&[r9, r10]);
        pb.writes(&[rb]);
        render_graph.commit_render_pass(pb);

        render_graph.persist_resource(&rb);

        let render_graph = render_graph.compile().unwrap();
        for pass in &render_graph.passes {
            println!("{:?}", pass);
        }
        assert_eq!(render_graph.passes.len(), 4);
        assert_eq!(render_graph.passes[0].label, "p1");
        assert_eq!(render_graph.passes[1].label, "p3");
        assert_eq!(render_graph.passes[2].label, "p4");
        assert_eq!(render_graph.passes[3].label, "pb");
        assert!(render_graph
            .passes
            .iter()
            .find(|p| p.label == "u1")
            .is_none());
        assert_eq!(render_graph.resources_used.len(), 7);
        assert!(render_graph
            .resources_used
            .iter()
            .find(|(id, _)| id == &&ru1)
            .is_none());
        assert!(render_graph
            .resources_used
            .iter()
            .find(|(id, _)| id == &&ru2)
            .is_none());

        gpu.run_graph(&render_graph);
    }
}
