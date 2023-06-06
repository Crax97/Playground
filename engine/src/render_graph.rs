/*


*/

use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    error::Error,
    hash::{Hash, Hasher},
};

use ash::{
    prelude::VkResult,
    vk::{
        self, AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor,
        BlendOp, ClearValue, ColorComponentFlags, ComponentMapping, DependencyFlags,
        ImageAspectFlags, ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageViewType,
        Offset2D, PipelineBindPoint, PipelineStageFlags, Rect2D, SampleCountFlags,
        SubpassDependency, SubpassDescriptionFlags,
    },
};
use gpu::{
    BeginRenderPassInfo, BlendState, CommandBuffer, FramebufferCreateInfo, Gpu, GpuFramebuffer,
    GpuImage, GpuImageView, ImageCreateInfo, ImageFormat, ImageViewCreateInfo, MemoryDomain,
    PipelineBarrierInfo, RenderPass, RenderPassAttachment, RenderPassCommand,
    RenderPassDescription, SubpassDescription, Swapchain, ToVk, TransitionInfo,
};
pub trait RenderGraphRunner {
    fn run_graph(&mut self, graph: &CompiledRenderGraph) -> VkResult<()>;
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
    ExternalImage(ImageDescription),
}

#[derive(Hash, Clone)]
pub struct ResourceInfo {
    label: String,
    ty: AllocationType,
}

pub struct RenderGraph {
    passes: Vec<RenderPassInfo>,
    allocations: HashMap<ResourceId, ResourceInfo>,
    persistent_resources: HashSet<ResourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompileError {
    ResourceAlreadyDefined(ResourceId, String),
    RenderPassAlreadyDefined(String),
    CyclicGraph,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Render Graph compilation error: {:?}", &self))
    }
}

impl Error for CompileError {}

pub type GraphResult<T> = Result<T, CompileError>;

#[derive(Default, Debug, Clone)]
pub struct RenderPassInfo {
    label: String,
    writes: HashSet<ResourceId>,
    reads: HashSet<ResourceId>,
}

impl Hash for RenderPassInfo {
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

impl RenderPassInfo {
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

pub struct RenderPassContext<'p, 'g> {
    pub render_pass: &'p RenderPass,
    pub render_pass_command: RenderPassCommand<'p, 'g>,
    pub framebuffer: &'p GpuFramebuffer,
}
pub struct EndContext<'p, 'g> {
    pub command_buffer: &'p mut CommandBuffer<'g>,
}

#[derive(Default)]
pub struct CompiledRenderGraph<'g> {
    pass_infos: Vec<RenderPassInfo>,
    resources_used: HashMap<ResourceId, ResourceInfo>,
    callbacks: HashMap<u64, Box<dyn Fn(&Gpu, &mut RenderPassContext) + 'g>>,
    graph_operations: Vec<GraphOperation>,
    end_callback: Option<Box<dyn Fn(&Gpu, &mut EndContext) + 'g>>,
}

impl<'g> CompiledRenderGraph<'g> {
    pub fn register_callback<F: Fn(&Gpu, &mut RenderPassContext) + 'g>(
        &mut self,
        handle: &RenderPassHandle,
        callback: F,
    ) {
        self.callbacks.insert(handle.id, Box::new(callback));
    }
    pub fn register_end_callback<F: Fn(&Gpu, &mut EndContext) + 'g>(&mut self, callback: F) {
        self.end_callback = Some(Box::new(callback));
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

    pub fn use_image(
        &mut self,
        label: &str,
        description: &ImageDescription,
    ) -> GraphResult<ResourceId> {
        let id = self.create_unique_id(label)?;

        let allocation = ResourceInfo {
            ty: AllocationType::Image(description.clone()),
            label: label.to_owned(),
        };
        self.allocations.insert(id, allocation);
        Ok(id)
    }

    fn create_unique_id(&mut self, label: &str) -> GraphResult<ResourceId> {
        let id = ResourceId::make(label);
        if self.allocations.contains_key(&id) {
            return Err(CompileError::ResourceAlreadyDefined(id, label.to_owned()));
        }
        Ok(id)
    }

    pub fn begin_render_pass(&self, label: &str) -> GraphResult<RenderPassInfo> {
        if self.render_pass_is_defined_already(&label) {
            return Err(CompileError::RenderPassAlreadyDefined(label.to_owned()));
        }
        Ok(RenderPassInfo {
            label: label.to_owned(),
            writes: Default::default(),
            reads: Default::default(),
        })
    }

    pub fn commit_render_pass(&mut self, pass: RenderPassInfo) -> RenderPassHandle {
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
        let mut write_resources: HashSet<ResourceId> = HashSet::default();
        while !working_set.is_empty() {
            let mut remove = None;

            for (index, pass) in self.passes.iter().enumerate().rev() {
                let mut found = false;
                for target in &used_resources {
                    if pass.writes.contains(target) {
                        compiled.pass_infos.push(pass.clone());
                        remove = Some(index);
                        working_set = pass.reads.clone();
                        used_resources.extend(&pass.reads.clone());
                        write_resources.extend(&pass.writes.clone());
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
                .insert(used_resource, self.allocations[&used_resource].clone());
        }
        for used_resource in write_resources {
            compiled
                .resources_used
                .insert(used_resource, self.allocations[&used_resource].clone());
        }

        compiled.pass_infos.reverse();
    }

    fn render_pass_is_defined_already(&self, label: &str) -> bool {
        self.passes.iter().find(|p| p.label == label).is_some()
    }

    fn ensure_graph_acyclic(&self, compiled: &CompiledRenderGraph) -> GraphResult<()> {
        let mut written_resources: HashSet<ResourceId> = Default::default();
        for pass in &compiled.pass_infos {
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
        let mut passes: Vec<_> = compiled.pass_infos.iter().enumerate().collect();

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
        for (i, pass) in compiled.pass_infos.iter().enumerate() {
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

pub struct GpuRunner<'a> {
    pub gpu: &'a Gpu,
    pub swapchain: &'a Swapchain,
}
impl<'a> GpuRunner<'a> {
    fn allocate_resource(
        &self,
        id: &ResourceId,
        graph: &CompiledRenderGraph,
    ) -> VkResult<GpuImage> {
        let info = graph.resources_used.get(&id).unwrap();
        if let AllocationType::Image(info) = &info.ty {
            return self.gpu.create_image(
                &ImageCreateInfo {
                    label: Some("GpuImage"),
                    width: info.width,
                    height: info.height,
                    format: info.format.to_vk(),
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
                },
                MemoryDomain::DeviceLocal,
            );
        }

        unreachable!()
    }

    fn create_render_passes(&self, graph: &CompiledRenderGraph) -> VkResult<Vec<RenderPass>> {
        let mut passes = vec![];
        for pass_info in &graph.pass_infos {
            let writes: Vec<_> = pass_info
                .writes
                .iter()
                .filter_map(|id| {
                    let resource_desc = graph.resources_used.get(id).unwrap();
                    match &resource_desc.ty {
                        AllocationType::Image(image_desc)
                        | AllocationType::ExternalImage(image_desc) => {
                            if image_desc.format == ImageFormat::Rgba8 {
                                Some(image_desc)
                            } else {
                                None
                            }
                        }
                    }
                })
                .map(|image_desc| RenderPassAttachment {
                    format: image_desc.format.to_vk(),
                    samples: SampleCountFlags::TYPE_1,
                    load_op: AttachmentLoadOp::DONT_CARE,
                    store_op: AttachmentStoreOp::STORE,
                    stencil_load_op: AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: AttachmentStoreOp::DONT_CARE,
                    initial_layout: ImageLayout::UNDEFINED,
                    final_layout: match image_desc.format {
                        ImageFormat::Rgba8 => ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        ImageFormat::Depth => ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    },
                    blend_state: BlendState {
                        blend_enable: true,
                        src_color_blend_factor: BlendFactor::ONE,
                        dst_color_blend_factor: BlendFactor::ZERO,
                        color_blend_op: BlendOp::ADD,
                        src_alpha_blend_factor: BlendFactor::ONE,
                        dst_alpha_blend_factor: BlendFactor::ZERO,
                        alpha_blend_op: BlendOp::ADD,
                        color_write_mask: ColorComponentFlags::RGBA,
                    },
                })
                .collect();

            let color_attachments: Vec<_> = writes
                .iter()
                .enumerate()
                .map(|(i, _)| AttachmentReference {
                    attachment: i as _,
                    layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                })
                .collect();
            let depth_attachments: Vec<_> = pass_info
                .writes
                .iter()
                .map(|id| graph.resources_used.get(id).unwrap())
                .enumerate()
                .filter_map(|(idx, info)| match info.ty {
                    AllocationType::Image(info) | AllocationType::ExternalImage(info) => {
                        if info.format == ImageFormat::Depth {
                            Some(idx)
                        } else {
                            None
                        }
                    }
                })
                .map(|i| AttachmentReference {
                    attachment: i as _,
                    layout: ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                })
                .collect();

            let description = RenderPassDescription {
                attachments: &writes,
                subpasses: &[SubpassDescription {
                    pipeline_bind_point: PipelineBindPoint::GRAPHICS,
                    flags: SubpassDescriptionFlags::empty(),
                    input_attachments: &[],
                    color_attachments: &color_attachments,
                    resolve_attachments: &[],
                    depth_stencil_attachment: &[],
                    preserve_attachments: &[],
                }],
                dependencies: &[SubpassDependency {
                    src_subpass: vk::SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: AccessFlags::empty(),
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: DependencyFlags::empty(),
                }],
            };
            let pass = RenderPass::new(self.gpu, &description)?;
            passes.push(pass);
        }
        Ok(passes)
    }

    fn create_framebuffers(
        &self,
        graph: &CompiledRenderGraph,
        passes: &Vec<RenderPass>,
        views: &HashMap<ResourceId, GpuImageView>,
    ) -> VkResult<Vec<GpuFramebuffer>> {
        let mut framebuffers: Vec<GpuFramebuffer> = vec![];
        for (pass_idx, _) in graph.pass_infos.iter().enumerate() {
            let info = graph.pass_infos.get(pass_idx).unwrap();
            let views: Vec<_> = info
                .writes
                .iter()
                .map(|ri| views.get(ri).unwrap())
                .collect();
            let framebuffer = self.gpu.create_framebuffer(&FramebufferCreateInfo {
                render_pass: passes.get(pass_idx).unwrap(),
                attachments: &views,
                width: self.swapchain.extents().width,
                height: self.swapchain.extents().height,
            })?;

            framebuffers.push(framebuffer);
        }
        Ok(framebuffers)
    }

    fn create_image_views(
        &self,
        resources: &HashMap<ResourceId, ResourceState>,
        graph: &CompiledRenderGraph,
    ) -> VkResult<HashMap<ResourceId, GpuImageView>> {
        let mut hm = HashMap::default();

        for info in &graph.pass_infos {
            for write in &info.writes {
                if hm.contains_key(write) {
                    continue;
                }

                let info = graph.resources_used.get(write).unwrap();
                let format = match info.ty {
                    AllocationType::Image(img) => img.format.to_vk(),
                    AllocationType::ExternalImage(_) => {
                        // External Images bust be injectex externally
                        continue;
                    }
                };

                let image = resources.get(write).unwrap();
                let view = self.gpu.create_image_view(&ImageViewCreateInfo {
                    image: &image.resource,
                    view_type: ImageViewType::TYPE_2D,
                    format,
                    components: ComponentMapping::default(),
                    subresource_range: ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                })?;
                hm.insert(*write, view);
            }
        }
        Ok(hm)
    }
}

struct ResourceState {
    resource: GpuImage,
}

impl<'a> RenderGraphRunner for GpuRunner<'a> {
    fn run_graph(&mut self, graph: &CompiledRenderGraph) -> VkResult<()> {
        let mut allocated_resources: HashMap<ResourceId, ResourceState> = HashMap::default();
        let mut resource_states: HashMap<ResourceId, TransitionInfo> = HashMap::default();

        let passes = self.create_render_passes(graph)?;
        let views = self.create_image_views(&allocated_resources, graph)?;
        let framebuffers = self.create_framebuffers(graph, &passes, &views)?;

        let mut command_buffer = CommandBuffer::new(self.gpu, gpu::QueueType::Graphics)?;

        for op in &graph.graph_operations {
            match op {
                GraphOperation::Allocate(resources) => {
                    for id in resources {
                        let image = self.allocate_resource(id, graph)?;
                        allocated_resources.insert(*id, ResourceState { resource: image });
                        resource_states.insert(
                            *id,
                            TransitionInfo {
                                layout: ImageLayout::UNDEFINED,
                                access_mask: AccessFlags::empty(),
                                stage_mask: PipelineStageFlags::FRAGMENT_SHADER,
                            },
                        );
                    }
                }
                GraphOperation::Destroy(resources) => {
                    for resource in resources {
                        allocated_resources.remove(resource);
                    }
                }
                GraphOperation::TransitionRead(resources) => {
                    for id in resources {
                        let resource = allocated_resources.get_mut(id).unwrap();
                        let resource_info = &graph.resources_used[id];
                        match resource_info.ty {
                            AllocationType::Image(desc) | AllocationType::ExternalImage(desc) => {
                                let access_flag = match desc.format {
                                    ImageFormat::Rgba8 => AccessFlags::COLOR_ATTACHMENT_READ,
                                    ImageFormat::Depth => {
                                        AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                    }
                                };

                                let old_layout = resource_states[id];
                                let new_layout = TransitionInfo {
                                    layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                    access_mask: AccessFlags::SHADER_READ | access_flag,
                                    stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                                };
                                self.gpu.transition_image_layout_in_command_buffer(
                                    &resource.resource,
                                    &mut command_buffer,
                                    old_layout,
                                    new_layout,
                                    ImageAspectFlags::COLOR,
                                );
                                resource_states.insert(*id, new_layout);
                            }
                        }
                    }
                }
                GraphOperation::TransitionWrite(resources) => {
                    for id in resources {
                        let resource = allocated_resources.get_mut(id).unwrap();
                        let resource_info = &graph.resources_used[id];
                        match resource_info.ty {
                            AllocationType::Image(desc) | AllocationType::ExternalImage(desc) => {
                                let access_flag = match desc.format {
                                    ImageFormat::Rgba8 => AccessFlags::COLOR_ATTACHMENT_WRITE,
                                    ImageFormat::Depth => {
                                        AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                                    }
                                };

                                let old_layout = resource_states[id];
                                let new_layout = TransitionInfo {
                                    layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                    access_mask: AccessFlags::SHADER_WRITE | access_flag,
                                    stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                                };
                                self.gpu.transition_image_layout_in_command_buffer(
                                    &resource.resource,
                                    &mut command_buffer,
                                    old_layout,
                                    new_layout,
                                    ImageAspectFlags::COLOR,
                                );
                                resource_states.insert(*id, new_layout);
                            }
                        }
                    }
                }
                GraphOperation::ExecuteRenderPass(rp) => {
                    let pass = passes.get(*rp).unwrap();
                    let info = graph.pass_infos.get(*rp).unwrap();
                    let cb = graph.callbacks.get(&info.id());
                    let clear_color_values = &[ClearValue::default()];
                    let render_pass_command =
                        command_buffer.begin_render_pass(&BeginRenderPassInfo {
                            framebuffer: framebuffers.get(*rp).unwrap(),
                            render_pass: pass,
                            clear_color_values,
                            render_area: Rect2D {
                                offset: Offset2D::default(),
                                extent: self.swapchain.extents(),
                            },
                        });
                    let mut context = RenderPassContext {
                        render_pass: &pass,
                        framebuffer: framebuffers.get(*rp).unwrap(),
                        render_pass_command,
                    };

                    if let Some(cb) = cb {
                        cb(self.gpu, &mut context);
                    }
                }
            }
        }
        if let Some(end_cb) = &graph.end_callback {
            end_cb(
                self.gpu,
                &mut EndContext {
                    command_buffer: &mut command_buffer,
                },
            );
        }
        command_buffer.submit(&gpu::CommandBufferSubmitInfo {
            wait_semaphores: &[&self.swapchain.image_available_semaphore],
            wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            signal_semaphores: &[&self.swapchain.render_finished_semaphore],
            fence: Some(&self.swapchain.in_flight_fence),
        })
    }
}

#[derive(Default)]
pub struct RenderGraphPrinter {}

impl RenderGraphRunner for RenderGraphPrinter {
    fn run_graph(&mut self, graph: &CompiledRenderGraph) -> VkResult<()> {
        println!(
            "Graph contains {} render passes, dumping pass info",
            graph.pass_infos.len()
        );
        for pass in &graph.pass_infos {
            println!(
                "\tName: {}, reads '{}', writes '{}'",
                pass.label,
                pass.reads.iter().fold(String::new(), |s, r| s + &format!(
                    "{},",
                    graph.resources_used[r].label
                )),
                pass.writes.iter().fold(String::new(), |s, r| s + &format!(
                    "{},",
                    graph.resources_used[r].label
                )),
            );
        }
        println!("Suggested execution order");

        for op in &graph.graph_operations {
            println!("\t{:?}", op);
        }
        Ok(())
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
            .use_image("Color component", &image_desc)
            .unwrap();
        let position_component = render_graph
            .use_image("Position component", &image_desc)
            .unwrap();
        let tangent_component = render_graph
            .use_image("Tangent component", &image_desc)
            .unwrap();
        let normal_component = render_graph
            .use_image("Normal component", &image_desc)
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
        assert_eq!(render_graph.pass_infos.len(), 0);
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
            .use_image("Color component", &image_desc)
            .unwrap();
        let color_component_2 = render_graph.use_image("Color component", &image_desc);

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
            .use_image("Color component", &image_desc)
            .unwrap();
        let position_component = render_graph
            .use_image("Position component", &image_desc)
            .unwrap();
        let tangent_component = render_graph
            .use_image("Tangent component", &image_desc)
            .unwrap();
        let normal_component = render_graph
            .use_image("Normal component", &image_desc)
            .unwrap();
        let output_image = render_graph.use_image("Output image", &image_desc).unwrap();

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
        assert_eq!(render_graph.pass_infos.len(), 2);
        assert_eq!(render_graph.pass_infos[0].label, "gbuffer");
        assert_eq!(render_graph.pass_infos[1].label, "compose_gbuffer");
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
            .use_image("Color component", &image_desc)
            .unwrap();
        let position_component = render_graph
            .use_image("Position component", &image_desc)
            .unwrap();
        let tangent_component = render_graph
            .use_image("Tangent component", &image_desc)
            .unwrap();
        let normal_component = render_graph
            .use_image("Normal component", &image_desc)
            .unwrap();
        let output_image = render_graph.use_image("Output image", &image_desc).unwrap();
        let unused = render_graph
            .use_image("Unused resource", &image_desc)
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

        assert_eq!(render_graph.pass_infos.len(), 2);
        assert_eq!(render_graph.pass_infos[0].label, "gbuffer");
        assert_eq!(render_graph.pass_infos[1].label, "compose_gbuffer");
    }

    fn alloc(name: &str, rg: &mut RenderGraph) -> ResourceId {
        let description = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
        };

        rg.use_image(name, &description).unwrap()
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
        for pass in &render_graph.pass_infos {
            println!("{:?}", pass);
        }
        assert_eq!(render_graph.pass_infos.len(), 4);
        assert_eq!(render_graph.pass_infos[0].label, "p1");
        assert_eq!(render_graph.pass_infos[1].label, "p3");
        assert_eq!(render_graph.pass_infos[2].label, "p4");
        assert_eq!(render_graph.pass_infos[3].label, "pb");
        assert!(render_graph
            .pass_infos
            .iter()
            .find(|p| p.label == "u1")
            .is_none());
        assert_eq!(render_graph.resources_used.len(), 10);
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
