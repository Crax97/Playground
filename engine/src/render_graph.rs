/*


*/

use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    error::Error,
    fmt::Debug,
    hash::{Hash, Hasher},
};

use ash::vk::{
    self, AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor,
    BlendOp, ClearDepthStencilValue, ClearValue, ColorComponentFlags, ComponentMapping,
    DependencyFlags, Extent2D, ImageAspectFlags, ImageLayout, ImageSubresourceRange,
    ImageUsageFlags, ImageViewType, Offset2D, PipelineBindPoint, PipelineStageFlags, Rect2D,
    SampleCountFlags, SubpassDependency, SubpassDescriptionFlags,
};
use gpu::{
    BeginRenderPassInfo, BlendState, CommandBuffer, FramebufferCreateInfo, Gpu, GpuFramebuffer,
    GpuImage, GpuImageView, ImageCreateInfo, ImageFormat, ImageMemoryBarrier, ImageViewCreateInfo,
    MemoryDomain, PipelineBarrierInfo, RenderPass, RenderPassAttachment, RenderPassCommand,
    RenderPassDescription, SubpassDescription, ToVk, TransitionInfo,
};
use log::trace;

use crate::app_state;

pub struct LifetimeAllocation<R, D: Label> {
    inner: R,
    desc: D,
    last_frame_used: u64,
}
impl<R, D: Label> LifetimeAllocation<R, D> {
    fn new<A>(inner: R, desc: D) -> LifetimeAllocation<R, D>
    where
        R: CreateFrom<D, A>,
    {
        Self {
            inner,
            desc,
            last_frame_used: 0,
        }
    }
}

pub trait CreateFrom<D, A>
where
    Self: Sized,
{
    fn create(gpu: &Gpu, desc: &D, additional: &A) -> anyhow::Result<Self>;
}

pub trait Label {
    fn label(&self) -> &str;
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderPassHandle {
    id: usize,
}

impl RenderPassHandle {}

pub struct ResourceAllocator<
    R: Sized,
    D: Eq + PartialEq + Clone + Label,
    ID: Hash + Eq + PartialEq + Ord + PartialOrd,
> {
    resources: HashMap<ID, LifetimeAllocation<R, D>>,
}

impl<
        R: Sized,
        D: Eq + PartialEq + Clone + Label,
        ID: Hash + Eq + PartialEq + Ord + PartialOrd + Clone,
    > Default for ResourceAllocator<R, D, ID>
{
    fn default() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }
}

impl<
        R: Sized,
        D: Eq + PartialEq + Clone + Label,
        ID: Hash + Eq + PartialEq + Ord + PartialOrd + Clone + Debug,
    > ResourceAllocator<R, D, ID>
{
    fn get<A>(&mut self, gpu: &Gpu, desc: &D, id: &ID, additional: &A) -> anyhow::Result<&R>
    where
        R: CreateFrom<D, A>,
    {
        self.ensure_resource_exists(gpu, desc, id, additional)?;
        self.ensure_resource_hasnt_changed(gpu, desc, id, additional)?;
        self.update_resource_access_time(id);
        Ok(self.get_unchecked(id))
    }
    fn get_unchecked(&self, id: &ID) -> &R {
        &self.resources[id].inner
    }

    fn ensure_resource_exists<A>(
        &mut self,
        gpu: &Gpu,
        desc: &D,
        id: &ID,
        additional: &A,
    ) -> anyhow::Result<()>
    where
        R: CreateFrom<D, A>,
    {
        if !self.resources.contains_key(id) {
            self.create_resource(gpu, desc, id, additional)?
        }

        Ok(())
    }

    fn create_resource<A>(
        &mut self,
        gpu: &Gpu,
        desc: &D,
        id: &ID,
        additional: &A,
    ) -> anyhow::Result<()>
    where
        R: CreateFrom<D, A>,
    {
        let resource = R::create(gpu, desc, additional)?;
        self.resources
            .insert(id.clone(), LifetimeAllocation::new(resource, desc.clone()));
        Ok(())
    }
    fn ensure_resource_hasnt_changed<A>(
        &mut self,
        gpu: &Gpu,
        desc: &D,
        id: &ID,
        additional: &A,
    ) -> anyhow::Result<()>
    where
        R: CreateFrom<D, A>,
    {
        let old_desc = &self.resources[id].desc;
        if old_desc != desc {
            self.create_resource(gpu, desc, id, additional)?;
        }
        Ok(())
    }
    fn update_resource_access_time(&mut self, id: &ID) {
        self.resources
            .get_mut(id)
            .expect("Failed to fetch resource")
            .last_frame_used = app_state().time().frames_since_start()
    }

    fn remove_unused_resources(&mut self) {
        const FRAMES_LIFETIME: u64 = 10;
        let now = app_state().time().frames_since_start();
        self.resources.retain(|_, res| {
            let can_live = now - res.last_frame_used < FRAMES_LIFETIME;
            if !can_live {
                trace!(
                    "Deallocating resource {:?} after {} frames",
                    res.desc.label(),
                    FRAMES_LIFETIME
                )
            }
            can_live
        })
    }
}

#[derive(Default)]
pub struct ExternalResources<'a> {
    external_images: HashMap<ResourceId, &'a GpuImage>,
    external_image_views: HashMap<ResourceId, &'a GpuImageView>,
    external_render_passes: HashMap<usize, &'a RenderPass>,
}

impl<'a> ExternalResources<'a> {
    pub fn inject_external_image(
        &mut self,
        id: &ResourceId,
        image: &'a GpuImage,
        view: &'a GpuImageView,
    ) {
        self.external_images.insert(*id, image);
        self.external_image_views.insert(*id, view);
    }

    pub fn inject_external_renderpass(
        &mut self,
        handle: RenderPassHandle,
        render_pass: &'a RenderPass,
    ) {
        self.external_render_passes.insert(handle.id, render_pass);
    }
}

impl CreateFrom<ImageDescription, ()> for GpuImage {
    fn create(gpu: &Gpu, desc: &ImageDescription, _: &()) -> anyhow::Result<Self> {
        Ok(gpu
            .create_image(
                &ImageCreateInfo {
                    label: None,
                    width: desc.width,
                    height: desc.height,
                    format: desc.format.to_vk(),
                    usage: match desc.format {
                        ImageFormat::Rgba8 => ImageUsageFlags::COLOR_ATTACHMENT,
                        ImageFormat::Depth => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    } | ImageUsageFlags::INPUT_ATTACHMENT
                        | ImageUsageFlags::SAMPLED,
                },
                MemoryDomain::DeviceLocal,
            )
            .expect("Failed to create image resource"))
    }
}

impl CreateFrom<ImageDescription, GpuImage> for GpuImageView {
    fn create(gpu: &Gpu, desc: &ImageDescription, additional: &GpuImage) -> anyhow::Result<Self> {
        Ok(gpu
            .create_image_view(&ImageViewCreateInfo {
                image: additional,
                view_type: ImageViewType::TYPE_2D,
                format: desc.format.to_vk(),
                components: ComponentMapping::default(),
                subresource_range: ImageSubresourceRange {
                    aspect_mask: match desc.format {
                        ImageFormat::Rgba8 => ImageAspectFlags::COLOR,
                        ImageFormat::Depth => ImageAspectFlags::DEPTH,
                    },
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            })
            .expect("Failed to create image resource"))
    }
}

struct RenderGraphFramebufferCreateInfo<'a> {
    render_pass: &'a RenderPass,
    render_targets: &'a [&'a GpuImageView],
    extents: Extent2D,
}

impl<'a> CreateFrom<RenderPassInfo, RenderGraphFramebufferCreateInfo<'a>> for GpuFramebuffer {
    fn create(
        gpu: &Gpu,
        _: &RenderPassInfo,
        additional: &RenderGraphFramebufferCreateInfo,
    ) -> anyhow::Result<Self> {
        Ok(gpu
            .create_framebuffer(&FramebufferCreateInfo {
                render_pass: additional.render_pass,
                attachments: additional.render_targets,
                width: additional.extents.width,
                height: additional.extents.height,
            })
            .expect("Failed to create framebuffer"))
    }
}

impl Label for ImageDescription {
    fn label(&self) -> &str {
        &self.label
    }
}

impl Label for RenderPassInfo {
    fn label(&self) -> &str {
        &self.label
    }
}

#[derive(Default)]
pub struct DefaultResourceAllocator {
    images: ResourceAllocator<GpuImage, ImageDescription, ResourceId>,
    image_views: ResourceAllocator<GpuImageView, ImageDescription, ResourceId>,
    framebuffers: ResourceAllocator<GpuFramebuffer, RenderPassInfo, u64>,
}

impl DefaultResourceAllocator {
    fn update(&mut self) {
        self.framebuffers.remove_unused_resources();
        self.image_views.remove_unused_resources();
        self.images.remove_unused_resources();
    }
}

pub trait RenderGraphRunner {
    fn run_graph(
        &mut self,
        graph: &CompiledRenderGraph,
        callbacks: &Callbacks,
        resource_allocator: &mut DefaultResourceAllocator,
        external_resources: &ExternalResources,
    ) -> anyhow::Result<()>;
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

#[derive(Hash, Clone, PartialEq, Eq)]
pub struct ImageDescription {
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub samples: u32,
    pub present: bool,
}

#[derive(Hash, Clone)]
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

    hasher: DefaultHasher,
    cached_graph_hash: u64,
    cached_graph: CompiledRenderGraph,
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

#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct RenderPassInfo {
    label: String,
    writes: Vec<ResourceId>,
    reads: Vec<ResourceId>,
    extents: Extent2D,
    is_external: bool,
}

impl Hash for RenderPassInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.extents.hash(state);
        self.is_external.hash(state);
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
        assert!(!self.writes.contains(&handle));
        self.writes.push(handle);
    }
    pub fn writes(&mut self, handles: &[ResourceId]) {
        self.writes.append(&mut handles.to_owned());
    }
    pub fn read(&mut self, handle: ResourceId) {
        assert!(!self.reads.contains(&handle));
        self.reads.push(handle);
    }
    pub fn reads(&mut self, handles: &[ResourceId]) {
        self.reads.append(&mut handles.to_owned());
    }

    pub fn mark_external(&mut self) {
        self.is_external = true;
    }
}

#[derive(Clone, Debug)]
pub enum GraphOperation {
    TransitionRead(Vec<ResourceId>),
    TransitionWrite(Vec<ResourceId>),
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

#[derive(Default, Clone)]
pub struct CompiledRenderGraph {
    pass_infos: Vec<RenderPassInfo>,
    resources_used: HashMap<ResourceId, ResourceInfo>,
    graph_operations: Vec<GraphOperation>,
}

#[derive(Default)]
pub struct Callbacks<'g> {
    callbacks: HashMap<usize, Box<dyn Fn(&Gpu, &mut RenderPassContext) + 'g>>,
    end_callback: Option<Box<dyn Fn(&Gpu, &mut EndContext) + 'g>>,
}

impl<'g> Callbacks<'g> {
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

            hasher: DefaultHasher::default(),
            cached_graph_hash: 0,
            cached_graph: CompiledRenderGraph::default(),
        }
    }

    pub fn use_image(&mut self, description: &ImageDescription) -> GraphResult<ResourceId> {
        let id = self.create_unique_id(&description.label)?;

        let allocation = ResourceInfo {
            ty: AllocationType::Image(description.clone()),
            label: description.label.clone(),
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

    pub fn begin_render_pass(&self, label: &str, extents: Extent2D) -> GraphResult<RenderPassInfo> {
        if self.render_pass_is_defined_already(&label) {
            return Err(CompileError::RenderPassAlreadyDefined(label.to_owned()));
        }
        Ok(RenderPassInfo {
            label: label.to_owned(),
            writes: Default::default(),
            reads: Default::default(),
            extents,
            is_external: false,
        })
    }

    pub fn commit_render_pass(&mut self, pass: RenderPassInfo) -> RenderPassHandle {
        let id = self.passes.len() as _;
        pass.hash(&mut self.hasher);
        self.passes.push(pass);

        RenderPassHandle { id }
    }

    pub fn persist_resource(&mut self, id: &ResourceId) {
        self.persistent_resources.insert(*id);
    }

    pub fn compile(&mut self) -> GraphResult<()> {
        if self.hasher.finish() == self.cached_graph_hash {
            return Ok(());
        }
        let mut compiled = CompiledRenderGraph::default();

        self.prune_passes(&mut compiled);
        self.ensure_graph_acyclic(&compiled)?;
        let merge_candidates = self.find_merge_candidates(&mut compiled);

        self.find_optimal_execution_order(&mut compiled, merge_candidates);

        self.cached_graph_hash = self.hasher.finish();
        self.cached_graph = compiled.clone();

        self.prepare_for_next_frame();
        Ok(())
    }

    pub fn run(
        &self,
        runner: &mut dyn RenderGraphRunner,
        callbacks: &Callbacks,
        resource_allocator: &mut DefaultResourceAllocator,
        external_resources: &ExternalResources,
    ) -> anyhow::Result<()> {
        runner.run_graph(
            &self.cached_graph,
            callbacks,
            resource_allocator,
            external_resources,
        )
    }

    fn prune_passes(&mut self, compiled: &mut CompiledRenderGraph) {
        let mut working_set: Vec<_> = self
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
                .filter(|(_, (_, p))| p.reads.iter().any(|read| pass.reads.contains(read)))
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
        for (i, pass) in compiled.pass_infos.iter().enumerate() {
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
    }

    fn prepare_for_next_frame(&mut self) {
        self.passes.clear();
        self.persistent_resources.clear();
        self.allocations.clear();
    }
}

pub struct GpuRunner {
    resource_states: HashMap<ResourceId, TransitionInfo>,
    render_passes: HashMap<usize, RenderPass>,
}
impl GpuRunner {
    pub fn new() -> Self {
        Self {
            resource_states: Default::default(),
            render_passes: Default::default(),
        }
    }

    pub fn get_image_view<'r, 'e>(
        gpu: &Gpu,
        graph: &CompiledRenderGraph,
        id: &ResourceId,
        allocator: &'r mut DefaultResourceAllocator,
        external_resources: &ExternalResources<'e>,
    ) -> anyhow::Result<&'e GpuImageView>
    where
        'r: 'e,
    {
        if external_resources.external_image_views.contains_key(id) {
            Ok(external_resources.external_image_views[id])
        } else {
            let desc = match &graph.resources_used[id].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            let image = allocator.images.get(gpu, &desc, id, &())?;
            allocator.image_views.get(gpu, &desc, id, image)
        }
    }
    pub fn get_image<'r, 'e>(
        gpu: &Gpu,
        graph: &CompiledRenderGraph,
        id: &ResourceId,
        allocator: &'r mut DefaultResourceAllocator,
        external_resources: &ExternalResources<'e>,
    ) -> anyhow::Result<&'e GpuImage>
    where
        'r: 'e,
    {
        if external_resources.external_images.contains_key(id) {
            Ok(external_resources.external_images[id])
        } else {
            let desc = match &graph.resources_used[id].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            allocator.images.get(gpu, &desc, id, &())
        }
    }

    fn transition_image_read(
        &mut self,
        graph: &CompiledRenderGraph,
        command_buffer: &mut CommandBuffer,
        id: &ResourceId,
        image: &GpuImage,
    ) {
        let resource_info = &graph.resources_used[id];
        match &resource_info.ty {
            AllocationType::Image(desc) | AllocationType::ExternalImage(desc) => {
                let access_flag = match desc.format {
                    ImageFormat::Rgba8 => AccessFlags::COLOR_ATTACHMENT_READ,
                    ImageFormat::Depth => AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                };

                let old_layout = self.resource_states.entry(*id).or_insert(TransitionInfo {
                    layout: ImageLayout::UNDEFINED,
                    access_mask: AccessFlags::empty(),
                    stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                });
                let new_layout = TransitionInfo {
                    layout: ImageLayout::READ_ONLY_OPTIMAL,
                    access_mask: AccessFlags::SHADER_READ | access_flag,
                    stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                };

                let aspect_mask = match desc.format {
                    ImageFormat::Rgba8 => ImageAspectFlags::COLOR,
                    ImageFormat::Depth => ImageAspectFlags::DEPTH,
                };

                let memory_barrier = ImageMemoryBarrier {
                    src_access_mask: old_layout.access_mask,
                    dst_access_mask: new_layout.access_mask,
                    old_layout: old_layout.layout,
                    new_layout: new_layout.layout,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: ImageSubresourceRange {
                        aspect_mask,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };
                command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                    src_stage_mask: old_layout.stage_mask,
                    dst_stage_mask: new_layout.stage_mask,
                    dependency_flags: DependencyFlags::empty(),
                    image_memory_barriers: &[memory_barrier],
                    ..Default::default()
                });
                self.resource_states.insert(*id, new_layout);
            }
        }
    }

    fn transition_image_write(
        &mut self,
        graph: &CompiledRenderGraph,
        command_buffer: &mut CommandBuffer,
        id: &ResourceId,
        image: &GpuImage,
    ) {
        let resource_info = &graph.resources_used[id];
        match &resource_info.ty {
            AllocationType::Image(desc) | AllocationType::ExternalImage(desc) => {
                let access_flag = match desc.format {
                    ImageFormat::Rgba8 => AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ImageFormat::Depth => AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                };

                let old_layout = self.resource_states.entry(*id).or_insert(TransitionInfo {
                    layout: ImageLayout::UNDEFINED,
                    access_mask: AccessFlags::empty(),
                    stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                });
                let new_layout = TransitionInfo {
                    layout: match desc.format {
                        ImageFormat::Rgba8 => {
                            if desc.present {
                                ImageLayout::PRESENT_SRC_KHR
                            } else {
                                ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                            }
                        }
                        ImageFormat::Depth => ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    },
                    access_mask: AccessFlags::SHADER_READ | access_flag,
                    stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                };

                let aspect_mask = match desc.format {
                    ImageFormat::Rgba8 => ImageAspectFlags::COLOR,
                    ImageFormat::Depth => ImageAspectFlags::DEPTH,
                };

                let memory_barrier = ImageMemoryBarrier {
                    src_access_mask: old_layout.access_mask,
                    dst_access_mask: new_layout.access_mask,
                    old_layout: old_layout.layout,
                    new_layout: new_layout.layout,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: ImageSubresourceRange {
                        aspect_mask,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };
                command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                    src_stage_mask: old_layout.stage_mask,
                    dst_stage_mask: new_layout.stage_mask,
                    dependency_flags: DependencyFlags::empty(),
                    image_memory_barriers: &[memory_barrier],
                    ..Default::default()
                });
                self.resource_states.insert(*id, new_layout);
            }
        }
    }

    fn create_render_pass(
        &mut self,
        graph: &CompiledRenderGraph,
        pass_info: &RenderPassInfo,
        id: usize,
    ) -> Result<&RenderPass, anyhow::Error> {
        let writes: Vec<_> = pass_info
            .writes
            .iter()
            .filter_map(|id| {
                let resource_desc = &graph.resources_used[&id];
                match &resource_desc.ty {
                    AllocationType::Image(image_desc)
                    | AllocationType::ExternalImage(image_desc) => Some(image_desc),
                }
            })
            .map(|image_desc| RenderPassAttachment {
                format: image_desc.format.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: match image_desc.format {
                    ImageFormat::Rgba8 => {
                        if image_desc.present {
                            ImageLayout::PRESENT_SRC_KHR
                        } else {
                            ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        }
                    }
                    ImageFormat::Depth => ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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
            .filter(|(_, attach)| attach.format != ImageFormat::Depth.to_vk())
            .map(|(i, _)| AttachmentReference {
                attachment: i as _,
                layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            })
            .collect();
        let depth_attachments: Vec<_> = pass_info
            .writes
            .iter()
            .map(|id| &graph.resources_used[id])
            .enumerate()
            .filter_map(|(idx, info)| match &info.ty {
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
                layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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
                depth_stencil_attachment: &depth_attachments,
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
        let pass = RenderPass::new(&crate::app_state().gpu, &description)?;
        self.render_passes.insert(id, pass);

        trace!("Created new render pass {}", pass_info.label);
        Ok(&self.render_passes[&id])
    }

    fn get_renderpass(&self, id: usize) -> &RenderPass {
        self.render_passes
            .get(&id)
            .expect(&format!("Failed to find renderpass with id {}", id))
    }

    fn ensure_render_pass_exists(
        &mut self,
        rp: &usize,
        graph: &CompiledRenderGraph,
        external_resources: &ExternalResources,
    ) -> Result<(), anyhow::Error> {
        {
            let pass_info = &graph.pass_infos[*rp];
            if pass_info.is_external && !external_resources.external_render_passes.contains_key(rp)
            {
                panic!(
                    "RenderPass {} is external, but it hasn't been injected",
                    pass_info.label
                );
            }

            if !self.render_passes.contains_key(rp) {
                self.create_render_pass(graph, pass_info, *rp)?;
            };
        }
        Ok(())
    }
}

impl RenderGraphRunner for GpuRunner {
    fn run_graph(
        &mut self,
        graph: &CompiledRenderGraph,
        callbacks: &Callbacks,
        resource_allocator: &mut DefaultResourceAllocator,
        external_resources: &ExternalResources,
    ) -> anyhow::Result<()> {
        self.resource_states.clear();
        resource_allocator.update();

        let mut command_buffer =
            CommandBuffer::new(&crate::app_state().gpu, gpu::QueueType::Graphics)?;

        let label = command_buffer.begin_debug_region(
            &format!(
                "Rendering frame {}",
                crate::app_state().time().frames_since_start()
            ),
            [0.0, 0.3, 0.0, 1.0],
        );
        for op in &graph.graph_operations {
            match op {
                GraphOperation::TransitionRead(resources) => {
                    command_buffer.insert_debug_label(
                        "Transitioning resources to read",
                        [0.0, 0.3, 0.3, 1.0],
                    );
                    for id in resources {
                        let image = Self::get_image(
                            &crate::app_state().gpu,
                            graph,
                            id,
                            resource_allocator,
                            external_resources,
                        )?;
                        self.transition_image_read(graph, &mut command_buffer, id, image);
                    }
                }
                GraphOperation::TransitionWrite(resources) => {
                    command_buffer.insert_debug_label(
                        "Transitioning resources to write",
                        [0.0, 0.3, 0.3, 1.0],
                    );

                    for id in resources {
                        let image = Self::get_image(
                            &crate::app_state().gpu,
                            graph,
                            id,
                            resource_allocator,
                            external_resources,
                        )?;
                        self.transition_image_write(graph, &mut command_buffer, id, image);
                    }
                }
                GraphOperation::ExecuteRenderPass(rp) => {
                    self.ensure_render_pass_exists(rp, graph, external_resources)?;
                    let info = &graph.pass_infos[*rp];

                    ensure_graph_allocated_image_views_exist(
                        info,
                        external_resources,
                        graph,
                        resource_allocator,
                    )?;
                    let views = resolve_image_views_unchecked(
                        info,
                        external_resources,
                        &mut resource_allocator.image_views,
                    );

                    let framebuffer_hash = compute_framebuffer_hash(&views);

                    let pass = self.get_renderpass(*rp);
                    let framebuffer = resource_allocator.framebuffers.get(
                        &app_state().gpu,
                        &info.clone(),
                        &framebuffer_hash,
                        &RenderGraphFramebufferCreateInfo {
                            render_pass: pass,
                            render_targets: &views,
                            extents: info.extents,
                        },
                    )?;

                    let cb = callbacks.callbacks.get(rp);
                    let clear_color_values: Vec<_> = info
                        .writes
                        .iter()
                        .map(|rd| {
                            let res_info = &graph.resources_used[rd];
                            match &res_info.ty {
                                AllocationType::Image(desc)
                                | AllocationType::ExternalImage(desc) => match desc.format {
                                    ImageFormat::Rgba8 => ClearValue {
                                        color: vk::ClearColorValue {
                                            float32: [0.0, 0.0, 0.0, 0.0],
                                        },
                                    },
                                    ImageFormat::Depth => ClearValue {
                                        depth_stencil: ClearDepthStencilValue {
                                            depth: 1.0,
                                            stencil: 255,
                                        },
                                    },
                                },
                            }
                        })
                        .collect();
                    let render_pass_label = command_buffer.begin_debug_region(
                        &format!("Begin Render Pass: {}", info.label),
                        [0.3, 0.0, 0.0, 1.0],
                    );
                    let render_pass_command =
                        command_buffer.begin_render_pass(&BeginRenderPassInfo {
                            framebuffer,
                            render_pass: pass,
                            clear_color_values: &clear_color_values,
                            render_area: Rect2D {
                                offset: Offset2D::default(),
                                extent: info.extents,
                            },
                        });
                    let mut context = RenderPassContext {
                        render_pass: &pass,
                        framebuffer,
                        render_pass_command,
                    };

                    if let Some(cb) = cb {
                        cb(&crate::app_state().gpu, &mut context);
                    }
                    render_pass_label.end();
                }
            }
        }
        if let Some(end_cb) = &callbacks.end_callback {
            end_cb(
                &crate::app_state().gpu,
                &mut EndContext {
                    command_buffer: &mut command_buffer,
                },
            );
        }
        label.end();
        command_buffer.submit(&gpu::CommandBufferSubmitInfo {
            wait_semaphores: &[&crate::app_state().swapchain.image_available_semaphore],
            wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            signal_semaphores: &[&crate::app_state().swapchain.render_finished_semaphore],
            fence: Some(&crate::app_state().swapchain.in_flight_fence),
        })?;

        crate::app_state().gpu.wait_device_idle()?;
        Ok(())
    }
}

fn compute_framebuffer_hash(views: &Vec<&GpuImageView>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for view in views {
        view.hash(&mut hasher);
    }

    let framebuffer_hash = hasher.finish();
    framebuffer_hash
}

fn ensure_graph_allocated_image_views_exist(
    info: &RenderPassInfo,
    external_resources: &ExternalResources,
    graph: &CompiledRenderGraph,
    resource_allocator: &mut DefaultResourceAllocator,
) -> Result<(), anyhow::Error> {
    Ok(for writes in &info.writes {
        let _ = if !external_resources.external_image_views.contains_key(writes) {
            let desc = match &graph.resources_used[writes].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            let image = resource_allocator
                .images
                .get(&app_state().gpu, &desc, writes, &())?;
            resource_allocator
                .image_views
                .get(&app_state().gpu, &desc, writes, image)?;
        };
    })
}

fn resolve_image_views_unchecked<'e, 'a>(
    info: &RenderPassInfo,
    external_resources: &ExternalResources<'e>,
    image_views_allocator: &'a mut ResourceAllocator<GpuImageView, ImageDescription, ResourceId>,
) -> Vec<&'e GpuImageView>
where
    'a: 'e,
{
    let mut views = vec![];
    for writes in &info.writes {
        let view = if external_resources.external_image_views.contains_key(writes) {
            &external_resources.external_image_views[writes]
        } else {
            image_views_allocator.get_unchecked(writes)
        };
        views.push(view);
    }
    views
}

#[derive(Default)]
pub struct RenderGraphPrinter {}

impl RenderGraphRunner for RenderGraphPrinter {
    fn run_graph(
        &mut self,
        graph: &CompiledRenderGraph,
        _callbacks: &Callbacks,
        _allocator: &mut DefaultResourceAllocator,
        _external_resources: &ExternalResources,
    ) -> anyhow::Result<()> {
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
    use ash::vk::Extent2D;

    use crate::{CompileError, ResourceId};

    use super::{ImageDescription, RenderGraph, RenderGraphPrinter, RenderGraphRunner};

    fn alloc(name: &str, rg: &mut RenderGraph) -> ResourceId {
        let description = ImageDescription {
            label: name.to_owned(),
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
            present: false,
        };

        rg.use_image(&description).unwrap()
    }
    #[test]
    pub fn prune_empty() {
        let gpu = RenderGraphPrinter::default();
        let mut render_graph = RenderGraph::new();

        let image_desc = ImageDescription {
            label: "".to_owned(),
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
            present: false,
        };

        let color_component = alloc("color", &mut render_graph);
        let position_component = alloc("position", &mut render_graph);
        let tangent_component = alloc("tangent", &mut render_graph);
        let normal_component = alloc("normal", &mut render_graph);

        let mut gbuffer = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap();
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
            label: "".to_owned(),
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
            present: false,
        };

        let color_component_1 = alloc("color1", &mut render_graph);
        let color_component_2 = alloc("color2", &mut render_graph);
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
        let p1 = render_graph
            .begin_render_pass("pass", Extent2D::default())
            .unwrap();
        render_graph.commit_render_pass(p1);
        let p2 = render_graph.begin_render_pass("pass", Extent2D::default());
        assert!(p2.is_err_and(|e| e == CompileError::RenderPassAlreadyDefined("pass".to_owned())));
    }

    #[test]
    pub fn survive_1() {
        let gpu = RenderGraphPrinter::default();
        let mut render_graph = RenderGraph::new();

        let image_desc = ImageDescription {
            label: "".to_owned(),
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
            present: false,
        };

        let color_component_1 = alloc("color1", &mut render_graph);
        let color_component_2 = alloc("color2", &mut render_graph);
        let position_component = alloc("position", &mut render_graph);
        let tangent_component = alloc("tangent", &mut render_graph);
        let normal_component = alloc("normal", &mut render_graph);
        let output_image = alloc("output", &mut render_graph);

        let mut gbuffer = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap();
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);

        let mut compose_gbuffer = render_graph
            .begin_render_pass("compose_gbuffer", Extent2D::default())
            .unwrap();
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
            label: "".to_owned(),
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
            present: false,
        };

        let color_component_1 = alloc("color1", &mut render_graph);
        let color_component_2 = alloc("color2", &mut render_graph);
        let position_component = alloc("position", &mut render_graph);
        let tangent_component = alloc("tangent", &mut render_graph);
        let normal_component = alloc("normal", &mut render_graph);
        let output_image = alloc("output", &mut render_graph);
        let output_image = alloc("unused", &mut render_graph);

        let mut gbuffer = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap();
        gbuffer.write(color_component);
        gbuffer.write(position_component);
        gbuffer.write(tangent_component);
        gbuffer.write(normal_component);
        let gbuffer = render_graph.commit_render_pass(gbuffer);

        let mut compose_gbuffer = render_graph
            .begin_render_pass("compose_gbuffer", Extent2D::default())
            .unwrap();
        compose_gbuffer.read(color_component);
        compose_gbuffer.read(position_component);
        compose_gbuffer.read(tangent_component);
        compose_gbuffer.read(normal_component);
        compose_gbuffer.write(output_image);
        let compose_gbuffer = render_graph.commit_render_pass(compose_gbuffer);

        // adding an empty pass that outputs to an unused buffer
        let mut unused_pass = render_graph
            .begin_render_pass("unused", Extent2D::default())
            .unwrap();
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

            let mut p1 = render_graph
                .begin_render_pass("p1", Extent2D::default())
                .unwrap();
            p1.writes(&[r1, r2]);
            render_graph.commit_render_pass(p1);

            let mut p2 = render_graph
                .begin_render_pass("p2", Extent2D::default())
                .unwrap();
            p2.reads(&[r1, r2]);
            p2.writes(&[r3]);
            render_graph.commit_render_pass(p2);

            let mut p3 = render_graph
                .begin_render_pass("p3", Extent2D::default())
                .unwrap();
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

            let mut p1 = render_graph
                .begin_render_pass("p1", Extent2D::default())
                .unwrap();
            p1.writes(&[r1, r2]);
            render_graph.commit_render_pass(p1);

            let mut p2 = render_graph
                .begin_render_pass("p2", Extent2D::default())
                .unwrap();
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

        let mut p1 = render_graph
            .begin_render_pass("p1", Extent2D::default())
            .unwrap();
        p1.writes(&[r1, r2, r3, r4]);
        render_graph.commit_render_pass(p1);

        let mut p2 = render_graph
            .begin_render_pass("p2", Extent2D::default())
            .unwrap();
        let r5 = alloc("r5", &mut render_graph);
        p2.reads(&[r1, r3]);
        p2.writes(&[r5]);
        render_graph.commit_render_pass(p2);

        let mut p3 = render_graph
            .begin_render_pass("p3", Extent2D::default())
            .unwrap();
        let r6 = alloc("r6", &mut render_graph);
        let r7 = alloc("r7", &mut render_graph);
        let r8 = alloc("r8", &mut render_graph);
        p3.reads(&[r2, r4]);
        p3.writes(&[r6, r7, r8]);
        render_graph.commit_render_pass(p3);

        // pruned
        let mut u1 = render_graph
            .begin_render_pass("u1", Extent2D::default())
            .unwrap();
        let ru1 = alloc("ru1", &mut render_graph);
        let ru2 = alloc("ru2", &mut render_graph);
        u1.reads(&[r7, r8]);
        u1.writes(&[ru1, ru2]);

        let mut p4 = render_graph
            .begin_render_pass("p4", Extent2D::default())
            .unwrap();
        let r9 = alloc("r9", &mut render_graph);
        let r10 = alloc("r10", &mut render_graph);
        p4.reads(&[r7, r8]);
        p4.writes(&[r9, r10]);
        render_graph.commit_render_pass(p4);

        let mut pb = render_graph
            .begin_render_pass("pb", Extent2D::default())
            .unwrap();
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
    }
}
