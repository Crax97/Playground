/*


*/

use std::{
    cell::RefCell,
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    error::Error,
    fmt::Debug,
    hash::{Hash, Hasher},
};

use ash::vk::{
    self, AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor,
    BlendOp, BorderColor, ClearDepthStencilValue, ClearValue, ColorComponentFlags, CompareOp,
    ComponentMapping, DependencyFlags, Extent2D, Filter, ImageAspectFlags, ImageLayout,
    ImageSubresourceRange, ImageUsageFlags, ImageViewType, Offset2D, PipelineBindPoint,
    PipelineStageFlags, Rect2D, SampleCountFlags, SamplerAddressMode, SamplerCreateFlags,
    SamplerCreateInfo, SamplerMipmapMode, StructureType, SubpassDependency,
    SubpassDescriptionFlags,
};
use gpu::{
    BeginRenderPassInfo, BlendState, CommandBuffer, DescriptorInfo, DescriptorSetInfo,
    FramebufferCreateInfo, Gpu, GpuDescriptorSet, GpuFramebuffer, GpuImage, GpuImageView,
    GpuSampler, ImageCreateInfo, ImageFormat, ImageMemoryBarrier, ImageViewCreateInfo,
    MemoryDomain, Pipeline, PipelineBarrierInfo, RenderPass, RenderPassAttachment,
    RenderPassCommand, RenderPassDescription, SubpassDescription, Swapchain, ToVk, TransitionInfo,
};

use ash::vk::PushConstantRange;
use gpu::{
    BindingElement, CullMode, DepthStencilAttachment, DepthStencilState, FragmentStageInfo,
    FrontFace, GlobalBinding, GpuShaderModule, LogicOp, PipelineDescription, PolygonMode,
    PrimitiveTopology, VertexBindingDescription, VertexStageInfo,
};
use indexmap::IndexSet;
use log::trace;

pub struct LifetimeAllocation<R, D> {
    inner: R,
    desc: D,
    last_frame_used: u64,
}
impl<R, D> LifetimeAllocation<R, D> {
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

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RenderPassHandle {
    label: &'static str,
}

impl RenderPassHandle {}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PipelineHandle {
    label: &'static str,
    owner: RenderPassHandle,
}
#[derive(Hash, PartialOrd, Ord, PartialEq, Eq, Clone, Copy, Debug)]
struct FramebufferHandle {
    render_pass_label: &'static str,
    hash: u64,
}

pub struct ResourceAllocator<
    R: Sized,
    D: Eq + PartialEq + Clone,
    ID: Hash + Eq + PartialEq + Ord + PartialOrd,
> {
    resources: HashMap<ID, LifetimeAllocation<R, D>>,
    lifetime: u64,
}

impl<R: Sized, D: Eq + PartialEq + Clone, ID: Hash + Eq + PartialEq + Ord + PartialOrd + Clone>
    ResourceAllocator<R, D, ID>
{
    fn new(lifetime: u64) -> Self {
        Self {
            resources: HashMap::new(),
            lifetime,
        }
    }
}

impl<
        R: Sized,
        D: Eq + PartialEq + Clone,
        ID: Hash + Eq + PartialEq + Ord + PartialOrd + Clone + Debug,
    > ResourceAllocator<R, D, ID>
{
    fn get<A>(
        &mut self,
        ctx: &GraphRunContext,
        desc: &D,
        id: &ID,
        additional: &A,
    ) -> anyhow::Result<&R>
    where
        R: CreateFrom<D, A>,
    {
        self.get_explicit(ctx.gpu, ctx.current_iteration, desc, id, additional)
    }

    fn get_explicit<A>(
        &mut self,
        gpu: &Gpu,
        current_iteration: u64,
        desc: &D,
        id: &ID,
        additional: &A,
    ) -> anyhow::Result<&R>
    where
        R: CreateFrom<D, A>,
    {
        self.ensure_resource_exists(gpu, desc, id, additional)?;
        self.ensure_resource_hasnt_changed(gpu, desc, id, additional)?;
        self.update_resource_access_time(id, current_iteration);
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
        trace!("Created new resource: {:?}", id);
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
    fn update_resource_access_time(&mut self, id: &ID, current_iteration: u64) {
        self.resources
            .get_mut(id)
            .expect("Failed to fetch resource")
            .last_frame_used = current_iteration;
    }

    fn remove_unused_resources(&mut self, current_iteration: u64) {
        if self.lifetime == 0 {
            return;
        }

        self.resources.retain(|id, res| {
            let can_live = current_iteration - res.last_frame_used < self.lifetime;
            if !can_live {
                trace!(
                    "Deallocating resource {:?} after {} frames",
                    id,
                    self.lifetime
                )
            }
            can_live
        })
    }
}

#[derive(Default)]
struct ExternalResources<'a> {
    external_images: HashMap<ResourceId, &'a GpuImage>,
    external_image_views: HashMap<ResourceId, &'a GpuImageView>,
    external_render_passes: HashMap<RenderPassHandle, &'a RenderPass>,
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
        handle: &RenderPassHandle,
        render_pass: &'a RenderPass,
    ) {
        self.external_render_passes.insert(*handle, render_pass);
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
                        ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                            ImageUsageFlags::COLOR_ATTACHMENT
                        }
                        ImageFormat::Depth => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    } | ImageUsageFlags::INPUT_ATTACHMENT
                        | ImageUsageFlags::SAMPLED,
                },
                MemoryDomain::DeviceLocal,
            )
            .expect("Failed to create image resource"))
    }
}

impl CreateFrom<ImageDescription, ()> for GpuSampler {
    fn create(gpu: &Gpu, _: &ImageDescription, _: &()) -> anyhow::Result<Self> {
        Ok(gpu
            .create_sampler(&SamplerCreateInfo {
                s_type: StructureType::SAMPLER_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: SamplerCreateFlags::empty(),
                mag_filter: Filter::LINEAR,
                min_filter: Filter::LINEAR,
                mipmap_mode: SamplerMipmapMode::LINEAR,
                address_mode_u: SamplerAddressMode::REPEAT,
                address_mode_v: SamplerAddressMode::REPEAT,
                address_mode_w: SamplerAddressMode::REPEAT,
                mip_lod_bias: 0.0,
                anisotropy_enable: vk::TRUE,
                max_anisotropy: gpu
                    .physical_device_properties()
                    .limits
                    .max_sampler_anisotropy,
                compare_enable: vk::FALSE,
                compare_op: CompareOp::ALWAYS,
                min_lod: 0.0,
                max_lod: 0.0,
                border_color: BorderColor::default(),
                unnormalized_coordinates: vk::FALSE,
            })
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
                        ImageFormat::Rgba8 | ImageFormat::RgbaFloat => ImageAspectFlags::COLOR,
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

struct DescriptorSetCreateInfo<'a> {
    inputs: &'a [DescriptorInfo<'a>],
}
impl<'a> CreateFrom<RenderPassInfo, RenderGraph> for RenderPass {
    fn create(gpu: &Gpu, pass_info: &RenderPassInfo, graph: &RenderGraph) -> anyhow::Result<Self> {
        let mut color_attachments = vec![];
        let mut depth_attachments = vec![];

        let mut all_attachments: Vec<_> = vec![];
        let mut index = 0;
        for write in &pass_info.attachment_writes {
            let image_desc = &graph.allocations[&write];
            let image_desc = match &image_desc.ty {
                AllocationType::Image(image_desc) | AllocationType::ExternalImage(image_desc) => {
                    image_desc
                }
            };
            let attachment = RenderPassAttachment {
                format: image_desc.format.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: match image_desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                        if image_desc.present {
                            ImageLayout::PRESENT_SRC_KHR
                        } else {
                            ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        }
                    }
                    ImageFormat::Depth => ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                },
                blend_state: if let Some(state) = pass_info.blend_state {
                    state
                } else {
                    BlendState {
                        blend_enable: true,
                        src_color_blend_factor: BlendFactor::ONE,
                        dst_color_blend_factor: BlendFactor::ZERO,
                        color_blend_op: BlendOp::ADD,
                        src_alpha_blend_factor: BlendFactor::ONE,
                        dst_alpha_blend_factor: BlendFactor::ZERO,
                        alpha_blend_op: BlendOp::ADD,
                        color_write_mask: ColorComponentFlags::RGBA,
                    }
                },
            };
            all_attachments.push(attachment);

            match image_desc.format {
                ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                    color_attachments.push(AttachmentReference {
                        attachment: index as _,
                        layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    })
                }
                ImageFormat::Depth => depth_attachments.push(AttachmentReference {
                    attachment: index as _,
                    layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                }),
            }
            index += 1;
        }
        for read in &pass_info.attachment_reads {
            let image_desc = &graph.allocations[&read];
            let image_desc = match &image_desc.ty {
                AllocationType::Image(image_desc) | AllocationType::ExternalImage(image_desc) => {
                    image_desc
                }
            };
            let attachment = RenderPassAttachment {
                format: image_desc.format.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::LOAD,
                store_op: AttachmentStoreOp::NONE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: match image_desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                        if image_desc.present {
                            ImageLayout::PRESENT_SRC_KHR
                        } else {
                            ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        }
                    }
                    ImageFormat::Depth => ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                },
                final_layout: match image_desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                        if image_desc.present {
                            ImageLayout::PRESENT_SRC_KHR
                        } else {
                            ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        }
                    }
                    ImageFormat::Depth => ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                },
                blend_state: if let Some(state) = pass_info.blend_state {
                    state
                } else {
                    BlendState {
                        blend_enable: true,
                        src_color_blend_factor: BlendFactor::ONE,
                        dst_color_blend_factor: BlendFactor::ZERO,
                        color_blend_op: BlendOp::ADD,
                        src_alpha_blend_factor: BlendFactor::ONE,
                        dst_alpha_blend_factor: BlendFactor::ZERO,
                        alpha_blend_op: BlendOp::ADD,
                        color_write_mask: ColorComponentFlags::RGBA,
                    }
                },
            };
            all_attachments.push(attachment);

            match image_desc.format {
                ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                    color_attachments.push(AttachmentReference {
                        attachment: index as _,
                        layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    })
                }
                ImageFormat::Depth => depth_attachments.push(AttachmentReference {
                    attachment: index as _,
                    layout: ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                }),
            }
            index += 1;
        }

        let description = RenderPassDescription {
            attachments: &all_attachments,
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
        let pass = RenderPass::new(&gpu, &description)?;

        Ok(pass)
    }
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

impl<'a> CreateFrom<RenderPassInfo, DescriptorSetCreateInfo<'a>> for GpuDescriptorSet {
    fn create(
        gpu: &Gpu,
        _: &RenderPassInfo,
        additional: &DescriptorSetCreateInfo,
    ) -> anyhow::Result<Self> {
        Ok(gpu
            .create_descriptor_set(&DescriptorSetInfo {
                descriptors: additional.inputs,
            })
            .expect("Failed to create framebuffer"))
    }
}

type RenderPassAllocator = ResourceAllocator<RenderPass, RenderPassInfo, RenderPassHandle>;

pub struct DefaultResourceAllocator {
    images: ResourceAllocator<GpuImage, ImageDescription, ResourceId>,
    image_views: ResourceAllocator<GpuImageView, ImageDescription, ResourceId>,
    samplers: ResourceAllocator<GpuSampler, ImageDescription, ResourceId>,
    framebuffers: ResourceAllocator<GpuFramebuffer, RenderPassInfo, FramebufferHandle>,
    descriptors: ResourceAllocator<GpuDescriptorSet, RenderPassInfo, u64>,
    render_passes: RenderPassAllocator,
}

impl DefaultResourceAllocator {
    pub fn new() -> Self {
        Self {
            images: ResourceAllocator::new(2),
            image_views: ResourceAllocator::new(2),
            framebuffers: ResourceAllocator::new(5),
            render_passes: RenderPassAllocator::new(0),
            samplers: ResourceAllocator::new(2),
            descriptors: ResourceAllocator::new(2),
        }
    }
}

impl DefaultResourceAllocator {
    fn update(&mut self, current_iteration: u64) {
        self.framebuffers.remove_unused_resources(current_iteration);
        self.image_views.remove_unused_resources(current_iteration);
        self.images.remove_unused_resources(current_iteration);
        self.samplers.remove_unused_resources(current_iteration);
        self.descriptors.remove_unused_resources(current_iteration);
        self.render_passes
            .remove_unused_resources(current_iteration);
    }
}
pub struct GraphRunContext<'a, 'e> {
    gpu: &'a Gpu,
    current_iteration: u64,
    swapchain: &'a mut Swapchain,

    callbacks: Callbacks<'e>,
    external_resources: ExternalResources<'e>,
}

impl<'a, 'e> GraphRunContext<'a, 'e> {
    pub fn new(gpu: &'a Gpu, swapchain: &'a mut Swapchain, current_iteration: u64) -> Self {
        Self {
            gpu,
            current_iteration,
            swapchain,
            callbacks: Callbacks::default(),
            external_resources: ExternalResources::default(),
        }
    }

    pub(crate) fn register_callback<F: FnMut(&Gpu, &mut RenderPassContext) + 'e>(
        &mut self,
        handle: &RenderPassHandle,
        callback: F,
    ) {
        self.callbacks.register_callback(handle, callback)
    }

    pub fn register_end_callback<F: FnMut(&Gpu, &mut EndContext) + 'e>(&mut self, callback: F) {
        self.callbacks.register_end_callback(callback)
    }

    pub(crate) fn inject_external_renderpass(
        &mut self,
        handle: &RenderPassHandle,
        pass: &'e RenderPass,
    ) {
        self.external_resources
            .inject_external_renderpass(handle, pass);
    }

    pub(crate) fn inject_external_image(
        &mut self,
        handle: &ResourceId,
        image: &'e GpuImage,
        view: &'e GpuImageView,
    ) {
        self.external_resources
            .inject_external_image(handle, image, view);
    }
}

pub trait RenderGraphRunner {
    fn run_graph(
        &mut self,
        context: &mut GraphRunContext,
        graph: &RenderGraph,
        resource_allocator: &mut DefaultResourceAllocator,
    ) -> anyhow::Result<()>;
}

#[derive(Copy, Clone, Eq, Ord, Debug)]
pub struct ResourceId {
    label: &'static str,
    raw: u64,
}

impl std::hash::Hash for ResourceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl PartialEq for ResourceId {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl PartialOrd for ResourceId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.raw.partial_cmp(&other.raw)
    }
}

impl ResourceId {
    fn make(label: &'static str) -> ResourceId {
        let mut hasher = DefaultHasher::new();
        label.hash(&mut hasher);
        Self {
            label,
            raw: hasher.finish(),
        }
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Eq)]
pub struct ImageDescription {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub samples: u32,
    pub present: bool,
}

#[derive(Hash, Copy, Clone)]
pub enum AllocationType {
    Image(ImageDescription),
    ExternalImage(ImageDescription),
}

#[derive(Hash, Copy, Clone)]
pub struct ResourceInfo {
    pub label: &'static str,
    pub ty: AllocationType,

    defined_this_frame: bool,
}

#[derive(Hash)]
pub struct ModuleInfo<'a> {
    pub module: &'a GpuShaderModule,
    pub entry_point: &'a str,
}

#[derive(Hash)]
pub enum RenderStage<'a> {
    Graphics {
        vertex: ModuleInfo<'a>,
        fragment: ModuleInfo<'a>,
    },
    Compute {
        shader: ModuleInfo<'a>,
    },
}

// This struct contains all the repetitive stuff that has mostly does not change in pipelines, so that
// it can be ..default()ed when needed

pub struct FragmentState<'a> {
    pub input_topology: PrimitiveTopology,
    pub primitive_restart: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub depth_stencil_state: DepthStencilState,
    pub logic_op: Option<LogicOp>,
    pub push_constant_ranges: &'a [PushConstantRange],
}

impl<'a> Default for FragmentState<'a> {
    fn default() -> Self {
        Self {
            input_topology: Default::default(),
            primitive_restart: false,
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::None,
            front_face: FrontFace::ClockWise,
            depth_stencil_state: DepthStencilState {
                depth_test_enable: false,
                ..Default::default()
            },
            logic_op: None,
            push_constant_ranges: Default::default(),
        }
    }
}

pub struct RenderGraphPipelineDescription<'a> {
    pub vertex_inputs: &'a [VertexBindingDescription<'a>],
    pub stage: RenderStage<'a>,
    pub fragment_state: FragmentState<'a>,
}

pub(crate) fn create_pipeline_for_graph_renderpass(
    graph: &RenderGraph,
    pass_info: &RenderPassInfo,
    vk_renderpass: &RenderPass,
    gpu: &Gpu,
    description: &RenderGraphPipelineDescription,
) -> anyhow::Result<Pipeline> {
    let mut set_zero_bindings = vec![];
    for (idx, read) in pass_info.shader_reads.iter().enumerate() {
        let resource = graph.get_resource_info(read)?;
        set_zero_bindings.push(BindingElement {
            binding_type: match resource.ty {
                crate::AllocationType::Image(_) | crate::AllocationType::ExternalImage(_) => {
                    gpu::BindingType::CombinedImageSampler
                }
            },
            index: idx as _,
            stage: gpu::ShaderStage::VertexFragment,
        });
    }

    let (mut color_attachments, mut depth_stencil_attachments) = (vec![], vec![]);

    for (_, write) in pass_info.attachment_writes.iter().enumerate() {
        let resource = graph.get_resource_info(write)?;

        match resource.ty {
            crate::AllocationType::Image(desc) | crate::AllocationType::ExternalImage(desc) => {
                let format = desc.format.to_vk();
                let samples = match desc.samples {
                    1 => SampleCountFlags::TYPE_1,
                    2 => SampleCountFlags::TYPE_2,
                    4 => SampleCountFlags::TYPE_4,
                    8 => SampleCountFlags::TYPE_8,
                    16 => SampleCountFlags::TYPE_16,
                    32 => SampleCountFlags::TYPE_32,
                    64 => SampleCountFlags::TYPE_64,
                    _ => panic!("Invalid sample count! {}", desc.samples),
                };
                match &desc.format {
                    gpu::ImageFormat::Rgba8 | gpu::ImageFormat::RgbaFloat => color_attachments
                        .push(RenderPassAttachment {
                            format,
                            samples,
                            load_op: AttachmentLoadOp::DONT_CARE,
                            store_op: AttachmentStoreOp::STORE,
                            stencil_load_op: AttachmentLoadOp::DONT_CARE,
                            stencil_store_op: AttachmentStoreOp::DONT_CARE,
                            initial_layout: ImageLayout::UNDEFINED,
                            final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            blend_state: if let Some(state) = pass_info.blend_state {
                                state
                            } else {
                                BlendState::default()
                            },
                        }),
                    gpu::ImageFormat::Depth => {
                        depth_stencil_attachments.push(DepthStencilAttachment {})
                    }
                }
            }
        }
    }

    let description = PipelineDescription {
        global_bindings: &[GlobalBinding {
            set_index: 0,
            elements: &set_zero_bindings,
        }],
        vertex_inputs: &description.vertex_inputs,
        vertex_stage: if let RenderStage::Graphics { vertex, .. } = &description.stage {
            Some(VertexStageInfo {
                entry_point: vertex.entry_point,
                module: vertex.module,
            })
        } else {
            None
        },
        fragment_stage: if let RenderStage::Graphics {
            vertex: _,
            fragment,
        } = &description.stage
        {
            Some(FragmentStageInfo {
                entry_point: fragment.entry_point,
                module: fragment.module,
                color_attachments: &color_attachments,
                depth_stencil_attachments: &depth_stencil_attachments,
            })
        } else {
            None
        },

        input_topology: description.fragment_state.input_topology,
        primitive_restart: description.fragment_state.primitive_restart,
        polygon_mode: description.fragment_state.polygon_mode,
        cull_mode: description.fragment_state.cull_mode,
        front_face: description.fragment_state.front_face,
        depth_stencil_state: description.fragment_state.depth_stencil_state,
        logic_op: description.fragment_state.logic_op,
        push_constant_ranges: &description.fragment_state.push_constant_ranges,
    };

    Ok(Pipeline::new(gpu, vk_renderpass, &description)?)
}

pub struct RenderGraph {
    passes: HashMap<RenderPassHandle, RenderPassInfo>,
    allocations: HashMap<ResourceId, ResourceInfo>,
    persistent_resources: HashSet<ResourceId>,
    resource_allocator: RefCell<DefaultResourceAllocator>,

    hasher: DefaultHasher,
    cached_graph_hash: u64,
    cached_graph: CompiledRenderGraph,

    render_pass_pipelines: HashMap<PipelineHandle, Pipeline>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompileError {
    ResourceAlreadyDefined(ResourceId, String),
    RenderPassAlreadyDefined(String),
    CyclicGraph,
    RenderPassNotFound(RenderPassHandle),
    ResourceNotFound(ResourceId),
    PipelineNotDefined(PipelineHandle),
    GraphNotCompiledYet,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Render Graph compilation error: {:?}", &self))
    }
}

impl Error for CompileError {}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourceLayout {
    #[default]
    Unknown,
    ShaderRead,
    ShaderOutput,
    AttachmentRead,
    AttachmentWrite,
    Present,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceUsage {
    input: ResourceLayout,
    output: ResourceLayout,
}

pub type GraphResult<T> = Result<T, CompileError>;
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RenderPassInfo {
    pub label: &'static str,
    pub attachment_writes: IndexSet<ResourceId>,
    pub shader_reads: IndexSet<ResourceId>,
    pub attachment_reads: IndexSet<ResourceId>,
    pub resource_usages: HashMap<ResourceId, ResourceUsage>,
    pub extents: Extent2D,
    pub blend_state: Option<BlendState>,
    pub is_external: bool,

    defined_this_frame: bool,
}

impl RenderPassInfo {
    fn uses_as_write_attachment(&self, resource: &ResourceId) -> bool {
        self.attachment_writes.contains(resource)
    }

    fn has_any_as_write_attachment<'s, R: IntoIterator<Item = &'s ResourceId>>(
        &self,
        resources: R,
    ) -> bool {
        resources
            .into_iter()
            .any(|r| self.uses_as_write_attachment(r))
    }
}

impl Hash for RenderPassInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.extents.hash(state);
        self.is_external.hash(state);
        for write in &self.attachment_writes {
            write.hash(state);
        }
        for read in &self.shader_reads {
            read.hash(state);
        }
    }
}

pub struct RenderPassBuilder<'g> {
    pass: RenderPassInfo,

    graph: &'g mut RenderGraph,
}

impl<'g> RenderPassBuilder<'g> {
    pub fn write(mut self, handle: ResourceId) -> Self {
        assert!(!self.pass.attachment_writes.contains(&handle));
        self.pass.attachment_writes.insert(handle);
        self
    }
    pub fn writes_attachments(mut self, handles: &[ResourceId]) -> Self {
        for handle in handles {
            assert!(!self.pass.attachment_writes.contains(handle));
        }

        self.pass.attachment_writes.extend(handles.into_iter());
        self
    }
    pub fn reads_attachments(mut self, handles: &[ResourceId]) -> Self {
        for handle in handles {
            assert!(!self.pass.attachment_reads.contains(handle));
        }

        self.pass.attachment_reads.extend(handles.into_iter());
        self
    }

    pub fn read(mut self, handle: ResourceId) -> Self {
        assert!(!self.pass.shader_reads.contains(&handle));
        self.pass.shader_reads.insert(handle);
        self
    }
    pub fn reads(mut self, handles: &[ResourceId]) -> Self {
        for handle in handles {
            assert!(!self.pass.attachment_writes.contains(handle));
        }
        self.pass.shader_reads.extend(handles.into_iter());
        self
    }

    pub fn with_blend_state(mut self, blend_state: BlendState) -> Self {
        self.pass.blend_state = Some(blend_state);
        self
    }

    pub fn mark_external(mut self) -> Self {
        self.pass.is_external = true;
        self
    }

    pub fn commit(self) -> RenderPassHandle {
        self.graph.commit_render_pass(self.pass)
    }
}

#[derive(Clone, Debug)]
pub enum GraphOperation {
    TransitionShaderRead(IndexSet<ResourceId>),
    TransitionAttachmentWrite(IndexSet<ResourceId>),
    TransitionAttachmentRead(IndexSet<ResourceId>),
    ExecuteRenderPass(RenderPassHandle),
}

pub struct RenderPassContext<'p, 'g> {
    pub render_graph: &'p RenderGraph,
    pub render_pass: &'p RenderPass,
    pub render_pass_command: RenderPassCommand<'p, 'g>,
    pub framebuffer: &'p GpuFramebuffer,
    pub read_descriptor_set: Option<&'p GpuDescriptorSet>,
}
pub struct EndContext<'p, 'g> {
    pub command_buffer: &'p mut CommandBuffer<'g>,
}

#[derive(Default, Clone)]
pub struct CompiledRenderGraph {
    pass_sequence: Vec<RenderPassHandle>,
    resources_used: HashSet<ResourceId>,
    graph_operations: Vec<GraphOperation>,
}

impl CompiledRenderGraph {
    fn schedule_pass(&mut self, handle: RenderPassHandle) {
        self.pass_sequence.push(handle);
    }
}

#[derive(Default)]
struct Callbacks<'g> {
    callbacks: HashMap<RenderPassHandle, Box<dyn FnMut(&Gpu, &mut RenderPassContext) + 'g>>,
    end_callback: Option<Box<dyn FnMut(&Gpu, &mut EndContext) + 'g>>,
}

impl<'g> Callbacks<'g> {
    pub fn register_callback<F: FnMut(&Gpu, &mut RenderPassContext) + 'g>(
        &mut self,
        handle: &RenderPassHandle,
        callback: F,
    ) {
        self.callbacks.insert(*handle, Box::new(callback));
    }
    pub fn register_end_callback<F: FnMut(&Gpu, &mut EndContext) + 'g>(&mut self, callback: F) {
        self.end_callback = Some(Box::new(callback));
    }
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Default::default(),
            allocations: HashMap::default(),
            persistent_resources: HashSet::default(),
            resource_allocator: RefCell::new(DefaultResourceAllocator::new()),

            hasher: DefaultHasher::default(),
            cached_graph_hash: 0,
            cached_graph: CompiledRenderGraph::default(),

            render_pass_pipelines: Default::default(),
        }
    }

    pub fn use_image(
        &mut self,
        label: &'static str,
        description: &ImageDescription,
    ) -> GraphResult<ResourceId> {
        let id = self.create_unique_id(label)?;

        let allocation = ResourceInfo {
            ty: AllocationType::Image(description.clone()),
            label,
            defined_this_frame: true,
        };
        self.allocations.insert(id, allocation);
        Ok(id)
    }

    fn create_unique_id(&mut self, label: &'static str) -> GraphResult<ResourceId> {
        let id = ResourceId::make(label);
        if self
            .allocations
            .get(&id)
            .is_some_and(|info| info.defined_this_frame)
        {
            return Err(CompileError::ResourceAlreadyDefined(id, label.to_owned()));
        }
        Ok(id)
    }

    pub fn begin_render_pass(
        &mut self,
        label: &'static str,
        extents: Extent2D,
    ) -> GraphResult<RenderPassBuilder> {
        if self.render_pass_is_defined_already(&label) {
            return Err(CompileError::RenderPassAlreadyDefined(label.to_owned()));
        }
        Ok(RenderPassBuilder {
            pass: RenderPassInfo {
                label,
                attachment_writes: Default::default(),
                shader_reads: Default::default(),
                attachment_reads: Default::default(),
                resource_usages: Default::default(),
                extents,
                is_external: false,
                blend_state: None,
                defined_this_frame: true,
            },
            graph: self,
        })
    }

    pub fn commit_render_pass(&mut self, mut pass: RenderPassInfo) -> RenderPassHandle {
        pass.hash(&mut self.hasher);
        let handle = RenderPassHandle { label: &pass.label };
        pass.defined_this_frame = true;
        self.passes.insert(handle, pass);
        handle
    }

    pub fn persist_resource(&mut self, id: &ResourceId) {
        self.persistent_resources.insert(*id);
    }

    pub fn compile(&mut self) -> GraphResult<()> {
        if self.hasher.finish() == self.cached_graph_hash {
            return Ok(());
        }
        let mut compiled = CompiledRenderGraph::default();

        self.prune_passes(&mut compiled)?;
        self.mark_resource_usages(&compiled);
        let merge_candidates = self.find_merge_candidates(&mut compiled);

        self.find_optimal_execution_order(&mut compiled, merge_candidates);

        self.cached_graph_hash = self.hasher.finish();
        self.cached_graph = compiled.clone();

        self.prepare_for_next_frame();
        Ok(())
    }

    pub fn run<R: RenderGraphRunner>(
        &self,
        mut ctx: GraphRunContext,
        runner: &mut R,
    ) -> anyhow::Result<()> {
        runner.run_graph(&mut ctx, &self, &mut self.resource_allocator.borrow_mut())
    }

    fn prune_passes(&self, compiled: &mut CompiledRenderGraph) -> GraphResult<()> {
        let mut render_passes = self.passes.clone();

        let mut working_set: HashSet<_> = self.persistent_resources.iter().cloned().collect();

        while !working_set.is_empty() {
            // Find a render pass that writes any of the attachments in the working set
            // or reads any of the attachments in the working set
            let writing_passes: Vec<_> = render_passes
                .iter()
                .filter_map(|(h, p)| {
                    if p.has_any_as_write_attachment(working_set.iter()) {
                        Some(*h)
                    } else {
                        None
                    }
                })
                .collect();

            // If there's more than one pass that writes to any of the working set, the graph
            // is not acyclic: refuse it
            if writing_passes.len() > 1 {
                return Err(CompileError::CyclicGraph);
            }

            // If we found a pass that writes to the working set
            // update the pass's reads with the working set
            if let Some(handle) = writing_passes.first() {
                let writing_pass = render_passes.remove(&handle).unwrap();

                working_set = writing_pass.shader_reads.iter().cloned().collect();
                working_set.extend(writing_pass.attachment_reads.iter());

                compiled.schedule_pass(*handle);

                // 3. Record all the resources used by the pass
                for read in writing_pass.shader_reads {
                    compiled.resources_used.insert(read);
                }
                for write in writing_pass.attachment_writes {
                    compiled.resources_used.insert(write);
                }
            } else {
                working_set.clear();
            }
        }
        compiled.pass_sequence.reverse();
        Ok(())
    }

    fn render_pass_is_defined_already(&self, label: &str) -> bool {
        self.passes
            .values()
            .find(|p| p.label == label && p.defined_this_frame)
            .is_some()
    }

    fn find_merge_candidates(&self, compiled: &mut CompiledRenderGraph) -> Vec<Vec<usize>> {
        let mut passes: Vec<_> = compiled.pass_sequence.iter().enumerate().collect();

        let mut merge_candidates = vec![];

        while let Some((pass_i, pass)) = passes.pop() {
            let matching_passes: Vec<_> = passes
                .iter()
                .enumerate()
                .filter(|(_, (_, p))| {
                    self.passes[*p]
                        .shader_reads
                        .iter()
                        .any(|read| self.passes[pass].shader_reads.contains(read))
                })
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
        for handle in compiled.pass_sequence.iter() {
            let pass = &self.passes[handle];
            compiled
                .graph_operations
                .push(GraphOperation::TransitionShaderRead(
                    pass.shader_reads.clone(),
                ));
            compiled
                .graph_operations
                .push(GraphOperation::TransitionAttachmentRead(
                    pass.attachment_reads.clone(),
                ));
            compiled
                .graph_operations
                .push(GraphOperation::TransitionAttachmentWrite(
                    pass.attachment_writes.clone(),
                ));
            compiled
                .graph_operations
                .push(GraphOperation::ExecuteRenderPass(*handle));
        }
    }

    pub fn prepare_for_next_frame(&mut self) {
        self.persistent_resources.clear();

        for pass in self.passes.values_mut() {
            pass.defined_this_frame = false;
        }

        for resource in self.allocations.values_mut() {
            resource.defined_this_frame = false;
        }
    }

    pub fn get_renderpass_info(
        &self,
        handle: &RenderPassHandle,
    ) -> Result<RenderPassInfo, CompileError> {
        self.passes
            .get(handle)
            .cloned()
            .ok_or(CompileError::RenderPassNotFound(handle.clone()))
    }

    pub fn get_resource_info(&self, resource: &ResourceId) -> GraphResult<ResourceInfo> {
        self.allocations
            .get(resource)
            .cloned()
            .ok_or(CompileError::ResourceNotFound(resource.clone()))
    }

    pub(crate) fn get_pipeline(&self, pipeline_handle: &PipelineHandle) -> GraphResult<&Pipeline> {
        self.render_pass_pipelines
            .get(pipeline_handle)
            .ok_or(CompileError::PipelineNotDefined(*pipeline_handle))
    }

    pub(crate) fn create_pipeline_for_render_pass(
        &mut self,
        gpu: &Gpu,
        pass_handle: &RenderPassHandle,
        pipeline_label: &'static str,
        pipeline_description: &RenderGraphPipelineDescription<'_>,
    ) -> anyhow::Result<PipelineHandle> {
        let handle = PipelineHandle {
            label: pipeline_label,
            owner: *pass_handle,
        };

        if !self.render_pass_pipelines.contains_key(&handle) {
            let mut allocator = self.resource_allocator.borrow_mut();
            let pass_info = &self.passes[pass_handle];
            let pass =
                allocator
                    .render_passes
                    .get_explicit(gpu, 0, &pass_info, pass_handle, self)?;
            let pipeline = create_pipeline_for_graph_renderpass(
                self,
                pass_info,
                pass,
                gpu,
                pipeline_description,
            )?;

            trace!(
                "Created new pipeline '{}' for render pass '{}'",
                pipeline_label,
                pass_handle.label
            );
            self.render_pass_pipelines.insert(handle, pipeline);
        }

        Ok(handle)
    }

    fn mark_resource_usages(&mut self, compiled: &CompiledRenderGraph) {
        self.mark_output_resource_usages(compiled);
        self.mark_input_resource_usages(compiled);
    }

    fn mark_output_resource_usages(&mut self, compiled: &CompiledRenderGraph) {
        let mut resource_usages = HashMap::new();
        for persistent in &self.persistent_resources {
            resource_usages.insert(*persistent, ResourceLayout::Present);
        }
        for pass_id in compiled.pass_sequence.iter().rev() {
            let pass_info = self.passes.get_mut(pass_id).expect("Failed to find pass");
            for write in &pass_info.attachment_writes {
                pass_info.resource_usages.entry(*write).or_default().output =
                    *resource_usages.entry(*write).or_default();
            }

            for shader_read in &pass_info.shader_reads {
                resource_usages.insert(*shader_read, ResourceLayout::ShaderRead);
            }
            for attachment_read in &pass_info.attachment_reads {
                resource_usages.insert(*attachment_read, ResourceLayout::AttachmentRead);
            }
        }
    }

    fn mark_input_resource_usages(&mut self, compiled: &CompiledRenderGraph) {
        let mut resource_usages = HashMap::new();
        for persistent in &self.persistent_resources {
            resource_usages.insert(*persistent, ResourceLayout::Present);
        }
        for pass_id in compiled.pass_sequence.iter() {
            let pass_info = self.passes.get_mut(pass_id).expect("Failed to find pass");
            for read in &pass_info.attachment_reads {
                pass_info.resource_usages.entry(*read).or_default().output =
                    *resource_usages.entry(*read).or_default();
            }
            for read in &pass_info.shader_reads {
                pass_info.resource_usages.entry(*read).or_default().output =
                    *resource_usages.entry(*read).or_default();
            }
            for write in &pass_info.attachment_writes {
                resource_usages.insert(*write, ResourceLayout::AttachmentWrite);
            }
        }
    }
}

pub struct GpuRunner {
    resource_states: HashMap<ResourceId, TransitionInfo>,
}
impl GpuRunner {
    pub fn new() -> Self {
        Self {
            resource_states: Default::default(),
        }
    }

    pub fn get_image<'r, 'e>(
        ctx: &'e GraphRunContext,
        graph: &RenderGraph,
        id: &ResourceId,
        allocator: &'r mut DefaultResourceAllocator,
    ) -> anyhow::Result<&'e GpuImage>
    where
        'r: 'e,
    {
        if ctx.external_resources.external_images.contains_key(id) {
            Ok(ctx.external_resources.external_images[id])
        } else {
            let desc = match &graph.allocations[id].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            allocator.images.get(ctx, &desc, id, &())
        }
    }

    fn transition_shader_read(
        &mut self,
        graph: &RenderGraph,
        command_buffer: &mut CommandBuffer,
        id: &ResourceId,
        image: &GpuImage,
    ) {
        let resource_info = &graph.allocations[id];
        match &resource_info.ty {
            AllocationType::Image(desc) | AllocationType::ExternalImage(desc) => {
                let access_flag = match desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                        AccessFlags::COLOR_ATTACHMENT_READ
                    }
                    ImageFormat::Depth => AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                };

                let old_layout = self.resource_states.entry(*id).or_insert(TransitionInfo {
                    layout: ImageLayout::UNDEFINED,
                    access_mask: AccessFlags::empty(),
                    stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                });
                let new_layout = TransitionInfo {
                    layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    access_mask: AccessFlags::SHADER_READ | access_flag,
                    stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                };

                let aspect_mask = match desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => ImageAspectFlags::COLOR,
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

    fn transition_attachment_read(
        &mut self,
        graph: &RenderGraph,
        command_buffer: &mut CommandBuffer,
        id: &ResourceId,
        image: &GpuImage,
    ) {
        let resource_info = &graph.allocations[id];
        match &resource_info.ty {
            AllocationType::Image(desc) | AllocationType::ExternalImage(desc) => {
                let access_flag = match desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                        AccessFlags::COLOR_ATTACHMENT_READ
                    }
                    ImageFormat::Depth => AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                };

                let old_layout = self.resource_states.entry(*id).or_insert(TransitionInfo {
                    layout: ImageLayout::UNDEFINED,
                    access_mask: AccessFlags::empty(),
                    stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                });
                let new_layout = TransitionInfo {
                    layout: match desc.format {
                        ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                            ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        }
                        ImageFormat::Depth => ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    },
                    access_mask: AccessFlags::INPUT_ATTACHMENT_READ | access_flag,
                    stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                };

                let aspect_mask = match desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => ImageAspectFlags::COLOR,
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
    fn transition_attachment_write(
        &mut self,
        graph: &RenderGraph,
        command_buffer: &mut CommandBuffer,
        id: &ResourceId,
        image: &GpuImage,
    ) {
        let resource_info = &graph.allocations[id];
        match &resource_info.ty {
            AllocationType::Image(desc) | AllocationType::ExternalImage(desc) => {
                let access_flag = match desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                        AccessFlags::COLOR_ATTACHMENT_WRITE
                    }
                    ImageFormat::Depth => AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                };

                let old_layout = self.resource_states.entry(*id).or_insert(TransitionInfo {
                    layout: ImageLayout::UNDEFINED,
                    access_mask: AccessFlags::empty(),
                    stage_mask: PipelineStageFlags::TOP_OF_PIPE,
                });
                let new_layout = TransitionInfo {
                    layout: if desc.present {
                        ImageLayout::PRESENT_SRC_KHR
                    } else {
                        match desc.format {
                            ImageFormat::Rgba8 | ImageFormat::RgbaFloat => {
                                ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                            }
                            ImageFormat::Depth => ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        }
                    },
                    access_mask: AccessFlags::SHADER_READ | access_flag,
                    stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                };

                let aspect_mask = match desc.format {
                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => ImageAspectFlags::COLOR,
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

    fn get_renderpass<'e, 'a>(
        &'a self,
        id: &RenderPassHandle,
        external_resources: &ExternalResources<'e>,
        pass_allocator: &'e RenderPassAllocator,
    ) -> &'e RenderPass
    where
        'a: 'e,
    {
        if external_resources.external_render_passes.contains_key(&id) {
            return &external_resources.external_render_passes[&id];
        }
        pass_allocator.get_unchecked(id)
    }

    fn ensure_render_pass_exists(
        &mut self,
        ctx: &GraphRunContext,
        rp: &RenderPassHandle,
        pass_allocator: &mut RenderPassAllocator,
        graph: &RenderGraph,
    ) -> Result<(), anyhow::Error> {
        {
            let pass_info = &graph.passes[rp];
            if pass_info.is_external {
                if !ctx
                    .external_resources
                    .external_render_passes
                    .contains_key(rp)
                {
                    panic!(
                        "RenderPass {} is external, but it hasn't been injected",
                        pass_info.label
                    );
                }
            } else {
                pass_allocator.ensure_resource_exists(ctx.gpu, pass_info, rp, graph)?;
            };
        }
        Ok(())
    }
}

impl RenderGraphRunner for GpuRunner {
    fn run_graph(
        &mut self,
        ctx: &mut GraphRunContext,
        graph: &RenderGraph,
        resource_allocator: &mut DefaultResourceAllocator,
    ) -> anyhow::Result<()> {
        self.resource_states.clear();
        resource_allocator.update(ctx.current_iteration);

        let mut command_buffer = CommandBuffer::new(&ctx.gpu, gpu::QueueType::Graphics)?;

        let label = command_buffer.begin_debug_region(
            &format!("Rendering frame {}", ctx.current_iteration),
            [0.0, 0.3, 0.0, 1.0],
        );
        for op in &graph.cached_graph.graph_operations {
            match op {
                GraphOperation::TransitionShaderRead(resources) => {
                    command_buffer.insert_debug_label(
                        "Transitioning resources to shader inputs",
                        [0.0, 0.3, 0.3, 1.0],
                    );
                    for id in resources {
                        let image = Self::get_image(ctx, &graph, id, resource_allocator)?;
                        self.transition_shader_read(&graph, &mut command_buffer, id, image);
                    }
                }
                GraphOperation::TransitionAttachmentRead(resources) => {
                    command_buffer.insert_debug_label(
                        "Transitioning resources to attachment inputs",
                        [0.0, 0.3, 0.3, 1.0],
                    );
                    for id in resources {
                        let image = Self::get_image(ctx, &graph, id, resource_allocator)?;
                        self.transition_attachment_read(&graph, &mut command_buffer, id, image);
                    }
                }
                GraphOperation::TransitionAttachmentWrite(resources) => {
                    command_buffer.insert_debug_label(
                        "Transitioning resources to attachment outputs",
                        [0.0, 0.3, 0.3, 1.0],
                    );

                    for id in resources {
                        let image = Self::get_image(ctx, &graph, id, resource_allocator)?;
                        self.transition_attachment_write(&graph, &mut command_buffer, id, image);
                    }
                }
                GraphOperation::ExecuteRenderPass(rp) => {
                    self.ensure_render_pass_exists(
                        ctx,
                        rp,
                        &mut resource_allocator.render_passes,
                        &graph,
                    )?;
                    let info = &graph.passes[rp];

                    ensure_graph_allocated_image_views_exist(
                        ctx,
                        info,
                        &graph,
                        resource_allocator,
                    )?;
                    ensure_graph_allocated_samplers_exists(ctx, info, &graph, resource_allocator)?;
                    let views = resolve_image_views_unchecked(
                        info,
                        &ctx.external_resources,
                        &resource_allocator.image_views,
                    );

                    let read_descriptor_set = resolve_input_descriptor_set(
                        ctx,
                        info,
                        &resource_allocator.image_views,
                        &resource_allocator.samplers,
                        &mut resource_allocator.descriptors,
                    );

                    let framebuffer_hash = compute_framebuffer_hash(info, &views);

                    let pass = self.get_renderpass(
                        rp,
                        &ctx.external_resources,
                        &resource_allocator.render_passes,
                    );
                    let framebuffer = resource_allocator.framebuffers.get(
                        ctx,
                        &info.clone(),
                        &framebuffer_hash,
                        &RenderGraphFramebufferCreateInfo {
                            render_pass: pass,
                            render_targets: &views,
                            extents: info.extents,
                        },
                    )?;

                    let cb = ctx.callbacks.callbacks.get_mut(rp);
                    let clear_color_values: Vec<_> = info
                        .attachment_writes
                        .iter()
                        .chain(info.attachment_reads.iter())
                        .map(|rd| {
                            let res_info = &graph.allocations[rd];
                            match &res_info.ty {
                                AllocationType::Image(desc)
                                | AllocationType::ExternalImage(desc) => match desc.format {
                                    ImageFormat::Rgba8 | ImageFormat::RgbaFloat => ClearValue {
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
                        render_graph: graph,
                        render_pass: &pass,
                        framebuffer,
                        render_pass_command,
                        read_descriptor_set,
                    };

                    if let Some(cb) = cb {
                        cb(&ctx.gpu, &mut context);
                    }
                    render_pass_label.end();
                }
            }
        }
        if let Some(end_cb) = &mut ctx.callbacks.end_callback {
            end_cb(
                &ctx.gpu,
                &mut EndContext {
                    command_buffer: &mut command_buffer,
                },
            );
        }
        label.end();
        command_buffer.submit(&gpu::CommandBufferSubmitInfo {
            wait_semaphores: &[&ctx.swapchain.image_available_semaphore],
            wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            signal_semaphores: &[&ctx.swapchain.render_finished_semaphore],
            fence: Some(&ctx.swapchain.in_flight_fence),
        })?;

        crate::app_state().gpu.wait_device_idle()?;
        Ok(())
    }
}

fn compute_framebuffer_hash(
    info: &RenderPassInfo,
    views: &Vec<&GpuImageView>,
) -> FramebufferHandle {
    let framebuffer_hash = hash_image_views(views);
    FramebufferHandle {
        render_pass_label: info.label,
        hash: framebuffer_hash,
    }
}

fn hash_image_views(views: &Vec<&GpuImageView>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for view in views {
        view.hash(&mut hasher);
    }

    hasher.finish()
}

fn ensure_graph_allocated_image_views_exist(
    ctx: &GraphRunContext,
    info: &RenderPassInfo,
    graph: &RenderGraph,
    resource_allocator: &mut DefaultResourceAllocator,
) -> Result<(), anyhow::Error> {
    for writes in &info.attachment_writes {
        let _ = if !ctx
            .external_resources
            .external_image_views
            .contains_key(writes)
        {
            let desc = match &graph.allocations[writes].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            let image = resource_allocator.images.get(ctx, &desc, writes, &())?;
            resource_allocator
                .image_views
                .get(ctx, &desc, writes, image)?;
        };
    }
    for res in &info.attachment_reads {
        let _ = if !ctx
            .external_resources
            .external_image_views
            .contains_key(res)
        {
            let desc = match &graph.allocations[res].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            let image = resource_allocator.images.get(ctx, &desc, res, &())?;
            resource_allocator.image_views.get(ctx, &desc, res, image)?;
        };
    }
    for writes in &info.shader_reads {
        let _ = if !ctx
            .external_resources
            .external_image_views
            .contains_key(writes)
        {
            let desc = match &graph.allocations[writes].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            let image = resource_allocator.images.get(ctx, &desc, writes, &())?;
            resource_allocator
                .image_views
                .get(ctx, &desc, writes, image)?;
        };
    }
    Ok(())
}

fn ensure_graph_allocated_samplers_exists(
    ctx: &GraphRunContext,
    info: &RenderPassInfo,
    graph: &RenderGraph,
    resource_allocator: &mut DefaultResourceAllocator,
) -> Result<(), anyhow::Error> {
    for writes in &info.shader_reads {
        let _ = if !ctx
            .external_resources
            .external_image_views
            .contains_key(writes)
        {
            let desc = match &graph.allocations[writes].ty {
                AllocationType::Image(d) | AllocationType::ExternalImage(d) => d.clone(),
            };
            resource_allocator.samplers.get(ctx, &desc, writes, &())?;
        };
    }
    Ok(())
}

fn resolve_image_views_unchecked<'e, 'a>(
    info: &RenderPassInfo,
    external_resources: &ExternalResources<'e>,
    image_views_allocator: &'a ResourceAllocator<GpuImageView, ImageDescription, ResourceId>,
) -> Vec<&'e GpuImageView>
where
    'a: 'e,
{
    let mut views = vec![];
    for writes in &info.attachment_writes {
        let view = if external_resources.external_image_views.contains_key(writes) {
            &external_resources.external_image_views[writes]
        } else {
            image_views_allocator.get_unchecked(writes)
        };
        views.push(view);
    }
    for reads in &info.attachment_reads {
        let view = if external_resources.external_image_views.contains_key(reads) {
            &external_resources.external_image_views[reads]
        } else {
            image_views_allocator.get_unchecked(reads)
        };
        views.push(view);
    }
    views
}

fn resolve_input_descriptor_set<'e, 'a, 'd>(
    ctx: &GraphRunContext,
    info: &RenderPassInfo,
    image_view_allocator: &'a ResourceAllocator<GpuImageView, ImageDescription, ResourceId>,
    sampler_allocator: &'a ResourceAllocator<GpuSampler, ImageDescription, ResourceId>,
    descriptor_view_allocator: &'a mut ResourceAllocator<GpuDescriptorSet, RenderPassInfo, u64>,
) -> Option<&'a GpuDescriptorSet> {
    let mut hasher = DefaultHasher::new();
    if info.shader_reads.is_empty() {
        return None;
    }
    let mut descriptors = vec![];
    for (idx, read) in info.shader_reads.iter().enumerate() {
        let view = if ctx
            .external_resources
            .external_image_views
            .contains_key(read)
        {
            ctx.external_resources.external_image_views[read]
        } else {
            image_view_allocator.get_unchecked(read)
        };
        view.hash(&mut hasher);
        descriptors.push(DescriptorInfo {
            binding: idx as _,
            element_type: gpu::DescriptorType::CombinedImageSampler(gpu::SamplerState {
                sampler: sampler_allocator.get_unchecked(read),
                image_view: view,
                image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }),
            binding_stage: gpu::ShaderStage::VertexFragment,
        })
    }

    let hash = hasher.finish();
    let res = descriptor_view_allocator
        .get(
            ctx,
            info,
            &hash,
            &DescriptorSetCreateInfo {
                inputs: &descriptors,
            },
        )
        .unwrap();
    Some(res)
}

#[cfg(test)]
mod test {
    use ash::vk::Extent2D;

    use crate::{CompileError, ResourceId, ResourceLayout, ResourceUsage};

    use super::{ImageDescription, RenderGraph};

    fn alloc(name: &'static str, rg: &mut RenderGraph) -> ResourceId {
        let description = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
            present: false,
        };

        rg.use_image(name, &description).unwrap()
    }
    #[test]
    pub fn prune_empty() {
        let mut render_graph = RenderGraph::new();

        let color_component = alloc("color", &mut render_graph);
        let position_component = alloc("position", &mut render_graph);
        let tangent_component = alloc("tangent", &mut render_graph);
        let normal_component = alloc("normal", &mut render_graph);

        let _ = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap()
            .write(color_component)
            .write(position_component)
            .write(tangent_component)
            .write(normal_component)
            .commit();

        render_graph.compile().unwrap();

        assert_eq!(render_graph.cached_graph.pass_sequence.len(), 0);
    }

    #[test]
    pub fn ensure_keys_are_unique() {
        let mut render_graph = RenderGraph::new();

        let color_component_1 = alloc("color1", &mut render_graph);
        let color_component_2 = {
            let description = ImageDescription {
                width: 1240,
                height: 720,
                format: gpu::ImageFormat::Rgba8,
                samples: 1,
                present: false,
            };

            render_graph.use_image("color1", &description)
        };
        let is_defined = color_component_2.is_err_and(|id| {
            id == CompileError::ResourceAlreadyDefined(color_component_1, "color1".to_owned())
        });
        assert!(is_defined)
    }
    #[test]
    pub fn ensure_passes_are_unique() {
        let mut render_graph = RenderGraph::new();
        let p1 = render_graph
            .begin_render_pass("pass", Extent2D::default())
            .unwrap();
        let _ = p1.commit();
        let p2 = render_graph.begin_render_pass("pass", Extent2D::default());
        assert!(p2.is_err_and(|e| e == CompileError::RenderPassAlreadyDefined("pass".to_owned())));

        // Defining a render pass after a compile should not panic
        render_graph.compile().unwrap();
        let p1 = render_graph.begin_render_pass("pass", Extent2D::default());
        assert!(p1.is_ok());
    }

    #[test]
    pub fn survive_1() {
        let mut render_graph = RenderGraph::new();

        let color_component = alloc("color1", &mut render_graph);
        let position_component = alloc("position", &mut render_graph);
        let tangent_component = alloc("tangent", &mut render_graph);
        let normal_component = alloc("normal", &mut render_graph);
        let output_image = alloc("output", &mut render_graph);

        let gb = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap()
            .write(color_component)
            .write(position_component)
            .write(tangent_component)
            .write(normal_component)
            .commit();

        let cm = render_graph
            .begin_render_pass("compose_gbuffer", Extent2D::default())
            .unwrap()
            .read(color_component)
            .read(position_component)
            .read(tangent_component)
            .read(normal_component)
            .write(output_image)
            .commit();

        // We need the color component: this will let the 'gbuffer' render pass live
        render_graph.persist_resource(&output_image);

        assert_eq!(render_graph.passes[&gb].label, "gbuffer");
        assert_eq!(render_graph.passes[&cm].label, "compose_gbuffer");
        render_graph.compile().unwrap();

        assert_eq!(render_graph.cached_graph.pass_sequence.len(), 2);
        assert_eq!(
            render_graph.passes[&render_graph.cached_graph.pass_sequence[0]].label,
            "gbuffer"
        );
        assert_eq!(
            render_graph.passes[&render_graph.cached_graph.pass_sequence[1]].label,
            "compose_gbuffer"
        );

        assert_eq!(render_graph.cached_graph.pass_sequence[0], gb);
        assert_eq!(render_graph.cached_graph.pass_sequence[1], cm);
    }

    #[test]
    pub fn survive_2() {
        let mut render_graph = RenderGraph::new();

        let color_component = alloc("color1", &mut render_graph);
        let position_component = alloc("position", &mut render_graph);
        let tangent_component = alloc("tangent", &mut render_graph);
        let normal_component = alloc("normal", &mut render_graph);
        let output_image = alloc("output", &mut render_graph);
        let unused = alloc("unused", &mut render_graph);

        let gb = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap()
            .write(color_component)
            .write(position_component)
            .write(tangent_component)
            .write(normal_component)
            .commit();

        let cm = render_graph
            .begin_render_pass("compose_gbuffer", Extent2D::default())
            .unwrap()
            .read(color_component)
            .read(position_component)
            .read(tangent_component)
            .read(normal_component)
            .write(output_image)
            .commit();
        // adding an empty pass that outputs to an unused buffer
        let _ = render_graph
            .begin_render_pass("unused", Extent2D::default())
            .unwrap()
            .read(color_component)
            .read(position_component)
            .read(tangent_component)
            .read(normal_component)
            .write(unused)
            .commit();

        render_graph.persist_resource(&output_image);

        assert_eq!(render_graph.passes[&gb].label, "gbuffer");
        assert_eq!(render_graph.passes[&cm].label, "compose_gbuffer");
        render_graph.compile().unwrap();

        assert_eq!(render_graph.cached_graph.pass_sequence.len(), 2);
        assert_eq!(
            render_graph.passes[&render_graph.cached_graph.pass_sequence[0]].label,
            "gbuffer"
        );
        assert_eq!(
            render_graph.passes[&render_graph.cached_graph.pass_sequence[1]].label,
            "compose_gbuffer"
        );
    }

    #[test]
    pub fn survive_3() {
        let mut render_graph = RenderGraph::new();

        let depth_component = alloc("depth", &mut render_graph);
        let color_component = alloc("color1", &mut render_graph);
        let position_component = alloc("position", &mut render_graph);
        let tangent_component = alloc("tangent", &mut render_graph);
        let normal_component = alloc("normal", &mut render_graph);
        let output_image = alloc("output", &mut render_graph);
        let unused = alloc("unused", &mut render_graph);

        let d = render_graph
            .begin_render_pass("depth", Extent2D::default())
            .unwrap()
            .writes_attachments(&[depth_component])
            .commit();

        let gb = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap()
            .writes_attachments(&[
                color_component,
                position_component,
                tangent_component,
                normal_component,
            ])
            .reads_attachments(&[depth_component])
            .commit();

        let cm = render_graph
            .begin_render_pass("compose_gbuffer", Extent2D::default())
            .unwrap()
            .read(color_component)
            .read(position_component)
            .read(tangent_component)
            .read(normal_component)
            .write(output_image)
            .commit();
        // adding an empty pass that outputs to an unused buffer
        let _ = render_graph
            .begin_render_pass("unused", Extent2D::default())
            .unwrap()
            .read(color_component)
            .read(position_component)
            .read(tangent_component)
            .read(normal_component)
            .write(unused)
            .commit();

        render_graph.persist_resource(&output_image);

        assert_eq!(render_graph.passes[&d].label, "depth");
        assert_eq!(render_graph.passes[&gb].label, "gbuffer");
        assert_eq!(render_graph.passes[&cm].label, "compose_gbuffer");
        render_graph.compile().unwrap();

        assert_eq!(
            render_graph.passes[&d].resource_usages[&depth_component].output,
            ResourceLayout::AttachmentRead
        );

        assert_eq!(
            render_graph.passes[&gb].resource_usages[&color_component].output,
            ResourceLayout::ShaderRead
        );
        assert_eq!(
            render_graph.passes[&gb].resource_usages[&position_component].output,
            ResourceLayout::ShaderRead
        );
        assert_eq!(
            render_graph.passes[&gb].resource_usages[&tangent_component].output,
            ResourceLayout::ShaderRead
        );
        assert_eq!(
            render_graph.passes[&gb].resource_usages[&normal_component].output,
            ResourceLayout::ShaderRead
        );

        assert_eq!(
            render_graph.passes[&cm].resource_usages[&output_image].output,
            ResourceLayout::Present
        );
        assert_eq!(render_graph.cached_graph.pass_sequence.len(), 3);
        assert_eq!(
            render_graph.passes[&render_graph.cached_graph.pass_sequence[0]].label,
            "depth"
        );
        assert_eq!(
            render_graph.passes[&render_graph.cached_graph.pass_sequence[1]].label,
            "gbuffer"
        );
        assert_eq!(
            render_graph.passes[&render_graph.cached_graph.pass_sequence[2]].label,
            "compose_gbuffer"
        );
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

            let _ = render_graph
                .begin_render_pass("p1", Extent2D::default())
                .unwrap()
                .writes_attachments(&[r1, r2])
                .commit();

            let _ = render_graph
                .begin_render_pass("p2", Extent2D::default())
                .unwrap()
                .reads(&[r1, r2])
                .writes_attachments(&[r3])
                .commit();

            let _ = render_graph
                .begin_render_pass("p3", Extent2D::default())
                .unwrap()
                .reads(&[r1, r2])
                .writes_attachments(&[r3])
                .commit();

            render_graph.persist_resource(&r3);

            let error = render_graph.compile();

            assert!(error.is_err_and(|e| e == CompileError::CyclicGraph));
        }
        {
            let mut render_graph = RenderGraph::new();

            let r1 = alloc("r1", &mut render_graph);
            let r2 = alloc("r2", &mut render_graph);

            let _ = render_graph
                .begin_render_pass("p1", Extent2D::default())
                .unwrap()
                .writes_attachments(&[r1, r2])
                .commit();

            let _ = render_graph
                .begin_render_pass("p2", Extent2D::default())
                .unwrap()
                .reads(&[r1, r2])
                .writes_attachments(&[r1])
                .commit();

            render_graph.persist_resource(&r1);

            let error = render_graph.compile();

            assert!(error.is_err_and(|e| e == CompileError::CyclicGraph));
        }
    }

    #[test]
    pub fn big_graph() {
        let mut render_graph = RenderGraph::new();

        let r1 = alloc("r1", &mut render_graph);
        let r2 = alloc("r2", &mut render_graph);
        let r3 = alloc("r3", &mut render_graph);
        let r4 = alloc("r4", &mut render_graph);
        let r5 = alloc("r5", &mut render_graph);
        let r6 = alloc("r6", &mut render_graph);
        let r7 = alloc("r7", &mut render_graph);
        let r8 = alloc("r8", &mut render_graph);
        let r9 = alloc("r9", &mut render_graph);
        let r10 = alloc("r10", &mut render_graph);
        let rb = alloc("rb", &mut render_graph);

        let ru1 = alloc("ru1", &mut render_graph);
        let ru2 = alloc("ru2", &mut render_graph);

        let _ = render_graph
            .begin_render_pass("p1", Extent2D::default())
            .unwrap()
            .writes_attachments(&[r1, r2, r3, r4])
            .commit();
        let _ = render_graph
            .begin_render_pass("p2", Extent2D::default())
            .unwrap()
            .reads(&[r1, r3])
            .writes_attachments(&[r5])
            .commit();

        let _p3 = render_graph
            .begin_render_pass("p3", Extent2D::default())
            .unwrap()
            .reads(&[r2, r4])
            .writes_attachments(&[r6, r7, r8])
            .commit();

        // pruned
        let _ = render_graph
            .begin_render_pass("u1", Extent2D::default())
            .unwrap()
            .reads(&[r7, r8])
            .writes_attachments(&[ru1, ru2])
            .commit();

        let _ = render_graph
            .begin_render_pass("p4", Extent2D::default())
            .unwrap()
            .reads(&[r7, r8])
            .writes_attachments(&[r9, r10])
            .commit();

        let _ = render_graph
            .begin_render_pass("pb", Extent2D::default())
            .unwrap()
            .reads(&[r9, r10])
            .writes_attachments(&[rb])
            .commit();

        render_graph.persist_resource(&rb);

        render_graph.compile().unwrap();
        for pass in &render_graph.cached_graph.pass_sequence {
            println!("{:?}", pass.label);
        }
        assert_eq!(render_graph.cached_graph.pass_sequence.len(), 4);
        assert_eq!(render_graph.cached_graph.pass_sequence[0].label, "p1");
        assert_eq!(render_graph.cached_graph.pass_sequence[1].label, "p3");
        assert_eq!(render_graph.cached_graph.pass_sequence[2].label, "p4");
        assert_eq!(render_graph.cached_graph.pass_sequence[3].label, "pb");
        assert!(render_graph
            .cached_graph
            .pass_sequence
            .iter()
            .find(|p| p.label == "u1")
            .is_none());
        assert_eq!(render_graph.cached_graph.resources_used.len(), 10);
        assert!(render_graph
            .cached_graph
            .resources_used
            .iter()
            .find(|id| id == &&ru1)
            .is_none());
        assert!(render_graph
            .cached_graph
            .resources_used
            .iter()
            .find(|id| id == &&ru2)
            .is_none());
    }
}
