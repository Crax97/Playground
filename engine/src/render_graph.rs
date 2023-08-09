use std::{
    cell::RefCell,
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    error::Error,
    fmt::Debug,
    hash::{Hash, Hasher},
};

use ash::vk::{
    self, AccessFlags, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor,
    BlendOp, BorderColor, ColorComponentFlags, ComponentMapping, DependencyFlags, Extent2D, Filter,
    ImageAspectFlags, ImageSubresourceRange, ImageUsageFlags, ImageViewType, Offset2D,
    PipelineBindPoint, Rect2D, SampleCountFlags, SamplerAddressMode, SamplerCreateFlags,
    SamplerCreateInfo, SamplerMipmapMode, StructureType, SubpassDependency,
    SubpassDescriptionFlags,
};
use gpu::{
    BeginRenderPassInfo, BindingType, BlendState, BufferCreateInfo, BufferRange, BufferUsageFlags,
    ColorAttachment, ColorLoadOp, CommandBuffer, DepthAttachment, DepthLoadOp, DescriptorInfo,
    DescriptorSetInfo, FramebufferCreateInfo, Gpu, GpuBuffer, GpuDescriptorSet, GpuFramebuffer,
    GpuImage, GpuImageView, GpuSampler, GraphicsPipeline, ImageCreateInfo, ImageFormat,
    ImageMemoryBarrier, ImageViewCreateInfo, MemoryDomain, PipelineBarrierInfo, RenderPass,
    RenderPassAttachment, RenderPassCommand, RenderPassDescription, StencilAttachment,
    StencilLoadOp, SubpassDescription, ToVk, TransitionInfo,
};

use ash::vk::PushConstantRange;
use gpu::{
    BindingElement, CullMode, DepthStencilAttachment, DepthStencilState, FragmentStageInfo,
    FrontFace, GlobalBinding, GpuShaderModule, GraphicsPipelineDescription, ImageLayout, LogicOp,
    PipelineStageFlags, PolygonMode, PrimitiveTopology, VertexBindingDescription, VertexStageInfo,
};
use indexmap::IndexSet;
use log::trace;

/*
 How to add another resource to DefaultResourceAllocator?
    let ResT the resource you want to add
    1. Wrap it into a GrapResT struct, which holds:
        a. the resource (ResT)
        b some kind of description (e.g ImageView => ImageDescription)
    2. create a CreationInfo struct, which holds the description D + additional stuff to create the resource
    3. implement AsRef<D> for CreationInfo
    4. implement GraphResource for GraphResT
    5. implement CreateFrom<'a, CreationInfo> for GrapResT

    now you can add a ResourceAllocator<GraphResT, ID> to DefaultResourceAllocator

    /// Note to self: Simplify this god damn mess, there HAS to be a simpler way to just create graph allocated resources
    ///       while automating resource lifetime/etc...
*/

pub trait CreateFrom<'a, D>
where
    Self: Sized,
{
    fn create(gpu: &Gpu, desc: &'a D) -> anyhow::Result<Self>;
}

pub trait GraphResource {
    type Inner;
    type Desc;
    fn construct(inner: Self::Inner, desc: Self::Desc) -> Self
    where
        Self: Sized;
    fn matches_description(&self, new_desc: &Self::Desc) -> bool;
    fn resource(&self) -> &Self::Inner;
}

pub struct LifetimeAllocation<R: GraphResource> {
    inner: R,
    last_frame_used: u64,
}
impl<R: GraphResource> LifetimeAllocation<R> {
    fn new<'a, D>(inner: R) -> LifetimeAllocation<R>
    where
        R: CreateFrom<'a, D>,
    {
        Self {
            inner,
            last_frame_used: 0,
        }
    }
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
pub struct FramebufferHandle {
    hash: u64,
}

pub struct ResourceAllocator<R: Sized + GraphResource, ID: Hash + Eq + PartialEq + Ord + PartialOrd>
{
    resources: HashMap<ID, LifetimeAllocation<R>>,
    lifetime: u64,
}

impl<R: Sized + GraphResource, ID: Hash + Eq + PartialEq + Ord + PartialOrd + Clone>
    ResourceAllocator<R, ID>
{
    fn new(lifetime: u64) -> Self {
        Self {
            resources: HashMap::new(),
            lifetime,
        }
    }
}

impl<R: Sized + GraphResource, ID: Hash + Eq + PartialEq + Ord + PartialOrd + Clone + Debug>
    ResourceAllocator<R, ID>
{
    fn get<'a, D: AsRef<R::Desc>>(
        &mut self,
        ctx: &GraphRunContext,
        desc: &'a D,
        id: &ID,
    ) -> anyhow::Result<&R>
    where
        R: CreateFrom<'a, D>,
    {
        self.get_explicit(ctx.gpu, ctx.current_iteration, desc, id)
    }

    fn get_explicit<'a, D: AsRef<R::Desc>>(
        &mut self,
        gpu: &Gpu,
        current_iteration: u64,
        desc: &'a D,
        id: &ID,
    ) -> anyhow::Result<&R>
    where
        R: CreateFrom<'a, D>,
    {
        self.ensure_resource_exists(gpu, desc, id)?;
        self.ensure_resource_hasnt_changed(gpu, desc, id)?;
        self.update_resource_access_time(id, current_iteration);
        Ok(self.get_unchecked(id))
    }

    fn get_unchecked(&self, id: &ID) -> &R {
        assert!(
            self.resources.contains_key(id),
            "tried to get_unchecked non existent resource: {:?}",
            id
        );
        &self.resources[id].inner
    }

    fn ensure_resource_exists<'a, D: AsRef<R::Desc>>(
        &mut self,
        gpu: &Gpu,
        desc: &'a D,
        id: &ID,
    ) -> anyhow::Result<()>
    where
        R: CreateFrom<'a, D>,
    {
        if !self.resources.contains_key(id) {
            self.create_resource(gpu, desc, id)?
        }

        Ok(())
    }

    fn create_resource<'a, D: AsRef<R::Desc>>(
        &mut self,
        gpu: &Gpu,
        desc: &'a D,
        id: &ID,
    ) -> anyhow::Result<()>
    where
        R: CreateFrom<'a, D>,
    {
        let resource = R::create(gpu, desc)?;
        trace!("Created new resource: {:?}", id);
        self.resources
            .insert(id.clone(), LifetimeAllocation::new(resource));
        Ok(())
    }
    fn ensure_resource_hasnt_changed<'a, D: AsRef<R::Desc>>(
        &mut self,
        gpu: &Gpu,
        desc: &'a D,
        id: &ID,
    ) -> anyhow::Result<()>
    where
        R: CreateFrom<'a, D>,
    {
        if !&self.resources[id].inner.matches_description(desc.as_ref()) {
            self.create_resource(gpu, desc, id)?;
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

enum ExternalShaderResource<'a> {
    ImageView(&'a GpuImageView),
    Buffer(&'a GpuBuffer),
}

impl<'a> ExternalShaderResource<'a> {
    fn as_image_view(&self) -> &GpuImageView {
        match self {
            ExternalShaderResource::ImageView(v) => v,
            _ => panic!("This resource is not an image view!"),
        }
    }
    fn as_buffer(&self) -> &GpuBuffer {
        match self {
            ExternalShaderResource::Buffer(b) => b,
            _ => panic!("This resource is not a buffer!"),
        }
    }
}

#[derive(Default)]
struct ExternalResources<'a> {
    external_images: HashMap<ResourceId, &'a GpuImage>,
    external_shader_resources: HashMap<ResourceId, ExternalShaderResource<'a>>,
}

impl<'a> ExternalResources<'a> {
    fn get_shader_resource(&self, resource_id: &ResourceId) -> &ExternalShaderResource {
        self.external_shader_resources
            .get(resource_id)
            .expect("Resource not found!")
    }
}

impl<'a> ExternalResources<'a> {
    pub fn inject_external_image(
        &mut self,
        id: &ResourceId,
        image: &'a GpuImage,
        view: &'a GpuImageView,
    ) {
        self.external_images.insert(*id, image);
        self.external_shader_resources
            .insert(*id, ExternalShaderResource::ImageView(view));
    }

    pub fn inject_external_buffer(&mut self, id: &ResourceId, buffer: &'a GpuBuffer) {
        self.external_shader_resources
            .insert(*id, ExternalShaderResource::Buffer(buffer));
    }
}

pub struct GraphImage {
    image: GpuImage,
    desc: ImageDescription,
}

impl GraphResource for GraphImage {
    type Inner = GpuImage;
    type Desc = ImageDescription;

    fn construct(image: GpuImage, desc: Self::Desc) -> Self
    where
        Self: Sized,
    {
        Self { image, desc }
    }

    fn matches_description(&self, new_desc: &Self::Desc) -> bool {
        &self.desc == new_desc
    }

    fn resource(&self) -> &Self::Inner {
        &self.image
    }
}

impl<'a> CreateFrom<'a, ImageDescription> for GraphImage {
    fn create(gpu: &Gpu, desc: &'a ImageDescription) -> anyhow::Result<Self> {
        let image = gpu
            .create_image(
                &ImageCreateInfo {
                    label: None,
                    width: desc.width,
                    height: desc.height,
                    format: desc.format.to_vk(),
                    usage: desc.format.default_usage_flags()
                        | ImageUsageFlags::INPUT_ATTACHMENT
                        | ImageUsageFlags::SAMPLED,
                },
                MemoryDomain::DeviceLocal,
                None,
            )
            .expect("Failed to create image resource");
        Ok(GraphImage::construct(image, *desc))
    }
}

#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
pub struct SamplerState {
    pub compare_op: Option<gpu::CompareOp>,
}

impl AsRef<SamplerState> for SamplerState {
    fn as_ref(&self) -> &SamplerState {
        self
    }
}

pub struct GraphSampler {
    image: GpuSampler,
    desc: SamplerState,
}

impl GraphResource for GraphSampler {
    type Inner = GpuSampler;
    type Desc = SamplerState;

    fn construct(image: GpuSampler, desc: Self::Desc) -> Self
    where
        Self: Sized,
    {
        Self { image, desc }
    }

    fn matches_description(&self, new_desc: &Self::Desc) -> bool {
        &self.desc == new_desc
    }

    fn resource(&self) -> &Self::Inner {
        &self.image
    }
}

impl<'a> CreateFrom<'a, SamplerState> for GraphSampler {
    fn create(gpu: &Gpu, samp: &'a SamplerState) -> anyhow::Result<Self> {
        let sam = gpu
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
                compare_enable: if samp.compare_op.is_some() {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
                compare_op: samp.compare_op.map(|s| s.to_vk()).unwrap_or_default(),
                min_lod: 0.0,
                max_lod: 0.0,
                border_color: BorderColor::default(),
                unnormalized_coordinates: vk::FALSE,
            })
            .expect("Failed to create image resource");
        Ok(GraphSampler::construct(sam, *samp))
    }
}

pub struct GraphImageView {
    image: GpuImageView,
    desc: ImageDescription,
}

impl GraphResource for GraphImageView {
    type Inner = GpuImageView;
    type Desc = ImageDescription;

    fn construct(image: GpuImageView, desc: Self::Desc) -> Self
    where
        Self: Sized,
    {
        Self { image, desc }
    }

    fn matches_description(&self, new_desc: &Self::Desc) -> bool {
        &self.desc == new_desc
    }

    fn resource(&self) -> &Self::Inner {
        &self.image
    }
}

pub struct GraphImageViewCreateInfo<'a> {
    desc: &'a ImageDescription,
    image: &'a GpuImage,
}
impl<'a> AsRef<ImageDescription> for GraphImageViewCreateInfo<'a> {
    fn as_ref(&self) -> &ImageDescription {
        self.desc
    }
}
impl<'a> CreateFrom<'a, GraphImageViewCreateInfo<'_>> for GraphImageView {
    fn create(gpu: &Gpu, desc: &'a GraphImageViewCreateInfo) -> anyhow::Result<Self> {
        let view = gpu
            .create_image_view(&ImageViewCreateInfo {
                image: desc.image,
                view_type: ImageViewType::TYPE_2D,
                format: desc.desc.format.to_vk(),
                components: ComponentMapping::default(),
                subresource_range: ImageSubresourceRange {
                    aspect_mask: desc.desc.format.aspect_mask(),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            })
            .expect("Failed to create image resource");
        Ok(GraphImageView::construct(view, *desc.desc))
    }
}

pub struct GraphBuffer {
    inner: GpuBuffer,
    desc: BufferDescription,
}

impl AsRef<BufferDescription> for BufferDescription {
    fn as_ref(&self) -> &BufferDescription {
        self
    }
}

impl GraphResource for GraphBuffer {
    type Inner = GpuBuffer;
    type Desc = BufferDescription;

    fn construct(inner: Self::Inner, desc: Self::Desc) -> Self
    where
        Self: Sized,
    {
        Self { inner, desc }
    }

    fn matches_description(&self, new_desc: &Self::Desc) -> bool {
        self.desc == *new_desc
    }

    fn resource(&self) -> &Self::Inner {
        &self.inner
    }
}

impl<'a> CreateFrom<'a, BufferDescription> for GraphBuffer {
    fn create(gpu: &Gpu, desc: &'a BufferDescription) -> anyhow::Result<Self> {
        let buffer = gpu.create_buffer(
            &BufferCreateInfo {
                label: None,
                size: desc.length as _,
                usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::STORAGE_BUFFER,
            },
            MemoryDomain::DeviceLocal,
        )?;
        Ok(GraphBuffer::construct(buffer, *desc))
    }
}
struct RenderGraphPassCreateInfo<'a> {
    graph: &'a RenderGraph,
    pass_info: &'a RenderPassInfo,
}

impl<'a> AsRef<RenderPassInfo> for RenderGraphPassCreateInfo<'a> {
    fn as_ref(&self) -> &RenderPassInfo {
        self.pass_info
    }
}

impl AsRef<ImageDescription> for ImageDescription {
    fn as_ref(&self) -> &ImageDescription {
        self
    }
}

pub struct GraphPass {
    inner: RenderPass,
    desc: RenderPassInfo,
}

impl GraphResource for GraphPass {
    type Inner = RenderPass;
    type Desc = RenderPassInfo;

    fn construct(inner: Self::Inner, desc: Self::Desc) -> Self
    where
        Self: Sized,
    {
        Self { inner, desc }
    }

    fn matches_description(&self, new_desc: &Self::Desc) -> bool {
        self.desc == *new_desc
    }

    fn resource(&self) -> &Self::Inner {
        &self.inner
    }
}

impl<'a> CreateFrom<'a, RenderGraphPassCreateInfo<'_>> for GraphPass {
    fn create(gpu: &Gpu, create_info: &'a RenderGraphPassCreateInfo) -> anyhow::Result<Self> {
        let mut color_attachments = vec![];
        let mut depth_attachments = vec![];

        let mut all_attachments: Vec<_> = vec![];
        let mut index = 0;
        for write in &create_info.pass_info.attachment_writes {
            let image_desc = &create_info.graph.allocations[write];
            let image_desc = match &image_desc.ty {
                AllocationType::Image(image_desc) => image_desc,
                AllocationType::Buffer { .. } => {
                    panic!("A buffer cannot be treated as a render target!")
                }
            };

            let resource_usage = create_info.pass_info.resource_usage(write);

            let final_layout = match resource_usage.output {
                ResourceLayout::Unknown => ImageLayout::General,
                ResourceLayout::ShaderRead => ImageLayout::ShaderReadOnly,
                ResourceLayout::AttachmentRead => {
                    image_desc.format.preferred_attachment_read_layout()
                }
                ResourceLayout::Present => ImageLayout::PresentSrc,
                _ => unreachable!(),
            };
            let attachment = RenderPassAttachment {
                format: image_desc.format.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: match resource_usage.input {
                    ResourceLayout::Unknown => AttachmentLoadOp::DONT_CARE,

                    _ => AttachmentLoadOp::CLEAR,
                },
                store_op: match resource_usage.output {
                    ResourceLayout::Unknown => AttachmentStoreOp::DONT_CARE,
                    _ => AttachmentStoreOp::STORE,
                },
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: match resource_usage.input {
                    ResourceLayout::Unknown => ImageLayout::Undefined,
                    ResourceLayout::ShaderWrite => {
                        image_desc.format.preferred_shader_write_layout()
                    }
                    ResourceLayout::AttachmentWrite => {
                        image_desc.format.preferred_attachment_write_layout()
                    }
                    _ => unreachable!(),
                }
                .to_vk(),
                final_layout: final_layout.to_vk(),
                blend_state: if let Some(state) = create_info.pass_info.blend_state {
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

            if image_desc.format.is_color() {
                color_attachments.push(AttachmentReference {
                    attachment: index as _,
                    layout: ImageLayout::ColorAttachment.to_vk(),
                });
            } else {
                depth_attachments.push(AttachmentReference {
                    attachment: index as _,
                    layout: ImageLayout::DepthStencilAttachment.to_vk(),
                });
            }

            index += 1;
        }
        for read in &create_info.pass_info.attachment_reads {
            let image_desc = &create_info.graph.allocations[read];
            let image_desc = match &image_desc.ty {
                AllocationType::Image(image_desc) => image_desc,
                AllocationType::Buffer { .. } => {
                    panic!("A buffer cannot be treated as a render target!")
                }
            };
            let resource_usage = create_info.pass_info.resource_usage(read);
            let attachment = RenderPassAttachment {
                format: image_desc.format.to_vk(),
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::LOAD,
                store_op: AttachmentStoreOp::NONE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: match resource_usage.input {
                    ResourceLayout::Unknown => ImageLayout::Undefined,
                    ResourceLayout::ShaderWrite => {
                        image_desc.format.preferred_shader_write_layout()
                    }
                    ResourceLayout::AttachmentWrite => {
                        image_desc.format.preferred_attachment_write_layout()
                    }
                    _ => unreachable!(),
                }
                .to_vk(),
                final_layout: match resource_usage.output {
                    ResourceLayout::Unknown => ImageLayout::General,
                    ResourceLayout::ShaderRead => ImageLayout::ShaderReadOnly,
                    ResourceLayout::AttachmentRead => {
                        image_desc.format.preferred_attachment_read_layout()
                    }
                    ResourceLayout::Present => ImageLayout::PresentSrc,
                    _ => unreachable!(),
                }
                .to_vk(),
                blend_state: if let Some(state) = create_info.pass_info.blend_state {
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

            if image_desc.format.is_color() {
                color_attachments.push(AttachmentReference {
                    attachment: index as _,
                    layout: ImageLayout::ColorAttachment.to_vk(),
                });
            } else {
                depth_attachments.push(AttachmentReference {
                    attachment: index as _,
                    layout: ImageLayout::DepthStencilAttachment.to_vk(),
                });
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
                src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.to_vk(),
                dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.to_vk(),
                src_access_mask: AccessFlags::empty(),
                dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: DependencyFlags::empty(),
            }],
        };
        let pass = RenderPass::new(gpu, &description)?;

        Ok(GraphPass::construct(pass, create_info.pass_info.clone()))
    }
}

pub struct GraphFramebuffer {
    inner: GpuFramebuffer,
    desc: FramebufferHandle,
}

impl GraphResource for GraphFramebuffer {
    type Inner = GpuFramebuffer;
    type Desc = FramebufferHandle;

    fn construct(inner: Self::Inner, desc: Self::Desc) -> Self
    where
        Self: Sized,
    {
        Self { inner, desc }
    }

    fn matches_description(&self, new_desc: &Self::Desc) -> bool {
        self.desc == *new_desc
    }

    fn resource(&self) -> &Self::Inner {
        &self.inner
    }
}

struct RenderGraphFramebufferCreateInfo<'a> {
    render_pass: &'a RenderPass,
    render_targets: &'a [&'a GpuImageView],
    extents: Extent2D,
    framebuffer_hash: FramebufferHandle,
}

impl<'a> AsRef<FramebufferHandle> for RenderGraphFramebufferCreateInfo<'a> {
    fn as_ref(&self) -> &FramebufferHandle {
        &self.framebuffer_hash
    }
}

impl<'a> CreateFrom<'a, RenderGraphFramebufferCreateInfo<'a>> for GraphFramebuffer {
    fn create(gpu: &Gpu, desc: &'a RenderGraphFramebufferCreateInfo) -> anyhow::Result<Self> {
        let fb = gpu
            .create_framebuffer(&FramebufferCreateInfo {
                render_pass: desc.render_pass,
                attachments: desc.render_targets,
                width: desc.extents.width,
                height: desc.extents.height,
            })
            .expect("Failed to create framebuffer");

        Ok(GraphFramebuffer::construct(fb, desc.framebuffer_hash))
    }
}

pub struct GraphDescriptorSet {
    inner: GpuDescriptorSet,
    desc: u64,
}

impl GraphResource for GraphDescriptorSet {
    type Inner = GpuDescriptorSet;
    type Desc = u64;

    fn construct(inner: Self::Inner, desc: Self::Desc) -> Self
    where
        Self: Sized,
    {
        Self { inner, desc }
    }

    fn matches_description(&self, new_desc: &Self::Desc) -> bool {
        self.desc == *new_desc
    }

    fn resource(&self) -> &Self::Inner {
        &self.inner
    }
}

struct DescriptorSetCreateInfo<'a> {
    hash: u64,
    inputs: &'a [DescriptorInfo<'a>],
}

impl<'a> AsRef<u64> for DescriptorSetCreateInfo<'a> {
    fn as_ref(&self) -> &u64 {
        &self.hash
    }
}

impl<'a> CreateFrom<'a, DescriptorSetCreateInfo<'a>> for GraphDescriptorSet {
    fn create(gpu: &Gpu, desc: &'a DescriptorSetCreateInfo) -> anyhow::Result<Self> {
        let ds = gpu
            .create_descriptor_set(&DescriptorSetInfo {
                descriptors: desc.inputs,
            })
            .expect("Failed to create descriptor set!");
        Ok(GraphDescriptorSet::construct(ds, desc.hash))
    }
}

type ImageAllocator = ResourceAllocator<GraphImage, ResourceId>;
type ImageViewAllocator = ResourceAllocator<GraphImageView, ResourceId>;
type BufferAllocator = ResourceAllocator<GraphBuffer, ResourceId>;
type RenderPassAllocator = ResourceAllocator<GraphPass, RenderPassHandle>;
type SampleAllocator = ResourceAllocator<GraphSampler, ResourceId>;
type FramebufferAllocator = ResourceAllocator<GraphFramebuffer, FramebufferHandle>;
type DescriptorSetAllocator = ResourceAllocator<GraphDescriptorSet, u64>;
pub struct DefaultResourceAllocator {
    images: ImageAllocator,
    image_views: ImageViewAllocator,
    buffers: BufferAllocator,
    samplers: SampleAllocator,
    framebuffers: FramebufferAllocator,
    descriptors: DescriptorSetAllocator,
    render_passes: RenderPassAllocator,
}

impl Default for DefaultResourceAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultResourceAllocator {
    pub fn new() -> Self {
        Self {
            images: ResourceAllocator::new(2),
            image_views: ResourceAllocator::new(2),
            buffers: ResourceAllocator::new(2),
            framebuffers: ResourceAllocator::new(5),
            render_passes: RenderPassAllocator::new(0),
            samplers: ResourceAllocator::new(2),
            descriptors: ResourceAllocator::new(3),
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

    callbacks: Callbacks<'e>,
    external_resources: ExternalResources<'e>,
    command_buffer: &'e mut CommandBuffer<'a>,
}

impl<'a, 'e> GraphRunContext<'a, 'e> {
    pub fn new(
        gpu: &'a Gpu,
        command_buffer: &'e mut CommandBuffer<'a>,
        current_iteration: u64,
    ) -> Self {
        Self {
            gpu,
            current_iteration,
            command_buffer,
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

    pub(crate) fn inject_external_image(
        &mut self,
        handle: &ResourceId,
        image: &'e GpuImage,
        view: &'e GpuImageView,
    ) {
        self.external_resources
            .inject_external_image(handle, image, view);
    }
    pub(crate) fn injext_external_buffer(&mut self, handle: &ResourceId, buffer: &'e GpuBuffer) {
        self.external_resources
            .inject_external_buffer(handle, buffer);
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

#[derive(Copy, Clone, Eq, Ord, PartialOrd, PartialEq, Debug)]
pub struct ResourceId {
    label: &'static str,
    raw: u64,
}

impl std::hash::Hash for ResourceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
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

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub enum ClearValue {
    #[default]
    DontCare,
    Color([f32; 4]),
    Depth(f32),
    Stencil(u8),
}

impl std::hash::Hash for ClearValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state)
    }
}

impl Eq for ClearValue {}

impl ClearValue {
    pub(crate) fn color_op(&self) -> ColorLoadOp {
        match self {
            ClearValue::DontCare => ColorLoadOp::DontCare,
            ClearValue::Color(c) => ColorLoadOp::Clear(*c),
            _ => unreachable!(),
        }
    }
    pub(crate) fn depth_op(&self) -> DepthLoadOp {
        match self {
            ClearValue::DontCare => DepthLoadOp::DontCare,
            ClearValue::Depth(d) => DepthLoadOp::Clear(*d),
            _ => unreachable!(),
        }
    }
    pub(crate) fn stencil_op(&self) -> StencilLoadOp {
        match self {
            ClearValue::DontCare => StencilLoadOp::DontCare,
            ClearValue::Stencil(s) => StencilLoadOp::Clear(*s),
            _ => unreachable!(),
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
    pub clear_value: ClearValue,
    pub sampler_state: Option<SamplerState>,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub enum BufferType {
    Storage,
    Uniform,
}

impl From<BufferType> for BindingType {
    fn from(value: BufferType) -> Self {
        match value {
            BufferType::Storage => BindingType::Storage,
            BufferType::Uniform => BindingType::Uniform,
        }
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub struct BufferDescription {
    pub length: u64,
    pub ty: BufferType,
}

#[derive(Hash, Copy, Clone)]
pub enum AllocationType {
    Image(ImageDescription),
    Buffer(BufferDescription),
}

#[derive(Hash, Copy, Clone)]
pub struct ResourceInfo {
    pub label: &'static str,
    pub ty: AllocationType,
    pub external: bool,

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
            front_face: FrontFace::CounterClockWise,
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
    gpu: &Gpu,
    description: &RenderGraphPipelineDescription,
) -> anyhow::Result<GraphicsPipeline> {
    let mut set_zero_bindings = vec![];
    for (idx, read) in pass_info.shader_reads.iter().enumerate() {
        let resource = graph.get_resource_info(read)?;
        set_zero_bindings.push(BindingElement {
            binding_type: match resource.ty {
                AllocationType::Image(_) => gpu::BindingType::CombinedImageSampler,
                AllocationType::Buffer(d) => d.ty.into(),
            },
            index: idx as _,
            stage: gpu::ShaderStage::VertexFragment,
        });
    }

    let (mut color_attachments, mut depth_stencil_attachments) = (vec![], vec![]);

    for (_, write) in pass_info.attachment_writes.iter().enumerate() {
        let resource = graph.get_resource_info(write)?;

        match resource.ty {
            AllocationType::Image(desc) => {
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
                if desc.format.is_color() {
                    color_attachments.push(RenderPassAttachment {
                        format,
                        samples,
                        load_op: AttachmentLoadOp::DONT_CARE,
                        store_op: AttachmentStoreOp::STORE,
                        stencil_load_op: AttachmentLoadOp::DONT_CARE,
                        stencil_store_op: AttachmentStoreOp::DONT_CARE,
                        initial_layout: ImageLayout::Undefined.to_vk(),
                        final_layout: ImageLayout::ColorAttachment.to_vk(),
                        blend_state: if let Some(state) = pass_info.blend_state {
                            state
                        } else {
                            BlendState::default()
                        },
                    });
                } else {
                    depth_stencil_attachments.push(DepthStencilAttachment {});
                }
            }
            AllocationType::Buffer { .. } => {
                todo!("Add support for storage buffers")
            }
        }
    }

    let description = GraphicsPipelineDescription {
        global_bindings: &[GlobalBinding {
            set_index: 0,
            elements: &set_zero_bindings,
        }],
        vertex_inputs: description.vertex_inputs,
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
        push_constant_ranges: description.fragment_state.push_constant_ranges,
    };

    Ok(GraphicsPipeline::new(gpu, &description)?)
}

pub struct RenderGraph {
    passes: HashMap<RenderPassHandle, RenderPassInfo>,
    allocations: HashMap<ResourceId, ResourceInfo>,
    persistent_resources: HashSet<ResourceId>,
    resource_allocator: RefCell<DefaultResourceAllocator>,

    hasher: DefaultHasher,
    cached_graph_hash: u64,
    cached_graph: CompiledRenderGraph,

    render_pass_pipelines: HashMap<RenderPassHandle, GraphicsPipeline>,
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
    ShaderWrite,
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

#[derive(Debug, Clone, Eq)]
pub struct RenderPassInfo {
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

    fn resource_usage(&self, resource: &ResourceId) -> ResourceUsage {
        *self
            .resource_usages
            .get(resource)
            .unwrap_or(&ResourceUsage {
                input: ResourceLayout::Unknown,
                output: ResourceLayout::Unknown,
            })
    }
}

impl PartialEq for RenderPassInfo {
    fn eq(&self, other: &Self) -> bool {
        if self.extents != other.extents {
            return false;
        }
        if self.blend_state != other.blend_state {
            return false;
        }
        if self.is_external != other.is_external {
            return false;
        }

        for r in &self.attachment_writes {
            if !other.attachment_writes.contains(r) {
                return false;
            }
        }
        for r in &self.attachment_reads {
            if !other.attachment_reads.contains(r) {
                return false;
            }
        }
        for r in &self.shader_reads {
            if !other.shader_reads.contains(r) {
                return false;
            }
        }

        for (r, u) in &self.resource_usages {
            if !other.resource_usages.get(r).is_some_and(|ou| *u == *ou) {
                return false;
            }
        }

        true
    }
}

impl Hash for RenderPassInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
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
    label: &'static str,
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

        self.pass.attachment_writes.extend(handles.iter());
        self
    }
    pub fn reads_attachments(mut self, handles: &[ResourceId]) -> Self {
        for handle in handles {
            assert!(!self.pass.attachment_reads.contains(handle));
        }

        self.pass.attachment_reads.extend(handles.iter());
        self
    }

    pub fn read(mut self, handle: ResourceId) -> Self {
        assert!(!self.pass.shader_reads.contains(&handle));
        self.pass.shader_reads.insert(handle);
        self
    }
    pub fn shader_reads(mut self, handles: &[ResourceId]) -> Self {
        for handle in handles {
            assert!(!self.pass.attachment_writes.contains(handle));
        }
        self.pass.shader_reads.extend(handles.iter());
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
        self.graph.commit_render_pass(self.pass, self.label)
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
    pub render_pass_command: RenderPassCommand<'p, 'g>,
    pub read_descriptor_set: Option<&'p GpuDescriptorSet>,
    pub pipeline: Option<&'p GraphicsPipeline>,
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

type RenderCallback<'g> = Box<dyn FnMut(&Gpu, &mut RenderPassContext) + 'g>;
type EndCallback<'g> = Box<dyn FnMut(&Gpu, &mut EndContext) + 'g>;

#[derive(Default)]
struct Callbacks<'g> {
    callbacks: HashMap<RenderPassHandle, RenderCallback<'g>>,
    end_callback: Option<EndCallback<'g>>,
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

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
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
        external: bool,
    ) -> GraphResult<ResourceId> {
        let id = self.create_unique_id(label)?;

        let allocation = ResourceInfo {
            ty: AllocationType::Image(*description),
            label,
            external,
            defined_this_frame: true,
        };
        self.allocations.insert(id, allocation);
        Ok(id)
    }

    pub fn use_buffer(
        &mut self,
        label: &'static str,
        description: &BufferDescription,
        external: bool,
    ) -> GraphResult<ResourceId> {
        let id = self.create_unique_id(label)?;

        let allocation = ResourceInfo {
            ty: AllocationType::Buffer(*description),
            label,
            external,
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
        if self.render_pass_is_defined_already(label) {
            return Err(CompileError::RenderPassAlreadyDefined(label.to_owned()));
        }
        Ok(RenderPassBuilder {
            label,
            pass: RenderPassInfo {
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

    pub fn commit_render_pass(
        &mut self,
        mut pass: RenderPassInfo,
        label: &'static str,
    ) -> RenderPassHandle {
        pass.hash(&mut self.hasher);
        let handle = RenderPassHandle { label };
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
        runner.run_graph(&mut ctx, self, &mut self.resource_allocator.borrow_mut())
    }

    fn prune_passes(&self, compiled: &mut CompiledRenderGraph) -> GraphResult<()> {
        let mut render_passes = self.passes.clone();

        let mut working_set: IndexSet<_> = self.persistent_resources.iter().cloned().collect();
        let mut all_written_resources = IndexSet::new();

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

            working_set.clear();

            for writing_pass_handle in writing_passes {
                let writing_pass_info = render_passes.remove(&writing_pass_handle).unwrap();

                let pass_writes = writing_pass_info.attachment_writes;

                for pass_write in pass_writes {
                    // If a resource is written more than once, the graph is acyclic
                    if all_written_resources.contains(&pass_write) {
                        return Err(CompileError::CyclicGraph);
                    }
                    compiled.resources_used.insert(pass_write);
                    all_written_resources.insert(pass_write);
                }
                // Schedule the pass

                working_set.extend(writing_pass_info.attachment_reads.iter());
                working_set.extend(writing_pass_info.shader_reads.iter());

                compiled.schedule_pass(writing_pass_handle);

                // 3. Record all the resources used by the pass
                for read in writing_pass_info.shader_reads {
                    compiled.resources_used.insert(read);
                }
            }

            // If we found a pass that writes to the working set
            // update the pass's reads with the working set
        }
        compiled.pass_sequence.reverse();
        Ok(())
    }

    fn render_pass_is_defined_already(&self, label: &str) -> bool {
        self.passes
            .iter()
            .any(|(h, p)| h.label == label && p.defined_this_frame)
    }

    fn find_merge_candidates(&self, _compiled: &mut CompiledRenderGraph) -> Vec<Vec<usize>> {
        vec![]
        /*
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
        */
    }

    fn find_optimal_execution_order(
        &self,
        compiled: &mut CompiledRenderGraph,
        _merge_candidates: Vec<Vec<usize>>,
    ) {
        // TODO: Upgrade to merge candidates
        for handle in compiled.pass_sequence.iter() {
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
            .ok_or(CompileError::RenderPassNotFound(*handle))
    }

    pub fn get_resource_info(&self, resource: &ResourceId) -> GraphResult<ResourceInfo> {
        self.allocations
            .get(resource)
            .cloned()
            .ok_or(CompileError::ResourceNotFound(*resource))
    }

    pub(crate) fn get_pipeline(
        &self,
        pipeline_handle: &RenderPassHandle,
    ) -> Option<&GraphicsPipeline> {
        self.render_pass_pipelines.get(pipeline_handle)
    }

    pub(crate) fn define_pipeline_for_renderpass(
        &mut self,
        gpu: &Gpu,
        pass_handle: &RenderPassHandle,
        pipeline_label: &'static str,
        pipeline_description: &RenderGraphPipelineDescription<'_>,
    ) -> anyhow::Result<()> {
        if !self.render_pass_pipelines.contains_key(pass_handle) {
            let pass_info = &self.passes[pass_handle];

            let pipeline =
                create_pipeline_for_graph_renderpass(self, pass_info, gpu, pipeline_description)?;

            trace!(
                "Created new pipeline '{}' for render pass '{}'",
                pipeline_label,
                pass_handle.label
            );
            self.render_pass_pipelines.insert(*pass_handle, pipeline);
        }

        Ok(())
    }

    fn mark_resource_usages(&mut self, compiled: &CompiledRenderGraph) {
        self.mark_input_resource_usages(compiled);
        self.mark_output_resource_usages(compiled);
    }

    fn mark_input_resource_usages(&mut self, compiled: &CompiledRenderGraph) {
        let mut resource_usages = HashMap::new();
        for pass_id in compiled.pass_sequence.iter() {
            let pass_info = self.passes.get_mut(pass_id).expect("Failed to find pass");
            for write in &pass_info.attachment_writes {
                resource_usages.insert(*write, ResourceLayout::AttachmentWrite);
            }
            for read in &pass_info.attachment_reads {
                pass_info.resource_usages.entry(*read).or_default().input =
                    *resource_usages.entry(*read).or_default();
            }
            for read in &pass_info.shader_reads {
                pass_info.resource_usages.entry(*read).or_default().input =
                    *resource_usages.entry(*read).or_default();
            }
        }
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
}

pub struct GpuRunner {
    resource_states: HashMap<ResourceId, TransitionInfo>,
}

impl Default for GpuRunner {
    fn default() -> Self {
        Self::new()
    }
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
                AllocationType::Image(d) => *d,
                _ => panic!("Type is not an image!"),
            };
            Ok(allocator.images.get(ctx, &desc, id)?.resource())
        }
    }
    fn get_image_unchecked<'r, 'e>(
        external_resources: &'e ExternalResources,
        id: &ResourceId,
        allocator: &'r DefaultResourceAllocator,
    ) -> &'e GpuImage
    where
        'r: 'e,
    {
        if external_resources.external_images.contains_key(id) {
            external_resources.external_images[id]
        } else {
            allocator.images.get_unchecked(id).resource()
        }
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

        let label = ctx.command_buffer.begin_debug_region(
            &format!("Rendering frame {}", ctx.current_iteration),
            [0.0, 0.3, 0.0, 1.0],
        );
        for op in &graph.cached_graph.graph_operations {
            if let GraphOperation::ExecuteRenderPass(rp) = op {
                let info = &graph.passes[rp];
                ensure_graph_allocated_resources_exist(ctx, info, graph, resource_allocator)?;
                ensure_graph_allocated_samplers_exists(ctx, info, graph, resource_allocator)?;

                // Transition shader reads
                {
                    let mut color_transitions = vec![];
                    let mut depth_stencil_transitions = vec![];
                    for read in &info.shader_reads {
                        let info = graph.get_resource_info(read)?;
                        let image_desc = if let AllocationType::Image(d) = info.ty {
                            d
                        } else {
                            continue;
                        };
                        let old_layout =
                            *self.resource_states.entry(*read).or_insert(TransitionInfo {
                                layout: ImageLayout::Undefined,
                                access_mask: AccessFlags::empty(),
                                stage_mask: if image_desc.format.is_color() {
                                    PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                                } else {
                                    PipelineStageFlags::EARLY_FRAGMENT_TESTS
                                },
                            });

                        let new_layout = TransitionInfo {
                            layout: ImageLayout::ShaderReadOnly,
                            access_mask: AccessFlags::SHADER_READ,
                            stage_mask: PipelineStageFlags::FRAGMENT_SHADER
                                | PipelineStageFlags::VERTEX_SHADER,
                        };

                        self.resource_states.insert(*read, new_layout);

                        let image = Self::get_image_unchecked(
                            &ctx.external_resources,
                            read,
                            resource_allocator,
                        );
                        if image_desc.format.is_color() {
                            color_transitions.push(ImageMemoryBarrier {
                                src_access_mask: old_layout.access_mask,
                                dst_access_mask: new_layout.access_mask,
                                old_layout: old_layout.layout,
                                new_layout: new_layout.layout,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                image,
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                            })
                        } else {
                            depth_stencil_transitions.push(ImageMemoryBarrier {
                                src_access_mask: old_layout.access_mask,
                                dst_access_mask: new_layout.access_mask,
                                old_layout: old_layout.layout,
                                new_layout: new_layout.layout,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                image,
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: ImageAspectFlags::DEPTH,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                            })
                        }
                    }

                    if !color_transitions.is_empty() {
                        ctx.command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dst_stage_mask: PipelineStageFlags::VERTEX_SHADER
                                | PipelineStageFlags::FRAGMENT_SHADER,
                            dependency_flags: Default::default(),
                            memory_barriers: &[],
                            buffer_memory_barriers: &[],
                            image_memory_barriers: &color_transitions,
                        })
                    }
                    if !depth_stencil_transitions.is_empty() {
                        ctx.command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                            src_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                            dst_stage_mask: PipelineStageFlags::VERTEX_SHADER
                                | PipelineStageFlags::FRAGMENT_SHADER,
                            dependency_flags: Default::default(),
                            memory_barriers: &[],
                            buffer_memory_barriers: &[],
                            image_memory_barriers: &depth_stencil_transitions,
                        })
                    }
                }

                // Transition attach write
                {
                    let mut color_transitions = vec![];
                    let mut depth_stencil_transitions = vec![];
                    for read in &info.attachment_writes {
                        let info = graph.get_resource_info(read)?;
                        let image_desc = if let AllocationType::Image(d) = info.ty {
                            d
                        } else {
                            continue;
                        };
                        let old_layout =
                            *self.resource_states.entry(*read).or_insert(TransitionInfo {
                                layout: ImageLayout::Undefined,
                                access_mask: AccessFlags::empty(),
                                stage_mask: if image_desc.format.is_color() {
                                    PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                                } else {
                                    PipelineStageFlags::EARLY_FRAGMENT_TESTS
                                },
                            });

                        let new_layout = TransitionInfo {
                            layout: if image_desc.present {
                                ImageLayout::PresentSrc
                            } else if image_desc.format.is_color() {
                                ImageLayout::ColorAttachment
                            } else {
                                ImageLayout::DepthStencilAttachment
                            },
                            access_mask: if image_desc.format.is_color() {
                                AccessFlags::COLOR_ATTACHMENT_WRITE
                            } else {
                                AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                            },
                            stage_mask: if image_desc.format.is_color() {
                                PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                            } else {
                                PipelineStageFlags::LATE_FRAGMENT_TESTS
                            },
                        };
                        self.resource_states.insert(*read, new_layout);

                        let image = Self::get_image_unchecked(
                            &ctx.external_resources,
                            read,
                            resource_allocator,
                        );
                        if image_desc.format.is_color() {
                            color_transitions.push(ImageMemoryBarrier {
                                src_access_mask: old_layout.access_mask,
                                dst_access_mask: new_layout.access_mask,
                                old_layout: old_layout.layout,
                                new_layout: new_layout.layout,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                image,
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                            })
                        } else {
                            depth_stencil_transitions.push(ImageMemoryBarrier {
                                src_access_mask: old_layout.access_mask,
                                dst_access_mask: new_layout.access_mask,
                                old_layout: old_layout.layout,
                                new_layout: new_layout.layout,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                image,
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: ImageAspectFlags::DEPTH,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                            })
                        }
                    }

                    if !color_transitions.is_empty() {
                        ctx.command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dependency_flags: Default::default(),
                            memory_barriers: &[],
                            buffer_memory_barriers: &[],
                            image_memory_barriers: &color_transitions,
                        })
                    }
                    if !depth_stencil_transitions.is_empty() {
                        ctx.command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                            src_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                            dst_stage_mask: PipelineStageFlags::LATE_FRAGMENT_TESTS,
                            dependency_flags: Default::default(),
                            memory_barriers: &[],
                            buffer_memory_barriers: &[],
                            image_memory_barriers: &depth_stencil_transitions,
                        })
                    }
                }

                // Transition attach read
                {
                    let mut color_transitions = vec![];
                    let mut depth_stencil_transitions = vec![];
                    for read in &info.attachment_reads {
                        let info = graph.get_resource_info(read)?;
                        let image_desc = if let AllocationType::Image(d) = info.ty {
                            d
                        } else {
                            continue;
                        };
                        let old_layout =
                            *self.resource_states.entry(*read).or_insert(TransitionInfo {
                                layout: ImageLayout::Undefined,
                                access_mask: AccessFlags::empty(),
                                stage_mask: if image_desc.format.is_color() {
                                    PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                                } else {
                                    PipelineStageFlags::EARLY_FRAGMENT_TESTS
                                },
                            });

                        let new_layout = TransitionInfo {
                            layout: if image_desc.format.is_color() {
                                ImageLayout::ColorAttachment
                            } else {
                                ImageLayout::DepthStencilReadOnly
                            },
                            access_mask: if image_desc.format.is_color() {
                                AccessFlags::COLOR_ATTACHMENT_READ
                            } else {
                                AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            },
                            stage_mask: if image_desc.format.is_color() {
                                PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                            } else {
                                PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            },
                        };
                        self.resource_states.insert(*read, new_layout);

                        let image = Self::get_image_unchecked(
                            &ctx.external_resources,
                            read,
                            resource_allocator,
                        );
                        if image_desc.format.is_color() {
                            color_transitions.push(ImageMemoryBarrier {
                                src_access_mask: old_layout.access_mask,
                                dst_access_mask: new_layout.access_mask,
                                old_layout: old_layout.layout,
                                new_layout: new_layout.layout,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                image,
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                            })
                        } else {
                            depth_stencil_transitions.push(ImageMemoryBarrier {
                                src_access_mask: old_layout.access_mask,
                                dst_access_mask: new_layout.access_mask,
                                old_layout: old_layout.layout,
                                new_layout: new_layout.layout,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                image,
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: ImageAspectFlags::DEPTH,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                            })
                        }
                    }

                    if !color_transitions.is_empty() {
                        ctx.command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dependency_flags: Default::default(),
                            memory_barriers: &[],
                            buffer_memory_barriers: &[],
                            image_memory_barriers: &color_transitions,
                        })
                    }
                    if !depth_stencil_transitions.is_empty() {
                        ctx.command_buffer.pipeline_barrier(&PipelineBarrierInfo {
                            src_stage_mask: PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                            dst_stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                            dependency_flags: Default::default(),
                            memory_barriers: &[],
                            buffer_memory_barriers: &[],
                            image_memory_barriers: &depth_stencil_transitions,
                        })
                    }
                }
                let (color_views, depth_view, stencil_view) = resolve_render_image_views_unchecked(
                    info,
                    graph,
                    &ctx.external_resources,
                    &resource_allocator.image_views,
                );

                let read_descriptor_set = resolve_input_descriptor_set(
                    ctx,
                    graph,
                    info,
                    &resource_allocator.image_views,
                    &resource_allocator.buffers,
                    &resource_allocator.samplers,
                    &mut resource_allocator.descriptors,
                );

                let cb = ctx.callbacks.callbacks.get_mut(rp);
                let render_pass_label = ctx.command_buffer.begin_debug_region(
                    &format!("Begin Render Pass: {}", rp.label),
                    [0.3, 0.0, 0.0, 1.0],
                );

                let mut render_pass_command =
                    ctx.command_buffer.begin_render_pass(&BeginRenderPassInfo {
                        color_attachments: &color_views,
                        depth_attachment: depth_view,
                        stencil_attachment: stencil_view,
                        render_area: Rect2D {
                            offset: Offset2D::default(),
                            extent: info.extents,
                        },
                    });

                let pipeline = graph.get_pipeline(rp);
                if let Some(pipeline) = pipeline {
                    render_pass_command.bind_pipeline(pipeline);
                    if let Some(resource) = read_descriptor_set {
                        render_pass_command.bind_descriptor_sets(
                            pipeline,
                            0,
                            &[resource.resource()],
                        )
                    }
                }
                let mut context = RenderPassContext {
                    render_graph: graph,
                    render_pass_command,
                    pipeline,
                    read_descriptor_set: read_descriptor_set.map(|r| r.resource()),
                };

                if let Some(cb) = cb {
                    cb(ctx.gpu, &mut context);
                }
                render_pass_label.end();
            }
        }
        if let Some(end_cb) = &mut ctx.callbacks.end_callback {
            end_cb(
                ctx.gpu,
                &mut EndContext {
                    command_buffer: ctx.command_buffer,
                },
            );
        }
        label.end();
        Ok(())
    }
}

fn ensure_graph_allocated_resources_exist(
    ctx: &GraphRunContext,
    info: &RenderPassInfo,
    graph: &RenderGraph,
    resource_allocator: &mut DefaultResourceAllocator,
) -> Result<(), anyhow::Error> {
    for writes in &info.attachment_writes {
        if !ctx
            .external_resources
            .external_shader_resources
            .contains_key(writes)
        {
            match &graph.allocations[writes].ty {
                AllocationType::Image(d) => {
                    let image = resource_allocator.images.get(ctx, d, writes)?.resource();
                    resource_allocator.image_views.get(
                        ctx,
                        &GraphImageViewCreateInfo { desc: d, image },
                        writes,
                    )?;
                }
                AllocationType::Buffer { .. } => panic!("Cannot treat buffer as write attachment!"),
            };
        };
    }
    for res in &info.attachment_reads {
        if !ctx
            .external_resources
            .external_shader_resources
            .contains_key(res)
        {
            match &graph.allocations[res].ty {
                AllocationType::Image(d) => {
                    let image = resource_allocator.images.get(ctx, d, res)?.resource();
                    resource_allocator.image_views.get(
                        ctx,
                        &GraphImageViewCreateInfo { desc: d, image },
                        res,
                    )?;
                }
                AllocationType::Buffer { .. } => panic!("Cannot treat buffer as read attachment!"),
            };
        };
    }
    for res in &info.shader_reads {
        if !ctx
            .external_resources
            .external_shader_resources
            .contains_key(res)
        {
            match &graph.allocations[res].ty {
                AllocationType::Image(d) => {
                    let image = resource_allocator.images.get(ctx, d, res)?.resource();
                    resource_allocator.image_views.get(
                        ctx,
                        &GraphImageViewCreateInfo { desc: d, image },
                        res,
                    )?;
                }
                AllocationType::Buffer(desc) => {
                    resource_allocator
                        .buffers
                        .ensure_resource_exists(ctx.gpu, desc, res)?;
                }
            };
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
        if !ctx
            .external_resources
            .external_shader_resources
            .contains_key(writes)
        {
            let desc = match &graph.allocations[writes].ty {
                AllocationType::Image(d) => *d,
                AllocationType::Buffer { .. } => panic!("A buffer cannot have a sampler! Bug?"),
            };

            let sampler_description = resource_allocator.samplers.get(
                ctx,
                &desc.sampler_state.unwrap_or_default(),
                writes,
            )?;
        };
    }
    Ok(())
}

fn resolve_render_image_views_unchecked<'e, 'a>(
    info: &RenderPassInfo,
    graph: &RenderGraph,
    external_resources: &'e ExternalResources<'e>,
    image_views_allocator: &'a ImageViewAllocator,
) -> (
    Vec<ColorAttachment<'e>>,
    Option<DepthAttachment<'e>>,
    Option<StencilAttachment<'e>>,
)
where
    'a: 'e,
{
    let mut colors = vec![];
    let mut depth = None;
    let mut stencil = None;
    for writes in &info.attachment_writes {
        let resource_info = graph
            .get_resource_info(writes)
            .expect("Resource not found!");

        let view = if resource_info.external {
            external_resources
                .get_shader_resource(writes)
                .as_image_view()
        } else {
            image_views_allocator.get_unchecked(writes).resource()
        };

        let image_desc = if let AllocationType::Image(d) = resource_info.ty {
            d
        } else {
            continue;
        };

        if view.format().is_color() {
            colors.push(ColorAttachment {
                image_view: view,
                load_op: image_desc.clear_value.color_op(),
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::ColorAttachment,
            });
        } else if view.format().is_depth() {
            depth = Some(DepthAttachment {
                image_view: view,
                load_op: image_desc.clear_value.depth_op(),
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::DepthStencilAttachment,
            });
        } else {
            stencil = Some(StencilAttachment {
                image_view: view,
                load_op: image_desc.clear_value.stencil_op(),
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::DepthStencilAttachment,
            });
        }
    }
    for reads in &info.attachment_reads {
        let view = if external_resources
            .external_shader_resources
            .contains_key(reads)
        {
            external_resources
                .get_shader_resource(reads)
                .as_image_view()
        } else {
            image_views_allocator.get_unchecked(reads).resource()
        };

        if view.format().is_color() {
            colors.push(ColorAttachment {
                image_view: view,
                load_op: ColorLoadOp::Load,
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::ColorAttachment,
            });
        } else if view.format().is_depth() {
            depth = Some(DepthAttachment {
                image_view: view,
                load_op: DepthLoadOp::Load,
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::DepthStencilAttachment,
            });
        } else {
            stencil = Some(StencilAttachment {
                image_view: view,
                load_op: StencilLoadOp::Load,
                store_op: gpu::AttachmentStoreOp::Store,
                initial_layout: ImageLayout::DepthStencilAttachment,
            });
        }
    }
    (colors, depth, stencil)
}

fn resolve_input_descriptor_set<'a>(
    ctx: &GraphRunContext,
    graph: &RenderGraph,
    info: &RenderPassInfo,
    image_view_allocator: &'a ImageViewAllocator,
    buffer_allocator: &'a BufferAllocator,
    sampler_allocator: &'a SampleAllocator,
    descriptor_view_allocator: &'a mut DescriptorSetAllocator,
) -> Option<&'a GraphDescriptorSet> {
    let mut hasher = DefaultHasher::new();
    if info.shader_reads.is_empty() {
        return None;
    }
    let mut descriptors = vec![];
    for (idx, read) in info.shader_reads.iter().enumerate() {
        let resource_info = graph.get_resource_info(read).expect("No resource found");

        match resource_info.ty {
            AllocationType::Image(_) => {
                let view = if resource_info.external {
                    ctx.external_resources.external_shader_resources[read].as_image_view()
                } else {
                    image_view_allocator.get_unchecked(read).resource()
                };
                view.hash(&mut hasher);
                descriptors.push(DescriptorInfo {
                    binding: idx as _,
                    element_type: gpu::DescriptorType::CombinedImageSampler(gpu::SamplerState {
                        sampler: sampler_allocator.get_unchecked(read).resource(),
                        image_view: view,
                        image_layout: ImageLayout::ShaderReadOnly,
                    }),
                    binding_stage: gpu::ShaderStage::VertexFragment,
                })
            }
            AllocationType::Buffer(desc) => {
                let buffer = if resource_info.external {
                    ctx.external_resources.external_shader_resources[read].as_buffer()
                } else {
                    buffer_allocator.get_unchecked(read).resource()
                };
                buffer.hash(&mut hasher);
                let range = BufferRange {
                    handle: buffer,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                };
                descriptors.push(DescriptorInfo {
                    binding: idx as _,
                    element_type: if desc.ty == BufferType::Uniform {
                        gpu::DescriptorType::UniformBuffer(range)
                    } else {
                        gpu::DescriptorType::StorageBuffer(range)
                    },
                    binding_stage: gpu::ShaderStage::VertexFragment,
                })
            }
        }
    }

    let hash = hasher.finish();
    let res = descriptor_view_allocator
        .get(
            ctx,
            &DescriptorSetCreateInfo {
                hash,
                inputs: &descriptors,
            },
            &hash,
        )
        .unwrap();
    Some(res)
}

#[cfg(test)]
mod test {
    use ash::vk::Extent2D;

    use crate::ClearValue::{Color, DontCare};
    use crate::{CompileError, ResourceId, ResourceLayout};

    use super::{ImageDescription, RenderGraph};

    fn alloc(name: &'static str, rg: &mut RenderGraph) -> ResourceId {
        let description = ImageDescription {
            width: 1240,
            height: 720,
            format: gpu::ImageFormat::Rgba8,
            samples: 1,
            present: false,
            clear_value: Color([0.0, 0.0, 0.0, 0.0]),
            sampler_state: None,
        };

        rg.use_image(name, &description, false).unwrap()
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
                clear_value: DontCare,
                sampler_state: None,
            };

            render_graph.use_image("color1", &description, false)
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

        render_graph.compile().unwrap();

        assert_eq!(render_graph.cached_graph.pass_sequence.len(), 2);
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

        let _ = render_graph
            .begin_render_pass("gbuffer", Extent2D::default())
            .unwrap()
            .write(color_component)
            .write(position_component)
            .write(tangent_component)
            .write(normal_component)
            .commit();

        let _ = render_graph
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

        render_graph.compile().unwrap();

        assert_eq!(render_graph.cached_graph.pass_sequence.len(), 2);
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
                .shader_reads(&[r1, r2])
                .writes_attachments(&[r3])
                .commit();

            let _ = render_graph
                .begin_render_pass("p3", Extent2D::default())
                .unwrap()
                .shader_reads(&[r1, r2])
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
                .shader_reads(&[r1, r2])
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
            .shader_reads(&[r1, r3])
            .writes_attachments(&[r5])
            .commit();

        let _p3 = render_graph
            .begin_render_pass("p3", Extent2D::default())
            .unwrap()
            .shader_reads(&[r2, r4])
            .writes_attachments(&[r6, r7, r8])
            .commit();

        // pruned
        let _ = render_graph
            .begin_render_pass("u1", Extent2D::default())
            .unwrap()
            .shader_reads(&[r7, r8])
            .writes_attachments(&[ru1, ru2])
            .commit();

        let _ = render_graph
            .begin_render_pass("p4", Extent2D::default())
            .unwrap()
            .shader_reads(&[r7, r8])
            .writes_attachments(&[r9, r10])
            .commit();

        let _ = render_graph
            .begin_render_pass("pb", Extent2D::default())
            .unwrap()
            .shader_reads(&[r9, r10])
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
