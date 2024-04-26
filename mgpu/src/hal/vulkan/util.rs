use ash::vk::{self};
use gpu_allocator::vulkan::Allocation;

use crate::{
    hal::{
        GraphicsPipelineLayout, OwnedFragmentStageInfo, OwnedVertexStageInfo, ResourceAccessMode,
    },
    util::{define_resource_resolver, Handle},
    AddressMode, BindingSet, BindingSetElementKind, BindingSetLayout, BindingSetLayoutInfo,
    BlendFactor, BlendOp, BorderColor, Buffer, BufferUsageFlags, ColorWriteMask, CompareOp,
    CullMode, Extents2D, Extents3D, FilterMode, FrontFace, GraphicsPipeline,
    GraphicsPipelineDescription, Image, ImageDimension, ImageFormat, ImageSubresource,
    ImageUsageFlags, ImageView, MipmapMode, Offset2D, Offset3D, PolygonMode, PresentMode,
    PrimitiveTopology, Rect2D, SampleCount, Sampler, ShaderModule, ShaderModuleLayout,
    ShaderStageFlags, Swapchain, VertexAttributeFormat, VertexInputFrequency,
};

#[cfg(feature = "swapchain")]
use super::swapchain::VulkanSwapchain;
use super::{get_allocation_callbacks, VulkanHal, VulkanHalError};

pub(crate) trait ToVk {
    type Target;
    fn to_vk(self) -> Self::Target;
}

pub(crate) trait FromVk
where
    Self: Copy,
{
    type Target;
    fn to_mgpu(self) -> Self::Target;
}

impl ToVk for ImageFormat {
    type Target = vk::Format;

    fn to_vk(self) -> Self::Target {
        match self {
            ImageFormat::Unknown => vk::Format::UNDEFINED,
            ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            ImageFormat::Depth32 => vk::Format::D32_SFLOAT,
        }
    }
}

impl FromVk for vk::Format {
    type Target = ImageFormat;
    fn to_mgpu(self) -> Self::Target {
        match self {
            vk::Format::R8G8B8A8_UNORM => ImageFormat::Rgba8,
            vk::Format::D32_SFLOAT => ImageFormat::Depth32,
            vk::Format::UNDEFINED => ImageFormat::Unknown,
            _ => unreachable!("Format not known"),
        }
    }
}

impl ToVk for ShaderStageFlags {
    type Target = vk::ShaderStageFlags;

    fn to_vk(self) -> Self::Target {
        let mut res = Self::Target::default();

        if self.contains(Self::VERTEX) {
            res |= Self::Target::VERTEX;
        }

        if self.contains(Self::FRAGMENT) {
            res |= Self::Target::FRAGMENT;
        }

        res
    }
}

impl ToVk for Extents2D {
    type Target = vk::Extent2D;

    fn to_vk(self) -> Self::Target {
        Self::Target {
            width: self.width,
            height: self.height,
        }
    }
}
impl FromVk for vk::Extent2D {
    type Target = Extents2D;

    fn to_mgpu(self) -> Self::Target {
        Self::Target {
            width: self.width,
            height: self.height,
        }
    }
}

impl ToVk for Offset2D {
    type Target = vk::Offset2D;

    fn to_vk(self) -> Self::Target {
        Self::Target {
            x: self.x,
            y: self.y,
        }
    }
}
impl FromVk for vk::Offset2D {
    type Target = Offset2D;

    fn to_mgpu(self) -> Self::Target {
        Self::Target {
            x: self.x,
            y: self.y,
        }
    }
}

impl ToVk for Offset3D {
    type Target = vk::Offset3D;

    fn to_vk(self) -> Self::Target {
        Self::Target {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}
impl FromVk for vk::Offset3D {
    type Target = Offset3D;

    fn to_mgpu(self) -> Self::Target {
        Self::Target {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl ToVk for Rect2D {
    type Target = vk::Rect2D;

    fn to_vk(self) -> Self::Target {
        Self::Target {
            offset: self.offset.to_vk(),
            extent: self.extents.to_vk(),
        }
    }
}
impl FromVk for vk::Rect2D {
    type Target = Rect2D;

    fn to_mgpu(self) -> Self::Target {
        Self::Target {
            offset: self.offset.to_mgpu(),
            extents: self.extent.to_mgpu(),
        }
    }
}

impl ToVk for Extents3D {
    type Target = vk::Extent3D;

    fn to_vk(self) -> Self::Target {
        Self::Target {
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }
}

impl FromVk for vk::Extent3D {
    type Target = Extents3D;
    fn to_mgpu(self) -> Self::Target {
        Self::Target {
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }
}

impl ToVk for SampleCount {
    type Target = vk::SampleCountFlags;

    fn to_vk(self) -> Self::Target {
        match self {
            SampleCount::One => vk::SampleCountFlags::TYPE_1,
        }
    }
}
impl FromVk for vk::SampleCountFlags {
    type Target = SampleCount;
    fn to_mgpu(self) -> Self::Target {
        match self {
            vk::SampleCountFlags::TYPE_1 => SampleCount::One,
            _ => todo!(),
        }
    }
}

impl ToVk for ImageDimension {
    type Target = vk::ImageType;

    fn to_vk(self) -> Self::Target {
        match self {
            ImageDimension::D1 => vk::ImageType::TYPE_1D,
            ImageDimension::D2 => vk::ImageType::TYPE_2D,
            ImageDimension::D3 => vk::ImageType::TYPE_3D,
        }
    }
}
impl FromVk for vk::ImageType {
    type Target = ImageDimension;

    fn to_mgpu(self) -> Self::Target {
        match self {
            vk::ImageType::TYPE_1D => ImageDimension::D1,
            vk::ImageType::TYPE_2D => ImageDimension::D2,
            vk::ImageType::TYPE_3D => ImageDimension::D3,
            _ => unreachable!(),
        }
    }
}

impl ToVk for ImageUsageFlags {
    type Target = vk::ImageUsageFlags;

    fn to_vk(self) -> Self::Target {
        let mut flags = Self::Target::default();

        if self.contains(Self::TRANSFER_SRC) {
            flags |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
            flags |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if self.contains(Self::SAMPLED) {
            flags |= vk::ImageUsageFlags::SAMPLED;
        }
        if self.contains(Self::STORAGE) {
            flags |= vk::ImageUsageFlags::STORAGE;
        }
        if self.contains(Self::COLOR_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if self.contains(Self::DEPTH_STENCIL_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if self.contains(Self::TRANSIENT_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        }
        if self.contains(Self::INPUT_ATTACHMENT) {
            flags |= vk::ImageUsageFlags::INPUT_ATTACHMENT;
        }
        flags
    }
}

impl FromVk for vk::ImageUsageFlags {
    type Target = ImageUsageFlags;

    fn to_mgpu(self) -> Self::Target {
        let mut flags = Self::Target::default();

        if self.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
            flags |= Self::Target::TRANSFER_SRC;
        }
        if self.contains(vk::ImageUsageFlags::TRANSFER_DST) {
            flags |= Self::Target::TRANSFER_DST;
        }
        if self.contains(vk::ImageUsageFlags::SAMPLED) {
            flags |= Self::Target::SAMPLED;
        }
        if self.contains(vk::ImageUsageFlags::STORAGE) {
            flags |= Self::Target::STORAGE;
        }
        if self.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
            flags |= Self::Target::COLOR_ATTACHMENT;
        }
        if self.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            flags |= Self::Target::DEPTH_STENCIL_ATTACHMENT;
        }
        if self.contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT) {
            flags |= Self::Target::TRANSIENT_ATTACHMENT;
        }
        if self.contains(vk::ImageUsageFlags::INPUT_ATTACHMENT) {
            flags |= Self::Target::INPUT_ATTACHMENT;
        }
        flags
    }
}

impl ToVk for VertexInputFrequency {
    type Target = vk::VertexInputRate;

    fn to_vk(self) -> Self::Target {
        match self {
            VertexInputFrequency::PerVertex => Self::Target::VERTEX,
            VertexInputFrequency::PerInstance => Self::Target::INSTANCE,
        }
    }
}

impl ToVk for VertexAttributeFormat {
    type Target = vk::Format;

    fn to_vk(self) -> Self::Target {
        match self {
            VertexAttributeFormat::Int => vk::Format::R32_SINT,
            VertexAttributeFormat::Int2 => vk::Format::R32G32_SINT,
            VertexAttributeFormat::Int3 => vk::Format::R32G32B32_SINT,
            VertexAttributeFormat::Int4 => vk::Format::R32G32B32A32_SINT,
            VertexAttributeFormat::Uint => vk::Format::R32_UINT,
            VertexAttributeFormat::Uint2 => vk::Format::R32G32_UINT,
            VertexAttributeFormat::Uint3 => vk::Format::R32G32B32_UINT,
            VertexAttributeFormat::Uint4 => vk::Format::R32G32B32A32_UINT,
            VertexAttributeFormat::Float => vk::Format::R32_SFLOAT,
            VertexAttributeFormat::Float2 => vk::Format::R32G32_SFLOAT,
            VertexAttributeFormat::Float3 => vk::Format::R32G32B32_SFLOAT,
            VertexAttributeFormat::Float4 => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

impl ToVk for PolygonMode {
    type Target = vk::PolygonMode;

    fn to_vk(self) -> Self::Target {
        match self {
            PolygonMode::Filled => vk::PolygonMode::FILL,
            PolygonMode::Line(_) => vk::PolygonMode::LINE,
            PolygonMode::Point => vk::PolygonMode::POINT,
        }
    }
}

impl ToVk for PrimitiveTopology {
    type Target = vk::PrimitiveTopology;

    fn to_vk(self) -> Self::Target {
        match self {
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
            PrimitiveTopology::Line => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::Point => vk::PrimitiveTopology::POINT_LIST,
        }
    }
}

impl ToVk for CullMode {
    type Target = vk::CullModeFlags;

    fn to_vk(self) -> Self::Target {
        match self {
            CullMode::Back => vk::CullModeFlags::BACK,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::None => vk::CullModeFlags::NONE,
        }
    }
}

impl ToVk for FrontFace {
    type Target = vk::FrontFace;

    fn to_vk(self) -> Self::Target {
        match self {
            FrontFace::ClockWise => vk::FrontFace::CLOCKWISE,
            FrontFace::CounterClockWise => vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }
}

impl ToVk for CompareOp {
    type Target = vk::CompareOp;

    fn to_vk(self) -> Self::Target {
        match self {
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Always => vk::CompareOp::ALWAYS,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
        }
    }
}

impl ToVk for FilterMode {
    type Target = vk::Filter;

    fn to_vk(self) -> Self::Target {
        match self {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        }
    }
}

impl ToVk for MipmapMode {
    type Target = vk::SamplerMipmapMode;

    fn to_vk(self) -> Self::Target {
        match self {
            MipmapMode::Linear => vk::SamplerMipmapMode::LINEAR,
            MipmapMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        }
    }
}

impl ToVk for AddressMode {
    type Target = vk::SamplerAddressMode;

    fn to_vk(self) -> Self::Target {
        match self {
            AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            AddressMode::ClampToBorder(_) => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        }
    }
}

impl ToVk for BorderColor {
    type Target = vk::BorderColor;

    fn to_vk(self) -> Self::Target {
        match self {
            BorderColor::White => vk::BorderColor::INT_OPAQUE_WHITE,
            BorderColor::Black => vk::BorderColor::INT_OPAQUE_BLACK,
        }
    }
}

impl ToVk for BindingSetElementKind {
    type Target = vk::DescriptorType;

    fn to_vk(self) -> Self::Target {
        match self {
            BindingSetElementKind::Buffer { ty, .. } => match ty {
                crate::BufferType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                crate::BufferType::Storage => vk::DescriptorType::STORAGE_BUFFER,
            },
            BindingSetElementKind::Sampler { .. } => vk::DescriptorType::SAMPLER,
            BindingSetElementKind::SampledImage { .. } => vk::DescriptorType::SAMPLED_IMAGE,
            BindingSetElementKind::StorageImage { .. } => vk::DescriptorType::STORAGE_IMAGE,
            BindingSetElementKind::CombinedImageSampler { .. } => {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }
            BindingSetElementKind::Unknown => unreachable!(),
        }
    }
}

impl ToVk for BufferUsageFlags {
    type Target = vk::BufferUsageFlags;

    fn to_vk(self) -> Self::Target {
        let mut flags = Self::Target::default();

        if self.contains(Self::TRANSFER_SRC) {
            flags |= Self::Target::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
            flags |= Self::Target::TRANSFER_DST;
        }
        if self.contains(Self::UNIFORM_TEXEL_BUFFER) {
            flags |= Self::Target::UNIFORM_TEXEL_BUFFER;
        }
        if self.contains(Self::STORAGE_TEXEL_BUFFER) {
            flags |= Self::Target::STORAGE_TEXEL_BUFFER;
        }
        if self.contains(Self::UNIFORM_BUFFER) {
            flags |= Self::Target::UNIFORM_BUFFER;
        }
        if self.contains(Self::STORAGE_BUFFER) {
            flags |= Self::Target::STORAGE_BUFFER;
        }
        if self.contains(Self::INDEX_BUFFER) {
            flags |= Self::Target::INDEX_BUFFER;
        }
        if self.contains(Self::VERTEX_BUFFER) {
            flags |= Self::Target::VERTEX_BUFFER;
        }
        if self.contains(Self::INDIRECT_BUFFER) {
            flags |= Self::Target::INDIRECT_BUFFER;
        }
        flags
    }
}

impl FromVk for vk::BufferUsageFlags {
    type Target = BufferUsageFlags;

    fn to_mgpu(self) -> Self::Target {
        let mut flags = Self::Target::default();
        if self.contains(Self::TRANSFER_SRC) {
            flags |= Self::Target::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
            flags |= Self::Target::TRANSFER_DST;
        }
        if self.contains(Self::UNIFORM_TEXEL_BUFFER) {
            flags |= Self::Target::UNIFORM_TEXEL_BUFFER;
        }
        if self.contains(Self::STORAGE_TEXEL_BUFFER) {
            flags |= Self::Target::STORAGE_TEXEL_BUFFER;
        }
        if self.contains(Self::UNIFORM_BUFFER) {
            flags |= Self::Target::UNIFORM_BUFFER;
        }
        if self.contains(Self::STORAGE_BUFFER) {
            flags |= Self::Target::STORAGE_BUFFER;
        }
        if self.contains(Self::INDEX_BUFFER) {
            flags |= Self::Target::INDEX_BUFFER;
        }
        if self.contains(Self::VERTEX_BUFFER) {
            flags |= Self::Target::VERTEX_BUFFER;
        }
        if self.contains(Self::INDIRECT_BUFFER) {
            flags |= Self::Target::INDIRECT_BUFFER;
        }
        flags
    }
}

impl ToVk for BlendFactor {
    type Target = vk::BlendFactor;

    fn to_vk(self) -> Self::Target {
        match self {
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::SourceColor => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::DestColor => vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDestColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DestAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDestAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        }
    }
}

impl ToVk for BlendOp {
    type Target = vk::BlendOp;

    fn to_vk(self) -> Self::Target {
        match self {
            BlendOp::Add => vk::BlendOp::ADD,
            BlendOp::Subtract => vk::BlendOp::SUBTRACT,
            BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Min => vk::BlendOp::MIN,
            BlendOp::Max => vk::BlendOp::MAX,
        }
    }
}

impl ToVk for ColorWriteMask {
    type Target = vk::ColorComponentFlags;

    fn to_vk(self) -> Self::Target {
        let mut res = Self::Target::default();
        if self.contains(Self::R) {
            res |= vk::ColorComponentFlags::R;
        }
        if self.contains(Self::G) {
            res |= vk::ColorComponentFlags::G;
        }
        if self.contains(Self::B) {
            res |= vk::ColorComponentFlags::B
        }
        if self.contains(Self::A) {
            res |= vk::ColorComponentFlags::A
        }
        res
    }
}

#[cfg(feature = "swapchain")]
impl ToVk for PresentMode {
    type Target = vk::PresentModeKHR;

    fn to_vk(self) -> Self::Target {
        match self {
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        }
    }
}

#[cfg(feature = "swapchain")]
impl FromVk for vk::PresentModeKHR {
    type Target = PresentMode;

    fn to_mgpu(self) -> Self::Target {
        match self {
            vk::PresentModeKHR::IMMEDIATE => PresentMode::Immediate,
            vk::PresentModeKHR::FIFO => PresentMode::Fifo,
            _ => unreachable!("Format not known"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct DescriptorPoolInfo {
    pub pool: vk::DescriptorPool,
    pub allocated: usize,
    pub max: usize,
}
#[derive(Clone, Copy)]
pub struct DescriptorSetAllocation {
    pub descriptor_set: vk::DescriptorSet,
    pub layout: vk::DescriptorSetLayout,
    pub pool_index: usize,
}

#[derive(Default)]
pub struct DescriptorPoolInfos {
    pub pools: Vec<DescriptorPoolInfo>,
    pub freed: Vec<DescriptorSetAllocation>,
}

#[derive(Default, Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub(super) struct LayoutInfo {
    pub(super) image_layout: vk::ImageLayout,
    pub(super) access_mask: vk::AccessFlags2,
    pub(super) stage_mask: vk::PipelineStageFlags2,
}

pub(super) struct VulkanImage {
    pub(super) label: Option<String>,
    pub(super) handle: vk::Image,
    pub(super) external: bool,
    pub(super) allocation: Option<Allocation>,
    // layer x mips
    pub(super) subresource_layouts: Vec<Vec<LayoutInfo>>,
}

pub(super) struct VulkanBuffer {
    pub(super) label: Option<String>,
    pub(super) handle: vk::Buffer,
    pub(super) allocation: Allocation,
    pub(super) current_access_mask: vk::AccessFlags2,
    pub(super) current_stage_mask: vk::PipelineStageFlags2,
}

#[derive(Clone)]
pub(super) struct VulkanImageView {
    pub(super) label: Option<String>,
    pub(super) handle: vk::ImageView,
    pub(super) owner: vk::Image,
    /// if true then the image was created outside of the vulkan instance
    /// e.g it could be a swapchain image
    pub(super) external: bool,
}

#[derive(Clone)]
pub(super) struct VulkanSampler {
    pub(super) label: Option<String>,
    pub(super) handle: vk::Sampler,
}

#[derive(Clone)]
pub(super) struct VulkanBindingSet {
    pub(super) label: Option<String>,
    pub(super) handle: vk::DescriptorSet,
    pub(super) allocation: DescriptorSetAllocation,
    pub(super) layout: BindingSetLayout,
}

#[derive(Clone)]
pub(super) struct VulkanGraphicsPipelineInfo {
    pub(super) label: Option<String>,
    pub(super) layout: GraphicsPipelineLayout,
    pub(super) vk_layout: vk::PipelineLayout,
    pub(super) pipelines: Vec<vk::Pipeline>,
}

#[derive(Clone)]
pub(super) struct SpirvShaderModule {
    pub(super) label: Option<String>,
    pub(super) layout: ShaderModuleLayout,
    pub(super) handle: vk::ShaderModule,
}

impl VulkanImage {
    // Panics if the layouts in the subresource are different
    pub(crate) fn get_subresource_layout(&self, subresource: ImageSubresource) -> LayoutInfo {
        let layout_info = self.subresource_layouts[subresource.base_array_layer as usize]
            [subresource.mip as usize];
        for layer in subresource.base_array_layer + 1..subresource.num_layers.get() {
            for mip in subresource.mip + 1..subresource.num_mips.get() {
                let subres_layout_info = self.subresource_layouts[layer as usize][mip as usize];
                assert!(layout_info.access_mask == subres_layout_info.access_mask);
                assert!(layout_info.stage_mask == subres_layout_info.stage_mask);
                assert!(layout_info.image_layout == subres_layout_info.image_layout);
            }
        }

        layout_info
    }

    pub(crate) fn set_subresource_layout(
        &mut self,
        region: ImageSubresource,
        layout_info: LayoutInfo,
    ) {
        for layer in 0..region.num_layers.get() {
            for mip in 0..region.num_mips.get() {
                self.subresource_layouts[(region.base_array_layer + layer) as usize]
                    [(region.mip + mip) as usize] = layout_info;
            }
        }
    }
}

impl<'a> GraphicsPipelineDescription<'a> {
    pub(super) fn to_vk_owned(
        self,
        binding_sets_infos: Vec<BindingSetLayoutInfo>,
    ) -> GraphicsPipelineLayout {
        GraphicsPipelineLayout {
            label: self.label.map(ToOwned::to_owned),
            binding_sets_infos,
            vertex_stage: OwnedVertexStageInfo {
                shader: *self.vertex_stage.shader,
                entry_point: self.vertex_stage.entry_point.to_owned(),
                vertex_inputs: self.vertex_stage.vertex_inputs.to_vec(),
            },
            fragment_stage: self.fragment_stage.map(|s| OwnedFragmentStageInfo {
                shader: *s.shader,
                entry_point: s.entry_point.to_owned(),
                render_targets: s.render_targets.to_vec(),
                depth_stencil_target: s.depth_stencil_target.map(ToOwned::to_owned),
            }),
            primitive_restart_enabled: self.primitive_restart_enabled,
            primitive_topology: self.primitive_topology,
            polygon_mode: self.polygon_mode,
            cull_mode: self.cull_mode,
            front_face: self.front_face,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
        }
    }
}

impl ImageFormat {
    pub(super) fn aspect_mask(&self) -> ash::vk::ImageAspectFlags {
        match self {
            ImageFormat::Unknown => vk::ImageAspectFlags::empty(),
            ImageFormat::Rgba8 => vk::ImageAspectFlags::COLOR,
            ImageFormat::Depth32 => vk::ImageAspectFlags::DEPTH,
        }
    }
}

impl ResourceAccessMode {
    pub(super) fn image_layout(self) -> vk::ImageLayout {
        match self {
            ResourceAccessMode::Undefined => vk::ImageLayout::UNDEFINED,
            crate::hal::ResourceAccessMode::AttachmentRead(ty) => match ty {
                crate::hal::AttachmentType::Color => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                crate::hal::AttachmentType::DepthStencil => {
                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                }
            },
            crate::hal::ResourceAccessMode::AttachmentWrite(ty) => match ty {
                crate::hal::AttachmentType::Color => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                crate::hal::AttachmentType::DepthStencil => {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                }
            },
            crate::hal::ResourceAccessMode::VertexInput => unreachable!(),
            crate::hal::ResourceAccessMode::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            crate::hal::ResourceAccessMode::ShaderWrite => vk::ImageLayout::GENERAL,
            crate::hal::ResourceAccessMode::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            crate::hal::ResourceAccessMode::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        }
    }

    pub(super) fn access_mask(self) -> vk::AccessFlags2 {
        match self {
            ResourceAccessMode::Undefined => Default::default(),
            crate::hal::ResourceAccessMode::AttachmentRead(ty) => match ty {
                crate::hal::AttachmentType::Color => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                crate::hal::AttachmentType::DepthStencil => {
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ_KHR
                }
            },
            crate::hal::ResourceAccessMode::AttachmentWrite(ty) => match ty {
                crate::hal::AttachmentType::Color => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                crate::hal::AttachmentType::DepthStencil => {
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                }
            },
            crate::hal::ResourceAccessMode::VertexInput => vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            crate::hal::ResourceAccessMode::ShaderRead => vk::AccessFlags2::SHADER_READ,
            crate::hal::ResourceAccessMode::ShaderWrite => vk::AccessFlags2::SHADER_WRITE,
            crate::hal::ResourceAccessMode::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
            crate::hal::ResourceAccessMode::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
        }
    }

    pub(super) fn pipeline_flags(self) -> vk::PipelineStageFlags2 {
        match self {
            ResourceAccessMode::Undefined => Default::default(),
            crate::hal::ResourceAccessMode::AttachmentRead(ty) => match ty {
                crate::hal::AttachmentType::Color => {
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                }
                crate::hal::AttachmentType::DepthStencil => {
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                }
            },
            crate::hal::ResourceAccessMode::AttachmentWrite(ty) => match ty {
                crate::hal::AttachmentType::Color => {
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                }
                crate::hal::AttachmentType::DepthStencil => {
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                }
            },
            crate::hal::ResourceAccessMode::VertexInput => {
                vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
            }
            crate::hal::ResourceAccessMode::ShaderRead => {
                vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER
            }
            crate::hal::ResourceAccessMode::ShaderWrite => {
                vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER
            }
            crate::hal::ResourceAccessMode::TransferSrc => vk::PipelineStageFlags2::TRANSFER,
            crate::hal::ResourceAccessMode::TransferDst => vk::PipelineStageFlags2::TRANSFER,
        }
    }
}

impl ImageDimension {
    pub(super) fn image_view_type(self) -> vk::ImageViewType {
        match self {
            ImageDimension::D1 => vk::ImageViewType::TYPE_1D,
            ImageDimension::D2 => vk::ImageViewType::TYPE_2D,
            ImageDimension::D3 => vk::ImageViewType::TYPE_3D,
        }
    }
}

define_resource_resolver!(
    VulkanHal,
    (VulkanSwapchain, |hal, swapchain| {
        use crate::hal::Hal;
        for image in swapchain.data.images {
            hal.destroy_image_view(image.view)?;
            hal.destroy_image(image.image)?;
        }
        Ok(())
    }) => swapchains,
    (VulkanImageView, |hal, view| {
        if view.external {
            return Ok(());
        }
        unsafe {
            hal.logical_device
                .handle
                .destroy_image_view(view.handle, get_allocation_callbacks());
        }
        Ok(())
    }) => image_views,
    (VulkanImage, |hal, image| unsafe {
        if image.external {
            return Ok(());
        }
        if let Some(allocation) = image.allocation {
            let mut allocator = hal
                .memory_allocator
                .write()
                .expect("Failed to lock memory allocator");
            allocator
                .free(allocation)
                .map_err(|e| MgpuError::VulkanError(VulkanHalError::GpuAllocatorError(e)))?;
        }
        hal.logical_device.handle.destroy_image(image.handle, get_allocation_callbacks()); Ok(()) } ) => images,
    (VulkanBuffer, |hal, buffer| {

        let mut allocator = hal
            .memory_allocator
            .write()
            .expect("Failed to lock memory allocator");
        allocator
            .free(buffer.allocation)
            .map_err(|e| MgpuError::VulkanError(VulkanHalError::GpuAllocatorError(e)))?;

        unsafe {
            hal.logical_device
                .handle
                .destroy_buffer(buffer.handle, get_allocation_callbacks());
        }
        Ok(())
    }) => buffers,
    (VulkanGraphicsPipelineInfo, |hal, pipeline| {
          for pipeline in pipeline.pipelines {
            unsafe {
                hal.logical_device.handle.destroy_pipeline(pipeline, get_allocation_callbacks());
            }
          }
          Ok(())
        }) => graphics_pipeline_infos,
    (SpirvShaderModule, |hal, module| unsafe { hal.logical_device.handle.destroy_shader_module(module.handle, get_allocation_callbacks()); Ok(()) }) => shader_modules,
    (VulkanSampler, |hal, sampler| unsafe { hal.logical_device.handle.destroy_sampler(sampler.handle, get_allocation_callbacks()); Ok(())}) => samplers,
    (VulkanBindingSet,  |hal, bs| {
        let mut infos = hal.descriptor_pool_infos.lock().unwrap();
        let ds_pool_info = infos.get_mut(&bs.allocation.layout).unwrap();
        ds_pool_info.freed.push(bs.allocation);
        ds_pool_info.pools[bs.allocation.pool_index].allocated -= 1;
        Ok(())
    }) => binding_sets,
);

pub(super) trait ResolveVulkan<T, H>
where
    Self: Sized,
{
    fn resolve_vulkan(&self, handle: H) -> Option<T>;
}

pub(in crate::hal::vulkan) type VulkanResolver = ResourceResolver;

macro_rules! impl_util_methods {
    ($handle:ty, $object:ty) => {
        impl From<$handle> for Handle<$object> {
            fn from(handle: $handle) -> Handle<$object> {
                unsafe { Handle::from_u64(handle.id) }
            }
        }
        impl From<&$handle> for Handle<$object> {
            fn from(handle: &$handle) -> Handle<$object> {
                unsafe { Handle::from_u64(handle.id) }
            }
        }

        impl crate::util::HasLabel for $object {
            fn label(&self) -> Option<&str> {
                self.label.as_deref()
            }
        }
    };
    ($handle:ty, $object:ty, $vulkan_ty:ty) => {
        impl_util_methods!($handle, $object);

        impl<H> ResolveVulkan<$vulkan_ty, H> for VulkanResolver
        where
            H: Into<Handle<$object>>,
        {
            fn resolve_vulkan(&self, handle: H) -> Option<$vulkan_ty> {
                self.get::<$object>()
                    .resolve(handle.into())
                    .map(|v| v.handle)
            }
        }

        impl<H: crate::util::HasLabel> ResolveVulkan<$vulkan_ty, H>
            for ResourceArena<VulkanHal, $object>
        where
            H: Into<Handle<$object>>,
        {
            fn resolve_vulkan(&self, handle: H) -> Option<$vulkan_ty> {
                self.resolve(handle.into()).map(|v| v.handle)
            }
        }
    };
}

impl_util_methods!(Image, VulkanImage, vk::Image);
impl_util_methods!(ImageView, VulkanImageView, vk::ImageView);
impl_util_methods!(Buffer, VulkanBuffer, vk::Buffer);
impl_util_methods!(ShaderModule, SpirvShaderModule, vk::ShaderModule);
impl_util_methods!(Sampler, VulkanSampler, vk::Sampler);
impl_util_methods!(BindingSet, VulkanBindingSet, vk::DescriptorSet);
impl_util_methods!(GraphicsPipeline, VulkanGraphicsPipelineInfo);
impl_util_methods!(Swapchain, VulkanSwapchain, vk::SwapchainKHR);
