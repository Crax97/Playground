use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::{
    hal::{GraphicsPipelineLayout, OwnedFragmentStageInfo, OwnedVertexStageInfo},
    util::{define_resource_resolver, Handle},
    BindingSetElementKind, BindingSetLayout, BlendFactor, BlendOp, Buffer, BufferUsageFlags,
    ColorWriteMask, CompareOp, CullMode, DepthStencilState, DepthStencilTargetInfo, Extents2D,
    Extents3D, FrontFace, GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDimension,
    ImageFormat, ImageUsageFlags, ImageView, MultisampleState, Offset2D, PolygonMode, PresentMode,
    PrimitiveTopology, Rect2D, RenderTargetInfo, SampleCount, ShaderModule, ShaderModuleLayout,
    ShaderStageFlags, Swapchain, VertexAttributeFormat, VertexInputDescription,
    VertexInputFrequency,
};

#[cfg(feature = "swapchain")]
use super::swapchain::VulkanSwapchain;
use super::{get_allocation_callbacks, VulkanHal, VulkanHalError};

pub(crate) trait ToVk {
    type Target;
    fn to_vk(self) -> Self::Target;
}

pub(crate) trait FromVk {
    type Target;
    fn from_vk(self) -> Self::Target;
}

impl ToVk for ImageFormat {
    type Target = vk::Format;

    fn to_vk(self) -> Self::Target {
        match self {
            ImageFormat::Unknown => vk::Format::UNDEFINED,
            ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        }
    }
}

impl FromVk for vk::Format {
    type Target = ImageFormat;
    fn from_vk(self) -> Self::Target {
        match self {
            vk::Format::R8G8B8A8_UNORM => ImageFormat::Rgba8,
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

    fn from_vk(self) -> Self::Target {
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

    fn from_vk(self) -> Self::Target {
        Self::Target {
            x: self.x,
            y: self.y,
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

    fn from_vk(self) -> Self::Target {
        Self::Target {
            offset: self.offset.from_vk(),
            extents: self.extent.from_vk(),
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
    fn from_vk(self) -> Self::Target {
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
    fn from_vk(self) -> Self::Target {
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

    fn from_vk(self) -> Self::Target {
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

    fn from_vk(self) -> Self::Target {
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

    fn from_vk(self) -> Self::Target {
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

    fn from_vk(self) -> Self::Target {
        match self {
            vk::PresentModeKHR::IMMEDIATE => PresentMode::Immediate,
            vk::PresentModeKHR::FIFO => PresentMode::Fifo,
            _ => unreachable!("Format not known"),
        }
    }
}

pub(super) struct VulkanImage {
    pub(super) label: Option<String>,
    pub(super) handle: vk::Image,
    pub(super) external: bool,
    pub(super) allocation: Option<Allocation>,
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
pub(super) struct SpirvShaderModule {
    pub(super) label: Option<String>,
    pub(super) layout: ShaderModuleLayout,
    pub(super) handle: vk::ShaderModule,
}

impl<'a> GraphicsPipelineDescription<'a> {
    pub(super) fn to_vk_owned(
        &self,
        binding_sets: Vec<BindingSetLayout>,
    ) -> GraphicsPipelineLayout {
        GraphicsPipelineLayout {
            label: self.label.map(ToOwned::to_owned),
            binding_sets,
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

define_resource_resolver!(
    VulkanHal,
    (VulkanImageView, |hal, view| unsafe {
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
    (VulkanSwapchain, |_, _| Ok(())) => swapchains,
    (VulkanBuffer, |hal, buffer| unsafe {

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
    (GraphicsPipelineLayout, |_, _| { todo!() }) => graphics_pipeline_infos,
    (SpirvShaderModule, |hal, module| unsafe { hal.logical_device.handle.destroy_shader_module(module.handle, get_allocation_callbacks()); Ok(()) }) => shader_modules,
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

        impl<H> ResolveVulkan<$vulkan_ty, H> for ResourceArena<VulkanHal, $object>
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
impl_util_methods!(GraphicsPipeline, GraphicsPipelineLayout);
impl_util_methods!(Swapchain, VulkanSwapchain, vk::SwapchainKHR);
