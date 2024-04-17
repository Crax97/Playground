use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::{
    util::{define_resource_resolver, Handle},
    BindingSet, Buffer, BufferUsageFlags, DepthStencilTarget, DepthStencilTargetInfo, Extents2D,
    Extents3D, GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDimension, ImageFormat,
    ImageUsageFlags, ImageView, Offset2D, PresentMode, Rect2D, RenderTargetInfo, SampleCount,
    ShaderModule, Swapchain, VertexInputDescription, VertexStageInfo,
};

#[cfg(feature = "swapchain")]
use super::swapchain::VulkanSwapchain;

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
            ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        }
    }
}

impl FromVk for vk::Format {
    type Target = ImageFormat;
    fn from_vk(self) -> Self::Target {
        match self {
            vk::Format::R8G8B8A8_UNORM => ImageFormat::Rgba8,
            _ => unreachable!("Format not known"),
        }
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
pub(super) struct VulkanShaderModule {
    pub(super) label: Option<String>,
    pub(super) handle: vk::ShaderModule,
}

#[derive(Clone)]
pub(super) struct OwnedVertexStageInfo {
    pub(super) shader: ShaderModule,
    pub(super) entry_point: String,
    pub(super) vertex_inputs: Vec<VertexInputDescription>,
}

#[derive(Clone)]
pub(super) struct OwnedFragmentStageInfo {
    pub(super) shader: ShaderModule,
    pub(super) entry_point: String,
    pub(super) render_targets: Vec<RenderTargetInfo>,
    pub(super) depth_stencil_target: Option<DepthStencilTargetInfo>,
}

#[derive(Clone)]
pub(super) struct VulkanGraphicsPipelineDescription {
    pub(super) label: Option<String>,
    pub(super) vertex_stage: OwnedVertexStageInfo,
    pub(super) fragment_stage: Option<OwnedFragmentStageInfo>,
    pub(super) binding_sets: Vec<BindingSet>,
}

impl<'a> GraphicsPipelineDescription<'a> {
    pub(super) fn to_vk_owned(&self) -> VulkanGraphicsPipelineDescription {
        VulkanGraphicsPipelineDescription {
            label: self.label.map(ToOwned::to_owned),
            vertex_stage: OwnedVertexStageInfo {
                shader: self.vertex_stage.shader.clone(),
                entry_point: self.vertex_stage.entry_point.to_owned(),
                vertex_inputs: self.vertex_stage.vertex_inputs.to_vec(),
            },
            fragment_stage: self.fragment_stage.map(|s| OwnedFragmentStageInfo {
                shader: s.shader.clone(),
                entry_point: s.entry_point.to_owned(),
                render_targets: s.render_targets.to_vec(),
                depth_stencil_target: s.depth_stencil_target.map(ToOwned::to_owned),
            }),
            binding_sets: self.binding_sets.to_vec(),
        }
    }
}

define_resource_resolver!(
    VulkanImage => images,
    VulkanImageView => image_views,
    VulkanSwapchain => swapchains,
    VulkanBuffer => buffers,
    VulkanShaderModule => shader_modules,
    VulkanGraphicsPipelineDescription => graphics_pipeline_infos,
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
    };
}

impl_util_methods!(Image, VulkanImage, vk::Image);
impl_util_methods!(ImageView, VulkanImageView, vk::ImageView);
impl_util_methods!(Buffer, VulkanBuffer, vk::Buffer);
impl_util_methods!(ShaderModule, VulkanShaderModule, vk::ShaderModule);
impl_util_methods!(GraphicsPipeline, VulkanGraphicsPipelineDescription);
impl_util_methods!(Swapchain, VulkanSwapchain, vk::SwapchainKHR);
