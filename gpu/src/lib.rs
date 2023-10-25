mod allocator;
mod command_buffer;
mod descriptor_set;
mod gpu;
mod gpu_resource_manager;
mod handle;
mod swapchain;
mod types;

pub use crate::gpu::*;
pub use allocator::*;
pub use bitflags::bitflags;
pub use command_buffer::*;
pub use gpu_resource_manager::*;
pub use handle::*;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
pub use swapchain::VkSwapchain;
pub use types::*;
use winit::window::Window;

pub const WHOLE_SIZE: u64 = u64::MAX;
pub const QUEUE_FAMILY_IGNORED: u32 = u32::MAX;

pub struct GpuConfiguration<'a> {
    pub app_name: &'a str,
    pub pipeline_cache_path: Option<&'a str>,
    pub enable_debug_utilities: bool,
    pub window: Option<&'a Window>,
}

pub trait Gpu {
    fn update(&self);

    fn make_buffer(
        &self,
        buffer_info: &BufferCreateInfo,
        memory_domain: MemoryDomain,
    ) -> anyhow::Result<BufferHandle>;
    fn write_buffer(&self, buffer: &BufferHandle, offset: u64, data: &[u8]) -> anyhow::Result<()>;

    fn make_image(
        &self,
        info: &ImageCreateInfo,
        memory_domain: MemoryDomain,
        data: Option<&[u8]>,
    ) -> anyhow::Result<ImageHandle>;
    fn write_image(
        &self,
        handle: &ImageHandle,
        data: &[u8],
        region: Rect2D,
        layer: u32,
    ) -> anyhow::Result<()>;

    fn make_image_view(&self, info: &ImageViewCreateInfo) -> anyhow::Result<ImageViewHandle>;

    fn make_sampler(&self, info: &SamplerCreateInfo) -> anyhow::Result<SamplerHandle>;

    fn make_shader_module(
        &self,
        info: &ShaderModuleCreateInfo,
    ) -> anyhow::Result<ShaderModuleHandle>;
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct FenceCreateFlags : u32 {
        const SIGNALED = 0b00000001;
    }
}

#[derive(Clone, Copy, Hash, Eq, Ord, PartialOrd, PartialEq, Debug, Default)]
pub enum ImageLayout {
    #[default]
    Undefined,
    General,
    ColorAttachment,
    DepthStencilAttachment,
    DepthStencilReadOnly,
    ShaderReadOnly,
    TransferSrc,
    TransferDst,
    PreinitializedByCpu,

    PresentSrc,
}

#[derive(Clone, Copy, Hash, Eq, Ord, PartialOrd, PartialEq, Debug)]
pub struct PipelineColorBlendAttachmentState {
    pub blend_enable: bool,
    pub src_color_blend_factor: BlendMode,
    pub dst_color_blend_factor: BlendMode,
    pub color_blend_op: BlendOp,
    pub src_alpha_blend_factor: BlendMode,
    pub dst_alpha_blend_factor: BlendMode,
    pub alpha_blend_op: BlendOp,
    pub color_write_mask: ColorComponentFlags,
}

#[derive(Clone, Hash, Eq, Ord, PartialOrd, PartialEq, Debug)]
pub struct VertexBindingInfo {
    pub handle: BufferHandle,
    pub location: u32,
    pub offset: u32,
    pub stride: u32,
    pub format: ImageFormat,
    pub input_rate: InputRate,
}
#[derive(Clone, Copy, Hash, Eq, Ord, PartialOrd, PartialEq, Debug)]
pub struct ImageCreateInfo<'a> {
    pub label: Option<&'a str>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mips: u32,
    pub layers: u32,
    pub samples: SampleCount,
    pub format: ImageFormat,
    pub usage: ImageUsageFlags,
}

pub struct ImageViewCreateInfo {
    pub image: ImageHandle,
    pub view_type: ImageViewType,
    pub format: ImageFormat,
    pub components: ComponentMapping,
    pub subresource_range: ImageSubresourceRange,
}

pub struct BufferCreateInfo<'a> {
    pub label: Option<&'a str>,
    pub size: usize,
    pub usage: BufferUsageFlags,
}

#[derive(Clone, Copy)]
pub struct TransitionInfo {
    pub layout: ImageLayout,
    pub access_mask: AccessFlags,
    pub stage_mask: PipelineStageFlags,
}

#[derive(Clone, Hash)]
pub struct BufferImageCopyInfo {
    pub source: BufferHandle,
    pub dest: ImageHandle,
    pub dest_layout: ImageLayout,
    pub image_offset: Offset3D,
    pub image_extent: Extent3D,
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub mip_level: u32,
    pub base_layer: u32,
    pub num_layers: u32,
}

pub struct ShaderModuleCreateInfo<'a> {
    pub code: &'a [u8],
}

bitflags! {
    #[repr(transparent)]
    #[derive(Default, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
    pub struct PipelineStageFlags : u32 {
        #[doc = "Before subsequent commands are processed"]
        const TOP_OF_PIPE = 0b1;
        #[doc = "Draw/DispatchIndirect command fetch"]
        const DRAW_INDIRECT = 0b10;
        #[doc = "Vertex/index fetch"]
        const VERTEX_INPUT = 0b100;
        #[doc = "Vertex shading"]
        const VERTEX_SHADER = 0b1000;
        #[doc = "Tessellation control shading"]
        const TESSELLATION_CONTROL_SHADER = 0b1_0000;
        #[doc = "Tessellation evaluation shading"]
        const TESSELLATION_EVALUATION_SHADER = 0b10_0000;
        #[doc = "Geometry shading"]
        const GEOMETRY_SHADER = 0b100_0000;
        #[doc = "Fragment shading"]
        const FRAGMENT_SHADER = 0b1000_0000;
        #[doc = "Early fragment (depth and stencil) tests"]
        const EARLY_FRAGMENT_TESTS = 0b1_0000_0000;
        #[doc = "Late fragment (depth and stencil) tests"]
        const LATE_FRAGMENT_TESTS = 0b10_0000_0000;
        #[doc = "Color attachment writes"]
        const COLOR_ATTACHMENT_OUTPUT = 0b100_0000_0000;
        #[doc = "Compute shading"]
        const COMPUTE_SHADER = 0b1000_0000_0000;
        #[doc = "Transfer/copy operations"]
        const TRANSFER = 0b1_0000_0000_0000;
        #[doc = "After previous commands have completed"]
        const BOTTOM_OF_PIPE = 0b10_0000_0000_0000;
        #[doc = "Indicates host (CPU) is a source/sink of the dependency"]
        const HOST = 0b100_0000_0000_0000;
        #[doc = "All stages of the graphics pipeline"]
        const ALL_GRAPHICS = 0b1000_0000_0000_0000;
        #[doc = "All stages supported on the queue"]
        const ALL_COMMANDS = 0b1_0000_0000_0000_0000;
    }
}
bitflags! {
    #[repr(transparent)]
    #[derive(Default, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
    pub struct AccessFlags : u64 {
        const NONE = 0;
        const INDIRECT_COMMAND_READ = 0x00000001;
        const INDEX_READ = 0x00000002;
        const VERTEX_ATTRIBUTE_READ = 0x00000004;
        const UNIFORM_READ = 0x00000008;
        const INPUT_ATTACHMENT_READ = 0x00000010;
        const SHADER_READ = 0x00000020;
        const SHADER_WRITE = 0x00000040;
        const COLOR_ATTACHMENT_READ = 0x00000080;
        const COLOR_ATTACHMENT_WRITE = 0x00000100;
        const DEPTH_STENCIL_ATTACHMENT_READ = 0x00000200;
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 0x00000400;
        const TRANSFER_READ = 0x00000800;
        const TRANSFER_WRITE = 0x00001000;
        const HOST_READ = 0x00002000;
        const HOST_WRITE = 0x00004000;
        const MEMORY_READ = 0x00008000;
        const MEMORY_WRITE = 0x00010000;
    }
}

bitflags! {
#[repr(transparent)]
#[derive(Default, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct BufferUsageFlags : u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC = 0b1;
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST = 0b10;
        #[doc = "Can be used as TBO"]
        const UNIFORM_TEXEL_BUFFER = 0b100;
        #[doc = "Can be used as IBO"]
        const STORAGE_TEXEL_BUFFER = 0b1000;
        #[doc = "Can be used as UBO"]
        const UNIFORM_BUFFER = 0b1_0000;
        #[doc = "Can be used as SSBO"]
        const STORAGE_BUFFER = 0b10_0000;
        #[doc = "Can be used as source of fixed-function index fetch (index buffer)"]
        const INDEX_BUFFER = 0b100_0000;
        #[doc = "Can be used as source of fixed-function vertex fetch (VBO)"]
        const VERTEX_BUFFER = 0b1000_0000;
        #[doc = "Can be the source of indirect parameters (e.g. indirect buffer, parameter buffer)"]
        const INDIRECT_BUFFER = 0b1_0000_0000;
    }
}

bitflags! {
#[repr(transparent)]
#[derive(Default, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ImageUsageFlags : u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC = 0b1;
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST = 0b10;
        #[doc = "Can be sampled from (SAMPLED_IMAGE and COMBINED_IMAGE_SAMPLER descriptor types)"]
        const SAMPLED = 0b100;
        #[doc = "Can be used as storage image (STORAGE_IMAGE descriptor type)"]
        const STORAGE = 0b1000;
        #[doc = "Can be used as framebuffer color attachment"]
        const COLOR_ATTACHMENT = 0b1_0000;
        #[doc = "Can be used as framebuffer depth/stencil attachment"]
        const DEPTH_STENCIL_ATTACHMENT = 0b10_0000;
        #[doc = "Image data not needed outside of rendering"]
        const TRANSIENT_ATTACHMENT = 0b100_0000;
        #[doc = "Can be used as framebuffer input attachment"]
        const INPUT_ATTACHMENT = 0b1000_0000;
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CompareOp {
    #[default]
    Always,
    Never,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreatereEqual,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ComponentMapping {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ComponentSwizzle {
    #[default]
    Identity,
    Zero,
    One,
    R,
    G,
    B,
    A,
}

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ImageFormat {
    #[default]
    Rgba8,
    Bgra8,
    SRgba8,
    Rgb8,
    RFloat32,
    RgFloat32,
    RgbFloat32,
    RgbaFloat32,
    RFloat16,
    RgFloat16,
    RgbFloat16,
    RgbaFloat16,
    Depth,
}

impl ImageFormat {
    pub fn texel_size_bytes(&self) -> u32 {
        match self {
            ImageFormat::Rgba8
            | ImageFormat::Bgra8
            | ImageFormat::SRgba8
            | ImageFormat::RFloat32 => 4,
            ImageFormat::RgFloat32 => 8,
            ImageFormat::Rgb8 => 3,
            ImageFormat::RgbFloat32 => 12,
            ImageFormat::RgbaFloat32 => 16,
            ImageFormat::RFloat16 => 2,
            ImageFormat::RgFloat16 => 4,
            ImageFormat::RgbFloat16 => 6,
            ImageFormat::RgbaFloat16 => 8,
            ImageFormat::Depth => 3,
        }
    }
    pub fn is_color(&self) -> bool {
        match self {
            ImageFormat::RFloat16
            | ImageFormat::RgFloat16
            | ImageFormat::RgbFloat16
            | ImageFormat::RgbaFloat16
            | ImageFormat::Rgba8
            | ImageFormat::Bgra8
            | ImageFormat::SRgba8
            | ImageFormat::Rgb8
            | ImageFormat::RFloat32
            | ImageFormat::RgFloat32
            | ImageFormat::RgbFloat32
            | ImageFormat::RgbaFloat32 => true,
            ImageFormat::Depth => false,
        }
    }

    pub fn is_depth(&self) -> bool {
        ImageFormat::Depth == *self
    }
    pub fn default_usage_flags(&self) -> ImageUsageFlags {
        if self.is_color() {
            ImageUsageFlags::COLOR_ATTACHMENT
        } else if self.is_depth() {
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            unreachable!()
        }
    }
    pub fn aspect_mask(&self) -> ImageAspectFlags {
        if self.is_color() {
            ImageAspectFlags::COLOR
        } else if self.is_depth() {
            ImageAspectFlags::DEPTH
        } else {
            unreachable!()
        }
    }
    pub fn preferred_attachment_read_layout(&self) -> ImageLayout {
        if self.is_color() {
            ImageLayout::ShaderReadOnly
        } else if self.is_depth() {
            ImageLayout::DepthStencilReadOnly
        } else {
            unreachable!()
        }
    }
    pub fn preferred_attachment_write_layout(&self) -> ImageLayout {
        if self.is_color() {
            ImageLayout::ColorAttachment
        } else if self.is_depth() {
            ImageLayout::DepthStencilAttachment
        } else {
            unreachable!()
        }
    }

    pub fn preferred_shader_read_layout(&self) -> ImageLayout {
        if self.is_color() {
            ImageLayout::ShaderReadOnly
        } else if self.is_depth() {
            ImageLayout::DepthStencilReadOnly
        } else {
            unreachable!()
        }
    }
    pub fn preferred_shader_write_layout(&self) -> ImageLayout {
        if self.is_color() {
            ImageLayout::ShaderReadOnly
        } else if self.is_depth() {
            ImageLayout::DepthStencilAttachment
        } else {
            unreachable!()
        }
    }
}

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImageViewType {
    Type1D,
    Type2D,
    Type3D,
    Cube,
    Type1DArray,
    Type2DArray,
    TypeCubeArray,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PushConstantRange {
    pub stage_flags: ShaderStage,
    pub offset: u32,
    pub size: u32,
}

bitflags! {
    #[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct ImageAspectFlags : u32 {
        const COLOR = 0b1;
        const DEPTH = 0b10;
        const STENCIL = 0b100;
    }
}

bitflags! {
    #[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct ShaderStage : u32{
        const VERTEX = 0b1;
        const TESSELLATION_CONTROL = 0b10;
        const TESSELLATION_EVALUATION = 0b100;
        const GEOMETRY = 0b1000;
        const FRAGMENT = 0b1_0000;
        const COMPUTE = 0b10_0000;
        const ALL_GRAPHICS = 0x0000_001F;
        const ALL = 0x7FFF_FFFF;
    }
}

impl Default for ShaderStage {
    fn default() -> Self {
        ShaderStage::empty()
    }
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndexType {
    #[default]
    Uint16,
    Uint32,
    Uint64,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace(u32),
    ClampedIncrement,
    ClampedDecrement,
    Invert,
    WrappedIncrement,
    WrappedDecrement,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct StencilOpState {
    fail: StencilOp,
    pass: StencilOp,
    depth_fail: StencilOp,
    compare: CompareOp,
    compare_mask: u32,
    write_mask: u32,
    reference: u32,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum SamplerAddressMode {
    #[default]
    ClampToBorder,
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Filter {
    #[default]
    Linear,
    Nearest,
}

#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlendMode {
    #[default]
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

#[derive(Default, Clone, Debug, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum BlendOp {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Default, Clone, Debug, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum SampleCount {
    #[default]
    Sample1,
    Sample2,
    Sample4,
    Sample8,
    Sample16,
    Sample32,
    Sample64,
}

bitflags! {
#[derive(Default, Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColorComponentFlags : u32 {
    const R = 0b1;
    const G = 0b10;
    const B = 0b100;
    const A = 0b1000;
}}

impl ColorComponentFlags {
    pub const RGBA: Self =
        Self::from_bits_truncate(Self::R.bits() | Self::G.bits() | Self::B.bits() | Self::A.bits());
}

#[derive(Default, Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct SamplerCreateInfo {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub address_u: SamplerAddressMode,
    pub address_v: SamplerAddressMode,
    pub address_w: SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub compare_function: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: [f32; 4],
}

#[derive(Default, Clone, Debug, Copy, PartialEq, PartialOrd)]
pub enum PipelineBindPoint {
    #[default]
    Graphics,
    Compute,
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum QueueType {
    #[default]
    Graphics,
    AsyncCompute,
    Transfer,
}
impl QueueType {
    fn get_vk_queue(&self, gpu: &VkGpu) -> ash::vk::Queue {
        match self {
            QueueType::Graphics => gpu.state.graphics_queue,
            QueueType::AsyncCompute => gpu.state.async_compute_queue,
            QueueType::Transfer => gpu.state.transfer_queue,
        }
    }

    fn get_vk_queue_index(&self, families: &QueueFamilies) -> u32 {
        match self {
            QueueType::Graphics => families.graphics_family.index,
            QueueType::AsyncCompute => families.async_compute_family.index,
            QueueType::Transfer => families.transfer_family.index,
        }
    }
}

#[derive(Clone, Hash)]
pub struct BufferRange {
    pub handle: BufferHandle,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Hash)]
pub struct SamplerState {
    pub sampler: SamplerHandle,
    pub image_view: ImageViewHandle,
    pub image_layout: ImageLayout,
}

#[derive(Clone)]
pub enum DescriptorType {
    UniformBuffer(BufferRange),
    StorageBuffer(BufferRange),
    Sampler(SamplerState),
    CombinedImageSampler(SamplerState),
}

impl std::hash::Hash for DescriptorType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl Debug for DescriptorType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", core::mem::discriminant(self)))
    }
}

#[derive(Clone, Hash, Debug)]
pub struct DescriptorInfo {
    pub binding: u32,
    pub element_type: DescriptorType,
    pub binding_stage: ShaderStage,
}

#[derive(Clone, Hash)]
pub struct DescriptorSetInfo<'a> {
    pub descriptors: &'a [DescriptorInfo],
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AttachmentReference {
    pub attachment: u32,
    pub layout: ImageLayout,
}
#[derive(Clone, Copy, Hash, Debug, PartialEq, Ord, PartialOrd, Eq)]
pub enum PresentMode {
    #[doc = "Present images immediately"]
    Immediate,
    #[doc = "Wait for next vsync, new images are discarded"]
    Fifo,
    #[doc = "May not be supported, wait for next vsync, discard old images in queue in favour for new images"]
    Mailbox,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SubpassDescription<'a> {
    pub pipeline_bind_point: PipelineBindPoint,
    pub input_attachments: &'a [AttachmentReference],
    pub color_attachments: &'a [AttachmentReference],
    pub resolve_attachments: &'a [AttachmentReference],
    pub depth_stencil_attachment: &'a [AttachmentReference],
    pub preserve_attachments: &'a [u32],
}

#[derive(Debug, Clone, Copy)]
pub struct SubpassDependency {
    pub src_subpass: u32,
    pub dst_subpass: u32,
    pub src_stage_mask: PipelineStageFlags,
    pub dst_stage_mask: PipelineStageFlags,
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
}

impl SubpassDependency {
    pub const EXTERNAL: u32 = u32::MAX;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderPassDescription<'a> {
    pub attachments: &'a [RenderPassAttachment],
    pub subpasses: &'a [SubpassDescription<'a>],
    pub dependencies: &'a [SubpassDependency],
}

#[derive(Copy, Clone, Default)]
pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: CompareOp,
    pub stencil_test_enable: bool,
    pub front: StencilOpState,
    pub back: StencilOpState,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

#[derive(Copy, Clone, Default, Hash)]
pub enum LogicOp {
    #[default]
    Clear,
    And,
    AndReverse,
    Copy,
    AndInverted,
    NoOp,
    Xor,
    Or,
    Nor,
    Equivalent,
    Invert,
    OrReverse,
    CopyInverted,
    OrInverted,
    Nand,
    Set,
}

#[derive(Clone, Copy, Debug, Default, Hash)]
pub enum FrontFace {
    #[default]
    CounterClockWise,
    ClockWise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum InputRate {
    PerVertex,
    PerInstance,
}
impl ToVk for InputRate {
    type Inner = ash::vk::VertexInputRate;
    fn to_vk(&self) -> ash::vk::VertexInputRate {
        match self {
            InputRate::PerVertex => Self::Inner::VERTEX,
            InputRate::PerInstance => Self::Inner::INSTANCE,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash)]
pub struct VertexAttributeDescription {
    pub location: u32,
    pub format: ImageFormat,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, Hash)]
pub struct VertexBindingDescription<'a> {
    pub binding: u32,
    pub input_rate: InputRate,
    pub stride: u32,
    pub attributes: &'a [VertexAttributeDescription],
}

#[derive(Clone)]
pub struct VertexStageInfo<'a> {
    pub entry_point: &'a str,
    pub module: ShaderModuleHandle,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct BlendState {
    pub blend_enable: bool,
    pub src_color_blend_factor: BlendMode,
    pub dst_color_blend_factor: BlendMode,
    pub color_blend_op: BlendOp,
    pub src_alpha_blend_factor: BlendMode,
    pub dst_alpha_blend_factor: BlendMode,
    pub alpha_blend_op: BlendOp,
    pub color_write_mask: ColorComponentFlags,
}

impl Default for BlendState {
    fn default() -> Self {
        Self {
            blend_enable: true,
            src_color_blend_factor: BlendMode::One,
            dst_color_blend_factor: BlendMode::Zero,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendMode::One,
            dst_alpha_blend_factor: BlendMode::Zero,
            alpha_blend_op: BlendOp::Add,
            color_write_mask: ColorComponentFlags::RGBA,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderPassAttachment {
    pub format: ImageFormat,
    pub samples: SampleCount,
    pub load_op: ColorLoadOp,
    pub store_op: AttachmentStoreOp,
    pub stencil_load_op: StencilLoadOp,
    pub stencil_store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
    pub final_layout: ImageLayout,
    pub blend_state: BlendState,
}

#[derive(Clone, Copy, Debug)]
pub struct DepthStencilAttachment {}
#[derive(Clone)]
pub struct FragmentStageInfo<'a> {
    pub entry_point: &'a str,
    pub module: ShaderModuleHandle,
    pub color_attachments: &'a [RenderPassAttachment],
    pub depth_stencil_attachments: &'a [DepthStencilAttachment],
}

#[derive(Clone, Copy, Debug, Default, Hash)]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
    TriangleStrip,
    TriangleFan,
    PointList,
    LineList,
    LineStrip,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line(f32),
    Point,
}

#[derive(Clone, Copy, Debug, Default, Hash)]
pub enum CullMode {
    #[default]
    None,
    Back,
    Front,
    FrontAndBack,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum BindingType {
    Uniform,
    Storage,
    Sampler,
    CombinedImageSampler,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct BindingElement {
    pub binding_type: BindingType,
    pub index: u32,
    pub stage: ShaderStage,
}

#[derive(Clone)]
pub struct GlobalBinding<'a> {
    pub set_index: u32,
    pub elements: &'a [BindingElement],
}

#[derive(Clone, Default)]
pub struct GraphicsPipelineDescription<'a> {
    pub global_bindings: &'a [GlobalBinding<'a>],
    pub vertex_inputs: &'a [VertexBindingDescription<'a>],
    pub vertex_stage: Option<VertexStageInfo<'a>>,
    pub fragment_stage: Option<FragmentStageInfo<'a>>,
    pub input_topology: PrimitiveTopology,
    pub primitive_restart: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub depth_stencil_state: DepthStencilState,
    pub logic_op: Option<LogicOp>,
    pub push_constant_ranges: &'a [PushConstantRange],
}

#[derive(Clone)]
pub struct ComputePipelineDescription<'a> {
    pub module: ShaderModuleHandle,
    pub entry_point: &'a str,
    pub bindings: &'a [GlobalBinding<'a>],
    pub push_constant_ranges: &'a [PushConstantRange],
}

#[derive(Clone, Copy)]
pub struct MemoryBarrier {
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
}

#[derive(Clone, Copy)]
pub struct ImageSubresourceRange {
    pub aspect_mask: ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

bitflags! {
    #[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
    pub struct CommandPoolCreateFlags :  u8 {
        const TRANSIENT = 0b0;
    }
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct CommandPoolCreateInfo {
    pub queue_type: QueueType,
    pub flags: CommandPoolCreateFlags,
}

#[derive(Default)]
pub struct CommandBufferSubmitInfo<'a> {
    pub wait_semaphores: &'a [&'a GPUSemaphore],
    pub wait_stages: &'a [PipelineStageFlags],
    pub signal_semaphores: &'a [&'a GPUSemaphore],
    pub fence: Option<&'a GPUFence>,
}

pub struct BufferMemoryBarrier {
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: BufferHandle,
    pub offset: u64,
    pub size: u64,
}

pub struct ImageMemoryBarrier {
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
    pub old_layout: ImageLayout,
    pub new_layout: ImageLayout,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: ImageHandle,
    pub subresource_range: ImageSubresourceRange,
}

#[derive(Default)]
pub struct PipelineBarrierInfo<'a> {
    pub src_stage_mask: PipelineStageFlags,
    pub dst_stage_mask: PipelineStageFlags,
    pub memory_barriers: &'a [MemoryBarrier],
    pub buffer_memory_barriers: &'a [BufferMemoryBarrier],
    pub image_memory_barriers: &'a [ImageMemoryBarrier],
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ColorLoadOp {
    #[default]
    DontCare,
    Load,
    Clear([f32; 4]),
}

#[derive(Clone, Copy, Debug, Default)]
pub enum DepthLoadOp {
    #[default]
    DontCare,
    Load,
    Clear(f32),
}
#[derive(Clone, Copy, Debug, Default)]
pub enum StencilLoadOp {
    #[default]
    DontCare,
    Load,
    Clear(u8),
}

#[derive(Clone, Copy, Debug, Default)]
pub enum AttachmentStoreOp {
    #[default]
    DontCare,
    Store,
}

#[derive(Clone)]
pub struct ColorAttachment {
    pub image_view: ImageViewHandle,
    pub load_op: ColorLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
}

#[derive(Clone)]
pub struct DepthAttachment {
    pub image_view: ImageViewHandle,
    pub load_op: DepthLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
}

#[derive(Clone)]
pub struct StencilAttachment {
    pub image_view: ImageViewHandle,
    pub load_op: StencilLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
}

#[derive(Clone)]
pub struct BeginRenderPassInfo<'a> {
    pub color_attachments: &'a [ColorAttachment],
    pub depth_attachment: Option<DepthAttachment>,
    pub stencil_attachment: Option<StencilAttachment>,
    pub render_area: Rect2D,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}
