mod allocator;
mod command_buffer;
mod descriptor_set;
mod gpu;
mod pipeline;
mod swapchain;
mod types;

pub use bitflags::bitflags;
pub use crate::gpu::*;
pub use allocator::*;
pub use command_buffer::*;
pub use pipeline::*;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
pub use swapchain::Swapchain;
pub use types::*;

pub const WHOLE_SIZE: u64 = u64::MAX;
pub const QUEUE_FAMILY_IGNORED: u32 = u32::MAX;

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
    Depth,
}

impl ImageFormat {
    pub fn is_color(&self) -> bool {
        match self {
            ImageFormat::Rgba8
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


#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
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
    Sample64
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
    pub const RGBA : Self = Self::from_bits_truncate(Self::R.bits() | Self::G.bits() | Self::B.bits() | Self::A.bits());
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

#[derive(Default)]
pub enum QueueType {
    #[default]
    Graphics,
    AsyncCompute,
    Transfer,
}
impl QueueType {
    fn get_vk_command_pool(&self, gpu: &Gpu) -> ash::vk::CommandPool {
        let thread_local_state = &gpu.thread_local_state;
        match self {
            QueueType::Graphics => thread_local_state.graphics_command_pool,
            QueueType::AsyncCompute => thread_local_state.compute_command_pool,
            QueueType::Transfer => thread_local_state.transfer_command_pool,
        }
    }
    fn get_vk_queue(&self, gpu: &Gpu) -> ash::vk::Queue {
        match self {
            QueueType::Graphics => gpu.state.graphics_queue,
            QueueType::AsyncCompute => gpu.state.async_compute_queue,
            QueueType::Transfer => gpu.state.transfer_queue,
        }
    }
}

#[derive(Clone, Hash)]
pub struct BufferRange<'a> {
    pub handle: &'a GpuBuffer,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Hash)]
pub struct SamplerState<'a> {
    pub sampler: &'a GpuSampler,
    pub image_view: &'a GpuImageView,
    pub image_layout: ImageLayout,
}

#[derive(Clone)]
pub enum DescriptorType<'a> {
    UniformBuffer(BufferRange<'a>),
    StorageBuffer(BufferRange<'a>),
    Sampler(SamplerState<'a>),
    CombinedImageSampler(SamplerState<'a>),
}

impl<'a> std::hash::Hash for DescriptorType<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl<'a> Debug for DescriptorType<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", core::mem::discriminant(self)))
    }
}

#[derive(Clone, Hash, Debug)]
pub struct DescriptorInfo<'a> {
    pub binding: u32,
    pub element_type: DescriptorType<'a>,
    pub binding_stage: ShaderStage,
}

#[derive(Clone, Hash)]
pub struct DescriptorSetInfo<'a> {
    pub descriptors: &'a [DescriptorInfo<'a>],
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
