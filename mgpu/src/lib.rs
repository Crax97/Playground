mod command_recorder;
mod device;
mod hal;
mod rdg;
mod swapchain;

#[macro_use]
pub mod util;
pub(crate) mod staging_buffer_allocator;

use std::num::NonZeroU32;

use bitflags::bitflags;

pub use command_recorder::*;
pub use device::*;
pub use hal::{GraphicsPipelineLayout, OwnedFragmentStageInfo, OwnedVertexStageInfo};
pub use swapchain::*;

#[cfg(feature = "vulkan")]
use hal::vulkan::VulkanHalError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub const MAX_PUSH_CONSTANT_RANGE_SIZE_BYTES: usize = 128;

#[derive(Debug)]
pub enum MgpuError {
    InvalidParams {
        params_name: &'static str,
        label: Option<String>,
        reason: String,
    },
    CheckFailed {
        check: &'static str,
        message: String,
    },
    InvalidHandle,
    Dynamic(String),

    #[cfg(feature = "vulkan")]
    VulkanError(VulkanHalError),
}

pub type MgpuResult<T> = Result<T, MgpuError>;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub enum MemoryDomain {
    Cpu,
    Gpu,
}

bitflags! {
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature="serde", derive(Serialize, Deserialize))]
pub struct ImageCreationFlags: u32 {
    #[doc = "Image should support sparse backing"]
    const SPARSE_BINDING = 0b1;
    #[doc = "Image should support sparse backing with partial residency"]
    const SPARSE_RESIDENCY = 0b10;
    #[doc = "Image should support constant data access to physical memory ranges mapped into multiple locations of sparse images"]
    const SPARSE_ALIASED = 0b100;
    #[doc = "Allows image views to have different format than the base image"]
    const MUTABLE_FORMAT = 0b1000;
    #[doc = "Allows creating image views with cube type from the created image"]
    const CUBE_COMPATIBLE = 0b1_0000;
}
}
bitflags! {
    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
    #[cfg_attr(feature="serde", derive(Serialize, Deserialize))]
    pub struct ImageUsageFlags : u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC = 0b1;
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST= 0b10;
        #[doc = "Can be sampled from (SAMPLED_IMAGE and COMBINED_IMAGE_SAMPLER descriptor types)"]
        const SAMPLED= 0b100;
        #[doc = "Can be used as storage image (STORAGE_IMAGE descriptor type)"]
        const STORAGE= 0b1000;
        #[doc = "Can be used as framebuffer color attachment"]
        const COLOR_ATTACHMENT= 0b1_0000;
        #[doc = "Can be used as framebuffer depth/stencil attachment"]
        const DEPTH_STENCIL_ATTACHMENT= 0b10_0000;
        #[doc = "Image data not needed outside of rendering"]
        const TRANSIENT_ATTACHMENT= 0b100_0000;
        #[doc = "Can be used as framebuffer input attachment"]
        const INPUT_ATTACHMENT= 0b1000_0000;
    }
}

bitflags! {
    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
    #[cfg_attr(feature="serde", derive(Serialize, Deserialize))]
    pub struct BufferUsageFlags : u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC= 0b1;
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST= 0b10;
        #[doc = "Can be used as TBO"]
        const UNIFORM_TEXEL_BUFFER= 0b100;
        #[doc = "Can be used as IBO"]
        const STORAGE_TEXEL_BUFFER= 0b1000;
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
    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
    #[cfg_attr(feature="serde", derive(Serialize, Deserialize))]
    pub struct RenderPassFlags : u32 {
        #[doc = "Don't flip the viewport, only valid when using Vulkan"]
        const DONT_FLIP_VIEWPORT = 0b1;
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageFormat {
    Unknown,
    R32f,
    Rg32f,
    Rgba32f,

    R8,
    Rg8,
    Rgba8,
    R8Signed,
    Rg8Signed,
    Rgba8Signed,
    R16f,
    Rg16f,
    Rgba16f,
    Bgra8,
    Bgra8_sRGB,
    Depth32,
    A2B10G10R10,
}
impl ImageFormat {
    pub fn byte_size(&self) -> usize {
        use ImageFormat::*;
        match self {
            Unknown => 0,
            R16f => 2,
            Rg16f => 4,
            Rgba16f => 8,
            R32f => 4,
            Rg32f => 8,
            Rgba32f => 16,
            R8 | R8Signed => 1,
            Rg8 | Rg8Signed => 2,
            Rgba8 | Rgba8Signed => 4,
            Bgra8 => 4,
            Bgra8_sRGB => 4,
            Depth32 => 4,
            A2B10G10R10 => 4,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageAspect {
    Color,
    Depth,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Extents3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Extents2D {
    pub width: u32,
    pub height: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extents: Extents2D,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageRegion {
    pub offset: Offset3D,
    pub extents: Extents3D,
    pub mip: u32,
    pub num_mips: NonZeroU32,
    pub base_array_layer: u32,
    pub num_layers: NonZeroU32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// Defines a subresource of an image
pub struct ImageSubresource {
    pub mip: u32,
    pub num_mips: NonZeroU32,
    pub base_array_layer: u32,
    pub num_layers: NonZeroU32,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageDimension {
    D1,
    D2,
    D3,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageViewType {
    D1,
    D2,
    D3,
    Cube,
    Array1D,
    Array2D,
    CubeArray,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SampleCount {
    One,
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FilterMode {
    #[default]
    Linear,
    Nearest,
}

#[derive(Default, Copy, Clone, PartialEq, PartialOrd, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BorderColor {
    #[default]
    White,
    Black,
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MipmapMode {
    #[default]
    Linear,
    Nearest,
}

#[derive(Default, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AddressMode {
    #[default]
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

pub struct ShaderModuleDescription<'a> {
    pub label: Option<&'a str>,
    pub source: &'a [u32],
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RenderTargetLoadOp {
    #[default]
    DontCare,
    Clear([f32; 4]),
    Load,
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DepthStencilTargetLoadOp {
    #[default]
    DontCare,
    Clear(f32, u32),
    Load,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AttachmentStoreOp {
    #[default]
    DontCare,
    Store,
    Present,
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Debug, Hash)]
pub struct RenderTarget {
    pub view: ImageView,
    pub sample_count: SampleCount,
    pub load_op: RenderTargetLoadOp,
    pub store_op: AttachmentStoreOp,
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Hash)]
pub struct DepthStencilTarget {
    pub view: ImageView,
    pub sample_count: SampleCount,
    pub load_op: DepthStencilTargetLoadOp,
    pub store_op: AttachmentStoreOp,
}

pub struct RenderPassDescription<'a> {
    pub label: Option<&'a str>,
    pub flags: RenderPassFlags,
    pub render_targets: &'a [RenderTarget],
    pub depth_stencil_attachment: Option<&'a DepthStencilTarget>,
    pub render_area: Rect2D,
}

pub struct ComputePassDescription<'a> {
    pub label: Option<&'a str>,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageDescription<'a> {
    pub label: Option<&'a str>,
    pub usage_flags: ImageUsageFlags,
    pub creation_flags: ImageCreationFlags,
    pub extents: Extents3D,
    pub dimension: ImageDimension,
    pub mips: NonZeroU32,
    pub array_layers: NonZeroU32,
    pub samples: SampleCount,
    pub format: ImageFormat,
    pub memory_domain: MemoryDomain,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct BufferDescription<'a> {
    pub label: Option<&'a str>,
    pub usage_flags: BufferUsageFlags,
    pub size: usize,
    pub memory_domain: MemoryDomain,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
/// This struct describes an image write operation
pub struct BufferWriteParams<'a> {
    pub data: &'a [u8],
    pub offset: usize,
    pub size: usize,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
/// This struct describes an image write operation
pub struct ImageWriteParams<'a> {
    pub data: &'a [u8],
    pub region: ImageRegion,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
/// This struct describes a blit operation
pub struct BlitParams {
    pub src_image: Image,
    pub src_region: ImageRegion,
    pub dst_image: Image,
    pub dst_region: ImageRegion,
    pub filter: FilterMode,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageViewDescription<'a> {
    pub label: Option<&'a str>,
    pub image: Image,
    pub format: ImageFormat,
    pub view_ty: ImageViewType,
    pub aspect: ImageAspect,
    pub image_subresource: ImageSubresource,
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub struct SamplerDescription<'a> {
    pub label: Option<&'a str>,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_mode: MipmapMode,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub lod_bias: f32,
    pub compare_op: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: BorderColor,
    pub unnormalized_coordinates: bool,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VertexInputFrequency {
    PerVertex,
    PerInstance,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VertexAttributeFormat {
    Int,
    Int2,
    Int3,
    Int4,
    Uint,
    Uint2,
    Uint3,
    Uint4,
    Float,
    Float2,
    Float3,
    Float4,
    Mat2x2,
    Mat3x3,
    Mat4x4,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VertexInputDescription {
    pub location: usize,
    pub stride: usize,
    pub offset: usize,
    pub format: VertexAttributeFormat,
    pub frequency: VertexInputFrequency,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    DestColor,
    OneMinusDestColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestAlpha,
    OneMinusDestAlpha,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BlendSettings {
    pub src_color_blend_factor: BlendFactor,
    pub dst_color_blend_factor: BlendFactor,
    pub color_blend_op: BlendOp,
    pub src_alpha_blend_factor: BlendFactor,
    pub dst_alpha_blend_factor: BlendFactor,
    pub alpha_blend_op: BlendOp,
    pub write_mask: ColorWriteMask,
}

bitflags! {
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColorWriteMask : u8 {
    const R = 0x01;
    const B = 0x02;
    const G = 0x03;
    const A = 0x04;

    const RGBA = Self::R.bits() | Self::B.bits() | Self::G.bits() | Self::A.bits();
}
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RenderTargetInfo {
    pub format: ImageFormat,
    pub blend: Option<BlendSettings>,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DepthStencilTargetInfo {
    pub format: ImageFormat,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct VertexStageInfo<'a> {
    pub shader: &'a ShaderModule,
    pub entry_point: &'a str,
    pub vertex_inputs: &'a [VertexInputDescription],
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
pub enum CompareOp {
    #[default]
    Always,
    Never,
    Less,
    LessOrEqual,
    Equal,
    Greater,
    GreaterOrEqual,
    NotEqual,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
pub struct DepthStencilState {
    pub depth_test_enabled: bool,
    pub depth_write_enabled: bool,
    pub depth_compare_op: CompareOp,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct StencilState {}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct FragmentStageInfo<'a> {
    pub shader: &'a ShaderModule,
    pub entry_point: &'a str,
    pub render_targets: &'a [RenderTargetInfo],
    pub depth_stencil_target: Option<&'a DepthStencilTargetInfo>,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
    TriangleFan,
    Line,
    LineList,
    LineStrip,
    Point,
}

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PolygonMode {
    #[default]
    Filled,
    Line(f32),
    Point,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CullMode {
    None,
    #[default]
    Back,
    Front,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FrontFace {
    #[default]
    CounterClockWise,
    ClockWise,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultisampleState {}

#[derive(Copy, Clone, Debug, Hash)]
pub struct PushConstantInfo {
    pub size: usize,
    pub visibility: ShaderStageFlags,
}

#[derive(Copy, Clone, Debug, Hash)]
pub struct GraphicsPipelineDescription<'a> {
    pub label: Option<&'a str>,
    pub vertex_stage: &'a VertexStageInfo<'a>,
    pub fragment_stage: Option<&'a FragmentStageInfo<'a>>,
    pub primitive_restart_enabled: bool,
    pub primitive_topology: PrimitiveTopology,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub multisample_state: Option<MultisampleState>,
    pub depth_stencil_state: DepthStencilState,
    pub binding_set_layouts: &'a [BindingSetLayoutInfo<'a>],
    pub push_constant_info: Option<PushConstantInfo>,
}

#[derive(Copy, Clone, Debug, Hash)]
pub struct ComputePipelineDescription<'a> {
    pub label: Option<&'a str>,
    pub shader: ShaderModule,
    pub entry_point: &'a str,
    pub binding_set_layouts: &'a [BindingSetLayoutInfo<'a>],
    pub push_constant_info: Option<PushConstantInfo>,
}

#[derive(Clone, Debug)]
pub struct ShaderAttribute {
    pub location: usize,
    pub format: VertexAttributeFormat,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BufferType {
    Uniform,
    Storage,
}

#[derive(Clone, Copy, PartialOrd, Ord, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StorageAccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Hash, Default, Eq, PartialEq)]
pub enum BindingSetElementKind {
    #[default]
    Unknown,
    Buffer {
        ty: BufferType,
        access_mode: StorageAccessMode,
    },
    Sampler,
    // This type can only be used with the Vulkan HAL
    CombinedImageSampler {
        format: ImageFormat,
        dimension: ImageDimension,
    },
    SampledImage,
    StorageImage {
        format: ImageFormat,
        access_mode: StorageAccessMode,
    },
}

#[derive(Default, Clone, Debug, Hash)]
pub struct BindingSetElement {
    pub binding: usize,
    pub array_length: usize,
    pub ty: BindingSetElementKind,
    pub shader_stage_flags: ShaderStageFlags,
}

#[derive(Default, Debug, Hash)]
pub struct BindingSetLayout<'a> {
    pub binding_set_elements: &'a [BindingSetElement],
}

#[derive(Clone, Default, Debug, Hash)]
pub struct OwnedBindingSetLayout {
    pub binding_set_elements: Vec<BindingSetElement>,
}

#[derive(Debug, Hash)]
pub struct BindingSetLayoutInfo<'a> {
    pub set: usize,
    pub layout: &'a BindingSetLayout<'a>,
}

#[derive(Clone, Default, Debug, Hash)]
pub struct OwnedBindingSetLayoutInfo {
    pub set: usize,
    pub layout: OwnedBindingSetLayout,
}

impl<'a> From<&BindingSetLayout<'a>> for OwnedBindingSetLayout {
    fn from(value: &BindingSetLayout<'a>) -> Self {
        OwnedBindingSetLayout {
            binding_set_elements: value.binding_set_elements.to_vec(),
        }
    }
}

impl<'a> From<&BindingSetLayoutInfo<'a>> for OwnedBindingSetLayoutInfo {
    fn from(value: &BindingSetLayoutInfo<'a>) -> OwnedBindingSetLayoutInfo {
        OwnedBindingSetLayoutInfo {
            set: value.set,
            layout: value.layout.into(),
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub enum BindingType {
    Sampler(Sampler),
    SampledImage {
        view: ImageView,
    },
    StorageImage {
        view: ImageView,
        access_mode: StorageAccessMode,
    },
    UniformBuffer {
        buffer: Buffer,
        offset: usize,
        range: usize,
    },
    StorageBuffer {
        buffer: Buffer,
        offset: usize,
        range: usize,
        access_mode: StorageAccessMode,
    },
}
impl BindingType {
    fn binding_type(&self) -> BindingSetElementKind {
        match self {
            BindingType::Sampler(_) => BindingSetElementKind::Sampler,
            BindingType::SampledImage { .. } => BindingSetElementKind::SampledImage,
            BindingType::UniformBuffer { .. } => BindingSetElementKind::Buffer {
                ty: BufferType::Uniform,
                access_mode: StorageAccessMode::Read,
            },
            BindingType::StorageBuffer {
                buffer: _,
                offset: _,
                range: _,
                access_mode,
            } => BindingSetElementKind::Buffer {
                ty: BufferType::Storage,
                access_mode: *access_mode,
            },
            BindingType::StorageImage { view, access_mode } => {
                BindingSetElementKind::StorageImage {
                    format: view.owner.format,
                    access_mode: *access_mode,
                }
            }
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct Binding {
    pub binding: usize,
    pub ty: BindingType,
    pub visibility: ShaderStageFlags,
}

#[derive(Debug)]
pub struct BindingSetDescription<'a> {
    pub label: Option<&'a str>,
    pub bindings: &'a [Binding],
}

bitflags! {
    #[derive(Clone, Copy, Default, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct ShaderStageFlags : u32 {
        const VERTEX = 0x01;
        const FRAGMENT = 0x02;
        const COMPUTE = 0x04;

        const ALL_GRAPHICS = Self::VERTEX.bits() | Self::FRAGMENT.bits();
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum VariableType {
    Compound(Vec<ShaderVariable>),
    Field {
        format: VertexAttributeFormat,
    },
    Array {
        members_layout: Box<VariableType>,
        length: Option<usize>,
    },
    Texture(StorageAccessMode),
}
impl VariableType {
    pub fn size(&self) -> usize {
        match self {
            VariableType::Compound(mems) => mems.iter().map(|m| m.ty.size()).sum(),
            VariableType::Field { format } => format.size_bytes(),
            VariableType::Array {
                members_layout,
                length: _,
            } => members_layout.size(),
            VariableType::Texture(_) => {
                todo!("Look up what's the size of a texture")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ShaderVariable {
    pub name: Option<String>,
    pub binding_set: usize,
    pub binding_index: usize,
    pub offset: usize,
    pub ty: VariableType,
}

#[derive(Clone, Debug)]
pub struct ShaderModuleLayout {
    pub entry_points: Vec<String>,
    pub inputs: Vec<ShaderAttribute>,
    pub outputs: Vec<ShaderAttribute>,
    pub binding_sets: Vec<OwnedBindingSetLayoutInfo>,
    pub variables: Vec<ShaderVariable>,
    pub push_constant: Option<ShaderStageFlags>,
}

/// A Buffer is a linear data buffer that can be read or written by a shader
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Buffer {
    id: u64,
    usage_flags: BufferUsageFlags,
    size: usize,
    memory_domain: MemoryDomain,
}
impl Buffer {
    pub fn write_all_params<'a>(&self, data: &'a [u8]) -> BufferWriteParams<'a> {
        BufferWriteParams {
            data,
            offset: 0,
            size: self.size,
        }
    }

    pub fn bind_whole_range_uniform_buffer(&self) -> BindingType {
        BindingType::UniformBuffer {
            buffer: *self,
            offset: 0,
            range: self.size,
        }
    }

    pub fn bind_whole_range_storage_buffer(&self, access_mode: StorageAccessMode) -> BindingType {
        BindingType::StorageBuffer {
            buffer: *self,
            offset: 0,
            range: self.size,
            access_mode,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

/// An image is a multidimensional buffer of data, with an associated format
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Image {
    id: u64,
    usage_flags: ImageUsageFlags,
    creation_flags: ImageCreationFlags,
    extents: Extents3D,
    dimension: ImageDimension,
    num_mips: NonZeroU32,
    array_layers: NonZeroU32,
    samples: SampleCount,
    format: ImageFormat,
}
impl Image {
    pub fn whole_region(&self) -> ImageRegion {
        ImageRegion {
            offset: Offset3D::default(),
            extents: self.extents,
            mip: 0,
            num_mips: self.num_mips,
            base_array_layer: 0,
            num_layers: self.array_layers,
        }
    }

    pub fn mip_region(&self, mip: u32) -> ImageRegion {
        let mut extents = self.extents;
        for _ in 0..mip {
            extents.width /= 2;
            extents.height /= 2;
            extents.depth /= 2;
            extents.depth = extents.depth.max(1);
        }
        ImageRegion {
            offset: Offset3D::default(),
            extents,
            mip,
            num_mips: 1.try_into().unwrap(),
            base_array_layer: 0,
            num_layers: self.array_layers,
        }
    }

    pub fn whole_subresource(&self) -> ImageSubresource {
        ImageSubresource {
            mip: 0,
            num_mips: self.num_mips,
            base_array_layer: 0,
            num_layers: self.array_layers,
        }
    }

    pub fn layer(&self, mip: u32, layer: u32) -> ImageSubresource {
        check!(
            mip < self.num_mips.get(),
            "Requested mip {} but only {} are available",
            mip,
            self.num_mips
        );
        check!(
            layer < self.array_layers.get(),
            "Requested layer {} but only {} are available",
            layer,
            self.array_layers
        );
        ImageSubresource {
            mip,
            num_mips: 1.try_into().unwrap(),
            base_array_layer: layer,
            num_layers: 1.try_into().unwrap(),
        }
    }

    pub fn extents(&self) -> Extents3D {
        self.extents
    }

    pub fn mips(&self) -> u32 {
        self.num_mips.get()
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }
}

/// An image view is a view over a portion of an image
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageView {
    owner: Image,
    subresource: ImageSubresource,
    id: u64,
}
impl ImageView {
    pub fn owner(self) -> Image {
        self.owner
    }
    pub fn extents_2d(&self) -> Extents2D {
        let exents = self.owner.mip_region(self.subresource.mip).extents;
        Extents2D {
            width: exents.width,
            height: exents.height,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Sampler {
    id: u64,
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ShaderModule {
    id: u64,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct GraphicsPipeline {
    id: u64,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ComputePipeline {
    id: u64,
}

#[derive(Clone, Debug, Hash)]
pub struct BindingSet {
    id: u64,
    bindings: Vec<Binding>,
}

impl ImageFormat {
    pub fn aspect(self) -> ImageAspect {
        match self {
            ImageFormat::Unknown => unreachable!(),
            ImageFormat::Depth32 => ImageAspect::Depth,
            _ => ImageAspect::Color,
        }
    }
}

impl PolygonMode {
    /// Gets the line width if the mode is Line, otherwise 0.0.
    pub fn line_width(self) -> f32 {
        match self {
            PolygonMode::Line(v) => v,
            _ => 0.0,
        }
    }
}

impl std::ops::Add<Extents2D> for Offset2D {
    type Output = Self;

    fn add(self, rhs: Extents2D) -> Self::Output {
        Self {
            x: self.x + rhs.width as i32,
            y: self.y + rhs.height as i32,
        }
    }
}

impl std::ops::Add<Extents3D> for Offset3D {
    type Output = Self;

    fn add(self, rhs: Extents3D) -> Self::Output {
        Self {
            x: self.x + rhs.width as i32,
            y: self.y + rhs.height as i32,
            z: self.z + rhs.depth as i32,
        }
    }
}

impl Extents3D {
    pub fn area(&self) -> u32 {
        self.width * self.height * self.depth
    }

    pub fn to_2d(&self) -> Extents2D {
        Extents2D {
            width: self.width,
            height: self.height,
        }
    }
}

impl Extents2D {
    pub fn area(&self) -> u32 {
        self.width * self.height
    }
}

impl ImageRegion {
    pub fn to_image_subresource(&self) -> ImageSubresource {
        ImageSubresource {
            mip: self.mip,
            num_mips: self.num_mips,
            base_array_layer: self.base_array_layer,
            num_layers: self.num_layers,
        }
    }
}

impl<'a> GraphicsPipelineDescription<'a> {
    pub fn new(label: Option<&'a str>, vertex_stage: &'a VertexStageInfo) -> Self {
        Self {
            label,
            vertex_stage,
            fragment_stage: None,
            primitive_restart_enabled: Default::default(),
            primitive_topology: Default::default(),
            polygon_mode: Default::default(),
            cull_mode: Default::default(),
            front_face: Default::default(),
            multisample_state: Default::default(),
            depth_stencil_state: Default::default(),
            binding_set_layouts: Default::default(),
            push_constant_info: Default::default(),
        }
    }

    pub fn fragment_stage(mut self, fragment_stage: &'a FragmentStageInfo) -> Self {
        self.fragment_stage = Some(fragment_stage);
        self
    }

    pub fn binding_set_layouts(mut self, binding_set_layouts: &'a [BindingSetLayoutInfo]) -> Self {
        self.binding_set_layouts = binding_set_layouts;
        self
    }

    pub fn depth_stencil_state(mut self, depth_stencil_state: DepthStencilState) -> Self {
        self.depth_stencil_state = depth_stencil_state;
        self
    }

    pub fn push_constant_info(mut self, push_constant_info: Option<PushConstantInfo>) -> Self {
        self.push_constant_info = push_constant_info;
        self
    }
}

impl VertexAttributeFormat {
    pub fn size_bytes(&self) -> usize {
        match self {
            VertexAttributeFormat::Int => 4,
            VertexAttributeFormat::Int2 => 8,
            VertexAttributeFormat::Int3 => 12,
            VertexAttributeFormat::Int4 => 16,
            VertexAttributeFormat::Uint => 4,
            VertexAttributeFormat::Uint2 => 8,
            VertexAttributeFormat::Uint3 => 12,
            VertexAttributeFormat::Uint4 => 16,
            VertexAttributeFormat::Float => 4,
            VertexAttributeFormat::Float2 => 8,
            VertexAttributeFormat::Float3 => 12,
            VertexAttributeFormat::Float4 => 16,
            VertexAttributeFormat::Mat2x2 => 8 * 8,
            VertexAttributeFormat::Mat3x3 => 12 * 12,
            VertexAttributeFormat::Mat4x4 => 16 * 16,
        }
    }
}

impl<'a> BufferWriteParams<'a> {
    pub fn total_bytes(&self) -> usize {
        self.size
    }
}

impl std::fmt::Display for MgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MgpuError::Dynamic(msg) => f.write_str(msg),
            MgpuError::InvalidHandle => f.write_str("Tried to resolve an invalid handle"),
            MgpuError::InvalidParams {
                params_name,
                label,
                reason,
            } => f.write_fmt(format_args!(
                "Invalid '{params_name}' for resource '{}': {reason}",
                label.as_ref().unwrap_or(&"Unnamed".to_string())
            )),

            MgpuError::CheckFailed { check, message } => f.write_fmt(format_args!(
                "A check failed, condition: {}, error: {}",
                check, message
            )),

            #[cfg(feature = "vulkan")]
            MgpuError::VulkanError(error) => f.write_fmt(format_args!("{:?}", error)),
        }
    }
}
impl std::error::Error for MgpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.source()
    }
}

impl std::hash::Hash for RenderTargetLoadOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}
impl std::hash::Hash for DepthStencilTargetLoadOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}
impl std::hash::Hash for PolygonMode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}
impl std::cmp::Eq for RenderTargetLoadOp {}
