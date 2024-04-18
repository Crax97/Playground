mod command_recorder;
mod device;
mod hal;
mod rdg;
mod swapchain;

#[macro_use]
pub(crate) mod util;
pub(crate) mod staging_buffer;

use std::num::NonZeroU32;

use bitflags::bitflags;

pub use command_recorder::*;
pub use device::*;
pub use swapchain::*;

#[cfg(feature = "vulkan")]
use hal::vulkan::VulkanHalError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
    HostVisible,
    HostCoherent,
    DeviceLocal,
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

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageFormat {
    Unknown,
    Rgba8,
}
impl ImageFormat {
    fn byte_size(&self) -> usize {
        match self {
            Self::Unknown => 0,
            ImageFormat::Rgba8 => 4,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageAspect {
    Color,
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
pub struct Rect2D {
    pub offset: Offset2D,
    pub extents: Extents2D,
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
pub enum SampleCount {
    One,
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
    pub render_targets: &'a [RenderTarget],
    pub depth_stencil_attachment: Option<&'a DepthStencilTarget>,
    pub render_area: Rect2D,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageDescription<'a> {
    pub label: Option<&'a str>,
    pub usage_flags: ImageUsageFlags,
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
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageViewDescription<'a> {
    pub label: Option<&'a str>,
    pub format: ImageFormat,
    pub aspect: ImageAspect,
    pub base_mip: u32,
    pub num_mips: u32,
    pub base_array_layer: u32,
    pub num_array_layers: u32,
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
pub struct BlendSettings {}

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
    pub write_mask: ColorWriteMask,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DepthStencilTargetInfo {}

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
    #[default]
    Back,
    Front,
    None,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FrontFace {
    #[default]
    ClockWise,
    CounterClockWise,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultisampleState {}

#[derive(Copy, Clone, Debug, Hash)]
pub struct GraphicsPipelineDescription<'a> {
    pub label: Option<&'a str>,
    pub vertex_stage: &'a VertexStageInfo<'a>,
    pub fragment_stage: Option<&'a FragmentStageInfo<'a>>,
    pub binding_sets: &'a [BindingSet],
    pub primitive_restart_enabled: bool,
    pub primitive_topology: PrimitiveTopology,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub multisample_state: Option<MultisampleState>,
    pub depth_stencil_state: DepthStencilState,
}

#[derive(Clone, Debug)]
pub struct ShaderAttribute {
    pub name: String,
    pub location: usize,
    pub format: VertexAttributeFormat,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BufferType {
    Uniform,
    Storage,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug)]
pub enum BindingSetElementKind {
    Buffer {
        ty: BufferType,
        access_mode: AccessMode,
    },
    Sampler {
        access_mode: AccessMode,
    },
    SampledImage {
        format: ImageFormat,
        dimension: ImageDimension,
    },
    StorageImage {
        format: ImageFormat,
        access_mode: AccessMode,
        dimension: ImageDimension,
    },
}

#[derive(Clone, Debug)]
pub struct BindingSetElement {
    pub name: String,
    pub binding: usize,
    pub ty: BindingSetElementKind,
}

#[derive(Clone, Default, Debug)]
pub struct BindingSetLayout {
    pub set: usize,
    pub bindings: Vec<BindingSetElement>,
}

#[derive(Clone, Default, Debug)]
pub struct ShaderModuleLayout {
    pub entry_points: Vec<String>,
    pub inputs: Vec<ShaderAttribute>,
    pub outputs: Vec<ShaderAttribute>,
    pub binding_sets: Vec<BindingSetLayout>,
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
}

/// An image is a multidimensional buffer of data, with an associated format
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Image {
    id: u64,
    usage_flags: ImageUsageFlags,
    extents: Extents3D,
    dimension: ImageDimension,
    mips: NonZeroU32,
    array_layers: NonZeroU32,
    samples: SampleCount,
    format: ImageFormat,
}

/// An image view is a view over a portion of an image
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ImageView {
    owner: Image,
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
pub struct BindingSet {
    id: u64,
}

#[derive(Clone)]
pub struct Swapchain {
    id: u64,
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

impl<'a> GraphicsPipelineDescription<'a> {
    pub fn new(label: Option<&'a str>, vertex_stage: &'a VertexStageInfo) -> Self {
        Self {
            label,
            vertex_stage,
            fragment_stage: None,
            binding_sets: &[],
            primitive_restart_enabled: Default::default(),
            primitive_topology: Default::default(),
            polygon_mode: Default::default(),
            cull_mode: Default::default(),
            front_face: Default::default(),
            multisample_state: Default::default(),
            depth_stencil_state: Default::default(),
        }
    }

    pub fn fragment_stage(mut self, fragment_stage: &'a FragmentStageInfo) -> Self {
        self.fragment_stage = Some(fragment_stage);
        self
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
