use core::panic;
use std::{ffi::CString, ops::Deref};

use ash::vk::{RenderingAttachmentInfoKHR, RenderingFlags, RenderingInfoKHR, ResolveModeFlags};
use ash::{
    extensions::ext::DebugUtils,
    prelude::VkResult,
    vk::{
        self, ClearDepthStencilValue, CommandBufferAllocateInfo, CommandBufferBeginInfo,
        CommandBufferLevel, CommandBufferUsageFlags, DebugUtilsLabelEXT, DependencyFlags,
        IndexType, PipelineBindPoint, StructureType, SubmitInfo,
    },
    RawPtr,
};

use crate::pipeline::GpuPipeline;
use crate::types::ImageLayout;
use crate::{
    AccessFlags, ComputePipeline, GPUFence, GPUSemaphore, GpuImage, GpuImageView, ImageAspectFlags,
    Offset2D, PipelineStageFlags, Rect2D, ToVk,
};

use super::{Gpu, GpuBuffer, GpuDescriptorSet, GraphicsPipeline, QueueType};

#[derive(Default)]
pub struct CommandBufferSubmitInfo<'a> {
    pub wait_semaphores: &'a [&'a GPUSemaphore],
    pub wait_stages: &'a [PipelineStageFlags],
    pub signal_semaphores: &'a [&'a GPUSemaphore],
    pub fence: Option<&'a GPUFence>,
}

pub struct CommandBuffer<'g> {
    gpu: &'g Gpu,
    inner_command_buffer: vk::CommandBuffer,
    has_recorded_anything: bool,
    has_been_submitted: bool,
    target_queue: vk::Queue,
}

pub struct RenderPassCommand<'c, 'g>
where
    'g: 'c,
{
    command_buffer: &'c mut CommandBuffer<'g>,
    viewport_area: Option<Viewport>,
    scissor_area: Option<Rect2D>,
    has_draw_command: bool,
    render_area: Rect2D,
    depth_bias_setup: Option<(f32, f32, f32)>,
}

pub struct ComputePassCommand<'c, 'g>
where
    'g: 'c,
{
    command_buffer: &'c mut CommandBuffer<'g>,
}

pub struct MemoryBarrier {
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
}

impl ToVk for MemoryBarrier {
    type Inner = vk::MemoryBarrier;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            s_type: StructureType::MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask: self.src_access_mask.to_vk(),
            dst_access_mask: self.dst_access_mask.to_vk(),
        }
    }
}

pub struct BufferMemoryBarrier<'a> {
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: &'a GpuBuffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

impl<'a> ToVk for BufferMemoryBarrier<'a> {
    type Inner = vk::BufferMemoryBarrier;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            s_type: StructureType::BUFFER_MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask: self.src_access_mask.to_vk(),
            dst_access_mask: self.dst_access_mask.to_vk(),
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            buffer: self.buffer.inner,
            offset: self.offset,
            size: self.size,
        }
    }
}

pub struct ImageSubresourceRange {
    pub aspect_mask: ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl ToVk for ImageSubresourceRange {
    type Inner = vk::ImageSubresourceRange;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            aspect_mask: self.aspect_mask.to_vk(),
            base_mip_level: self.base_mip_level,
            level_count: self.level_count,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        }
    }
}

pub struct ImageMemoryBarrier<'a> {
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
    pub old_layout: ImageLayout,
    pub new_layout: ImageLayout,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: &'a GpuImage,
    pub subresource_range: ImageSubresourceRange,
}

impl<'a> ToVk for ImageMemoryBarrier<'a> {
    type Inner = vk::ImageMemoryBarrier;

    fn to_vk(&self) -> Self::Inner {
        Self::Inner {
            s_type: StructureType::IMAGE_MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask: self.src_access_mask.to_vk(),
            dst_access_mask: self.dst_access_mask.to_vk(),
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            old_layout: self.old_layout.to_vk(),
            new_layout: self.new_layout.to_vk(),
            image: self.image.inner,
            subresource_range: self.subresource_range.to_vk(),
        }
    }
}

#[derive(Default)]
pub struct PipelineBarrierInfo<'a> {
    pub src_stage_mask: PipelineStageFlags,
    pub dst_stage_mask: PipelineStageFlags,
    pub memory_barriers: &'a [MemoryBarrier],
    pub buffer_memory_barriers: &'a [BufferMemoryBarrier<'a>],
    pub image_memory_barriers: &'a [ImageMemoryBarrier<'a>],
}

mod inner {
    use ash::vk::ShaderStageFlags;

    pub(super) fn push_constant<T: Copy + Sized>(
        command_buffer: &crate::CommandBuffer,
        pipeline: &crate::GraphicsPipeline,
        data: &T,
        offset: u32,
    ) {
        let device = command_buffer.gpu.vk_logical_device();
        unsafe {
            let ptr: *const u8 = data as *const T as *const u8;
            let slice = std::slice::from_raw_parts(ptr, std::mem::size_of::<T>());
            device.cmd_push_constants(
                command_buffer.inner_command_buffer,
                pipeline.pipeline_layout,
                ShaderStageFlags::ALL,
                offset,
                slice,
            );
        }
    }
}

impl<'g> CommandBuffer<'g> {
    pub fn new(gpu: &'g Gpu, target_queue: QueueType) -> VkResult<Self> {
        let device = gpu.vk_logical_device();
        let inner_command_buffer = unsafe {
            device.allocate_command_buffers(&CommandBufferAllocateInfo {
                s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                command_pool: target_queue.get_vk_command_pool(gpu),
                level: CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
            })
        }?[0];

        unsafe {
            device.begin_command_buffer(
                inner_command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )
        }?;

        Ok(Self {
            gpu,
            inner_command_buffer,
            has_recorded_anything: false,
            has_been_submitted: false,
            target_queue: target_queue.get_vk_queue(gpu),
        })
    }
    pub fn begin_render_pass<'p>(
        &'p mut self,
        info: &BeginRenderPassInfo<'p>,
    ) -> RenderPassCommand<'p, 'g> {
        RenderPassCommand::<'p, 'g>::new(self, info)
    }

    pub fn begin_compute_pass<'p>(&'p mut self) -> ComputePassCommand<'p, 'g> {
        ComputePassCommand::<'p, 'g>::new(self)
    }

    pub fn pipeline_barrier(&mut self, barrier_info: &PipelineBarrierInfo) {
        self.has_recorded_anything = true;
        let device = self.gpu.vk_logical_device();
        let memory_barriers: Vec<_> = barrier_info
            .memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        let buffer_memory_barriers: Vec<_> = barrier_info
            .buffer_memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        let image_memory_barriers: Vec<_> = barrier_info
            .image_memory_barriers
            .iter()
            .map(|b| b.to_vk())
            .collect();
        unsafe {
            device.cmd_pipeline_barrier(
                self.inner_command_buffer,
                barrier_info.src_stage_mask.to_vk(),
                barrier_info.dst_stage_mask.to_vk(),
                DependencyFlags::empty(),
                &memory_barriers,
                &buffer_memory_barriers,
                &image_memory_barriers,
            )
        };
    }

    pub fn bind_descriptor_sets<T: GpuPipeline>(
        &self,
        pipeline: &T,
        first_index: u32,
        descriptor_sets: &[&GpuDescriptorSet],
    ) {
        let descriptor_sets: Vec<_> = descriptor_sets
            .iter()
            .map(|d| d.allocation.descriptor_set)
            .collect();
        unsafe {
            self.gpu.vk_logical_device().cmd_bind_descriptor_sets(
                self.inner_command_buffer,
                T::bind_point().to_vk(),
                pipeline.vk_pipeline_layout(),
                first_index,
                &descriptor_sets,
                &[],
            );
        }
    }

    pub fn submit(mut self, submit_info: &CommandBufferSubmitInfo) -> VkResult<()> {
        self.has_been_submitted = true;
        if !self.has_recorded_anything {
            return Ok(());
        }

        let device = self.gpu.vk_logical_device();
        unsafe {
            device
                .end_command_buffer(self.inner())
                .expect("Failed to end inner command buffer");
            let target_queue = self.target_queue;

            let wait_semaphores: Vec<_> = submit_info
                .wait_semaphores
                .iter()
                .map(|s| s.inner)
                .collect();

            let signal_semaphores: Vec<_> = submit_info
                .signal_semaphores
                .iter()
                .map(|s| s.inner)
                .collect();

            let stage_masks: Vec<_> = submit_info.wait_stages.iter().map(|v| v.to_vk()).collect();

            device.queue_submit(
                target_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: wait_semaphores.len() as _,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    p_wait_dst_stage_mask: stage_masks.as_ptr(),
                    command_buffer_count: 1,
                    p_command_buffers: [self.inner_command_buffer].as_ptr(),
                    signal_semaphore_count: signal_semaphores.len() as _,
                    p_signal_semaphores: signal_semaphores.as_ptr(),
                }],
                if let Some(fence) = &submit_info.fence {
                    fence.inner
                } else {
                    vk::Fence::null()
                },
            )
        }
    }

    pub fn inner(&self) -> vk::CommandBuffer {
        self.inner_command_buffer
    }
}

// Debug utilities

pub struct ScopedDebugLabelInner {
    debug_utils: DebugUtils,
    command_buffer: vk::CommandBuffer,
}

impl ScopedDebugLabelInner {
    fn new(
        label: &str,
        color: [f32; 4],
        debug_utils: DebugUtils,
        command_buffer: vk::CommandBuffer,
    ) -> Self {
        unsafe {
            let c_label = CString::new(label).unwrap();
            debug_utils.cmd_begin_debug_utils_label(
                command_buffer,
                &DebugUtilsLabelEXT {
                    s_type: StructureType::DEBUG_UTILS_LABEL_EXT,
                    p_next: std::ptr::null(),
                    p_label_name: c_label.as_ptr(),
                    color,
                },
            );
        }
        Self {
            debug_utils,
            command_buffer,
        }
    }
    fn end(&self) {
        unsafe {
            self.debug_utils
                .cmd_end_debug_utils_label(self.command_buffer);
        }
    }
}

pub struct ScopedDebugLabel {
    inner: Option<ScopedDebugLabelInner>,
}

impl ScopedDebugLabel {
    pub fn end(mut self) {
        if let Some(label) = self.inner.take() {
            label.end();
        }
    }
}

impl Drop for ScopedDebugLabel {
    fn drop(&mut self) {
        if let Some(label) = self.inner.take() {
            label.end();
        }
    }
}

impl<'g> CommandBuffer<'g> {
    pub fn begin_debug_region(&self, label: &str, color: [f32; 4]) -> ScopedDebugLabel {
        ScopedDebugLabel {
            inner: self.gpu.state.debug_utilities.as_ref().map(|debug_utils| {
                ScopedDebugLabelInner::new(label, color, debug_utils.clone(), self.inner())
            }),
        }
    }

    pub fn insert_debug_label(&self, label: &str, color: [f32; 4]) {
        if let Some(debug_utils) = &self.gpu.state.debug_utilities {
            unsafe {
                let c_label = CString::new(label).unwrap();
                debug_utils.cmd_insert_debug_utils_label(
                    self.inner(),
                    &DebugUtilsLabelEXT {
                        s_type: StructureType::DEBUG_UTILS_LABEL_EXT,
                        p_next: std::ptr::null(),
                        p_label_name: c_label.as_ptr(),
                        color,
                    },
                );
            }
        }
    }
}

impl<'g> Drop for CommandBuffer<'g> {
    fn drop(&mut self) {
        if !self.has_been_submitted {
            panic!("CommandBuffer::submit hasn't been called!");
        }
    }
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

impl ToVk for ColorLoadOp {
    type Inner = vk::AttachmentLoadOp;

    fn to_vk(&self) -> Self::Inner {
        use ColorLoadOp::{Clear, DontCare, Load};
        match self {
            DontCare => Self::Inner::DONT_CARE,
            Load => Self::Inner::LOAD,
            Clear(_) => Self::Inner::CLEAR,
        }
    }
}

impl ToVk for DepthLoadOp {
    type Inner = vk::AttachmentLoadOp;

    fn to_vk(&self) -> Self::Inner {
        use DepthLoadOp::{Clear, DontCare, Load};
        match self {
            DontCare => Self::Inner::DONT_CARE,
            Load => Self::Inner::LOAD,
            Clear(_) => Self::Inner::CLEAR,
        }
    }
}

impl ToVk for StencilLoadOp {
    type Inner = vk::AttachmentLoadOp;

    fn to_vk(&self) -> Self::Inner {
        use StencilLoadOp::{Clear, DontCare, Load};
        match self {
            DontCare => Self::Inner::DONT_CARE,
            Load => Self::Inner::LOAD,
            Clear(_) => Self::Inner::CLEAR,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum AttachmentStoreOp {
    #[default]
    DontCare,
    Store,
}

impl ToVk for AttachmentStoreOp {
    type Inner = vk::AttachmentStoreOp;

    fn to_vk(&self) -> Self::Inner {
        match self {
            AttachmentStoreOp::DontCare => Self::Inner::DONT_CARE,
            AttachmentStoreOp::Store => Self::Inner::STORE,
        }
    }
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

impl ToVk for Viewport {
    type Inner = vk::Viewport;

    fn to_vk(&self) -> Self::Inner {
        vk::Viewport {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ColorAttachment<'a> {
    pub image_view: &'a GpuImageView,
    pub load_op: ColorLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
}

#[derive(Clone, Copy)]
pub struct DepthAttachment<'a> {
    pub image_view: &'a GpuImageView,
    pub load_op: DepthLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
}

#[derive(Clone, Copy)]
pub struct StencilAttachment<'a> {
    pub image_view: &'a GpuImageView,
    pub load_op: StencilLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
}

#[derive(Clone, Copy)]
pub struct BeginRenderPassInfo<'a> {
    pub color_attachments: &'a [ColorAttachment<'a>],
    pub depth_attachment: Option<DepthAttachment<'a>>,
    pub stencil_attachment: Option<StencilAttachment<'a>>,
    pub render_area: Rect2D,
}

impl<'c, 'g> RenderPassCommand<'c, 'g> {
    fn new(command_buffer: &'c mut CommandBuffer<'g>, info: &BeginRenderPassInfo<'c>) -> Self {
        let color_attachments: Vec<_> = info
            .color_attachments
            .iter()
            .map(|attch| RenderingAttachmentInfoKHR {
                s_type: StructureType::RENDERING_ATTACHMENT_INFO,
                p_next: std::ptr::null(),
                image_view: attch.image_view.inner,
                image_layout: attch.initial_layout.to_vk(),
                resolve_mode: ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(),
                resolve_image_layout: ImageLayout::Undefined.to_vk(),
                load_op: attch.load_op.to_vk(),
                store_op: attch.store_op.to_vk(),
                clear_value: match attch.load_op {
                    ColorLoadOp::Clear(color) => ash::vk::ClearValue {
                        color: ash::vk::ClearColorValue { float32: color },
                    },
                    _ => ash::vk::ClearValue::default(),
                },
            })
            .collect();

        let depth_attachment = info
            .depth_attachment
            .map(|attch| RenderingAttachmentInfoKHR {
                s_type: StructureType::RENDERING_ATTACHMENT_INFO,
                p_next: std::ptr::null(),
                image_view: attch.image_view.inner,
                image_layout: attch.initial_layout.to_vk(),
                resolve_mode: ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(),
                resolve_image_layout: ImageLayout::Undefined.to_vk(),
                load_op: attch.load_op.to_vk(),
                store_op: attch.store_op.to_vk(),
                clear_value: match attch.load_op {
                    DepthLoadOp::Clear(d) => ash::vk::ClearValue {
                        depth_stencil: ClearDepthStencilValue {
                            depth: d,
                            stencil: 255,
                        },
                    },
                    _ => ash::vk::ClearValue::default(),
                },
            });

        let stencil_attachment = info
            .stencil_attachment
            .map(|attch| RenderingAttachmentInfoKHR {
                s_type: StructureType::RENDERING_ATTACHMENT_INFO,
                p_next: std::ptr::null(),
                image_view: attch.image_view.inner,
                image_layout: attch.initial_layout.to_vk(),
                resolve_mode: ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(),
                resolve_image_layout: ImageLayout::Undefined.to_vk(),
                load_op: attch.load_op.to_vk(),
                store_op: attch.store_op.to_vk(),
                clear_value: match attch.load_op {
                    StencilLoadOp::Clear(s) => ash::vk::ClearValue {
                        depth_stencil: ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: s as _,
                        },
                    },
                    _ => ash::vk::ClearValue::default(),
                },
            });

        let create_info = RenderingInfoKHR {
            s_type: StructureType::RENDERING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: RenderingFlags::empty(),
            layer_count: 1,
            view_mask: 0,
            render_area: info.render_area.to_vk(),
            color_attachment_count: color_attachments.len() as _,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: depth_attachment.as_ref().as_raw_ptr(),
            p_stencil_attachment: stencil_attachment.as_ref().as_raw_ptr(),
        };
        unsafe {
            command_buffer
                .gpu
                .state
                .dynamic_rendering
                .cmd_begin_rendering(command_buffer.inner_command_buffer, &create_info);
        };

        Self {
            command_buffer,
            has_draw_command: false,
            viewport_area: None,
            scissor_area: None,
            render_area: info.render_area,
            depth_bias_setup: None,
        }
    }

    pub fn bind_pipeline(&mut self, material: &GraphicsPipeline) {
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_bind_pipeline(
                self.command_buffer.inner_command_buffer,
                PipelineBindPoint::GRAPHICS,
                material.pipeline,
            )
        }
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.prepare_draw();
        self.has_draw_command = true;
        self.command_buffer.has_recorded_anything = true;
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_draw_indexed(
                self.command_buffer.inner(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.prepare_draw();
        self.has_draw_command = true;
        self.command_buffer.has_recorded_anything = true;
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_draw(
                self.command_buffer.inner(),
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport_area = Some(viewport);
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.depth_bias_setup = Some((constant, clamp, slope));
    }

    fn prepare_draw(&self) {
        let device = self.command_buffer.gpu.vk_logical_device();

        // Negate height because of Khronos brain farts
        let height = self.render_area.extent.height as f32;
        let viewport = match self.viewport_area {
            Some(viewport) => viewport,
            None => Viewport {
                x: 0 as f32,
                y: 0.0,
                width: self.render_area.extent.width as f32,
                height,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        };
        let scissor = match self.scissor_area {
            Some(scissor) => scissor,
            None => Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: self.render_area.extent,
            },
        };
        let (depth_constant, depth_clamp, depth_slope) = match self.depth_bias_setup {
            Some((depth_constant, depth_clamp, depth_slope)) => {
                (depth_constant, depth_clamp, depth_slope)
            }
            _ => (0.0, 0.0, 0.0),
        };
        unsafe {
            device.cmd_set_depth_bias_enable(self.command_buffer.inner(), true);
            device.cmd_set_depth_bias(
                self.command_buffer.inner(),
                depth_constant,
                depth_clamp,
                depth_slope,
            );
            device.cmd_set_viewport(self.command_buffer.inner(), 0, &[viewport.to_vk()]);
            device.cmd_set_scissor(self.command_buffer.inner(), 0, &[scissor.to_vk()]);
        }
    }

    pub fn bind_index_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: vk::DeviceSize,
        index_type: IndexType,
    ) {
        let device = self.command_buffer.gpu.vk_logical_device();
        let index_buffer = buffer.inner;
        unsafe {
            device.cmd_bind_index_buffer(
                self.command_buffer.inner_command_buffer,
                index_buffer,
                offset,
                index_type,
            );
        }
    }
    pub fn bind_vertex_buffer(
        &self,
        first_binding: u32,
        buffers: &[&GpuBuffer],
        offsets: &[vk::DeviceSize],
    ) {
        assert!(buffers.len() == offsets.len());
        let device = self.command_buffer.gpu.vk_logical_device();
        let vertex_buffers: Vec<_> = buffers.iter().map(|b| b.inner).collect();
        unsafe {
            device.cmd_bind_vertex_buffers(
                self.command_buffer.inner_command_buffer,
                first_binding,
                &vertex_buffers,
                offsets,
            );
        }
    }

    pub fn push_constant<T: Copy + Sized>(
        &self,
        pipeline: &GraphicsPipeline,
        data: &T,
        offset: u32,
    ) {
        inner::push_constant(self, pipeline, data, offset)
    }
}

impl<'c, 'g> AsRef<CommandBuffer<'g>> for RenderPassCommand<'c, 'g> {
    fn as_ref(&self) -> &CommandBuffer<'g> {
        self.command_buffer
    }
}

impl<'c, 'g> Deref for RenderPassCommand<'c, 'g> {
    type Target = CommandBuffer<'g>;

    fn deref(&self) -> &Self::Target {
        self.command_buffer
    }
}

impl<'c, 'g> Drop for RenderPassCommand<'c, 'g> {
    fn drop(&mut self) {
        unsafe {
            self.command_buffer
                .gpu
                .state
                .dynamic_rendering
                .cmd_end_rendering(self.command_buffer.inner_command_buffer)
        };
    }
}

impl<'c, 'g> ComputePassCommand<'c, 'g> {
    pub fn new(command_buffer: &'c mut CommandBuffer<'g>) -> Self {
        Self { command_buffer }
    }

    pub fn bind_pipeline(&mut self, pipeline: &ComputePipeline) {
        let device = self.command_buffer.gpu.vk_logical_device();
        unsafe {
            device.cmd_bind_pipeline(
                self.command_buffer.inner_command_buffer,
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            )
        }
    }

    pub fn dispatch(&mut self, group_size_x: u32, group_size_y: u32, group_size_z: u32) {
        unsafe {
            self.command_buffer.gpu.vk_logical_device().cmd_dispatch(
                self.command_buffer.inner_command_buffer,
                group_size_x,
                group_size_y,
                group_size_z,
            );
        }
        self.command_buffer.has_recorded_anything = true;
    }
}

impl<'c, 'g> AsRef<CommandBuffer<'g>> for ComputePassCommand<'c, 'g> {
    fn as_ref(&self) -> &CommandBuffer<'g> {
        self.command_buffer
    }
}

impl<'c, 'g> Deref for ComputePassCommand<'c, 'g> {
    type Target = CommandBuffer<'g>;

    fn deref(&self) -> &Self::Target {
        self.command_buffer
    }
}
