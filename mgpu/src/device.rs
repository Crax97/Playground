use crate::hal::{
    Hal, QueueType, ResourceTransition, SubmissionGroup, SubmitInfo, SynchronizationInfo,
};
use crate::rdg::{Node, OwnershipTransfer, Rdg, Step};
use crate::staging_buffer_allocator::StagingBufferAllocator;
use crate::DrawType::Draw;
use crate::{
    hal, BindingSet, BindingSetDescription, BindingSetLayout, BlitParams, Buffer,
    BufferDescription, BufferWriteParams, CommandRecorder, CommandRecorderType, ComputePipeline,
    ComputePipelineDescription, DispatchCommand, DispatchType, DrawCommand, FilterMode,
    GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription, ImageView,
    ImageViewDescription, MemoryDomain, MgpuResult, Sampler, SamplerDescription, ShaderModule,
    ShaderModuleDescription, ShaderModuleLayout,
};
#[cfg(debug_assertions)]
use crate::{util::check, Extents3D, ImageAspect, ImageDimension, ImageUsageFlags};
use crate::{BufferUsageFlags, ImageWriteParams};
use bitflags::bitflags;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

static DUMP_RDG: AtomicBool = AtomicBool::new(false);

bitflags! {
    #[derive(Default, Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct DeviceFeatures : u32 {
        const HAL_DEBUG_LAYERS = 0b01;
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DevicePreference {
    HighPerformance,
    Software,
    AnyDevice,
}

#[derive(Debug)]
pub struct DeviceConfiguration<'a> {
    pub app_name: Option<&'a str>,
    pub features: DeviceFeatures,
    pub device_preference: Option<DevicePreference>,
    pub desired_frames_in_flight: usize,

    #[cfg(feature = "swapchain")]
    pub display_handle: Option<raw_window_handle::RawDisplayHandle>,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub api_description: String,
    pub swapchain_support: bool,
    pub frames_in_flight: usize,
}
#[derive(Clone)]
pub struct Device {
    pub(crate) device_info: DeviceInfo,

    pub(crate) rdg: Arc<Mutex<Rdg>>,
    #[allow(dead_code)]
    // This is only needed to ensure that the StagingBuffer data is destroyed with the last Device instance
    cleanup_context: Arc<DeviceCleanupContext>,
    pub(crate) staging_buffer_allocator: Arc<Mutex<StagingBufferAllocator>>,

    #[cfg(feature = "swapchain")]
    pub(crate) presentation_requests: Arc<RwLock<Vec<PresentationRequest>>>,

    pub(crate) hal: Arc<dyn Hal>,
}

struct DeviceCleanupContext {
    hal: Arc<dyn Hal>,
    staging_buffer: Buffer,
}

impl Device {
    const STAGING_BUFFER_SIZE: usize = 1024 * 1024 * 64;
    pub fn new(configuration: DeviceConfiguration) -> MgpuResult<Self> {
        let hal = hal::create(&configuration)?;
        let device_info = hal.device_info();

        let staging_buffer = hal.create_buffer(&BufferDescription {
            label: Some("Staging buffer"),
            usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::TRANSFER_SRC,
            size: Self::STAGING_BUFFER_SIZE,
            memory_domain: MemoryDomain::Cpu,
        })?;
        let staging_buffer_allocator =
            StagingBufferAllocator::new(staging_buffer, configuration.desired_frames_in_flight)?;

        let cleanup_context = DeviceCleanupContext {
            hal: hal.clone(),
            staging_buffer,
        };
        unsafe { hal.prepare_next_frame() }?;
        Ok(Self {
            hal,
            device_info,
            cleanup_context: Arc::new(cleanup_context),
            rdg: Default::default(),
            staging_buffer_allocator: Arc::new(Mutex::new(staging_buffer_allocator)),

            #[cfg(feature = "swapchain")]
            presentation_requests: Default::default(),
        })
    }

    pub fn dummy() -> Self {
        use crate::hal::dummy::DummyHal;

        let hal = Arc::new(DummyHal::default());
        let device_info = hal.device_info();

        let staging_buffer = hal
            .create_buffer(&BufferDescription {
                label: Some("Staging buffer"),
                usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::TRANSFER_SRC,
                size: Self::STAGING_BUFFER_SIZE,
                memory_domain: MemoryDomain::Cpu,
            })
            .unwrap();
        let staging_buffer_allocator = StagingBufferAllocator::new(staging_buffer, 3).unwrap();

        let cleanup_context = DeviceCleanupContext {
            hal: hal.clone(),
            staging_buffer,
        };
        unsafe { hal.prepare_next_frame() }.unwrap();
        Self {
            hal,
            device_info,
            cleanup_context: Arc::new(cleanup_context),
            rdg: Default::default(),
            staging_buffer_allocator: Arc::new(Mutex::new(staging_buffer_allocator)),

            #[cfg(feature = "swapchain")]
            presentation_requests: Default::default(),
        }
    }

    pub fn dump_rdg(&self) {
        DUMP_RDG.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn submit(&self) -> MgpuResult<()> {
        let mut staging_buffer = self.staging_buffer_allocator.lock().unwrap();
        staging_buffer.flush(self.hal.as_ref())?;

        let mut rdg = self.write_rdg();
        let compiled = rdg.compile()?;
        rdg.clear();

        if DUMP_RDG.load(std::sync::atomic::Ordering::Relaxed) {
            compiled.save_to_svg(&format!(
                "rdg_graph_{}.svg",
                std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ));
            DUMP_RDG.store(false, std::sync::atomic::Ordering::Relaxed);
        }

        let queues = unsafe { self.hal.begin_rendering()? };
        let mut graphics_command_recorders = vec![];
        let mut async_compute_command_recorders = vec![];
        let mut async_transfer_command_recorders = vec![];

        let mut submission_groups = Vec::<SubmissionGroup>::default();

        for step in &compiled.sequence {
            if let Step::ExecutePasses(passes) = step {
                if !passes.graphics_passes.is_empty() {
                    let command_recorder = unsafe {
                        self.hal
                            .request_command_recorder(queues.graphics_compute_allocator)?
                    };
                    graphics_command_recorders.push(command_recorder);
                }
                if !passes.async_compute_passes.is_empty() {
                    let command_recorder = unsafe {
                        self.hal
                            .request_command_recorder(queues.async_compute_allocator)?
                    };
                    async_compute_command_recorders.push(command_recorder);
                }

                if !passes.async_copy_passes.is_empty() {
                    let command_recorder = unsafe {
                        self.hal
                            .request_command_recorder(queues.async_transfer_allocator)?
                    };
                    async_transfer_command_recorders.push(command_recorder);
                }
            }
        }

        let mut current_graphics = 0;
        let mut current_compute = 0;
        let mut current_transfer = 0;

        let mut current_submission_group = SubmissionGroup::default();
        for step in &compiled.sequence {
            match step {
                Step::ExecutePasses(passes) => {
                    if !passes.graphics_passes.is_empty() {
                        let command_recorder = graphics_command_recorders[current_graphics];
                        current_submission_group
                            .command_recorders
                            .push(command_recorder);

                        for pass in &passes.graphics_passes {
                            let node = &compiled.nodes[pass.node_id];
                            self.hal
                                .transition_resources(command_recorder, &pass.prequisites)?;
                            match &node.ty {
                                Node::RenderPass { info } => {
                                    let render_pass_label =
                                        if let Some(label) = info.label.as_deref() {
                                            label
                                        } else {
                                            "Render Pass"
                                        };
                                    self.hal.begin_debug_region(
                                        command_recorder,
                                        render_pass_label,
                                        next_hsl_color(0.5),
                                    );
                                    unsafe { self.hal.begin_render_pass(command_recorder, info)? };
                                    for (step_idx, step) in info.steps.iter().enumerate() {
                                        for command in &step.commands {
                                            let DrawCommand {
                                                pipeline,
                                                vertex_buffers,
                                                index_buffer,
                                                binding_sets,
                                                push_constants,
                                                draw_type,
                                                scissor_rect,
                                                label,
                                            } = command;

                                            let scissor_rect =
                                                scissor_rect.unwrap_or(info.render_area);

                                            if let Some(label) = label {
                                                self.hal.begin_debug_region(
                                                    command_recorder,
                                                    label,
                                                    next_hsl_color(0.3),
                                                );
                                            }

                                            unsafe {
                                                self.hal.bind_graphics_pipeline(
                                                    command_recorder,
                                                    *pipeline,
                                                )?;
                                                self.hal.set_vertex_buffers(
                                                    command_recorder,
                                                    vertex_buffers,
                                                )?;

                                                self.hal.set_scissor_rect(
                                                    command_recorder,
                                                    scissor_rect,
                                                );

                                                if let Some(index_buffer) = index_buffer {
                                                    self.hal.set_index_buffer(
                                                        command_recorder,
                                                        *index_buffer,
                                                    )?;
                                                }

                                                if !binding_sets.is_empty() {
                                                    self.hal.bind_graphics_binding_sets(
                                                        command_recorder,
                                                        binding_sets,
                                                        *pipeline,
                                                    )?;
                                                }
                                                for pc in push_constants.iter() {
                                                    self.hal.set_graphics_push_constant(
                                                        command_recorder,
                                                        *pipeline,
                                                        &pc.data,
                                                        pc.visibility,
                                                    )?;
                                                }
                                                match *draw_type {
                                                    Draw {
                                                        vertices,
                                                        instances,
                                                        first_vertex,
                                                        first_instance,
                                                    } => {
                                                        self.hal.draw(
                                                            command_recorder,
                                                            vertices,
                                                            instances,
                                                            first_vertex,
                                                            first_instance,
                                                        )?;
                                                    }
                                                    crate::DrawType::DrawIndexed {
                                                        indices,
                                                        instances,
                                                        first_index,
                                                        vertex_offset,
                                                        first_instance,
                                                    } => {
                                                        self.hal.draw_indexed(
                                                            command_recorder,
                                                            indices,
                                                            instances,
                                                            first_index,
                                                            vertex_offset,
                                                            first_instance,
                                                        )?;
                                                    }
                                                }
                                            }

                                            if label.is_some() {
                                                self.hal.end_debug_region(command_recorder);
                                            }
                                        }

                                        if step_idx < info.steps.len() - 1 {
                                            unsafe {
                                                self.hal.advance_to_next_step(command_recorder)?
                                            }
                                        };
                                    }
                                    unsafe { self.hal.end_render_pass(command_recorder)? };
                                    self.hal.end_debug_region(command_recorder);
                                }
                                Node::CopyBufferToBuffer {
                                    source,
                                    dest,
                                    source_offset,
                                    dest_offset,
                                    size,
                                } => unsafe {
                                    self.hal.cmd_copy_buffer_to_buffer(
                                        command_recorder,
                                        *source,
                                        *dest,
                                        *source_offset,
                                        *dest_offset,
                                        *size,
                                    )?
                                },
                                Node::CopyBufferToImage {
                                    source,
                                    dest,
                                    source_offset,
                                    dest_region,
                                } => unsafe {
                                    self.hal.cmd_copy_buffer_to_image(
                                        command_recorder,
                                        *source,
                                        *dest,
                                        *source_offset,
                                        *dest_region,
                                    )?
                                },
                                Node::Blit {
                                    source,
                                    source_region,
                                    dest,
                                    dest_region,
                                    filter,
                                } => unsafe {
                                    self.hal.cmd_blit_image(
                                        command_recorder,
                                        *source,
                                        *source_region,
                                        *dest,
                                        *dest_region,
                                        *filter,
                                    )?;
                                },
                                Node::ComputePass { info } => {
                                    self.execute_compute_pass(info, command_recorder)?;
                                }
                                Node::Clear { target, color } => unsafe {
                                    self.hal.cmd_clear_image(command_recorder, *target, *color)
                                },
                            }
                        }

                        current_graphics += 1;
                    }
                    if !passes.async_compute_passes.is_empty() {
                        let command_recorder = async_compute_command_recorders[current_compute];
                        current_submission_group
                            .command_recorders
                            .push(command_recorder);
                        for pass in &passes.async_compute_passes {
                            let node = &compiled.nodes[pass.node_id];
                            self.hal
                                .transition_resources(command_recorder, &pass.prequisites)?;
                            match &node.ty {
                                Node::RenderPass { .. } => {
                                    unreachable!()
                                }
                                Node::CopyBufferToBuffer { .. } => {
                                    unreachable!()
                                }
                                Node::CopyBufferToImage { .. } => {
                                    unreachable!()
                                }
                                Node::Blit { .. } => {
                                    unreachable!()
                                }
                                Node::ComputePass { info } => {
                                    self.execute_compute_pass(info, command_recorder)?
                                }
                                Node::Clear { .. } => unreachable!(),
                            }
                        }
                        current_compute += 1;
                    }
                    if !passes.async_copy_passes.is_empty() {
                        let command_recorder = async_transfer_command_recorders[current_transfer];
                        current_submission_group
                            .command_recorders
                            .push(command_recorder);

                        for pass in &passes.async_copy_passes {
                            self.hal
                                .transition_resources(command_recorder, &pass.prequisites)?;
                            match &compiled.nodes[pass.node_id].ty {
                                Node::RenderPass { .. } => unreachable!(),
                                Node::CopyBufferToBuffer {
                                    source,
                                    dest,
                                    source_offset,
                                    dest_offset,
                                    size,
                                } => unsafe {
                                    self.hal.cmd_copy_buffer_to_buffer(
                                        command_recorder,
                                        *source,
                                        *dest,
                                        *source_offset,
                                        *dest_offset,
                                        *size,
                                    )?
                                },
                                Node::CopyBufferToImage {
                                    source,
                                    dest,
                                    source_offset,
                                    dest_region,
                                } => unsafe {
                                    self.hal.cmd_copy_buffer_to_image(
                                        command_recorder,
                                        *source,
                                        *dest,
                                        *source_offset,
                                        *dest_region,
                                    )?
                                },
                                Node::Blit { .. } => {
                                    unreachable!()
                                }
                                Node::ComputePass { .. } => unreachable!(),
                                Node::Clear { .. } => unreachable!(),
                            }
                        }
                        current_transfer += 1;
                    }
                }
                Step::OwnershipTransfer { transfers } => {
                    let submission_group = std::mem::take(&mut current_submission_group);
                    submission_groups.push(submission_group);

                    let mut resources =
                        HashMap::<(QueueType, QueueType), Vec<&OwnershipTransfer>>::default();
                    for transfer in transfers {
                        let pair = (transfer.source, transfer.destination);
                        resources.entry(pair).or_default().push(transfer)
                    }

                    let synchronization_infos = resources.into_iter().map(|((src, dest), res)| {
                        let source_command_recorder = match src {
                            QueueType::Graphics => {
                                if current_graphics > 0 {
                                    Some(graphics_command_recorders[current_graphics - 1])
                                } else {
                                    None
                                }
                            }
                            QueueType::AsyncCompute => {
                                if current_compute > 0 {
                                    Some(async_compute_command_recorders[current_compute - 1])
                                } else {
                                    None
                                }
                            }
                            QueueType::AsyncTransfer => {
                                if current_transfer > 0 {
                                    Some(async_transfer_command_recorders[current_transfer - 1])
                                } else {
                                    None
                                }
                            }
                        };

                        let destination_command_recorder = *match dest {
                            QueueType::Graphics => graphics_command_recorders.last().unwrap(),
                            QueueType::AsyncCompute => {
                                async_compute_command_recorders.last().unwrap()
                            }
                            QueueType::AsyncTransfer => {
                                async_transfer_command_recorders.last().unwrap()
                            }
                        };

                        let resources = res
                            .iter()
                            .map(|ot| ResourceTransition {
                                resource: ot.resource.resource,
                                old_usage: ot.source_usage,
                                new_usage: ot.dest_usage,
                            })
                            .collect();

                        SynchronizationInfo {
                            source_queue: src,
                            source_command_recorder,
                            destination_queue: dest,
                            destination_command_recorder,
                            resources,
                        }
                    });
                    let infos = synchronization_infos.collect::<Vec<_>>();

                    unsafe { self.hal.enqueue_synchronization(&infos)? };
                }
            }
        }

        if !current_submission_group.command_recorders.is_empty() {
            let submission_group = std::mem::take(&mut current_submission_group);
            submission_groups.push(submission_group);
        }

        for command_recorder in graphics_command_recorders
            .iter()
            .chain(async_compute_command_recorders.iter())
            .chain(async_transfer_command_recorders.iter())
        {
            unsafe { self.hal.finalize_command_recorder(*command_recorder)? };
        }

        let submit_info = SubmitInfo { submission_groups };
        unsafe { self.hal.submit(submit_info)? };

        #[cfg(feature = "swapchain")]
        {
            let presentation_requests: Vec<_> = {
                let mut lock = self
                    .presentation_requests
                    .write()
                    .expect("Failed to lock presentation requests");
                std::mem::take(lock.deref_mut())
            };

            for PresentationRequest { id, image } in presentation_requests {
                unsafe { self.hal.present_image(id, image.image)? };
            }
        }

        unsafe { self.hal.end_rendering()? };

        staging_buffer.end_frame();

        unsafe { self.hal.prepare_next_frame()? };

        Ok(())
    }

    pub fn wait_idle(&self) -> MgpuResult<()> {
        self.hal.device_wait_idle()
    }

    fn execute_compute_pass(
        &self,
        info: &crate::ComputePassInfo,
        command_recorder: hal::CommandRecorder,
    ) -> Result<(), crate::MgpuError> {
        let label = if let Some(label) = info.label.as_deref() {
            label
        } else {
            "Compute Pass"
        };
        self.hal
            .begin_debug_region(command_recorder, label, next_hsl_color(0.5));
        for step in info.steps.iter() {
            for command in &step.commands {
                let DispatchCommand {
                    pipeline,
                    binding_sets,
                    push_constants,
                    dispatch_type,
                    label,
                } = command;

                if let Some(label) = label {
                    self.hal
                        .begin_debug_region(command_recorder, label, next_hsl_color(0.3));
                }

                unsafe {
                    self.hal
                        .bind_compute_pipeline(command_recorder, *pipeline)?;

                    if let Some(pc) = &push_constants {
                        self.hal.set_compute_push_constant(
                            command_recorder,
                            *pipeline,
                            &pc.data,
                            pc.visibility,
                        )?;
                    }

                    if !binding_sets.is_empty() {
                        self.hal.bind_compute_binding_sets(
                            command_recorder,
                            binding_sets,
                            *pipeline,
                        )?;
                    }
                    match *dispatch_type {
                        DispatchType::Dispatch(gx, gy, gz) => {
                            self.hal.dispatch(command_recorder, gx, gy, gz)?;
                        }
                    }
                }

                if label.is_some() {
                    self.hal.end_debug_region(command_recorder);
                }
            }
        }

        self.hal.end_debug_region(command_recorder);
        Ok(())
    }

    pub fn get_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }

    #[cfg(feature = "swapchain")]
    pub fn create_swapchain(
        &self,
        swapchain_info: &SwapchainCreationInfo,
    ) -> MgpuResult<Swapchain> {
        let swapchain_info = self.hal.create_swapchain_impl(swapchain_info)?;
        Ok(Swapchain {
            info: swapchain_info,
            device: self.clone(),
            current_acquired_image: None,
        })
    }

    pub fn create_buffer(&self, buffer_description: &BufferDescription) -> MgpuResult<Buffer> {
        #[cfg(debug_assertions)]
        Self::validate_buffer_description(buffer_description);
        self.hal.create_buffer(buffer_description)
    }

    pub fn write_buffer_immediate(
        &self,
        buffer: Buffer,
        params: &BufferWriteParams,
    ) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_buffer_write_params(buffer, params);

        if buffer.memory_domain == MemoryDomain::Gpu {
            todo!()
        } else {
            // SAFETY: The write parameters have been validated
            unsafe { self.hal.write_host_visible_buffer(buffer, params) }
        }
    }

    pub fn write_buffer(&self, buffer: Buffer, params: &BufferWriteParams) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_buffer_write_params(buffer, params);
        let mut allocator = self
            .staging_buffer_allocator
            .lock()
            .expect("Failed to lock staging buffer allocator");
        allocator.write_buffer(
            self.hal.as_ref(),
            buffer,
            params.data,
            params.offset,
            params.size,
        )?;

        Ok(())
    }

    pub fn destroy_buffer(&self, buffer: Buffer) -> MgpuResult<()> {
        self.hal.destroy_buffer(buffer)
    }

    pub fn create_image(&self, image_description: &ImageDescription) -> MgpuResult<Image> {
        #[cfg(debug_assertions)]
        Self::validate_image_description(image_description);
        self.hal.create_image(image_description)
    }

    pub fn write_image_data(&self, image: Image, params: &ImageWriteParams) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_image_write_params(image, params);
        let mut allocator = self
            .staging_buffer_allocator
            .lock()
            .expect("Failed to lock staging buffer allocator");

        let total_size_to_write = params.region.extents.area() as usize * image.format.byte_size();

        if total_size_to_write > allocator.staging_buffer.size {
            let mut num_written = 0;
            let ausiliary_buffer = self.create_buffer(&BufferDescription {
                label: None,
                usage_flags: BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::TRANSFER_SRC,
                size: total_size_to_write,
                memory_domain: MemoryDomain::Gpu,
            })?;

            while num_written < total_size_to_write {
                let to_write_this_iteration =
                    if total_size_to_write - num_written > allocator.staging_buffer.size {
                        allocator.staging_buffer.size
                    } else {
                        total_size_to_write - num_written
                    };
                allocator.write_buffer(
                    self.hal.as_ref(),
                    ausiliary_buffer,
                    &params.data[num_written..num_written + to_write_this_iteration],
                    num_written,
                    to_write_this_iteration,
                )?;
                num_written += to_write_this_iteration;
            }

            allocator.flush(self.hal.as_ref())?;

            let one_submit_cmd_buffer = unsafe {
                self.hal
                    .request_oneshot_command_recorder(QueueType::Graphics)?
            };
            unsafe {
                self.hal.cmd_copy_buffer_to_image(
                    one_submit_cmd_buffer,
                    ausiliary_buffer,
                    image,
                    0,
                    params.region,
                )?;

                self.hal.finalize_command_recorder(one_submit_cmd_buffer)?;
                self.hal
                    .submit_command_recorder_immediate(one_submit_cmd_buffer)?;
                self.hal.device_wait_queue(QueueType::Graphics)?;
            };
        } else {
            allocator.write_image(self.hal.as_ref(), image, params.data, params.region)?;
        }

        Ok(())
    }

    pub fn blit_image(&self, params: &BlitParams) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        Self::validate_blit_params(params);
        let mut rdg = self.write_rdg();
        let BlitParams {
            src_image,
            src_region,
            dst_image,
            dst_region,
            filter,
        } = *params;
        rdg.add_graphics_pass(Node::Blit {
            source: src_image,
            source_region: src_region,
            dest: dst_image,
            dest_region: dst_region,
            filter,
        });
        Ok(())
    }

    pub fn destroy_image(&self, image: Image) -> MgpuResult<()> {
        self.hal.destroy_image(image)
    }

    pub fn create_image_view(
        &self,
        image_view_description: &ImageViewDescription,
    ) -> MgpuResult<ImageView> {
        #[cfg(debug_assertions)]
        Self::validate_image_view_description(image_view_description);
        self.hal.create_image_view(image_view_description)
    }

    pub fn create_shader_module(
        &self,
        shader_module_description: &ShaderModuleDescription,
    ) -> MgpuResult<ShaderModule> {
        #[cfg(debug_assertions)]
        self.validate_shader_module_description(shader_module_description)?;
        self.hal.create_shader_module(shader_module_description)
    }

    pub fn get_shader_module_layout(
        &self,
        shader_module: ShaderModule,
    ) -> MgpuResult<ShaderModuleLayout> {
        self.hal.get_shader_module_layout(shader_module)
    }

    pub fn create_sampler(&self, sampler_description: &SamplerDescription) -> MgpuResult<Sampler> {
        #[cfg(debug_assertions)]
        self.validate_sampler_description(sampler_description)?;
        self.hal.create_sampler(sampler_description)
    }

    pub fn destroy_sampler(&self, sampler: Sampler) -> MgpuResult<()> {
        self.hal.destroy_sampler(sampler)
    }

    pub fn create_binding_set(
        &self,
        description: &BindingSetDescription,
        layout: &BindingSetLayout,
    ) -> MgpuResult<BindingSet> {
        #[cfg(debug_assertions)]
        Self::validate_binding_set_description(description, layout);
        self.hal.create_binding_set(description, layout)
    }

    pub fn destroy_binding_set(&self, binding_set: BindingSet) -> MgpuResult<()> {
        self.hal.destroy_binding_set(binding_set)
    }

    pub fn create_graphics_pipeline(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<GraphicsPipeline> {
        #[cfg(debug_assertions)]
        self.validate_graphics_pipeline_description(graphics_pipeline_description)?;
        self.hal
            .create_graphics_pipeline(graphics_pipeline_description)
    }

    pub fn create_compute_pipeline(
        &self,
        compute_pipeline_description: &ComputePipelineDescription,
    ) -> MgpuResult<ComputePipeline> {
        #[cfg(debug_assertions)]
        self.validate_compute_pipeline_description(compute_pipeline_description)?;
        self.hal
            .create_compute_pipeline(compute_pipeline_description)
    }

    pub fn destroy_graphics_pipeline(&self, graphics_pipeline: GraphicsPipeline) -> MgpuResult<()> {
        self.hal.destroy_graphics_pipeline(graphics_pipeline)
    }

    pub fn destroy_compute_pipeline(&self, pipeline: ComputePipeline) -> MgpuResult<()> {
        self.hal.destroy_compute_pipeline(pipeline)
    }

    pub fn destroy_shader_module(&self, shader_module: ShaderModule) -> MgpuResult<()> {
        self.hal.destroy_shader_module(shader_module)
    }

    pub fn generate_mip_chain(&self, image: Image, filter: FilterMode) -> MgpuResult<()> {
        for mip_level in 1..image.num_mips.get() {
            let source = mip_level - 1;
            let operation = BlitParams {
                src_image: image,
                src_region: image.mip_region(source),
                dst_image: image,
                dst_region: image.mip_region(mip_level),
                filter,
            };
            self.blit_image(&operation)?;
        }
        Ok(())
    }

    pub fn destroy_image_view(&self, image_view: ImageView) -> MgpuResult<()> {
        self.hal.destroy_image_view(image_view)
    }

    pub fn create_command_recorder<T: CommandRecorderType>(&self) -> CommandRecorder<T> {
        CommandRecorder {
            _ph: std::marker::PhantomData,
            device: self.clone(),
            binding_sets: Default::default(),
            new_nodes: Default::default(),
            push_constants: Default::default(),
        }
    }

    pub(crate) fn write_rdg(&self) -> MutexGuard<'_, Rdg> {
        self.rdg.lock().expect("Failed to lock rdg")
    }

    #[cfg(debug_assertions)]
    fn validate_image_description(image_description: &ImageDescription) {
        use crate::ImageCreationFlags;

        check!(
            image_description.dimension != ImageDimension::D3
                || image_description.extents.depth > 0,
            "The depth of an image cannot be 0"
        );

        if image_description
            .creation_flags
            .contains(ImageCreationFlags::CUBE_COMPATIBLE)
        {
            check!(
                image_description.dimension == ImageDimension::D2,
                "If an image is CUBE_COMPATIBLE, it must be 2D"
            );
            check!(
                image_description.array_layers.get() == 6,
                "If an image is CUBE_COMPATIBLE, it must have exactly six layers"
            );
        }

        if image_description.dimension == ImageDimension::D3 {
            check!(
                image_description.array_layers.get() == 1,
                "A 3D image can only have one layer"
            );
        }

        let total_texels = image_description.extents.width
            * image_description.extents.height
            * image_description.extents.depth;
        check!(
            total_texels > 0,
            "The width, height and depth of an image cannot be 0"
        );

        match image_description.format.aspect() {
            ImageAspect::Color => {
                check!(
                    !image_description
                        .usage_flags
                        .contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                    "A color image cannot be used as a depth attachment!"
                );
            }
            ImageAspect::Depth => {
                check!(
                    !image_description
                        .usage_flags
                        .contains(ImageUsageFlags::COLOR_ATTACHMENT),
                    "A depth image cannot be used as a color attachment!"
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    fn validate_image_view_description(image_view_description: &ImageViewDescription) {
        use crate::{ImageCreationFlags, ImageViewType};

        if !image_view_description
            .image
            .creation_flags
            .contains(ImageCreationFlags::MUTABLE_FORMAT)
        {
            check!(image_view_description.format == image_view_description.image.format, "If an image was not created with the MUTABLE_FORMAT flag, creating a view whose format is different from the image's format is not allowed.");
        }

        if image_view_description.view_ty == ImageViewType::D2 {
            check!(
                image_view_description.image_subresource.num_layers.get() == 1,
                "When creating a 2D image view only one layer can be used"
            );
            check!(
                image_view_description.image.dimension == ImageDimension::D2,
                "If an image view is 2D then the image must be 2D too"
            );
        }

        if image_view_description.view_ty == ImageViewType::Cube {
            check!(
                image_view_description
                    .image
                    .creation_flags
                    .contains(ImageCreationFlags::CUBE_COMPATIBLE),
                "Cannot create an image view of an image which is not CUBE_COMPATIBLE"
            );
            check!(
                image_view_description.image_subresource.num_layers.get() == 6,
                "When creating a Cubemap the image view subresource must have 6 layers"
            );
        }
    }

    #[cfg(debug_assertions)]
    fn validate_buffer_description(buffer_description: &BufferDescription) {
        check!(
            buffer_description.size > 0,
            "Cannot create a buffer with size 0!"
        );
    }

    #[cfg(debug_assertions)]
    fn validate_image_write_params(&self, image: Image, params: &ImageWriteParams) {
        let total_texels_needed = params.region.extents.area();
        let texel_byte_size = image.format.byte_size();
        let total_bytes = total_texels_needed as usize * texel_byte_size;

        check!(params.data.len() >= total_bytes,
            &format!("Attempted to execute a write operation without enough source data, expected at least {total_bytes} bytes, got {}", params.data.len()));
        check!(
            params.region.extents.area() > 0,
            "Image write region is empty"
        );

        check!(
            params.region.offset.z + params.region.extents.depth as i32
                <= image.extents.depth as i32,
            "Image write region goes beyond the image"
        );
        check!(
            params.region.offset.x + params.region.extents.width as i32
                <= image.extents.width as i32,
            "Image write region goes beyond the image"
        );
        check!(
            params.region.offset.y + params.region.extents.height as i32
                <= image.extents.height as i32,
            "Image write region goes beyond the image"
        );
        check!(
            params.region.mip < image.num_mips.get(),
            "Trying to write image mip {}, but the image has only {} mips",
            params.region.mip,
            image.num_mips,
        );
        check!(
            params.region.base_array_layer < params.region.num_layers.get(),
            "base_array_layer must be <= num_layers"
        );
        check!(
            params.region.base_array_layer + params.region.num_layers.get()
                <= image.array_layers.get(),
            "Trying to write more image layers than supported by the image"
        );
    }

    #[cfg(debug_assertions)]
    fn validate_buffer_write_params(&self, buffer: Buffer, params: &BufferWriteParams) {
        check!(
            params.size > 0,
            "A buffer write operation cannot have data length of 0!"
        );
        let expected_data_len = params.size;
        check!(
            params.data.len() >= expected_data_len,
            &format!(
                "Not enough data in data buffer: expected {expected_data_len}, got {}",
                params.data.len()
            )
        );
        check!(
            buffer.memory_domain == MemoryDomain::Cpu ||
            buffer.usage_flags.contains(BufferUsageFlags::TRANSFER_DST),
            "If a buffer write operation is initiated, the buffer must either be host visible, or it must have the TRANSFER_DST flag"
        );
    }

    #[cfg(debug_assertions)]
    fn validate_shader_module_description(
        &self,
        shader_module_description: &ShaderModuleDescription,
    ) -> MgpuResult<()> {
        check!(!shader_module_description.source.is_empty(), "Empty source");

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn validate_sampler_description(
        &self,
        _sampler_description: &SamplerDescription,
    ) -> MgpuResult<()> {
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn validate_graphics_pipeline_description(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<()> {
        // Validate vertex inputs

        use std::collections::HashSet;
        let vertex_shader_layout =
            self.get_shader_module_layout(*graphics_pipeline_description.vertex_stage.shader)?;
        let fragment_shader_layout = graphics_pipeline_description
            .fragment_stage
            .map(|fs| self.get_shader_module_layout(*fs.shader))
            .transpose()?;
        let mut already_defined_inputs: HashSet<usize> = HashSet::default();
        for input in graphics_pipeline_description.vertex_stage.vertex_inputs {
            check!(
                !already_defined_inputs.contains(&input.location),
                "Vertex attribute {} was defined twice",
                input.location
            );
            already_defined_inputs.insert(input.location);
        }
        for input in &vertex_shader_layout.inputs {
            let pipeline_input = graphics_pipeline_description
                .vertex_stage
                .vertex_inputs
                .get(input.location);

            check!(pipeline_input.is_some(),
                &format!("Vertex shader expects input at location {} compatible with format {:?}, but the pipeline description does not provide it",
                input.location, input.format));
            let pipeline_input = pipeline_input.unwrap();
            check!(pipeline_input.format.channels() == input.format.channels(),
                &format!("Vertex shader expects at location {} input with '{}' channels (format {:?}), but pipeline provieds {}", input.location, input.format.channels(), input.format, pipeline_input.format.channels())
            )
        }

        if let Some(fs) = &graphics_pipeline_description.fragment_stage {
            if let Some(ds) = &fs.depth_stencil_target {
                check!(
                    ds.format.aspect() == ImageAspect::Depth,
                    "Depth stencil target format isn't a valid depth format"
                );
            }
        }

        let all_shader_binding_entries = vertex_shader_layout.binding_sets.iter().chain(
            fragment_shader_layout
                .iter()
                .flat_map(|fl| fl.binding_sets.iter()),
        );
        for binding_set_layout_info in graphics_pipeline_description.binding_set_layouts {
            for bs_element in binding_set_layout_info.layout.binding_set_elements {
                let matching_element = all_shader_binding_entries
                    .clone()
                    .filter(|entry| entry.set == binding_set_layout_info.set)
                    .flat_map(|shader_bs| {
                        shader_bs
                            .layout
                            .binding_set_elements
                            .iter()
                            .find(|l| l.binding == bs_element.binding)
                    })
                    .next();

                check!(
                    matching_element.is_some(),
                    "Pipeline defines an input at set {} binding {} of type {:#?}, but none was found in any shader",
                    binding_set_layout_info.set,
                    bs_element.binding,
                    bs_element
                );
                let matching_element = matching_element.unwrap();
                check!(
                    matching_element.ty == bs_element.ty,
                    "Mismatching types in set {}: expected {:?} got {:?}",
                    binding_set_layout_info.set,
                    matching_element,
                    bs_element
                );
                check!(bs_element.shader_stage_flags.contains(matching_element.shader_stage_flags), "The binding set describes an element with shader flags {:?}, which is not visibile to shader stage {:?}", bs_element.shader_stage_flags, matching_element.shader_stage_flags);
            }
        }

        if let Some(pc) = &graphics_pipeline_description.push_constant_info {
            let whole_push_constant_shader_stages =
                vertex_shader_layout.push_constant.unwrap_or_default()
                    | fragment_shader_layout
                        .as_ref()
                        .and_then(|v| v.push_constant)
                        .unwrap_or_default();
            check!(
                pc.size <= crate::MAX_PUSH_CONSTANT_RANGE_SIZE_BYTES,
                "A push constant range cannot exceed the maximum size of {} bytes",
                crate::MAX_PUSH_CONSTANT_RANGE_SIZE_BYTES
            );
            check!(pc.visibility.contains(whole_push_constant_shader_stages), "The pipeline's push constant visibility doesn't include all the shader stage where the
             push constant is used, expected {:?} got {:?}", whole_push_constant_shader_stages, pc.visibility);
        }

        if let Some(shader_layout) = &fragment_shader_layout {
            let fragment_stage = graphics_pipeline_description.fragment_stage.unwrap();
            check!(shader_layout.outputs.len() == fragment_stage.render_targets.len(), "While creating Graphics Pipeline '{}': mismatch between fragment shader outputs and render targets in fragment stage! They must match exactly, expected {} output(s), got {} render target(s)",
                graphics_pipeline_description.label.unwrap_or("Unknown"), shader_layout.outputs.len(), fragment_stage.render_targets.len());
        }
        Ok(())
    }
    #[cfg(debug_assertions)]
    fn validate_compute_pipeline_description(
        &self,
        compute_pipeline_description: &ComputePipelineDescription,
    ) -> MgpuResult<()> {
        let shader_layout = self
            .hal
            .get_shader_module_layout(compute_pipeline_description.shader)?;
        let all_shader_binding_entries = shader_layout.binding_sets.iter();
        for binding_set_layout_info in compute_pipeline_description.binding_set_layouts {
            for bs_element in binding_set_layout_info.layout.binding_set_elements {
                let matching_element = all_shader_binding_entries
                    .clone()
                    .filter(|entry| entry.set == binding_set_layout_info.set)
                    .flat_map(|shader_bs| {
                        shader_bs.layout.binding_set_elements.iter().find(|l| {
                            l.binding == bs_element.binding
                                && bs_element.shader_stage_flags.contains(l.shader_stage_flags)
                        })
                    })
                    .next();

                check!(
                    matching_element.is_some(),
                    "No matching element at set {} binding {} of type {:#?} found in any shader",
                    binding_set_layout_info.set,
                    bs_element.binding,
                    bs_element
                );

                let matching_element = matching_element.unwrap();
                check!(
                    matching_element.ty == bs_element.ty,
                    "Mismatching types in set {}: expected {:?} got {:?}",
                    binding_set_layout_info.set,
                    matching_element,
                    bs_element
                )
            }
        }

        if let Some(pc) = &compute_pipeline_description.push_constant_info {
            let whole_push_constant_shader_stages = shader_layout.push_constant.unwrap_or_default();
            check!(
                pc.size <= crate::MAX_PUSH_CONSTANT_RANGE_SIZE_BYTES,
                "A push constant range cannot exceed the maximum size of {} bytes",
                crate::MAX_PUSH_CONSTANT_RANGE_SIZE_BYTES
            );
            check!(pc.visibility.contains(whole_push_constant_shader_stages), "The pipeline's push constant visibility doesn't include all the shader stage where the
             push constant is used, expected {:?} got {:?}", whole_push_constant_shader_stages, pc.visibility);
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn validate_binding_set_description(
        description: &BindingSetDescription,
        layout: &BindingSetLayout,
    ) {
        check!(
            layout.binding_set_elements.len() == description.bindings.len(),
            "Layout expects {} elements, but {} were provided: they must match exactly",
            layout.binding_set_elements.len(),
            description.bindings.len()
        );
        // TODO: Ensure that bindings in description are present in layout
        for layout_binding in layout.binding_set_elements {
            let description_binding = description
                .bindings
                .iter()
                .find(|b| b.binding == layout_binding.binding);
            check!(description_binding.is_some(), "Layout defines a binding at index {} of type {:?}, but none was found in the description", layout_binding.binding, layout_binding.ty);

            let description_binding = description_binding.unwrap();

            check!(
                layout_binding.ty == description_binding.ty.binding_type(),
                "Layout binding and description binding differs in type, got {:?} expected {:?}",
                description_binding.ty,
                layout_binding.ty
            );
        }
    }

    #[cfg(debug_assertions)]
    pub(crate) fn validate_blit_params(params: &BlitParams) {
        check!(
            params
                .src_image
                .usage_flags
                .contains(ImageUsageFlags::TRANSFER_SRC),
            "Cannot blit from an image that doesn't have the TRANSFER_SRC flag"
        );
        check!(
            params
                .dst_image
                .usage_flags
                .contains(ImageUsageFlags::TRANSFER_DST),
            "Cannot blit to an image that doesn't have the TRANSFER_DST flag"
        );

        let mut dst_image_mip_extents = params.dst_image.extents;
        for _ in 1..params.dst_region.mip {
            dst_image_mip_extents = Extents3D {
                width: (dst_image_mip_extents.width as f32).sqrt() as u32,
                height: (dst_image_mip_extents.height as f32).sqrt() as u32,
                depth: (dst_image_mip_extents.depth as f32).sqrt() as u32,
            }
        }

        let mut src_image_mip_extents = params.src_image.extents;
        for _ in 1..params.src_region.mip {
            src_image_mip_extents = Extents3D {
                width: (src_image_mip_extents.width as f32).sqrt() as u32,
                height: (src_image_mip_extents.height as f32).sqrt() as u32,
                depth: (src_image_mip_extents.depth as f32).sqrt() as u32,
            }
        }
    }
}

impl Drop for DeviceCleanupContext {
    fn drop(&mut self) {
        self.hal.destroy_buffer(self.staging_buffer).unwrap();
    }
}

#[cfg(feature = "swapchain")]
pub(crate) struct PresentationRequest {
    pub(crate) id: u64,
    pub(crate) image: SwapchainImage,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Device name: {}\n", self.name))?;
        f.write_fmt(format_args!("Api description: {}\n", self.api_description))?;
        f.write_fmt(format_args!(
            "Supports swapchain: {}\n",
            self.swapchain_support
        ))?;
        f.write_fmt(format_args!(
            "Max frames in flight: {}\n",
            self.frames_in_flight
        ))
    }
}

fn next_hsl_color(saturation: f32) -> [f32; 3] {
    static CURRENT_HSL_COUNTER: AtomicUsize = AtomicUsize::new(0);
    hsl_color(
        CURRENT_HSL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        saturation,
    )
}

fn hsl_color(index: usize, saturation: f32) -> [f32; 3] {
    let hue_deg = (15.0 * index as f32) % 360.0;
    let lightness = 0.5f32;

    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let h1 = hue_deg / 60.0;
    let x = chroma * (1.0 - ((h1 % 2.0) - 1.0).abs());

    let (r, g, b) = if h1 <= 6.0 && h1 > 5.0 {
        (chroma, 0.0, x)
    } else if h1 <= 5.0 && h1 > 4.0 {
        (x, 0.0, chroma)
    } else if h1 <= 4.0 && h1 > 3.0 {
        (0.0, x, chroma)
    } else if h1 <= 3.0 && h1 > 2.0 {
        (0.0, chroma, x)
    } else if h1 <= 2.0 && h1 > 1.0 {
        (x, chroma, 0.0)
    } else {
        (chroma, x, 0.0)
    };

    let m = lightness - chroma * 0.5;
    [r + m, g + m, b + m]
}
