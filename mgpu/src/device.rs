use crate::hal::{
    Hal, QueueType, Resource, ResourceInfo, SubmissionGroup, SubmitInfo, SynchronizationInfo,
};
use crate::rdg::{Node, Rdg, Step};
use crate::staging_buffer_allocator::StagingBufferAllocator;
use crate::util::check;
use crate::DrawType::Draw;
use crate::{
    hal, Buffer, BufferDescription, BufferWriteParams, CommandRecorder, CommandRecorderType,
    DrawCommand, GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription,
    ImageDimension, ImageViewDescription, MemoryDomain, MgpuResult, ShaderModule,
    ShaderModuleDescription, ShaderModuleLayout,
};
use crate::{BufferUsageFlags, ImageWriteParams};
use ash::vk::{self, ImageView};
use bitflags::bitflags;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::ops::DerefMut;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct DeviceFeatures : u32 {
        const DEBUG_FEATURES = 0b01;
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
    pub display_handle: raw_window_handle::RawDisplayHandle,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub api_description: String,
    pub swapchain_support: bool,
}
#[derive(Clone)]
pub struct Device {
    pub(crate) hal: Arc<dyn Hal>,
    pub(crate) device_info: DeviceInfo,

    pub(crate) rdg: Arc<RwLock<Rdg>>,
    pub(crate) staging_buffer_allocator: Arc<Mutex<StagingBufferAllocator>>,

    #[cfg(feature = "swapchain")]
    pub(crate) presentation_requests: Arc<RwLock<Vec<PresentationRequest>>>,
}

impl Device {
    pub fn new(configuration: DeviceConfiguration) -> MgpuResult<Self> {
        let hal = hal::create(&configuration)?;
        let device_info = hal.device_info();
        Ok(Self {
            hal,
            device_info,
            rdg: Default::default(),
            staging_buffer_allocator: Default::default(),

            #[cfg(feature = "swapchain")]
            presentation_requests: Default::default(),
        })
    }

    pub fn submit(&self) -> MgpuResult<()> {
        let rdg = self.write_rdg().take();
        let compiled = rdg.compile()?;
        static DUMP: AtomicBool = AtomicBool::new(true);
        if DUMP.load(std::sync::atomic::Ordering::Relaxed) {
            println!("{}", compiled.dump_dot());
            DUMP.store(false, std::sync::atomic::Ordering::Relaxed);
        }

        let queues = unsafe { self.hal.begin_rendering()? };
        let mut graphics_command_recorders = vec![];
        let mut async_compute_command_recorders = vec![];
        let mut async_transfer_command_recorders = vec![];

        let mut submission_groups = Vec::<SubmissionGroup>::default();

        for step in &compiled.sequence {
            if let Step::ExecutePasses(passes) = step {
                if !passes.graphics_nodes.is_empty() {
                    let command_recorder = unsafe {
                        self.hal
                            .request_command_recorder(queues.graphics_compute_allocator)?
                    };
                    graphics_command_recorders.push(command_recorder);
                }
                if !passes.compute_nodes.is_empty() {
                    let command_recorder = unsafe {
                        self.hal
                            .request_command_recorder(queues.async_compute_allocator)?
                    };
                    async_compute_command_recorders.push(command_recorder);
                }

                if !passes.transfer_nodes.is_empty() {
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
                    if !passes.graphics_nodes.is_empty() {
                        let command_recorder = graphics_command_recorders[current_graphics];
                        current_submission_group
                            .command_recorders
                            .push(command_recorder);
                        for pass in &passes.graphics_nodes {
                            let node = &compiled.nodes[*pass];
                            match &node.ty {
                                Node::RenderPass { info } => {
                                    unsafe { self.hal.begin_render_pass(command_recorder, info)? };
                                    for (step_idx, step) in info.steps.iter().enumerate() {
                                        for command in &step.commands {
                                            let DrawCommand {
                                                pipeline,
                                                vertex_buffers,
                                                index_buffer,
                                                binding_sets,
                                                draw_type,
                                            } = command;

                                            unsafe {
                                                self.hal.bind_graphics_pipeline(
                                                    command_recorder,
                                                    *pipeline,
                                                )?;
                                                self.hal.set_vertex_buffers(
                                                    command_recorder,
                                                    vertex_buffers,
                                                )?;
                                                self.hal.set_binding_sets(
                                                    command_recorder,
                                                    binding_sets,
                                                )?;
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
                                                }
                                            }
                                        }

                                        if step_idx < info.steps.len() - 1 {
                                            unsafe {
                                                self.hal.advance_to_next_step(command_recorder)?
                                            }
                                        };
                                    }
                                    unsafe { self.hal.end_render_pass(command_recorder)? };
                                }
                                Node::CopyBufferToBuffer { .. } => unreachable!(),
                            }
                        }

                        current_graphics += 1;
                    }
                    if !passes.compute_nodes.is_empty() {
                        let command_recorder = async_compute_command_recorders[current_compute];
                        current_submission_group
                            .command_recorders
                            .push(command_recorder);
                        for pass in &passes.compute_nodes {
                            todo!();
                        }
                        current_compute += 1;
                    }
                    if !passes.transfer_nodes.is_empty() {
                        for pass in &passes.transfer_nodes {
                            let command_recorder =
                                async_transfer_command_recorders[current_transfer];

                            current_submission_group
                                .command_recorders
                                .push(command_recorder);
                            match &compiled.nodes[*pass].ty {
                                Node::RenderPass { info } => unreachable!(),
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
                            }
                        }
                        current_transfer += 1;
                    }
                }
                Step::OwnershipTransfer { transfers } => {
                    let submission_group = std::mem::take(&mut current_submission_group);
                    submission_groups.push(submission_group);

                    let mut resources =
                        HashMap::<(QueueType, QueueType), Vec<ResourceInfo>>::default();
                    for transfer in transfers {
                        let pair = (transfer.source, transfer.destination);
                        resources.entry(pair).or_default().push(transfer.resource)
                    }

                    let synchronization_infos = resources.into_iter().map(|((src, dest), res)| {
                        let source_command_recorder = match src {
                            QueueType::Graphics => graphics_command_recorders[current_graphics - 1],
                            QueueType::AsyncCompute => {
                                async_compute_command_recorders[current_compute - 1]
                            }
                            QueueType::AsyncTransfer => {
                                async_transfer_command_recorders[current_transfer - 1]
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

                        SynchronizationInfo {
                            source_queue: src,
                            source_command_recorder,
                            destination_queue: dest,
                            destination_command_recorder,
                            resources: res,
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
        let swapchain_id = self.hal.create_swapchain_impl(swapchain_info)?;
        Ok(Swapchain {
            device: self.clone(),
            id: swapchain_id,
            current_acquired_image: None,
        })
    }

    pub fn create_buffer(&self, buffer_description: &BufferDescription) -> MgpuResult<Buffer> {
        #[cfg(debug_assertions)]
        Self::validate_buffer_description(buffer_description);
        self.hal.create_buffer(buffer_description)
    }

    pub(crate) fn write_buffer_immediate(
        &self,
        buffer: Buffer,
        params: &BufferWriteParams,
    ) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_buffer_write_params(buffer, params);

        if buffer.memory_domain == MemoryDomain::DeviceLocal {
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
        let allocation = allocator.get_staging_buffer(self, params.size)?;
        unsafe {
            self.hal.write_host_visible_buffer(
                allocation.buffer,
                &BufferWriteParams {
                    data: params.data,
                    offset: allocation.offset,
                    size: params.size,
                },
            )
        }?;

        self.write_rdg()
            .add_async_transfer_node(Node::CopyBufferToBuffer {
                source: allocation.buffer,
                dest: buffer,
                source_offset: allocation.offset,
                dest_offset: params.offset,
                size: params.size,
            });
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

    pub fn write_image(&self, image: Image, params: &ImageWriteParams) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_image_write_params(image, params);

        todo!();

        Ok(())
    }

    pub fn destroy_image(&self, image: Image) -> MgpuResult<()> {
        self.hal.destroy_image(image)
    }

    pub fn create_image_view(
        &self,
        image_view_description: &ImageViewDescription,
    ) -> MgpuResult<ImageView> {
        todo!()
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

    pub fn create_graphics_pipeline(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<GraphicsPipeline> {
        self.validate_graphics_pipeline_description(graphics_pipeline_description)?;
        self.hal
            .create_graphics_pipeline(graphics_pipeline_description)
    }

    pub fn destroy_graphics_pipeline(&self, graphics_pipeline: GraphicsPipeline) -> MgpuResult<()> {
        self.hal.destroy_graphics_pipeline(graphics_pipeline)
    }

    pub fn destroy_shader_module(&self, shader_module: ShaderModule) -> MgpuResult<()> {
        self.hal.destroy_shader_module(shader_module)
    }

    pub fn destroy_image_view(&self, image_view: ImageView) {
        todo!()
    }

    pub fn create_command_recorder<T: CommandRecorderType>(&self) -> CommandRecorder<T> {
        CommandRecorder {
            _ph: std::marker::PhantomData,
            device: self.clone(),
            binding_sets: Default::default(),
            new_nodes: Default::default(),
        }
    }

    pub(crate) fn write_rdg(&self) -> RwLockWriteGuard<'_, Rdg> {
        self.rdg.write().expect("Failed to lock rdg")
    }

    pub(crate) fn read_rdg(&self) -> RwLockReadGuard<'_, Rdg> {
        self.rdg.read().expect("Failed to lock rdg")
    }

    fn validate_image_description(image_description: &ImageDescription) {
        check!(
            image_description.dimension != ImageDimension::D3
                || image_description.extents.depth > 0,
            "The depth of an image cannot be 0"
        );

        let total_texels = image_description.extents.width
            * image_description.extents.height
            * image_description.extents.depth;
        check!(
            total_texels > 0,
            "The width, height and depth of an image cannot be 0"
        );
    }

    fn validate_buffer_description(buffer_description: &BufferDescription) {
        check!(
            buffer_description.size > 0,
            "Cannot create a buffer with size 0!"
        );
    }

    fn validate_image_write_params(&self, image: Image, params: &ImageWriteParams) {
        let total_texels = image.extents.width * image.extents.height * image.extents.depth;
        let texel_byte_size = image.format.byte_size();
        let total_bytes = total_texels as usize * texel_byte_size;

        check!(params.data.len() >= total_bytes, 
            &format!("Attempted to execute a write operation without enough source data, expected at least {total_bytes} bytes, got {}", params.data.len()));
    }

    fn validate_buffer_write_params(&self, buffer: Buffer, params: &BufferWriteParams) {
        check!(
            params.size > 0,
            "A buffer write operation cannot have data length of 0!"
        );
        let expected_data_len = params.size - params.offset;
        check!(
            params.data.len() >= expected_data_len,
            &format!(
                "Not enough data in data buffer: expected {expected_data_len}, got {}",
                params.data.len()
            )
        );
        check!(
            buffer.memory_domain == MemoryDomain::HostVisible ||
            buffer.usage_flags.contains(BufferUsageFlags::TRANSFER_DST),
            "If a buffer write operation is initiated, the buffer must either be host visible, or it must have the TRANSFER_DST flag"
        );
    }

    fn validate_shader_module_description(
        &self,
        shader_module_description: &ShaderModuleDescription,
    ) -> MgpuResult<()> {
        check!(!shader_module_description.source.is_empty(), "Empty source");

        Ok(())
    }

    fn validate_graphics_pipeline_description(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<()> {
        // Validate vertex inputs
        let vertex_shader_layout =
            self.get_shader_module_layout(*graphics_pipeline_description.vertex_stage.shader)?;
        for input in &vertex_shader_layout.inputs {
            let pipeline_input = graphics_pipeline_description
                .vertex_stage
                .vertex_inputs
                .get(input.location);
            check!(pipeline_input.is_some(),
                &format!("Vertex shader expects input '{}' at location {} with format {:?}, but the pipeline description does not provide it",
                input.name, input.location, input.format));
            let pipeline_input = pipeline_input.unwrap();
            check!(pipeline_input.format == input.format,
                &format!("The format of vertex attribute '{}' differs from the one in the pipeline description! Expected {:?}, got {:?}", input.name, input.format, pipeline_input.format)
            )
        }

        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.hal.device_wait_idle().unwrap();
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
        ))
    }
}
