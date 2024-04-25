use crate::hal::{
    Hal, QueueType, ResourceInfo, SubmissionGroup, SubmitInfo, SynchronizationInfo,
};
use crate::rdg::{Node, Rdg, Step};
use crate::staging_buffer_allocator::StagingBufferAllocator;
use crate::util::check;
use crate::DrawType::Draw;
use crate::{
    hal, BindingSet, BindingSetDescription, BindingSetLayout, BlitParams, Buffer, BufferDescription, BufferWriteParams, CommandRecorder, CommandRecorderType, DrawCommand, Extents3D, FilterMode, GraphicsPipeline, GraphicsPipelineDescription, Image, ImageAspect, ImageDescription, ImageDimension, ImageUsageFlags, ImageView, ImageViewDescription, MemoryDomain, MgpuResult, Sampler, SamplerDescription, ShaderModule, ShaderModuleDescription, ShaderModuleLayout
};
use crate::{BufferUsageFlags, ImageWriteParams};
use bitflags::bitflags;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::ops::DerefMut;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

#[cfg(feature = "swapchain")]
use crate::swapchain::*;

bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
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

    pub(crate) rdg: Arc<Mutex<Rdg>>,
    pub(crate) staging_buffer_allocator: Arc<Mutex<StagingBufferAllocator>>,

    #[cfg(feature = "swapchain")]
    pub(crate) presentation_requests: Arc<RwLock<Vec<PresentationRequest>>>,
}

impl Device {
    const MB_128: usize = 1024 * 1024 * 128;
    pub fn new(configuration: DeviceConfiguration) -> MgpuResult<Self> {
        let hal = hal::create(&configuration)?;
        let device_info = hal.device_info();
        let staging_buffer_allocator = StagingBufferAllocator::new(hal.as_ref(), Self::MB_128, configuration.desired_frames_in_flight)?;
        unsafe { hal.prepare_next_frame() }?;
        Ok(Self {
            hal,
            device_info,
            rdg: Default::default(),
            staging_buffer_allocator: Arc::new(Mutex::new(staging_buffer_allocator)),
        
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
                if !passes.graphics_passes.is_empty() {
                    let command_recorder = unsafe {
                        self.hal
                            .request_command_recorder(queues.graphics_compute_allocator)?
                    };
                    graphics_command_recorders.push(command_recorder);
                }
                if !passes.compute_passes.is_empty() {
                    let command_recorder = unsafe {
                        self.hal
                            .request_command_recorder(queues.async_compute_allocator)?
                    };
                    async_compute_command_recorders.push(command_recorder);
                }

                if !passes.transfer_passes.is_empty() {
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
                            self.hal.transition_resources(command_recorder, &pass.prequisites)?;
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

                                                if let Some(index_buffer) = index_buffer {
                                                    self.hal.set_index_buffer(command_recorder, *index_buffer)?;
                                                }

                                                if !binding_sets.is_empty() {
                                                    self.hal.set_binding_sets(
                                                        command_recorder,
                                                        binding_sets,
                                                        *pipeline,
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
                                                    crate::DrawType::DrawIndexed { indices, instances, first_index, vertex_offset, first_instance } =>  {
                                                        self.hal.draw_indexed(command_recorder, indices, instances, first_index, vertex_offset, first_instance)?;
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
                                Node::CopyBufferToImage { source, dest, source_offset, dest_region } => unsafe {
                                    self.hal.cmd_copy_buffer_to_image(
                                        command_recorder,
                                        *source,
                                        *dest,
                                        *source_offset,
                                        *dest_region,
                                    )?
                                }
                                Node::Blit { source, source_region, dest, dest_region, filter } => unsafe {
                                    self.hal.cmd_blit_image(command_recorder, *source, *source_region, *dest, *dest_region, *filter)?;
                                },
                            }
                        }

                        current_graphics += 1;
                    }
                    if !passes.compute_passes.is_empty() {
                        let command_recorder = async_compute_command_recorders[current_compute];
                        current_submission_group
                            .command_recorders
                            .push(command_recorder);
                        for pass in &passes.compute_passes {
                        self.hal.transition_resources(command_recorder, &pass.prequisites)?;
                            todo!();
                        }
                        current_compute += 1;
                    }
                    if !passes.transfer_passes.is_empty() {
                        let command_recorder =
                            async_transfer_command_recorders[current_transfer];
                        current_submission_group
                            .command_recorders
                            .push(command_recorder);

                        for pass in &passes.transfer_passes {
                        self.hal.transition_resources(command_recorder, &pass.prequisites)?;
                            match &compiled.nodes[pass.node_id].ty {
                                Node::RenderPass {..} => unreachable!(),
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
                                Node::CopyBufferToImage { source, dest, source_offset, dest_region } => unsafe {
                                    self.hal.cmd_copy_buffer_to_image(
                                        command_recorder,
                                        *source,
                                        *dest,
                                        *source_offset,
                                        *dest_region,
                                    )?
                                }
                                Node::Blit {..} => {
                                   unreachable!()
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

        self.staging_buffer_allocator.lock().expect("Failed to lock staging buffer").end_frame();

        unsafe { self.hal.prepare_next_frame()? };

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

    pub fn write_buffer_immediate(
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
        let allocation = allocator.allocate_staging_buffer_region(params.size)?;
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

    pub fn write_image_data(&self, image: Image, params: &ImageWriteParams) -> MgpuResult<()> {
        #[cfg(debug_assertions)]
        self.validate_image_write_params(image, params);

        let size = params.region.extents.area() as usize * image.format.byte_size();
        let mut allocator = self
            .staging_buffer_allocator
            .lock()
            .expect("Failed to lock staging buffer allocator");
        let allocation = allocator.allocate_staging_buffer_region( size)?;
        unsafe {
            self.hal.write_host_visible_buffer(
                allocation.buffer,
                &BufferWriteParams {
                    data: params.data,
                    offset: allocation.offset,
                    size
                },
            )
        }?;

        self.write_rdg()
            .add_async_transfer_node(Node::CopyBufferToImage { 
                source: allocation.buffer,
                dest: image,
                source_offset: allocation.offset,
                dest_region: params.region
                
            });

        Ok(())
    }

    pub fn blit_image(&self, params: &BlitParams) -> MgpuResult<()> {
        Self::validate_blit_params(params);
        let mut rdg = self.write_rdg();
        let BlitParams { src_image, src_region, dst_image, dst_region, filter } = *params;
        rdg.add_graphics_pass(Node::Blit {
            source: src_image,
            source_region: src_region,
            dest: dst_image,
            dest_region: dst_region,
            filter
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
    
    pub fn create_binding_set(&self, description: &BindingSetDescription, layout: &BindingSetLayout) -> MgpuResult<BindingSet> {
        #[cfg(debug_assertions)]
        Self::validate_bdinging_set_description(description, layout);
        self.hal.create_binding_set(description, layout)
    }

    pub fn destroy_binding_set(&self, binding_set: BindingSet) -> MgpuResult<()> {
        self.hal.destroy_binding_set(binding_set)
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
    
    pub fn generate_mip_chain(&self, image: Image) -> MgpuResult<()> {
        for mip_level in 1..image.num_mips.get() {
            let source = mip_level - 1;
            let operation = BlitParams {
                src_image: image,
                src_region: image.mip_region(source),
                dst_image: image,
                dst_region: image.mip_region(mip_level),
                filter: FilterMode::Nearest,
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
        }
    }
    

    pub(crate) fn write_rdg(&self) -> MutexGuard<'_, Rdg> {
        self.rdg.lock().expect("Failed to lock rdg")
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

    fn validate_image_view_description(_image_view_description: &ImageViewDescription) {
    }

    fn validate_buffer_description(buffer_description: &BufferDescription) {
        check!(
            buffer_description.size > 0,
            "Cannot create a buffer with size 0!"
        );
    }

    fn validate_image_write_params(&self, image: Image, params: &ImageWriteParams) {
        let total_image_texels = image.extents.area();
        let texel_byte_size = image.format.byte_size();
        let total_bytes = total_image_texels as usize * texel_byte_size;

        check!(params.data.len() >= total_bytes, 
            &format!("Attempted to execute a write operation without enough source data, expected at least {total_bytes} bytes, got {}", params.data.len()));
        check!(params.region.extents.area() > 0, "Image write region is empty");

        check!(params.region.offset.z + params.region.extents.depth as i32 <= image.extents.depth as i32, "Image write region goes beyond the image");
        check!(params.region.offset.x + params.region.extents.width as i32 <= image.extents.width as i32, "Image write region goes beyond the image");
        check!(params.region.offset.y + params.region.extents.height as i32 <= image.extents.height as i32, "Image write region goes beyond the image");
        check!(params.region.mip < image.num_mips.get(), "Trying to write image mip {}, but the image has only {} mips", params.region.mip, image.num_mips,);
        check!(params.region.base_array_layer < params.region.num_layers.get(), "base_array_layer must be <= num_layers");
        check!(params.region.base_array_layer + params.region.num_layers.get() <= image.array_layers.get(), "Trying to write more image layers than supported by the image");
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

    fn validate_sampler_description(
        &self,
        _sampler_description: &SamplerDescription,
    ) -> MgpuResult<()> {

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

        if let Some(fs) = &graphics_pipeline_description.fragment_stage {
            if let Some(ds) = &fs.depth_stencil_target {
                check!(ds.format.aspect() == ImageAspect::Depth, "Depth stencil target format isn't a valid depth format");
            }
        }

        Ok(())
    }
    
    fn validate_bdinging_set_description(description: &BindingSetDescription, layout: &BindingSetLayout) {
        for layout_binding in &layout.binding_set_elements {
            let description_binding = description.bindings.iter().find(|b| b.binding == layout_binding.binding);
            check!(description_binding.is_some(), "Layout defines a binding at index {} of type {:?}, but none was found in the description", layout_binding.binding, layout_binding.ty);

            let description_binding = description_binding.unwrap();

            check!(layout_binding.ty == description_binding.ty.binding_type(), "eLayout binding and description binding differs in type, got {:?} expected {:?}", description_binding.ty, layout_binding.ty);
        }
    }


fn validate_blit_params(params: &BlitParams) {
    check!(params.src_image.usage_flags.contains(ImageUsageFlags::TRANSFER_SRC), "Cannot blit from an image that doesn't have the TRANSFER_SRC flag");
    check!(params.dst_image.usage_flags.contains(ImageUsageFlags::TRANSFER_DST), "Cannot blit to an image that doesn't have the TRANSFER_DST flag");

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
