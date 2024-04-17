use crate::hal::{Hal, SubmitInfo};
use crate::rdg::{Node, Rdg};
use crate::DrawType::Draw;
use crate::{
    hal, Buffer, BufferDescription, BufferWriteParams, CommandRecorder, CommandRecorderType,
    DrawCommand, GraphicsPipeline, GraphicsPipelineDescription, Image, ImageDescription,
    ImageDimension, ImageViewDescription, MemoryDomain, MgpuResult, ShaderModule,
    ShaderModuleDescription,
};
use crate::{ImageWriteParams, MgpuError};
use ash::vk::{self, ImageView};
use bitflags::bitflags;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::ops::DerefMut;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

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

#[cfg(feature = "swapchain")]
pub(crate) struct PresentationRequest {
    pub(crate) id: u64,
    pub(crate) image: SwapchainImage,
}

#[derive(Clone)]
pub struct Device {
    pub(crate) hal: Arc<dyn Hal>,
    pub(crate) device_info: DeviceInfo,

    pub(crate) rdg: Arc<RwLock<Rdg>>,

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

            #[cfg(feature = "swapchain")]
            presentation_requests: Default::default(),
        })
    }

    pub fn submit(&self) -> MgpuResult<()> {
        let rdg = self.write_rdg().take();
        let compiled = rdg.compile()?;

        let mut graphics_command_recorders = vec![];
        let mut async_compute_command_recorders = vec![];
        let queues = unsafe { self.hal.begin_rendering()? };
        for step in compiled.sequence {
            match step {
                crate::rdg::Step::Barrier { transfers } => {
                    todo!()
                }
                crate::rdg::Step::PassGroup(passes) => {
                    if !passes.graphics_nodes.is_empty() {
                        let command_recorder = unsafe {
                            self.hal
                                .request_command_recorder(queues.graphics_compute_allocator)?
                        };
                        for pass in passes.graphics_nodes {
                            let node = &compiled.nodes[pass];
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
                            }
                        }

                        for pass in passes.compute_nodes {
                            todo!();
                        }
                        unsafe { self.hal.finalize_command_recorder(command_recorder)? };
                        graphics_command_recorders.push(command_recorder);
                    }
                }
            }
        }
        let submit_info = SubmitInfo {
            graphics_command_recorders,
            async_compute_command_recorders,
        };
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
        Self::validate_buffer_description(buffer_description)?;
        self.hal.create_buffer(buffer_description)
    }

    pub(crate) fn write_buffer_immediate(
        &self,
        buffer: Buffer,
        params: &BufferWriteParams,
    ) -> MgpuResult<()> {
        self.validate_buffer_write_params(buffer, params)?;

        if buffer.memory_domain == MemoryDomain::DeviceLocal {
            todo!()
        } else {
            // SAFETY: The write parameters have been validated
            unsafe { self.hal.write_host_visible_buffer(buffer, params) }
        }
    }

    pub fn write_buffer(&self, buffer: Buffer, params: &BufferWriteParams) -> MgpuResult<()> {
        self.validate_buffer_write_params(buffer, params)?;
        todo!()
    }

    pub fn destroy_buffer(&self, buffer: Buffer) -> MgpuResult<()> {
        self.hal.destroy_buffer(buffer)
    }

    pub fn create_image(&self, image_description: &ImageDescription) -> MgpuResult<Image> {
        Self::validate_image_description(image_description)?;
        self.hal.create_image(image_description)
    }

    pub fn write_image(&self, image: Image, params: &ImageWriteParams) -> MgpuResult<()> {
        self.validate_image_write_params(image, params)?;

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
        self.validate_shader_module_description(shader_module_description)?;
        self.hal.create_shader_module(shader_module_description)
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

    fn validate_image_description(image_description: &ImageDescription) -> MgpuResult<()> {
        let check_condition = |condition: bool, error_message: &str| {
            if !condition {
                Err(MgpuError::InvalidParams {
                    params_name: "ImageDescription",
                    label: image_description.label.map(ToOwned::to_owned),
                    reason: error_message.to_string(),
                })
            } else {
                Ok(())
            }
        };

        check_condition(
            image_description.dimension != ImageDimension::D3
                || image_description.extents.depth > 0,
            "The depth of an image cannot be 0",
        )?;

        let total_texels = image_description.extents.width
            * image_description.extents.height
            * image_description.extents.depth;
        check_condition(
            total_texels > 0,
            "The width, height and depth of an image cannot be 0",
        )?;
        Ok(())
    }

    fn validate_buffer_description(buffer_description: &BufferDescription) -> MgpuResult<()> {
        let check_condition = |condition: bool, error_message: &str| {
            if !condition {
                Err(MgpuError::InvalidParams {
                    params_name: "BufferDescription",
                    label: buffer_description.label.map(ToOwned::to_owned),
                    reason: error_message.to_string(),
                })
            } else {
                Ok(())
            }
        };

        check_condition(
            buffer_description.size > 0,
            "Cannot create a buffer with size 0!",
        )?;
        Ok(())
    }

    fn validate_image_write_params(
        &self,
        image: Image,
        params: &ImageWriteParams,
    ) -> MgpuResult<()> {
        let check_condition = |condition: bool, error_message: &str| {
            if !condition {
                Err(MgpuError::InvalidParams {
                    params_name: "ImageWriteParams",
                    label: self.hal.image_name(image)?,
                    reason: error_message.to_string(),
                })
            } else {
                Ok(())
            }
        };

        let total_texels = image.extents.width * image.extents.height * image.extents.depth;
        let texel_byte_size = image.format.byte_size();
        let total_bytes = total_texels as usize * texel_byte_size;

        check_condition(params.data.len() >= total_bytes, &format!("Attempted to execute a write operation without enough source data, expected at least {total_bytes} bytes, got {}", params.data.len()))?;

        Ok(())
    }

    fn validate_buffer_write_params(
        &self,
        buffer: Buffer,
        params: &BufferWriteParams,
    ) -> MgpuResult<()> {
        let check_condition = |condition: bool, error_message: &str| {
            if !condition {
                Err(MgpuError::InvalidParams {
                    params_name: "BufferWriteParams",
                    label: self.hal.buffer_name(buffer)?,
                    reason: error_message.to_string(),
                })
            } else {
                Ok(())
            }
        };

        check_condition(
            params.size > 0,
            "A buffer write operation cannot have data length of 0!",
        )?;
        let expected_data_len = params.size - params.offset;
        check_condition(
            params.data.len() >= expected_data_len,
            &format!(
                "Not enough data in data buffer: expected {expected_data_len}, got {}",
                params.data.len()
            ),
        )?;
        Ok(())
    }

    fn validate_shader_module_description(
        &self,
        shader_module_description: &ShaderModuleDescription,
    ) -> MgpuResult<()> {
        let check_condition = |condition: bool, error_message: &str| {
            if !condition {
                Err(MgpuError::InvalidParams {
                    params_name: "ShaderModuleDescription",
                    label: shader_module_description.label.map(ToOwned::to_owned),
                    reason: error_message.to_string(),
                })
            } else {
                Ok(())
            }
        };

        check_condition(shader_module_description.source.len() > 0, "Empty source")?;

        Ok(())
    }

    fn validate_graphics_pipeline_description(
        &self,
        graphics_pipeline_description: &GraphicsPipelineDescription,
    ) -> MgpuResult<()> {
        let check_condition = |condition: bool, error_message: &str| {
            if !condition {
                Err(MgpuError::InvalidParams {
                    params_name: "GraphicsPipelineDescription",
                    label: graphics_pipeline_description.label.map(ToOwned::to_owned),
                    reason: error_message.to_string(),
                })
            } else {
                Ok(())
            }
        };

        for binding in graphics_pipeline_description.binding_sets {
            todo!()
        }

        Ok(())
    }
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
