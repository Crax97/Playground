use std::{ffi::c_void, sync::atomic::AtomicPtr};

use ash::vk::{
    self, BufferCreateFlags, BufferCreateInfo, BufferUsageFlags, CommandBufferBeginInfo,
    CommandPoolCreateFlags, FenceCreateInfo, MemoryAllocateInfo, MemoryPropertyFlags,
    PhysicalDevice,
};

use crate::get_allocation_callbacks;
/*
                            buffer_offset: src_offset,
                            buffer_row_length: todo!(),
                            buffer_image_height: todo!(),
                            image_subresource: todo!(),
                            image_offset: todo!(),
                            image_extent: todo!(),
*/
pub enum BufferOperation {
    CopyBuffer {
        dest_buffer: vk::Buffer,
        dest_offset: u64,
        src_offset: u64,
        size: u64,
    },
    CopyImage {
        image: vk::Image,
        image_subresource: vk::ImageSubresourceLayers,
        image_offset: vk::Offset3D,
        image_extent: vk::Extent3D,
        final_layout: vk::ImageLayout,

        src_offset: u64,
    },
}

const fn assume_send_sync<T: Send + Sync>() {}
const _TEST: () = assume_send_sync::<VkStagingBuffer>();

struct ImageToTransition {
    image: vk::Image,
    subresource: vk::ImageSubresourceRange,
}

pub struct VkStagingBuffer {
    copy_command_pool: vk::CommandPool,
    copy_command_buffer: vk::CommandBuffer,

    backing_buffer: vk::Buffer,
    memory_allocation: vk::DeviceMemory,
    operations: Vec<BufferOperation>,
    total_size: u64,
    current_offset: u64,
    images_to_transition: Vec<ImageToTransition>,
    persistent_pointer: AtomicPtr<c_void>,
    queue: vk::Queue,
    wait_fence: vk::Fence,
}

impl VkStagingBuffer {
    pub fn new(
        device: &ash::Device,
        physical_device: PhysicalDevice,
        instance: &ash::Instance,
        backbuffer_size: u64,
        queue_family_index: u32,
        queue: vk::Queue,
    ) -> anyhow::Result<Self> {
        unsafe {
            let copy_command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index,
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ..Default::default()
                },
                get_allocation_callbacks(),
            )?;
            let copy_command_buffer =
                device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    command_pool: copy_command_pool,
                    command_buffer_count: 1,
                    level: vk::CommandBufferLevel::PRIMARY,
                    ..Default::default()
                })?[0];

            let queues = [queue_family_index];
            let backing_buffer = device.create_buffer(
                &BufferCreateInfo {
                    flags: BufferCreateFlags::empty(),
                    size: backbuffer_size,
                    usage: BufferUsageFlags::TRANSFER_SRC,
                    queue_family_index_count: queues.len() as _,
                    p_queue_family_indices: queues.as_ptr(),
                    sharing_mode: vk::SharingMode::EXCLUSIVE,

                    ..Default::default()
                },
                get_allocation_callbacks(),
            )?;

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);

            let memory_requirements = device.get_buffer_memory_requirements(backing_buffer);

            let mut memory_index = None;
            for i in 0..memory_properties.memory_type_count {
                let memory_type = &memory_properties.memory_types[i as usize];
                if ((1 << i) & memory_requirements.memory_type_bits) != 0
                    && memory_type.property_flags
                        == (MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT)
                {
                    memory_index = Some(i);
                }
            }

            let memory_type_index =
                memory_index.ok_or(anyhow::format_err!("No suitable memory found"))?;

            let memory_allocation = device.allocate_memory(
                &MemoryAllocateInfo {
                    allocation_size: backbuffer_size,
                    memory_type_index,
                    ..Default::default()
                },
                get_allocation_callbacks(),
            )?;

            device.bind_buffer_memory(backing_buffer, memory_allocation, 0)?;
            let persistent_pointer = device.map_memory(
                memory_allocation,
                0,
                backbuffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            let wait_fence = device.create_fence(
                &FenceCreateInfo::builder().build(),
                get_allocation_callbacks(),
            )?;

            Ok(Self {
                backing_buffer,
                memory_allocation,
                operations: vec![],
                images_to_transition: vec![],

                total_size: backbuffer_size,
                current_offset: 0,
                persistent_pointer: AtomicPtr::new(persistent_pointer),
                copy_command_pool,
                copy_command_buffer,
                queue,
                wait_fence,
            })
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.memory_allocation);
            device.destroy_buffer(self.backing_buffer, get_allocation_callbacks());
            device.free_memory(self.memory_allocation, get_allocation_callbacks());
        }
    }

    pub fn write_buffer<T: Copy>(
        &mut self,
        device: &ash::Device,
        dest_buffer: vk::Buffer,
        dest_offset: u64,
        data: &[T],
    ) -> anyhow::Result<()> {
        let data_size = std::mem::size_of_val(data) as u64;
        if self.current_offset + data_size > self.total_size {
            self.flush_operations(device)?;
        }

        unsafe {
            let ptr = self
                .persistent_pointer
                .load(std::sync::atomic::Ordering::Relaxed)
                .add(self.current_offset as usize);
            ptr.copy_from(data.as_ptr().cast(), data_size as usize);
        }
        self.operations.push(BufferOperation::CopyBuffer {
            dest_buffer,
            dest_offset,
            src_offset: self.current_offset,
            size: data_size,
        });
        self.current_offset += data_size;

        Ok(())
    }
    pub fn write_image<T: Copy>(
        &mut self,
        device: &ash::Device,
        dest_image: vk::Image,
        dest_offset: vk::Offset3D,
        dest_extent: vk::Extent3D,
        texel_size_bytes: u32,
        image_subresource: vk::ImageSubresourceLayers,
        final_layout: vk::ImageLayout,
        data: &[T],
    ) -> anyhow::Result<()> {
        let data_size = std::mem::size_of_val(data) as u64;
        let expected_data_size =
            (dest_extent.width * dest_extent.height * dest_extent.depth * texel_size_bytes) as u64;
        if data_size < expected_data_size as u64 {
            anyhow::bail!("Not enough bytes in data!");
        }
        if self.current_offset + expected_data_size > self.total_size {
            self.flush_operations(device)?;
        }

        unsafe {
            let ptr = self
                .persistent_pointer
                .load(std::sync::atomic::Ordering::Relaxed)
                .add(self.current_offset as usize);
            ptr.copy_from(data.as_ptr().cast(), expected_data_size as usize);
        }
        self.operations.push(BufferOperation::CopyImage {
            image: dest_image,
            image_subresource,
            image_offset: dest_offset,
            image_extent: dest_extent,
            final_layout: final_layout,
            src_offset: self.current_offset,
        });
        self.images_to_transition.push(ImageToTransition {
            image: dest_image,
            subresource: vk::ImageSubresourceRange {
                aspect_mask: image_subresource.aspect_mask,
                base_mip_level: image_subresource.mip_level,
                level_count: 1,
                base_array_layer: image_subresource.base_array_layer,
                layer_count: image_subresource.layer_count,
            },
        });
        self.current_offset += expected_data_size as u64;

        Ok(())
    }

    pub fn flush_operations(&mut self, device: &ash::Device) -> anyhow::Result<()> {
        if self.operations.is_empty() {
            return Ok(());
        }
        unsafe {
            device.begin_command_buffer(
                self.copy_command_buffer,
                &CommandBufferBeginInfo::builder().build(),
            )?;
            {
                let pre_copy_barriers = std::mem::take(&mut self.images_to_transition)
                    .iter()
                    .map(
                        |ImageToTransition { image, subresource }| vk::ImageMemoryBarrier2 {
                            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                            dst_access_mask: vk::AccessFlags2::MEMORY_WRITE,
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: *image,
                            subresource_range: *subresource,
                            ..Default::default()
                        },
                    )
                    .collect::<Vec<_>>();
                device.cmd_pipeline_barrier2(
                    self.copy_command_buffer,
                    &vk::DependencyInfo {
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                        memory_barrier_count: 0,
                        p_memory_barriers: std::ptr::null(),
                        buffer_memory_barrier_count: 0,
                        p_buffer_memory_barriers: std::ptr::null(),
                        image_memory_barrier_count: pre_copy_barriers.len() as _,
                        p_image_memory_barriers: pre_copy_barriers.as_ptr(),
                        ..Default::default()
                    },
                );
            }
            let mut images_to_transition = vec![];
            for operation in std::mem::take(&mut self.operations) {
                match operation {
                    BufferOperation::CopyBuffer {
                        dest_buffer,
                        dest_offset,
                        src_offset,
                        size,
                    } => {
                        device.cmd_copy_buffer(
                            self.copy_command_buffer,
                            self.backing_buffer,
                            dest_buffer,
                            &[vk::BufferCopy {
                                src_offset,
                                dst_offset: dest_offset,
                                size,
                            }],
                        );
                    }
                    BufferOperation::CopyImage {
                        image,
                        image_subresource,
                        image_offset,
                        image_extent,
                        final_layout,

                        src_offset,
                    } => {
                        device.cmd_copy_buffer_to_image(
                            self.copy_command_buffer,
                            self.backing_buffer,
                            image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[vk::BufferImageCopy {
                                buffer_offset: src_offset,
                                buffer_row_length: 0,
                                buffer_image_height: 0,
                                image_subresource,
                                image_offset,
                                image_extent,
                            }],
                        );
                        images_to_transition.push(vk::ImageMemoryBarrier2 {
                            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                            src_access_mask: vk::AccessFlags2::MEMORY_WRITE,
                            dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                            dst_access_mask: vk::AccessFlags2::SHADER_READ,
                            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            new_layout: final_layout,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: image_subresource.aspect_mask,
                                base_mip_level: image_subresource.mip_level,
                                level_count: 1,
                                base_array_layer: image_subresource.base_array_layer,
                                layer_count: image_subresource.layer_count,
                            },
                            ..Default::default()
                        });
                    }
                }

                device.cmd_pipeline_barrier2(
                    self.copy_command_buffer,
                    &vk::DependencyInfo {
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                        memory_barrier_count: 0,
                        p_memory_barriers: std::ptr::null(),
                        buffer_memory_barrier_count: 0,
                        p_buffer_memory_barriers: std::ptr::null(),
                        image_memory_barrier_count: images_to_transition.len() as _,
                        p_image_memory_barriers: images_to_transition.as_ptr(),
                        ..Default::default()
                    },
                );
            }
            device.end_command_buffer(self.copy_command_buffer)?;
            let command_buffers = [self.copy_command_buffer];

            device.queue_submit(
                self.queue,
                &[vk::SubmitInfo {
                    p_command_buffers: command_buffers.as_ptr(),
                    command_buffer_count: 1,
                    ..Default::default()
                }],
                self.wait_fence,
            )?;

            device.wait_for_fences(&[self.wait_fence], true, u64::MAX)?;

            device.reset_command_pool(
                self.copy_command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
            self.current_offset = 0;
            Ok(())
        }
    }
}
