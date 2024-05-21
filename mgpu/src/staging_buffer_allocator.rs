use crate::{
    hal::{Hal, QueueType},
    Buffer, Image, ImageRegion, MgpuResult,
};
#[cfg(debug_assertions)]
use crate::{BufferUsageFlags, MemoryDomain};

pub(crate) struct StagingBufferAllocator {
    pub(crate) staging_buffer: Buffer,
    operations: Vec<StagingBufferOperation>,
    head: usize,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum OperationType {
    WriteBuffer {
        dest: Buffer,
        dest_offset: usize,
        size: usize,
    },
    WriteImage {
        dest: Image,
        region: ImageRegion,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StagingBufferOperation {
    pub source_offset: usize,
    pub op_type: OperationType,
}

impl StagingBufferAllocator {
    pub fn new(staging_buffer: Buffer, _frames_in_flight: usize) -> MgpuResult<Self> {
        #[cfg(debug_assertions)]
        check!(
            staging_buffer
                .usage_flags
                .contains(BufferUsageFlags::TRANSFER_SRC)
                && staging_buffer.memory_domain == MemoryDomain::Cpu,
            "Invalid staging buffer"
        );
        Ok(Self {
            staging_buffer,
            operations: vec![],
            head: 0,
        })
    }

    pub(crate) fn write_buffer(
        &mut self,
        hal: &dyn Hal,
        buffer: Buffer,
        data: &[u8],
        offset: usize,
        size: usize,
    ) -> MgpuResult<()> {
        self.write_buffer_inner(hal, data, buffer, offset, size, false)
    }

    pub(crate) fn write_image(
        &mut self,
        hal: &dyn Hal,
        image: Image,
        data: &[u8],
        region: ImageRegion,
    ) -> MgpuResult<()> {
        self.write_image_inner(hal, data, image, region, false)
    }

    pub(crate) fn flush(&mut self, hal: &dyn Hal) -> MgpuResult<()> {
        if self.operations.is_empty() {
            return Ok(());
        }
        let command_recorder =
            unsafe { hal.request_oneshot_command_recorder(crate::hal::QueueType::Graphics)? };

        for op in std::mem::take(&mut self.operations) {
            unsafe {
                match op.op_type {
                    OperationType::WriteBuffer {
                        dest,
                        dest_offset,
                        size,
                    } => {
                        hal.cmd_copy_buffer_to_buffer(
                            command_recorder,
                            self.staging_buffer,
                            dest,
                            op.source_offset,
                            dest_offset,
                            size,
                        )?;
                    }
                    OperationType::WriteImage { dest, region } => {
                        hal.cmd_copy_buffer_to_image(
                            command_recorder,
                            self.staging_buffer,
                            dest,
                            op.source_offset,
                            region,
                        )?;
                    }
                }
            }
        }

        unsafe {
            hal.finalize_command_recorder(command_recorder)?;
            hal.submit_command_recorder_immediate(command_recorder)?;
            hal.device_wait_queue(QueueType::Graphics)?;
        }
        Ok(())
    }

    fn write_buffer_inner(
        &mut self,
        hal: &dyn Hal,
        data: &[u8],
        dest: Buffer,
        offset: usize,
        size: usize,
        recursed: bool,
    ) -> MgpuResult<()> {
        debug_assert!(size <= self.staging_buffer.size);
        if self.head + size > self.staging_buffer.size {
            if recursed {
                panic!("Recursed twice in update loop");
            }
            self.flush(hal)?;
            self.head = 0;
            self.write_buffer_inner(hal, data, dest, offset, size, true)?;
        } else {
            let operation_type = OperationType::WriteBuffer {
                dest,
                size,
                dest_offset: offset,
            };
            unsafe {
                hal.write_host_visible_buffer(
                    self.staging_buffer,
                    &crate::BufferWriteParams {
                        data,
                        offset: self.head,
                        size,
                    },
                )?
            };

            self.operations.push(StagingBufferOperation {
                source_offset: self.head,
                op_type: operation_type,
            });

            self.head += size;
        }
        Ok(())
    }

    fn write_image_inner(
        &mut self,
        hal: &dyn Hal,
        data: &[u8],
        dest: Image,
        region: ImageRegion,
        recursed: bool,
    ) -> MgpuResult<()> {
        let size = region.extents.area() as usize * dest.format.byte_size();
        debug_assert!(size <= self.staging_buffer.size);
        if self.head + size > self.staging_buffer.size {
            if recursed {
                panic!("Recursed twice in update loop");
            }
            self.flush(hal)?;
            self.head = 0;
            self.write_image_inner(hal, data, dest, region, recursed)?;
        } else {
            let operation_type = OperationType::WriteImage { dest, region };
            unsafe {
                hal.write_host_visible_buffer(
                    self.staging_buffer,
                    &crate::BufferWriteParams {
                        data,
                        offset: self.head,
                        size,
                    },
                )?
            };

            self.operations.push(StagingBufferOperation {
                source_offset: self.head,
                op_type: operation_type,
            });

            self.head += size;
        }
        Ok(())
    }
    pub fn end_frame(&mut self) {}
}

impl Drop for StagingBufferAllocator {
    fn drop(&mut self) {
        // self.hal.destroy_buffer(self.staging_buffer).unwrap();
    }
}
