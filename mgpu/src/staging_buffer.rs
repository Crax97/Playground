use ash::vk::Image;

use crate::{Buffer, BufferUsageFlags, BufferWriteParams, Device, MemoryDomain, MgpuResult};

pub(crate) struct StagingBuffer {
    staging_buffer: Buffer,
    device: Device,
    operations: Vec<StagingBufferOperation>,
    current_offset: usize,
}

#[derive(Clone, Copy)]
struct StagingBufferSource {
    offset: usize,
    size: usize,
}
#[derive(Clone, Copy)]
enum StagingBufferOperationKind {
    WriteBuffer { dest: Buffer },
    WriteImage { dest: Image },
}

#[derive(Clone, Copy)]
struct StagingBufferOperation {
    source: StagingBufferSource,
    kind: StagingBufferOperationKind,
}

impl StagingBuffer {
    const MB_128: usize = 1024 * 1024 * 128;
    pub fn new(device: Device) -> MgpuResult<Self> {
        let buffer = device.create_buffer(&crate::BufferDescription {
            label: Some("Staging buffer"),
            usage_flags: BufferUsageFlags::TRANSFER_SRC,
            size: Self::MB_128,
            memory_domain: MemoryDomain::HostVisible,
        })?;
        Ok(Self {
            staging_buffer: buffer,
            device,
            operations: Default::default(),
            current_offset: 0,
        })
    }

    pub fn enqueue_write_buffer(
        &mut self,
        buffer: Buffer,
        params: &BufferWriteParams,
    ) -> MgpuResult<()> {
        if self.remaining_space() < params.total_bytes() {
            self.flush_operations()?;
        }
        let offset = self.current_offset;
        self.device.write_buffer_immediate(
            self.staging_buffer,
            &BufferWriteParams {
                data: params.data,
                offset,
                size: params.size,
            },
        )?;

        self.operations.push(StagingBufferOperation {
            source: StagingBufferSource {
                offset,
                size: params.size,
            },
            kind: StagingBufferOperationKind::WriteBuffer { dest: buffer },
        });

        self.current_offset += params.size;
        Ok(())
    }

    pub fn flush_operations(&mut self) -> MgpuResult<()> {
        todo!()
    }

    pub fn remaining_space(&self) -> usize {
        self.staging_buffer.size - self.current_offset
    }
}
