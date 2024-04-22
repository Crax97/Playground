use crate::{Buffer, BufferUsageFlags, Device, MemoryDomain, MgpuResult};

#[derive(Default)]
pub(crate) struct StagingBufferAllocator {
    staging_buffers: Vec<Buffer>,
    current_buffer_idx: usize,
    current_offset: usize,
}

pub(crate) struct StagingBufferAllocation {
    pub buffer: Buffer,
    pub offset: usize,
}

impl StagingBufferAllocator {
    const MB_128: usize = 1024 * 1024 * 128;

    // Returns a suitable, host-visible, buffer big enough to write 'size' bytes to it
    pub(crate) fn get_staging_buffer(
        &mut self,
        device: &Device,
        size: usize,
    ) -> MgpuResult<StagingBufferAllocation> {
        if let Some(buffer) = self.staging_buffers.get(self.current_buffer_idx).copied() {
            let remaining_space = buffer.size - self.current_offset;
            if remaining_space > size {
                let allocation = StagingBufferAllocation {
                    buffer,
                    offset: self.current_offset,
                };
                self.current_offset += size;
                return Ok(allocation);
            } else {
                self.current_buffer_idx += 1;
            }
        }

        self.current_offset = size;

        let buffer = if self.current_buffer_idx == self.staging_buffers.len() {
            let new_buffer = Self::allocate_staging_buffer(device)?;
            self.staging_buffers.push(new_buffer);
            new_buffer
        } else {
            self.staging_buffers[self.current_buffer_idx]
        };

        Ok(StagingBufferAllocation { buffer, offset: 0 })
    }

    fn allocate_staging_buffer(device: &Device) -> MgpuResult<Buffer> {
        let buffer = device.create_buffer(&crate::BufferDescription {
            label: Some("Staging buffer"),
            usage_flags: BufferUsageFlags::TRANSFER_SRC,
            size: Self::MB_128,
            memory_domain: MemoryDomain::HostVisible,
        })?;
        Ok(buffer)
    }

    pub fn clear(&mut self) {
        self.current_buffer_idx = 0;
        self.current_offset = 0;
    }
}
