use crate::{Buffer, MgpuResult};
#[cfg(debug_assertions)]
use crate::{BufferUsageFlags, MemoryDomain};

pub(crate) struct StagingBufferAllocator {
    staging_buffer: Buffer,
    allocated_regions: Vec<StagingBufferAllocatedRegion>,
    current_frame: usize,
    head: usize,
    tail: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StagingBufferAllocation {
    pub buffer: Buffer,
    pub offset: usize,
}

#[derive(Default, Clone, Copy)]
struct StagingBufferAllocatedRegion {
    pub tip: usize,
}

impl StagingBufferAllocator {
    pub fn new(staging_buffer: Buffer, frames_in_flight: usize) -> MgpuResult<Self> {
        #[cfg(debug_assertions)]
        check!(
            staging_buffer
                .usage_flags
                .contains(BufferUsageFlags::TRANSFER_SRC)
                && staging_buffer.memory_domain == MemoryDomain::Cpu,
            "Invalid staging buffer"
        );
        let allocated_regions = std::iter::repeat(StagingBufferAllocatedRegion::default())
            .take(frames_in_flight)
            .collect::<Vec<_>>();

        Ok(Self {
            staging_buffer,
            allocated_regions,
            current_frame: 0,
            head: 0,
            tail: 0,
        })
    }

    // Returns a suitable, host-visible, buffer big enough to write 'size' bytes to it
    pub(crate) fn allocate_staging_buffer_region(
        &mut self,
        size: usize,
    ) -> MgpuResult<StagingBufferAllocation> {
        self.allocate_inner(size, false)
    }

    fn allocate_inner(
        &mut self,
        size: usize,
        recursed: bool,
    ) -> Result<StagingBufferAllocation, crate::MgpuError> {
        if self.head < self.tail {
            if self.head + size > self.tail {
                panic!("Staging buffer ran out of space");
            } else {
                let allocation = StagingBufferAllocation {
                    buffer: self.staging_buffer,
                    offset: self.head,
                };
                self.head += size;
                Ok(allocation)
            }
        } else if self.head + size > self.staging_buffer.size {
            self.head = 0;
            if recursed {
                panic!("Bug! Recursed twice in allocate_inner")
            }
            return self.allocate_inner(size, true);
        } else {
            let allocation = StagingBufferAllocation {
                buffer: self.staging_buffer,
                offset: self.head,
            };
            self.head += size;
            return Ok(allocation);
        }
    }

    pub fn end_frame(&mut self) {
        self.allocated_regions[self.current_frame] =
            StagingBufferAllocatedRegion { tip: self.head };
        self.current_frame = (self.current_frame + 1) % self.allocated_regions.len();
        self.tail = self.allocated_regions[self.current_frame].tip;
        self.allocated_regions[self.current_frame] = Default::default();
    }
}

impl Drop for StagingBufferAllocator {
    fn drop(&mut self) {
        // self.hal.destroy_buffer(self.staging_buffer).unwrap();
    }
}
#[cfg(test)]
mod tests {

    use crate::BufferUsageFlags;

    use super::StagingBufferAllocator;

    #[test]
    fn test_staging_buffer_looping() {
        let mut staging_buffer = StagingBufferAllocator::new(
            crate::Buffer {
                id: 0,
                usage_flags: BufferUsageFlags::TRANSFER_SRC,
                size: 32,
                memory_domain: crate::MemoryDomain::Cpu,
            },
            4,
        )
        .unwrap();
        for _ in 0..10 {
            let alloc = staging_buffer.allocate_staging_buffer_region(5).unwrap();
            println!("{alloc:?}");
            staging_buffer.end_frame();
        }
    }
}
