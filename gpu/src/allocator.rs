use std::ffi::c_void;
use std::ptr::NonNull;

use ash::vk::{
    MemoryAllocateInfo, MemoryMapFlags, MemoryPropertyFlags, PhysicalDevice,
    PhysicalDeviceMemoryProperties, StructureType,
};
use ash::{
    prelude::VkResult,
    vk::{DeviceMemory, MemoryRequirements},
};
use ash::{Device, Instance};
use bitflags::bitflags;
use log::trace;

bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
    pub struct MemoryDomain: u32 {
        const DeviceLocal =     0b00000001;
        const HostVisible =     0b00000010;
        const HostCoherent =    0b00000100;
        const HostCached =      0b00001000;
    }
}

impl From<MemoryDomain> for MemoryPropertyFlags {
    fn from(domain: MemoryDomain) -> Self {
        let mut flags = MemoryPropertyFlags::empty();
        if domain.contains(MemoryDomain::DeviceLocal) {
            flags |= MemoryPropertyFlags::DEVICE_LOCAL;
        }
        if domain.contains(MemoryDomain::HostVisible) {
            flags |= MemoryPropertyFlags::HOST_VISIBLE;
        }
        if domain.contains(MemoryDomain::HostCoherent) {
            flags |= MemoryPropertyFlags::HOST_COHERENT;
        }
        if domain.contains(MemoryDomain::HostCached) {
            flags |= MemoryPropertyFlags::HOST_CACHED;
        }
        flags
    }
}

pub struct AllocationRequirements {
    pub memory_requirements: MemoryRequirements,
    pub memory_domain: MemoryDomain,
}

#[derive(Eq, Ord, PartialOrd, PartialEq)]
pub struct MemoryAllocation {
    pub device_memory: DeviceMemory,
    pub offset: u64,
    pub size: u64,
    pub persistent_ptr: Option<NonNull<c_void>>,
}

pub trait GpuAllocator {
    fn new(instance: &Instance, physical_device: PhysicalDevice, device: &Device) -> VkResult<Self>
    where
        Self: Sized;

    fn allocate(
        &mut self,
        allocation_requirements: AllocationRequirements,
    ) -> VkResult<MemoryAllocation>;

    fn deallocate(&mut self, allocation: &MemoryAllocation);
}

pub struct PasstroughAllocator {
    memory_properties: PhysicalDeviceMemoryProperties,
    device: Device,
    num_allocations: u32,
}
impl PasstroughAllocator {
    fn find_memory_type(&self, type_filter: u32, memory_domain: MemoryDomain) -> Option<u32> {
        let mem_properties = memory_domain.into();
        (0..self.memory_properties.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) > 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .intersects(mem_properties)
        })
    }
}

impl GpuAllocator for PasstroughAllocator {
    fn new(instance: &Instance, physical_device: PhysicalDevice, device: &Device) -> VkResult<Self>
    where
        Self: Sized,
    {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        Ok(Self {
            memory_properties,
            num_allocations: 0,
            device: device.clone(),
        })
    }

    fn allocate(
        &mut self,
        allocation_requirements: AllocationRequirements,
    ) -> VkResult<MemoryAllocation> {
        let memory_type_index = self.find_memory_type(
            allocation_requirements.memory_requirements.memory_type_bits,
            allocation_requirements.memory_domain,
        );
        let memory_type_index = if let Some(index) = memory_type_index {
            index
        } else {
            return Err(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        };
        let allocate_info = MemoryAllocateInfo {
            s_type: StructureType::MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: allocation_requirements.memory_requirements.size,
            memory_type_index,
        };
        let device_memory = unsafe { self.device.allocate_memory(&allocate_info, None) }?;
        self.num_allocations += 1;
        trace!(
            "PasstroughAllocator: Allocated {} bytes, there are {} allocations",
            allocate_info.allocation_size,
            self.num_allocations
        );

        let persistent_ptr = if allocation_requirements
            .memory_domain
            .contains(MemoryDomain::HostVisible)
        {
            NonNull::new(unsafe {
                self.device.map_memory(
                    device_memory,
                    0,
                    allocation_requirements.memory_requirements.size,
                    MemoryMapFlags::empty(),
                )?
            })
        } else {
            None
        };

        Ok(MemoryAllocation {
            device_memory,
            offset: 0,
            size: allocate_info.allocation_size,
            persistent_ptr,
        })
    }

    fn deallocate(&mut self, allocation: &MemoryAllocation) {
        unsafe {
            if allocation.persistent_ptr.is_some() {
                self.device.unmap_memory(allocation.device_memory);
            }
            self.device.free_memory(allocation.device_memory, None);
        }
        self.num_allocations -= 1;
        trace!(
            "PasstroughAllocator: Deallocated {} bytes, there are {} allocations",
            allocation.size,
            self.num_allocations
        );
    }
}
