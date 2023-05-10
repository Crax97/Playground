use ash::vk::{
    MemoryAllocateInfo, MemoryPropertyFlags, PhysicalDevice, PhysicalDeviceMemoryProperties,
    StructureType,
};
use ash::{
    prelude::VkResult,
    vk::{DeviceMemory, MemoryRequirements},
};
use ash::{Device, Instance};
use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

pub trait Allocator {
    fn new(instance: &Instance, physical_device: PhysicalDevice, device: &Device) -> VkResult<Self>
    where
        Self: Sized;

    fn allocate(
        &mut self,
        device: &Device,
        allocation_requirements: AllocationRequirements,
    ) -> VkResult<DeviceMemory>;

    fn deallocate(&mut self, device: Device, allocation: DeviceMemory);
}

pub struct PasstroughAllocator {
    memory_properties: PhysicalDeviceMemoryProperties,
}
impl PasstroughAllocator {
    fn find_memory_type(&self, type_filter: u32, memory_domain: MemoryDomain) -> Option<u32> {
        let mem_properties = memory_domain.into();
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) > 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .intersects(mem_properties)
            {
                return Some(i);
            }
        }
        None
    }
}

impl Allocator for PasstroughAllocator {
    fn new(instance: &Instance, physical_device: PhysicalDevice, _: &Device) -> VkResult<Self>
    where
        Self: Sized,
    {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        Ok(Self { memory_properties })
    }

    fn allocate(
        &mut self,
        device: &Device,
        allocation_requirements: AllocationRequirements,
    ) -> VkResult<DeviceMemory> {
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
        unsafe { device.allocate_memory(&allocate_info, None) }
    }

    fn deallocate(&mut self, device: Device, allocation: DeviceMemory) {
        unsafe {
            device.free_memory(allocation, None);
        }
    }
}
