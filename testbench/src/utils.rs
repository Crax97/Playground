use std::{cell::RefCell, path::Path, sync::Arc};

use ash::vk::ShaderModuleCreateFlags;
use egui_winit_ash_integration::{AllocationCreateInfoTrait, AllocationTrait, AllocatorTrait};
use log::info;

use gpu::{
    AllocationRequirements, Gpu, GpuAllocator, GpuShaderModule, MemoryAllocation, MemoryDomain,
    ShaderModuleCreateInfo,
};

pub fn read_file_to_vk_module<P: AsRef<Path>>(
    gpu: &Gpu,
    path: P,
) -> anyhow::Result<GpuShaderModule> {
    info!(
        "Reading path from {:?}",
        path.as_ref()
            .canonicalize()
            .expect("Failed to canonicalize path")
    );
    let input_file = std::fs::read(path)?;
    let create_info = ShaderModuleCreateInfo {
        flags: ShaderModuleCreateFlags::empty(),
        code: &input_file,
    };
    Ok(gpu.create_shader_module(&create_info)?)
}

pub struct EguiVkAllocator(pub Arc<RefCell<dyn GpuAllocator>>);
pub struct EguiVkAllocation(MemoryAllocation);
pub struct EguiVkAllocationCreateInfo(AllocationRequirements);
impl AllocationTrait for EguiVkAllocation {
    unsafe fn memory(&self) -> ash::vk::DeviceMemory {
        self.0.device_memory
    }

    fn offset(&self) -> u64 {
        self.0.offset
    }

    fn size(&self) -> u64 {
        self.0.size
    }

    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        self.0.persistent_ptr
    }
}

impl AllocationCreateInfoTrait for EguiVkAllocationCreateInfo {
    fn new(
        requirements: ash::vk::MemoryRequirements,
        location: egui_winit_ash_integration::MemoryLocation,
        _linear: bool,
    ) -> Self {
        Self(AllocationRequirements {
            memory_requirements: requirements.into(),
            memory_domain: match location {
                egui_winit_ash_integration::MemoryLocation::Unknown => MemoryDomain::HostVisible,
                egui_winit_ash_integration::MemoryLocation::GpuOnly => MemoryDomain::DeviceLocal,
                egui_winit_ash_integration::MemoryLocation::CpuToGpu => {
                    MemoryDomain::HostVisible | MemoryDomain::HostCoherent
                }
                egui_winit_ash_integration::MemoryLocation::GpuToCpu => {
                    MemoryDomain::HostVisible | MemoryDomain::DeviceLocal
                }
            },
        })
    }
}

impl AllocatorTrait for EguiVkAllocator {
    type Allocation = EguiVkAllocation;
    type AllocationCreateInfo = EguiVkAllocationCreateInfo;

    fn allocate(&self, desc: Self::AllocationCreateInfo) -> anyhow::Result<Self::Allocation> {
        let alloc = self.0.borrow_mut().allocate(desc.0)?;
        Ok(EguiVkAllocation(alloc))
    }

    fn free(&self, allocation: Self::Allocation) -> anyhow::Result<()> {
        self.0.borrow_mut().deallocate(&allocation.0);
        Ok(())
    }
}
