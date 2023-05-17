use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    ptr::addr_of,
};

use ash::{
    prelude::VkResult,
    vk::{
        self, DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
        DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateFlags, DescriptorType, ShaderStageFlags, StructureType,
    },
    Device,
};
use log::trace;

use super::DescriptorSetInfo;

pub struct DescriptorSetAllocation {
    pub owner_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

pub trait DescriptorSetAllocator {
    fn allocate(&mut self, info: &DescriptorSetInfo) -> VkResult<DescriptorSetAllocation>;
    fn deallocate(&mut self, descriptor_set: &DescriptorSetAllocation) -> VkResult<()>;
}

/*
This allocator simply creates a new pool with a fixed amount of descriptors
each time a descriptor set allocation fails
 */
pub struct PooledDescriptorSetAllocator {
    usable_descriptor_pools: Vec<DescriptorPool>,
    hashed_layouts: HashMap<u64, DescriptorSetLayout>,
    device: ash::Device,
}
impl PooledDescriptorSetAllocator {
    fn get_last_allocated_descriptor_pool(&self) -> DescriptorPool {
        self.usable_descriptor_pools.last().unwrap().clone()
    }

    fn allocate_new_descriptor_pool(&mut self) -> VkResult<()> {
        let descriptor_pool = unsafe {
            let pool_size_uniform_buffer = DescriptorPoolSize {
                ty: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 100,
            };
            let pool_size_storage_buffer = DescriptorPoolSize {
                ty: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 100,
            };
            let pool_size_sampler = DescriptorPoolSize {
                ty: DescriptorType::SAMPLER,
                descriptor_count: 100,
            };
            let pool_size_combined_image_sampler = DescriptorPoolSize {
                ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 100,
            };
            self.device.create_descriptor_pool(
                &DescriptorPoolCreateInfo {
                    s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                    max_sets: 100,
                    pool_size_count: 4,
                    p_pool_sizes: [
                        pool_size_uniform_buffer,
                        pool_size_storage_buffer,
                        pool_size_combined_image_sampler,
                        pool_size_sampler,
                    ]
                    .as_ptr(),
                },
                None,
            )?
        };

        self.usable_descriptor_pools.push(descriptor_pool);
        Ok(())
    }

    fn get_descriptor_set_layout(
        &mut self,
        info: &DescriptorSetInfo,
    ) -> VkResult<vk::DescriptorSetLayout> {
        let mut hasher = DefaultHasher::new();
        info.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(layout) = self.hashed_layouts.get(&hash) {
            Ok(layout.clone())
        } else {
            let new_layout = self.construct_descriptor_set_layout(info)?;
            self.hashed_layouts.insert(hash, new_layout.clone());
            trace!(
                "Created a new descriptor set layout! There are {} layouts allocated",
                self.hashed_layouts.len()
            );
            Ok(new_layout)
        }
    }

    fn construct_descriptor_set_layout(
        &self,
        info: &DescriptorSetInfo,
    ) -> VkResult<DescriptorSetLayout> {
        let mut descriptor_set_bindings = vec![];
        for descriptor_info in info.descriptors {
            let stage_flags = match descriptor_info.binding_stage {
                super::ShaderStage::Vertex => ShaderStageFlags::VERTEX,
                super::ShaderStage::Fragment => ShaderStageFlags::FRAGMENT,
                super::ShaderStage::Compute => ShaderStageFlags::COMPUTE,
            };
            let descriptor_type = match descriptor_info.element_type {
                super::DescriptorType::UniformBuffer(_) => DescriptorType::UNIFORM_BUFFER,
                super::DescriptorType::StorageBuffer(_) => DescriptorType::STORAGE_BUFFER,
                super::DescriptorType::Sampler(_) => DescriptorType::SAMPLER,
                super::DescriptorType::CombinedImageSampler(_) => {
                    DescriptorType::COMBINED_IMAGE_SAMPLER
                }
            };
            let binding = DescriptorSetLayoutBinding {
                binding: descriptor_info.binding,
                descriptor_type,
                descriptor_count: 1,
                stage_flags,
                p_immutable_samplers: std::ptr::null(),
            };

            descriptor_set_bindings.push(binding);
        }
        unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: DescriptorSetLayoutCreateFlags::empty(),
                    binding_count: descriptor_set_bindings.len() as _,
                    p_bindings: descriptor_set_bindings.as_ptr(),
                },
                None,
            )
        }
    }
}

impl PooledDescriptorSetAllocator {
    pub fn new(device: Device) -> VkResult<Self> {
        let mut me = Self {
            usable_descriptor_pools: vec![],
            device,
            hashed_layouts: HashMap::new(),
        };

        me.allocate_new_descriptor_pool()?;

        Ok(me)
    }
}

impl DescriptorSetAllocator for PooledDescriptorSetAllocator {
    fn allocate(&mut self, info: &DescriptorSetInfo) -> VkResult<DescriptorSetAllocation> {
        let descriptor_set_layout = self.get_descriptor_set_layout(info)?;

        let mut did_try_once = false;

        while !did_try_once {
            let descriptor_pool = self.get_last_allocated_descriptor_pool();
            let descriptor_set = unsafe {
                self.device
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                        s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        descriptor_pool,
                        descriptor_set_count: 1,
                        p_set_layouts: addr_of!(descriptor_set_layout),
                    })
            };

            match descriptor_set {
                Ok(descriptor_set) => {
                    return Ok(DescriptorSetAllocation {
                        descriptor_set: descriptor_set[0],
                        owner_pool: descriptor_pool,
                        descriptor_set_layout,
                    })
                }
                Err(e) => {
                    if e != vk::Result::ERROR_OUT_OF_POOL_MEMORY {
                        return Err(e);
                    } else {
                        self.allocate_new_descriptor_pool()?;
                        did_try_once = true;
                    }
                }
            };
        }

        Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
    }

    fn deallocate(&mut self, allocation: &DescriptorSetAllocation) -> VkResult<()> {
        unsafe {
            self.device
                .free_descriptor_sets(allocation.owner_pool, &[allocation.descriptor_set])
        }
    }
}

impl Drop for PooledDescriptorSetAllocator {
    fn drop(&mut self) {
        for layout in self.hashed_layouts.values() {
            unsafe {
                self.device
                    .destroy_descriptor_set_layout(layout.clone(), None);
                trace!(
                    "Destroyed a descriptor set layout! There are {} layouts allocated",
                    self.hashed_layouts.len()
                );
            }
        }
    }
}
