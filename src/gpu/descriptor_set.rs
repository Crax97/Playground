/*

    let descriptor_set_layout = {
        let uniform_buffer_binding = DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::VERTEX,
            p_immutable_samplers: null(),
        };
        let sampler_binding = DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: null(),
        };
        let create_info = DescriptorSetLayoutCreateInfo {
            s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: null(),
            flags: DescriptorSetLayoutCreateFlags::empty(),
            binding_count: 2,
            p_bindings: [uniform_buffer_binding, sampler_binding].as_ptr(),
        };
        unsafe { device.create_descriptor_set_layout(&create_info, None) }?
    };

    let descriptor_pool = unsafe {
        let pool_size_uniform_buffer = DescriptorPoolSize {
            ty: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        };
        let pool_size_sampler = DescriptorPoolSize {
            ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };
        device.create_descriptor_pool(
            &DescriptorPoolCreateInfo {
                s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                p_next: null(),
                flags: DescriptorPoolCreateFlags::empty(),
                max_sets: 1,
                pool_size_count: 2,
                p_pool_sizes: [pool_size_uniform_buffer, pool_size_sampler].as_ptr(),
            },
            None,
        )?
    };

    let descriptor_set = unsafe {
        let descriptor_set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
            s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: addr_of!(descriptor_set_layout),
        })?[0];

        let buffer_info = DescriptorBufferInfo {
            buffer: *gpu.resource_map.get(&uniform_buffer).unwrap().deref(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let image_info = DescriptorImageInfo {
            sampler: *gpu.resource_map.get(&sampler).unwrap().deref(),
            image_view: gpu.resource_map.get(&image).unwrap().view,
            image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        device.update_descriptor_sets(
            &[
                WriteDescriptorSet {
                    s_type: StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: DescriptorType::UNIFORM_BUFFER,
                    p_image_info: null(),
                    p_buffer_info: addr_of!(buffer_info),
                    p_texel_buffer_view: null(),
                },
                WriteDescriptorSet {
                    s_type: StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: addr_of!(image_info),
                    p_buffer_info: null(),
                    p_texel_buffer_view: null(),
                },
            ],
            &[],
        );

*/

/*
    let unif_buffer = ...
    let sampler = ...
    let storage_buffer = ...

    gpu.create_descriptor_set({
        DescriptorInfo {
            binding: 0,
            element_type: DescriptorType::UniformBuffer(unif_buffer)
            binding_stage: ShaderStage::Vertex,
        },
        DescriptorInfo {
            binding: 1,
            element_type: DescriptorType::Sampler(sampler)
            binding_stage: ShaderStage::Vertex,
        },
        DescriptorInfo {
            binding: 2,
            element_type: DescriptorType::StorageBuffer(storage_buffer)
            binding_stage: ShaderStage::Vertex,
        }
    });
*/

use std::ptr::addr_of;

use ash::{
    prelude::VkResult,
    vk::{
        self, DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
        DescriptorPoolSize, DescriptorSetLayoutCreateInfo, DescriptorType, StructureType,
    },
    Device,
};

use super::Gpu;

pub struct DescriptorSetAllocation {
    pub owner_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

pub trait DescriptorSetAllocator {
    fn allocate(
        &mut self,
        info: &DescriptorSetLayoutCreateInfo,
    ) -> VkResult<DescriptorSetAllocation>;
    fn deallocate(&mut self, descriptor_set: &DescriptorSetAllocation) -> VkResult<()>;
}

pub struct PooledDescriptorSetAllocator {
    usable_descriptor_pools: Vec<DescriptorPool>,
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
                    max_sets: 1,
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
}

impl PooledDescriptorSetAllocator {
    pub fn new(device: Device) -> VkResult<Self> {
        let mut me = Self {
            usable_descriptor_pools: vec![],
            device,
        };

        me.allocate_new_descriptor_pool()?;

        Ok(me)
    }
}

impl DescriptorSetAllocator for PooledDescriptorSetAllocator {
    fn allocate(
        &mut self,
        info: &DescriptorSetLayoutCreateInfo,
    ) -> VkResult<DescriptorSetAllocation> {
        let descriptor_set_layout =
            unsafe { self.device.create_descriptor_set_layout(&info, None) }?;

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
