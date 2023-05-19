use std::{
    cell::RefCell,
    ffi::{CStr, CString},
    ops::Deref,
    ptr::{addr_of, null},
    rc::Rc,
    sync::Arc,
};

use anyhow::Result;
use ash::{
    prelude::*,
    vk::{
        make_api_version, AccessFlags, ApplicationInfo, BufferCreateFlags, BufferUsageFlags,
        CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo,
        CommandPoolResetFlags, DependencyFlags, DescriptorBufferInfo, DescriptorImageInfo,
        DeviceCreateFlags, DeviceCreateInfo, DeviceQueueCreateFlags, DeviceQueueCreateInfo,
        Extent3D, Fence, FramebufferCreateFlags, ImageAspectFlags, ImageCreateFlags, ImageLayout,
        ImageMemoryBarrier, ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageType,
        ImageViewCreateFlags, ImageViewType, InstanceCreateFlags, InstanceCreateInfo, MemoryHeap,
        MemoryHeapFlags, Offset3D, PhysicalDevice, PhysicalDeviceFeatures,
        PhysicalDeviceProperties, PhysicalDeviceType, PipelineStageFlags, Queue, QueueFlags,
        SampleCountFlags, SamplerCreateInfo, ShaderModuleCreateFlags, SharingMode, StructureType,
        SubmitInfo, WriteDescriptorSet, API_VERSION_1_3,
    },
    *,
};

use log::{error, trace};
use raw_window_handle::HasRawDisplayHandle;
use thiserror::Error;
use winit::window::Window;

use crate::{GpuFramebuffer, GpuImageView, GpuShaderModule, RenderPass};

use super::descriptor_set::PooledDescriptorSetAllocator;

use super::{
    allocator::{GpuAllocator, PasstroughAllocator},
    descriptor_set::DescriptorSetAllocator,
    resource::{ResourceHandle, ResourceMap},
    AllocationRequirements, DescriptorSetInfo, GpuBuffer, GpuDescriptorSet, GpuImage, GpuSampler,
    MemoryDomain,
};

const KHRONOS_VALIDATION_LAYER: &'static str = "VK_LAYER_KHRONOS_validation";

pub struct GpuDescription {
    name: String,
}
impl GpuDescription {
    fn new(physical_device: &SelectedPhysicalDevice) -> Self {
        let name =
            unsafe { CStr::from_ptr(physical_device.device_properties.device_name.as_ptr()) };
        let name = name.to_str().expect("Invalid device name");
        let name = String::from(name);

        Self { name }
    }
}

pub struct GpuState {
    pub entry: Entry,
    pub instance: Instance,
    pub logical_device: Device,
    pub physical_device: SelectedPhysicalDevice,
    pub graphics_queue: Queue,
    pub async_compute_queue: Queue,
    pub transfer_queue: Queue,
    pub queue_families: QueueFamilies,
    pub description: GpuDescription,
    pub gpu_memory_allocator: Arc<RefCell<dyn GpuAllocator>>,
    pub descriptor_set_allocator: Arc<RefCell<dyn DescriptorSetAllocator>>,
}

pub struct GpuThreadLocalState {
    pub graphics_command_pool: vk::CommandPool,
    pub compute_command_pool: vk::CommandPool,
    pub transfer_command_pool: vk::CommandPool,

    shared_state: Arc<GpuState>,
}

impl GpuThreadLocalState {
    pub fn new(shared_state: Arc<GpuState>) -> VkResult<Self> {
        let graphics_command_pool = unsafe {
            shared_state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: null(),
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: shared_state.queue_families.graphics_family.index,
                },
                None,
            )
        }?;
        let compute_command_pool = unsafe {
            shared_state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: null(),
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: shared_state.queue_families.async_compute_family.index,
                },
                None,
            )
        }?;
        let transfer_command_pool = unsafe {
            shared_state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: null(),
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: shared_state.queue_families.transfer_family.index,
                },
                None,
            )
        }?;

        Ok(Self {
            graphics_command_pool,
            compute_command_pool,
            transfer_command_pool,
            shared_state,
        })
    }
}

impl Drop for GpuThreadLocalState {
    fn drop(&mut self) {
        let device = &self.shared_state.logical_device;
        unsafe {
            device
                .device_wait_idle()
                .expect("Failed to wait for device while dropping thread local state");
            device.destroy_command_pool(self.graphics_command_pool, None);
            device.destroy_command_pool(self.compute_command_pool, None);
            device.destroy_command_pool(self.transfer_command_pool, None);
        }
    }
}

pub struct Gpu {
    pub(super) state: Arc<GpuState>,
    pub(super) thread_local_state: GpuThreadLocalState,
    pub(super) staging_buffer: ResourceHandle<GpuBuffer>,
    pub(super) resource_map: Rc<ResourceMap>,
}

pub struct GpuConfiguration<'a> {
    pub app_name: &'a str,
    pub engine_name: &'a str,
    pub enable_validation_layer: bool,
    pub window: &'a Window,
}

#[derive(Error, Debug, Clone)]
pub enum GpuError {
    #[error("No physical devices were found on this machine")]
    NoPhysicalDevices,

    #[error("A suitable device with the requested capabilities was not found")]
    NoSuitableDevice,

    #[error("One or more queue families aren't supported")]
    NoQueueFamilyFound(
        Option<(u32, vk::QueueFamilyProperties)>,
        Option<(u32, vk::QueueFamilyProperties)>,
        Option<(u32, vk::QueueFamilyProperties)>,
    ),
}

#[derive(Clone, Copy, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct QueueFamilies {
    pub graphics_family: QueueFamily,
    pub async_compute_family: QueueFamily,
    pub transfer_family: QueueFamily,
}

#[derive(Clone, Copy, Debug)]
pub struct SelectedPhysicalDevice {
    pub physical_device: PhysicalDevice,
    pub device_properties: PhysicalDeviceProperties,
    pub device_features: PhysicalDeviceFeatures,
}

impl QueueFamilies {
    pub fn is_valid(&self) -> bool {
        self.graphics_family.count > 0
            && self.async_compute_family.count > 0
            && self.transfer_family.count > 0
    }
}

impl Gpu {
    pub fn new(configuration: GpuConfiguration) -> Result<Self> {
        let entry = Entry::linked();

        let instance_extensions =
            ash_window::enumerate_required_extensions(configuration.window.raw_display_handle())?
                .iter()
                .map(|c_ext| unsafe { CStr::from_ptr(*c_ext) })
                .map(|c_str| c_str.to_string_lossy().to_string())
                .collect::<Vec<_>>();

        Self::ensure_required_instance_extensions_are_available(&instance_extensions, &entry)?;

        let instance = Self::create_instance(&entry, &configuration, &instance_extensions)?;
        trace!("Created instance");

        let device_extensions = vec!["VK_KHR_swapchain".into()];

        let physical_device = Self::select_discrete_physical_device(&instance)?;
        trace!("Created physical device");

        Self::log_physical_device_memory(&physical_device, instance.clone());

        let description = GpuDescription::new(&physical_device);

        trace!("Created presentation surface");

        let queue_families = Self::select_queue_families_indices(&physical_device, &instance)?;
        if !queue_families.is_valid() {
            log::error!("Queue configurations are invalid!");
        }

        Self::ensure_required_device_extensions_are_available(
            &device_extensions,
            &instance,
            &physical_device,
        )?;

        let logical_device = Self::create_device(
            &configuration,
            &device_extensions,
            &instance,
            physical_device,
            &queue_families,
        )?;
        trace!("Created logical device");

        let (graphics_queue, async_compute_queue, transfer_queue) =
            Self::get_device_queues(&logical_device, &queue_families)?;
        trace!("Created queues");

        trace!(
            "Created a GPU from a device with name '{}'",
            &description.name
        );

        let gpu_memory_allocator =
            PasstroughAllocator::new(&instance, physical_device.physical_device, &logical_device)?;

        let descriptor_set_allocator = PooledDescriptorSetAllocator::new(logical_device.clone())?;

        let state = Arc::new(GpuState {
            entry,
            instance,
            logical_device,
            physical_device,
            graphics_queue,
            async_compute_queue,
            transfer_queue,
            description,
            queue_families,
            gpu_memory_allocator: Arc::new(RefCell::new(gpu_memory_allocator)),
            descriptor_set_allocator: Arc::new(RefCell::new(descriptor_set_allocator)),
        });

        let thread_local_state = GpuThreadLocalState::new(state.clone())?;

        let staging_buffer = create_staging_buffer(&state)?;
        let resource_map = ResourceMap::new();
        let staging_buffer = resource_map.add(staging_buffer);
        Ok(Gpu {
            state,
            thread_local_state,
            staging_buffer,
            resource_map: Rc::new(resource_map),
        })
    }

    pub fn reset_state(&self) -> VkResult<()> {
        unsafe {
            self.vk_logical_device().reset_command_pool(
                self.thread_local_state.graphics_command_pool,
                CommandPoolResetFlags::empty(),
            )
        }
    }

    fn create_instance(
        entry: &Entry,
        configuration: &GpuConfiguration,
        instance_extensions: &Vec<String>,
    ) -> VkResult<Instance> {
        let vk_layer_khronos_validation = CString::new(KHRONOS_VALIDATION_LAYER).unwrap();
        let vk_layer_khronos_validation = vk_layer_khronos_validation.as_ptr();

        let required_extensions: Vec<CString> = instance_extensions
            .iter()
            .map(|str| CString::new(str.clone()).unwrap())
            .collect();
        let required_extensions: Vec<*const i8> =
            required_extensions.iter().map(|ext| ext.as_ptr()).collect();

        let app_name =
            CString::new(configuration.app_name).expect("Failed to create valid App name");
        let engine_name =
            CString::new(configuration.engine_name).expect("Failed to create valid Engine Engine");

        let app_info = ApplicationInfo {
            s_type: StructureType::APPLICATION_INFO,
            p_next: null(),
            p_application_name: app_name.as_ptr(),
            application_version: make_api_version(0, 0, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: make_api_version(0, 0, 0, 0),
            api_version: API_VERSION_1_3,
        };
        let create_info = InstanceCreateInfo {
            s_type: StructureType::INSTANCE_CREATE_INFO,
            p_next: null(),
            flags: InstanceCreateFlags::empty(),
            p_application_info: addr_of!(app_info),
            enabled_layer_count: if configuration.enable_validation_layer {
                1
            } else {
                0
            },
            pp_enabled_layer_names: if configuration.enable_validation_layer {
                addr_of!(vk_layer_khronos_validation)
            } else {
                null()
            },
            enabled_extension_count: required_extensions.len() as u32,
            pp_enabled_extension_names: required_extensions.as_ptr(),
        };

        unsafe { entry.create_instance(&create_info, None) }
    }

    fn select_discrete_physical_device(
        instance: &Instance,
    ) -> Result<SelectedPhysicalDevice, GpuError> {
        unsafe {
            let devices = instance
                .enumerate_physical_devices()
                .map_err(|_| GpuError::NoPhysicalDevices)?;

            for physical_device in devices {
                let device_properties = instance.get_physical_device_properties(physical_device);
                let device_features = instance.get_physical_device_features(physical_device);

                if device_properties.device_type == PhysicalDeviceType::DISCRETE_GPU {
                    return Ok(SelectedPhysicalDevice {
                        physical_device,
                        device_properties,
                        device_features,
                    });
                }
            }
        }
        Err(GpuError::NoSuitableDevice)
    }

    fn select_queue_families_indices(
        device: &SelectedPhysicalDevice,
        instance: &Instance,
    ) -> Result<QueueFamilies, GpuError> {
        let all_queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device.physical_device) };

        let mut graphics_queue = None;
        let mut async_compute_queue = None;
        let mut transfer_queue = None;

        for (index, queue_family) in all_queue_families.iter().enumerate() {
            if queue_family.queue_count == 0 {
                continue;
            }

            if queue_family.queue_flags.intersects(QueueFlags::GRAPHICS) {
                graphics_queue = Some((index as u32, *queue_family));
            } else if queue_family.queue_flags.intersects(QueueFlags::COMPUTE) {
                async_compute_queue = Some((index as u32, *queue_family));
            } else if queue_family.queue_flags.intersects(QueueFlags::TRANSFER) {
                transfer_queue = Some((index as u32, *queue_family));
            }
        }

        match (graphics_queue, async_compute_queue, transfer_queue) {
            (Some(g), Some(a), Some(t)) => Ok(QueueFamilies {
                graphics_family: QueueFamily {
                    index: g.0,
                    count: g.1.queue_count,
                },
                async_compute_family: QueueFamily {
                    index: a.0,
                    count: a.1.queue_count,
                },
                transfer_family: QueueFamily {
                    index: t.0,
                    count: t.1.queue_count,
                },
            }),
            _ => Err(GpuError::NoQueueFamilyFound(
                graphics_queue,
                async_compute_queue,
                transfer_queue,
            )),
        }
    }

    fn create_device(
        configuration: &GpuConfiguration,
        device_extensions: &Vec<String>,
        instance: &Instance,
        selected_device: SelectedPhysicalDevice,
        queue_indices: &QueueFamilies,
    ) -> VkResult<Device> {
        let priority_one: f32 = 1.0;
        let vk_layer_khronos_validation = CString::new(KHRONOS_VALIDATION_LAYER).unwrap();
        let vk_layer_khronos_validation = vk_layer_khronos_validation.as_ptr();

        let c_string_device_extensions: Vec<CString> = device_extensions
            .iter()
            .map(|e| CString::new(e.as_str()).unwrap())
            .collect();

        let c_ptr_device_extensions: Vec<*const i8> = c_string_device_extensions
            .iter()
            .map(|cstr| cstr.as_ptr())
            .collect();

        let make_queue_create_info = |index| DeviceQueueCreateInfo {
            s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: null(),
            flags: DeviceQueueCreateFlags::empty(),
            queue_family_index: index,
            queue_count: 1,
            p_queue_priorities: addr_of!(priority_one),
        };
        let queue_create_infos = [
            make_queue_create_info(queue_indices.graphics_family.index),
            make_queue_create_info(queue_indices.async_compute_family.index),
            make_queue_create_info(queue_indices.transfer_family.index),
        ];

        let mut device_features = PhysicalDeviceFeatures::default();
        device_features.sampler_anisotropy = vk::TRUE;

        let create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            p_next: null(),
            flags: DeviceCreateFlags::empty(),
            queue_create_info_count: 3,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: if configuration.enable_validation_layer {
                1
            } else {
                0
            },
            pp_enabled_layer_names: if configuration.enable_validation_layer {
                addr_of!(vk_layer_khronos_validation)
            } else {
                null()
            },
            enabled_extension_count: c_ptr_device_extensions.len() as u32,
            pp_enabled_extension_names: c_ptr_device_extensions.as_ptr(),
            p_enabled_features: addr_of!(device_features),
        };

        unsafe { instance.create_device(selected_device.physical_device, &create_info, None) }
    }

    fn ensure_required_instance_extensions_are_available(
        requested_extensions: &[String],
        entry: &Entry,
    ) -> VkResult<()> {
        let all_extensions = entry.enumerate_instance_extension_properties(None)?;
        trace!(
            "Requested instance extensions: {}",
            requested_extensions.join(",")
        );
        let mut all_extensions_c_names = all_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) });

        for requested_extension in requested_extensions {
            let required_c_name =
                unsafe { CString::from_vec_unchecked(requested_extension.clone().into_bytes()) };
            if all_extensions_c_names
                .find(|name| *name == required_c_name.as_c_str())
                .is_none()
            {
                error!("Instance extension {:?} is not supported", required_c_name);
            }
        }

        Ok(())
    }

    fn get_device_queues(
        device: &Device,
        queues: &QueueFamilies,
    ) -> Result<(Queue, Queue, Queue), GpuError> {
        let graphics_queue = unsafe { device.get_device_queue(queues.graphics_family.index, 0) };
        let async_compute_queue =
            unsafe { device.get_device_queue(queues.async_compute_family.index, 0) };
        let transfer_queue = unsafe { device.get_device_queue(queues.transfer_family.index, 0) };
        Ok((graphics_queue, async_compute_queue, transfer_queue))
    }

    fn ensure_required_device_extensions_are_available(
        device_extensions: &[String],
        instance: &Instance,
        physical_device: &SelectedPhysicalDevice,
    ) -> VkResult<()> {
        trace!(
            "Requested device extensions: {}",
            device_extensions.join(",")
        );
        let all_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device.physical_device)
        }?;
        let mut all_extensions_c_names = all_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) });

        for requested_extension in device_extensions {
            let required_c_name =
                unsafe { CString::from_vec_unchecked(requested_extension.clone().into_bytes()) };

            if all_extensions_c_names
                .find(|name| *name == required_c_name.as_c_str())
                .is_none()
            {
                error!("Device extension {:?} is not supported", required_c_name);
            }
        }

        Ok(())
    }

    pub(crate) fn vk_logical_device(&self) -> Device {
        self.state.logical_device.clone()
    }

    pub(crate) fn graphics_queue_family_index(&self) -> u32 {
        self.state.queue_families.graphics_family.index
    }

    pub(crate) fn graphics_queue(&self) -> Queue {
        self.state.graphics_queue
    }

    fn log_physical_device_memory(physical_device: &SelectedPhysicalDevice, instance: Instance) {
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device.physical_device)
        };

        let stringify_memory_heap = |heap: MemoryHeap| {
            let flags_str = {
                let mut s = String::from("{ ");
                if heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL) {
                    s += "DEVICE_LOCAL | ";
                }
                if heap.flags.contains(MemoryHeapFlags::MULTI_INSTANCE) {
                    s += "MULTI_INSTANCE | ";
                }
                if heap.flags.contains(MemoryHeapFlags::MULTI_INSTANCE_KHR) {
                    s += "MULTI_INSTANCE_KHR | ";
                }
                if heap.flags.contains(MemoryHeapFlags::RESERVED_2_KHR) {
                    s += "RESERVED_2_KHR ";
                }

                s += "}";
                s
            };
            format!("size: {} flags {}", heap.size, flags_str)
        };

        trace!(
            "Device has {} memory types:",
            memory_properties.memory_type_count
        );
        let mut s = String::new();
        for i in 0..memory_properties.memory_type_count {
            let memory_type = memory_properties.memory_types[i as usize];
            let memory_heap = memory_properties.memory_heaps[memory_type.heap_index as usize];
            s += format!(
                "\n\t{}) Memory type {:?} Heap info: {}",
                i,
                memory_type,
                stringify_memory_heap(memory_heap)
            )
            .as_str();
        }
        trace!("{}", s);
    }

    fn initialize_descriptor_set(
        &self,
        descriptor_set: &vk::DescriptorSet,
        info: &DescriptorSetInfo,
    ) -> VkResult<()> {
        let mut buffer_descriptors = vec![];
        let mut image_descriptors = vec![];
        info.descriptors.iter().for_each(|i| match &i.element_type {
            super::DescriptorType::UniformBuffer(buf) => buffer_descriptors.push((
                i.binding,
                DescriptorBufferInfo {
                    buffer: self.resource_map.get(&buf.handle).unwrap().inner,
                    offset: buf.offset,
                    range: buf.size,
                },
                vk::DescriptorType::UNIFORM_BUFFER,
            )),
            super::DescriptorType::StorageBuffer(buf) => buffer_descriptors.push((
                i.binding,
                DescriptorBufferInfo {
                    buffer: self.resource_map.get(&buf.handle).unwrap().inner,
                    offset: buf.offset,
                    range: buf.size,
                },
                vk::DescriptorType::STORAGE_BUFFER,
            )),
            super::DescriptorType::Sampler(sam) => image_descriptors.push((
                i.binding,
                DescriptorImageInfo {
                    sampler: self.resource_map.get(&sam.sampler).unwrap().inner,
                    image_view: self.resource_map.get(&sam.image_view).unwrap().inner,
                    image_layout: sam.image_layout,
                },
                vk::DescriptorType::SAMPLER,
            )),
            super::DescriptorType::CombinedImageSampler(sam) => image_descriptors.push((
                i.binding,
                DescriptorImageInfo {
                    sampler: self.resource_map.get(&sam.sampler).unwrap().inner,
                    image_view: self.resource_map.get(&sam.image_view).unwrap().inner,
                    image_layout: sam.image_layout,
                },
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            )),
        });

        let mut write_descriptor_sets = vec![];

        for (bind, desc, ty) in &buffer_descriptors {
            write_descriptor_sets.push(WriteDescriptorSet {
                s_type: StructureType::WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: descriptor_set.clone(),
                dst_binding: *bind,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: *ty,
                p_image_info: std::ptr::null(),
                p_buffer_info: addr_of!(*desc),
                p_texel_buffer_view: std::ptr::null(),
            });
        }
        for (bind, desc, ty) in &image_descriptors {
            write_descriptor_sets.push(WriteDescriptorSet {
                s_type: StructureType::WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: descriptor_set.clone(),
                dst_binding: *bind,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: *ty,
                p_image_info: addr_of!(*desc),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            });
        }
        unsafe {
            self.vk_logical_device()
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }
        Ok(())
    }

    pub fn wait_device_idle(&self) -> VkResult<()> {
        unsafe { self.vk_logical_device().device_wait_idle() }
    }

    pub fn physical_device_properties(&self) -> PhysicalDeviceProperties {
        self.state.physical_device.device_properties
    }

    pub fn create_shader_module(
        &self,
        create_info: &ShaderModuleCreateInfo,
    ) -> VkResult<ResourceHandle<GpuShaderModule>> {
        let code = bytemuck::cast_slice(&create_info.code);
        let p_code = code.as_ptr();

        assert!(
            p_code as u32 % 4 == 0,
            "Pointers to shader modules code must be 4 byte aligned"
        );

        let create_info = vk::ShaderModuleCreateInfo {
            s_type: StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: create_info.flags,
            code_size: create_info.code.len() as _,
            p_code,
        };

        let shader = GpuShaderModule::create(self.vk_logical_device(), &create_info)?;

        Ok(self.resource_map.add(shader))
    }
}

fn create_staging_buffer(state: &Arc<GpuState>) -> VkResult<GpuBuffer> {
    let mb_64 = 1024 * 1024 * 64;
    let family = state.queue_families.clone();
    let create_info = vk::BufferCreateInfo {
        s_type: StructureType::BUFFER_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: BufferCreateFlags::empty(),
        size: mb_64 as u64,
        usage: BufferUsageFlags::TRANSFER_SRC,
        sharing_mode: SharingMode::CONCURRENT,
        queue_family_index_count: 3,
        p_queue_family_indices: [
            family.graphics_family.index,
            family.async_compute_family.index,
            family.transfer_family.index,
        ]
        .as_ptr(),
    };
    let buffer = unsafe { state.logical_device.create_buffer(&create_info, None) }?;
    let memory_requirements =
        unsafe { state.logical_device.get_buffer_memory_requirements(buffer) };
    let allocation_requirements = AllocationRequirements {
        memory_requirements,
        memory_domain: MemoryDomain::HostVisible,
    };
    let allocation = state
        .gpu_memory_allocator
        .borrow_mut()
        .allocate(&state.logical_device, allocation_requirements)?;
    unsafe {
        state
            .logical_device
            .bind_buffer_memory(buffer, allocation.device_memory, 0)
    }?;
    let buffer = GpuBuffer::create(
        state.logical_device.clone(),
        buffer,
        MemoryDomain::HostVisible,
        allocation,
        state.gpu_memory_allocator.clone(),
    )?;
    Ok(buffer)
}

pub struct ImageCreateInfo {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

pub struct ImageViewCreateInfo<'a> {
    pub image: &'a ResourceHandle<GpuImage>,
    pub view_type: ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub subresource_range: ImageSubresourceRange,
}
pub struct BufferCreateInfo {
    pub size: usize,
    pub usage: BufferUsageFlags,
}
pub struct TransitionInfo {
    pub layout: ImageLayout,
    pub access_mask: AccessFlags,
    pub stage_mask: PipelineStageFlags,
}

pub struct FramebufferCreateInfo<'a> {
    pub render_pass: &'a RenderPass,
    pub attachments: &'a [&'a ResourceHandle<GpuImageView>],
    pub width: u32,
    pub height: u32,
}

pub struct ShaderModuleCreateInfo<'a> {
    pub flags: ShaderModuleCreateFlags,
    pub code: &'a [u8],
}

impl Gpu {
    pub fn create_buffer(
        &self,
        create_info: &BufferCreateInfo,
        memory_domain: MemoryDomain,
    ) -> VkResult<ResourceHandle<GpuBuffer>> {
        let family = self.state.queue_families.clone();
        let create_info = vk::BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: BufferCreateFlags::empty(),
            size: create_info.size as u64,
            usage: create_info.usage
                | if memory_domain.contains(MemoryDomain::HostVisible) {
                    BufferUsageFlags::empty()
                } else {
                    BufferUsageFlags::TRANSFER_DST
                },
            sharing_mode: SharingMode::CONCURRENT,
            queue_family_index_count: 3,
            p_queue_family_indices: [
                family.graphics_family.index,
                family.async_compute_family.index,
                family.transfer_family.index,
            ]
            .as_ptr(),
        };
        let buffer = unsafe { self.state.logical_device.create_buffer(&create_info, None) }?;
        let memory_requirements = unsafe {
            self.state
                .logical_device
                .get_buffer_memory_requirements(buffer)
        };

        let allocation_requirements = AllocationRequirements {
            memory_requirements,
            memory_domain,
        };

        let allocation = self
            .state
            .gpu_memory_allocator
            .borrow_mut()
            .allocate(&self.state.logical_device, allocation_requirements)?;
        unsafe {
            self.state
                .logical_device
                .bind_buffer_memory(buffer, allocation.device_memory, 0)
        }?;
        let buffer = GpuBuffer::create(
            self.vk_logical_device(),
            buffer,
            memory_domain,
            allocation,
            self.state.gpu_memory_allocator.clone(),
        )?;

        let id = self.resource_map.add(buffer);
        Ok(id)
    }

    pub fn write_buffer_data<T: Copy>(
        &self,
        buffer: &ResourceHandle<GpuBuffer>,
        data: &[T],
    ) -> VkResult<()> {
        let gpu_buffer = self.resource_map.get(buffer).unwrap();
        if gpu_buffer.memory_domain.contains(MemoryDomain::HostVisible) {
            gpu_buffer.write_data(data);
        } else {
            self.resource_map
                .get(&self.staging_buffer)
                .unwrap()
                .write_data(data);
            self.copy_buffer(
                &self.staging_buffer,
                buffer,
                data.len() * std::mem::size_of::<T>(),
            )?;
        }
        Ok(())
    }
    pub fn create_image(
        &self,
        create_info: &ImageCreateInfo,
        memory_domain: MemoryDomain,
    ) -> VkResult<ResourceHandle<GpuImage>> {
        let image = unsafe {
            let create_info = vk::ImageCreateInfo {
                s_type: StructureType::IMAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ImageCreateFlags::empty(),
                image_type: ImageType::TYPE_2D,
                format: create_info.format,
                extent: Extent3D {
                    width: create_info.width,
                    height: create_info.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCountFlags::TYPE_1,
                tiling: if memory_domain.contains(MemoryDomain::HostVisible) {
                    ImageTiling::LINEAR
                } else {
                    ImageTiling::OPTIMAL
                },
                usage: create_info.usage,
                sharing_mode: SharingMode::EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: addr_of!(self.state.queue_families.graphics_family.index),
                initial_layout: ImageLayout::UNDEFINED,
            };
            self.state.logical_device.create_image(&create_info, None)?
        };
        let memory_requirements = unsafe {
            self.state
                .logical_device
                .get_image_memory_requirements(image)
        };
        let allocation_requirements = AllocationRequirements {
            memory_requirements,
            memory_domain,
        };
        let allocation = self
            .state
            .gpu_memory_allocator
            .borrow_mut()
            .allocate(&self.state.logical_device, allocation_requirements)?;
        unsafe {
            self.state
                .logical_device
                .bind_image_memory(image, allocation.device_memory, 0)
        }?;
        let image: GpuImage = GpuImage::create(
            self,
            image,
            allocation,
            self.state.gpu_memory_allocator.clone(),
        )?;

        let id = self.resource_map.add(image);
        Ok(id)
    }

    pub fn create_image_view(
        &self,
        create_info: &ImageViewCreateInfo,
    ) -> VkResult<ResourceHandle<GpuImageView>> {
        let image = self.resource_map.get(create_info.image).unwrap().inner;
        let create_info = vk::ImageViewCreateInfo {
            s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: ImageViewCreateFlags::empty(),
            image,
            view_type: create_info.view_type,
            format: create_info.format,
            components: create_info.components,
            subresource_range: create_info.subresource_range,
        };

        let resource = GpuImageView::create(self.vk_logical_device(), &create_info)?;
        Ok(self.resource_map.add(resource))
    }
    pub fn create_sampler(
        &self,
        create_info: &SamplerCreateInfo,
    ) -> VkResult<ResourceHandle<GpuSampler>> {
        let sampler = GpuSampler::create(self.vk_logical_device(), create_info)?;
        let id = self.resource_map.add(sampler);
        Ok(id)
    }

    pub fn create_framebuffer(
        &self,
        create_info: &FramebufferCreateInfo,
    ) -> VkResult<ResourceHandle<GpuFramebuffer>> {
        let attachments: Vec<_> = create_info
            .attachments
            .iter()
            .map(|a| self.resource_map.get(a).unwrap().inner)
            .collect();
        let create_info = vk::FramebufferCreateInfo {
            s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: FramebufferCreateFlags::empty(),
            render_pass: create_info.render_pass.inner,

            attachment_count: attachments.len() as _,
            p_attachments: attachments.as_ptr(),
            width: create_info.width,
            height: create_info.height,

            // We only support one single framebuffer
            layers: 1,
        };

        let fb = GpuFramebuffer::create(self.vk_logical_device(), &create_info)?;

        Ok(self.resource_map.add(fb))
    }

    pub fn copy_buffer(
        &self,
        source_buffer: &ResourceHandle<GpuBuffer>,
        dest_buffer: &ResourceHandle<GpuBuffer>,
        size: usize,
    ) -> VkResult<()> {
        unsafe {
            let command_pool = self.state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandPoolCreateFlags::empty(),
                    queue_family_index: self.graphics_queue_family_index(),
                },
                None,
            )?;

            let command_buffer =
                self.state
                    .logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })?[0];

            self.state.logical_device.begin_command_buffer(
                command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )?;

            let src_buffer = self.resource_map.get(source_buffer).unwrap().inner;
            let dst_buffer = self.resource_map.get(dest_buffer).unwrap().inner;

            self.state.logical_device.cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: size as u64,
                }],
            );
            self.state
                .logical_device
                .end_command_buffer(command_buffer)?;
            self.state.logical_device.queue_submit(
                self.state.graphics_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 0,
                    p_wait_semaphores: std::ptr::null(),
                    p_wait_dst_stage_mask: std::ptr::null(),
                    command_buffer_count: 1,
                    p_command_buffers: addr_of!(command_buffer),
                    signal_semaphore_count: 0,
                    p_signal_semaphores: std::ptr::null(),
                }],
                Fence::null(),
            )?;
            self.state
                .logical_device
                .queue_wait_idle(self.graphics_queue())?;

            self.state
                .logical_device
                .free_command_buffers(command_pool, &[command_buffer]);
            self.state
                .logical_device
                .destroy_command_pool(command_pool, None);
        }

        Ok(())
    }

    pub fn transition_image_layout(
        &self,
        image: &ResourceHandle<GpuImage>,
        old_layout: TransitionInfo,
        new_layout: TransitionInfo,
        aspect_mask: ImageAspectFlags,
    ) -> VkResult<()> {
        unsafe {
            let command_pool = self.state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandPoolCreateFlags::empty(),
                    queue_family_index: self.graphics_queue_family_index(),
                },
                None,
            )?;

            let command_buffer =
                self.state
                    .logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })?[0];

            self.state.logical_device.begin_command_buffer(
                command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )?;

            let memory_barrier = ImageMemoryBarrier {
                s_type: StructureType::IMAGE_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: old_layout.access_mask,
                dst_access_mask: new_layout.access_mask,
                old_layout: old_layout.layout,
                new_layout: new_layout.layout,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: *self.resource_map.get(image).unwrap().deref(),
                subresource_range: ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };

            self.state.logical_device.cmd_pipeline_barrier(
                command_buffer,
                old_layout.stage_mask,
                new_layout.stage_mask,
                DependencyFlags::empty(),
                &[],
                &[],
                &[memory_barrier],
            );

            self.state
                .logical_device
                .end_command_buffer(command_buffer)?;
            self.state.logical_device.queue_submit(
                self.state.graphics_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 0,
                    p_wait_semaphores: std::ptr::null(),
                    p_wait_dst_stage_mask: std::ptr::null(),
                    command_buffer_count: 1,
                    p_command_buffers: addr_of!(command_buffer),
                    signal_semaphore_count: 0,
                    p_signal_semaphores: std::ptr::null(),
                }],
                Fence::null(),
            )?;
            self.state
                .logical_device
                .queue_wait_idle(self.graphics_queue())?;

            self.state
                .logical_device
                .free_command_buffers(command_pool, &[command_buffer]);
            self.state
                .logical_device
                .destroy_command_pool(command_pool, None);
        };
        Ok(())
    }

    pub fn copy_buffer_to_image(
        &self,
        source_buffer: &ResourceHandle<GpuBuffer>,
        dest_image: &ResourceHandle<GpuImage>,
        width: u32,
        height: u32,
    ) -> VkResult<()> {
        unsafe {
            let command_pool = self.state.logical_device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandPoolCreateFlags::empty(),
                    queue_family_index: self.graphics_queue_family_index(),
                },
                None,
            )?;

            let command_buffer =
                self.state
                    .logical_device
                    .allocate_command_buffers(&CommandBufferAllocateInfo {
                        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                        p_next: std::ptr::null(),
                        command_pool,
                        level: CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                    })?[0];

            self.state.logical_device.begin_command_buffer(
                command_buffer,
                &CommandBufferBeginInfo {
                    s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )?;

            let src_buffer = self.resource_map.get(source_buffer).unwrap().inner;
            let dst_image = self.resource_map.get(dest_image).unwrap().inner;

            self.state.logical_device.cmd_copy_buffer_to_image(
                command_buffer,
                src_buffer,
                dst_image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: ImageSubresourceLayers {
                        aspect_mask: ImageAspectFlags::COLOR,
                        mip_level: 0,
                        layer_count: 1,
                        base_array_layer: 0,
                    },
                    image_offset: Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                }],
            );
            self.state
                .logical_device
                .end_command_buffer(command_buffer)?;
            self.state.logical_device.queue_submit(
                self.state.graphics_queue,
                &[SubmitInfo {
                    s_type: StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 0,
                    p_wait_semaphores: std::ptr::null(),
                    p_wait_dst_stage_mask: std::ptr::null(),
                    command_buffer_count: 1,
                    p_command_buffers: addr_of!(command_buffer),
                    signal_semaphore_count: 0,
                    p_signal_semaphores: std::ptr::null(),
                }],
                Fence::null(),
            )?;
            self.state
                .logical_device
                .queue_wait_idle(self.graphics_queue())?;

            self.state
                .logical_device
                .free_command_buffers(command_pool, &[command_buffer]);
            self.state
                .logical_device
                .destroy_command_pool(command_pool, None);
        }

        Ok(())
    }

    pub fn create_descriptor_set(
        &self,
        info: &DescriptorSetInfo,
    ) -> VkResult<ResourceHandle<GpuDescriptorSet>> {
        let allocated_descriptor_set = self
            .state
            .descriptor_set_allocator
            .borrow_mut()
            .allocate(info)?;
        self.initialize_descriptor_set(&allocated_descriptor_set.descriptor_set, info)?;
        let descriptor_set = GpuDescriptorSet::create(
            allocated_descriptor_set,
            self.state.descriptor_set_allocator.clone(),
        )?;
        Ok(self.resource_map.add(descriptor_set))
    }
}
