use std::{
    ffi::{CStr, CString},
    num::NonZeroU32,
    ops::{Deref, DerefMut},
    ptr::{addr_of, addr_of_mut, null},
    sync::Arc,
};

use anyhow::Result;
use ash::{
    prelude::*,
    vk::{
        make_api_version, ApplicationInfo, BufferCreateInfo, DeviceCreateFlags, DeviceCreateInfo,
        DeviceQueueCreateFlags, DeviceQueueCreateInfo, Extent2D, Format, ImageView,
        InstanceCreateFlags, InstanceCreateInfo, MemoryHeap, MemoryHeapFlags, PhysicalDevice,
        PhysicalDeviceFeatures, PhysicalDeviceProperties, PhysicalDeviceType, PresentModeKHR,
        Queue, QueueFlags, Semaphore, StructureType, SurfaceKHR, API_VERSION_1_3,
    },
    *,
};

use log::{error, trace};
use raw_window_handle::HasRawDisplayHandle;
use thiserror::Error;
use winit::window::Window;

use super::{
    allocator::{GpuAllocator, PasstroughAllocator},
    resource::{ResourceHandle, ResourceMap},
    types, AllocationRequirements, GpuBuffer, MemoryDomain,
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

pub struct Gpu<A: GpuAllocator> {
    pub entry: Entry,
    pub instance: Instance,
    pub logical_device: Device,
    pub physical_device: PhysicalDevice,
    pub graphics_queue: Queue,
    pub async_compute_queue: Queue,
    pub transfer_queue: Queue,
    pub queue_families: QueueFamilies,
    pub description: GpuDescription,
    pub allocator: A,

    pub resource_map: ResourceMap,
}

pub type SharedGpu = Arc<Gpu<PasstroughAllocator>>;

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

    #[error("Generic")]
    GenericGpuError(String),
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
struct SelectedPhysicalDevice {
    physical_device: PhysicalDevice,
    device_properties: PhysicalDeviceProperties,
    device_features: PhysicalDeviceFeatures,
}

impl QueueFamilies {
    pub fn is_valid(&self) -> bool {
        self.graphics_family.count > 0
            && self.async_compute_family.count > 0
            && self.transfer_family.count > 0
    }
}

impl<A: GpuAllocator> Gpu<A> {
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

        let allocator = A::new(&instance, physical_device.physical_device, &logical_device)?;

        Ok(Gpu {
            entry,
            instance,
            logical_device,
            physical_device: physical_device.physical_device,
            graphics_queue,
            async_compute_queue,
            transfer_queue,
            description,
            queue_families,
            allocator,
            resource_map: ResourceMap::new(),
        })
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
            p_enabled_features: null(),
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
        self.logical_device.clone()
    }

    pub(crate) fn graphics_queue_family_index(&self) -> u32 {
        self.queue_families.graphics_family.index
    }

    pub(crate) fn graphics_queue(&self) -> Queue {
        self.graphics_queue
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

    pub(crate) fn vk_physical_device(&self) -> PhysicalDevice {
        self.physical_device.clone()
    }
}

impl<A: GpuAllocator> Gpu<A> {
    pub fn create_buffer(
        &self,
        create_info: &BufferCreateInfo,
        memory_domain: MemoryDomain,
    ) -> VkResult<ResourceHandle<GpuBuffer>> {
        let buffer = unsafe { self.logical_device.create_buffer(create_info, None) }?;
        let memory_requirements =
            unsafe { self.logical_device.get_buffer_memory_requirements(buffer) };

        let allocation_requirements = AllocationRequirements {
            memory_requirements,
            memory_domain,
        };

        let allocation = self
            .allocator
            .allocate(&self.logical_device, allocation_requirements)?;
        unsafe {
            self.logical_device
                .bind_buffer_memory(buffer, allocation.device_memory, 0)
        }?;
        let buffer = GpuBuffer::create(self, buffer, allocation)?;

        let id = self.resource_map.add(buffer);
        Ok(id)
    }
}
