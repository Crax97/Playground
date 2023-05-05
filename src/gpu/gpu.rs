use std::{
    ffi::{CStr, CString},
    num::NonZeroU32,
    ops::{Deref, DerefMut},
    ptr::{addr_of, addr_of_mut, null},
    sync::Arc,
};

use anyhow::Result;
use ash::{
    extensions::khr::{Surface, Swapchain},
    prelude::*,
    vk::{
        make_api_version, ApplicationInfo, AttachmentDescription, AttachmentDescriptionFlags,
        AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Bool32, ComponentMapping,
        ComponentSwizzle, CompositeAlphaFlagsKHR, DeviceCreateFlags, DeviceCreateInfo,
        DeviceQueueCreateFlags, DeviceQueueCreateInfo, ExtensionProperties, Extent2D, Fence,
        FenceCreateFlags, Format, Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo,
        Image, ImageAspectFlags, ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageView,
        ImageViewCreateFlags, ImageViewCreateInfo, ImageViewType, InstanceCreateFlags,
        InstanceCreateInfo, KhrSurfaceFn, MemoryHeap, MemoryHeapFlags,
        PFN_vkGetPhysicalDeviceSurfaceSupportKHR, PhysicalDevice, PhysicalDeviceFeatures,
        PhysicalDeviceProperties, PhysicalDeviceType, PipelineBindPoint, PresentInfoKHR,
        PresentModeKHR, Queue, QueueFlags, RenderPass, RenderPassCreateFlags, RenderPassCreateInfo,
        SampleCountFlags, Semaphore, SemaphoreCreateFlags, SemaphoreCreateInfo, SharingMode,
        StructureType, SubpassDescription, SubpassDescriptionFlags, SurfaceCapabilitiesKHR,
        SurfaceFormatKHR, SurfaceKHR, SwapchainCreateFlagsKHR, SwapchainCreateInfoKHR,
        SwapchainKHR, API_VERSION_1_3,
    },
    *,
};

use log::{error, info, trace, warn};
use once_cell::sync::OnceCell;
use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use thiserror::Error;
use winit::window::Window;

const KHRONOS_VALIDATION_LAYER: &'static str = "VK_LAYER_KHRONOS_validation";

mod util {
    use ash::vk::{ColorSpaceKHR, Format, PresentModeKHR, SurfaceFormatKHR};

    pub(super) fn stringify_present_mode(mode: PresentModeKHR) -> &'static str {
        match mode {
            PresentModeKHR::FIFO => "FIFO",
            PresentModeKHR::FIFO_RELAXED => "FIFO_RELAXED",
            PresentModeKHR::MAILBOX => "MAILBOX",
            PresentModeKHR::IMMEDIATE => "IMMEDIATE",
            PresentModeKHR::SHARED_CONTINUOUS_REFRESH => "SHARED_CONTINUOUS_REFRESH",
            PresentModeKHR::SHARED_DEMAND_REFRESH => "SHARED_DEMAND_REFRESH",
            _ => unreachable!(),
        }
    }

    pub(super) fn stringify_presentation_format(format: SurfaceFormatKHR) -> String {
        format!(
            "Image format: {:?}, Color space: {:?}",
            format.format, format.color_space
        )
    }
}

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

pub struct Gpu {
    pub entry: Entry,
    pub instance: Instance,
    pub logical_device: Device,
    pub physical_device: PhysicalDevice,
    pub graphics_queue: Queue,
    pub async_compute_queue: Queue,
    pub transfer_queue: Queue,
    pub queue_families: QueueFamilies,
    pub description: GpuDescription,

    window: Window,

    extension_surface: Surface,
    swapchain_extension: Swapchain,
    khr_surface: SurfaceKHR,
    present_mode: PresentModeKHR,
    swapchain_image_count: NonZeroU32,
    present_extent: Extent2D,
    present_format: SurfaceFormatKHR,

    supported_present_modes: Vec<PresentModeKHR>,
    supported_presentation_formats: Vec<SurfaceFormatKHR>,
    device_capabilities: SurfaceCapabilitiesKHR,
    render_pass: RenderPass,
    current_swapchain: SwapchainKHR,
    current_swapchain_images: Vec<Image>,
    current_swapchain_image_views: Vec<ImageView>,
    current_swapchain_framebuffers: Vec<Framebuffer>,

    next_image_fence: Fence,
    in_flight_fence: Fence,
    render_finished_semaphore: Semaphore,
}

pub struct GpuConfiguration<'a> {
    pub app_name: &'a str,
    pub engine_name: &'a str,
    pub enable_validation_layer: bool,
    pub window: Window,
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

        let khr_surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                configuration.window.raw_display_handle(),
                configuration.window.raw_window_handle(),
                None,
            )
        }
        .unwrap();
        let swapchain_extension = Swapchain::new(&instance, &logical_device);

        let extension_surface = Surface::new(&entry, &instance);

        let (supported_formats, device_capabilities, supported_present_modes) = unsafe {
            let supported_formats = extension_surface.get_physical_device_surface_formats(
                physical_device.physical_device,
                khr_surface,
            )?;
            let device_capabilities = extension_surface.get_physical_device_surface_capabilities(
                physical_device.physical_device,
                khr_surface,
            )?;
            let supported_present_modes = extension_surface
                .get_physical_device_surface_present_modes(
                    physical_device.physical_device,
                    khr_surface,
                )?;
            (
                supported_formats,
                device_capabilities,
                supported_present_modes,
            )
        };

        let swapchain_format = supported_formats[0];

        let next_image_fence = unsafe {
            let create_info = vk::FenceCreateInfo {
                s_type: StructureType::FENCE_CREATE_INFO,
                p_next: null(),
                flags: FenceCreateFlags::empty(),
            };
            logical_device.create_fence(&create_info, None)
        }?;
        let in_flight_fence = unsafe {
            let create_info = vk::FenceCreateInfo {
                s_type: StructureType::FENCE_CREATE_INFO,
                p_next: null(),
                flags: FenceCreateFlags::empty(),
            };
            logical_device.create_fence(&create_info, None)
        }?;

        let render_finished_semaphore = unsafe {
            logical_device.create_semaphore(
                &SemaphoreCreateInfo {
                    s_type: StructureType::SEMAPHORE_CREATE_INFO,
                    p_next: null(),
                    flags: SemaphoreCreateFlags::empty(),
                },
                None,
            )?
        };

        let pass_info = RenderPassCreateInfo {
            s_type: StructureType::RENDER_PASS_CREATE_INFO,
            p_next: null(),
            flags: RenderPassCreateFlags::empty(),
            attachment_count: 1,
            p_attachments: &[AttachmentDescription {
                flags: AttachmentDescriptionFlags::empty(),
                format: swapchain_format.format,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }] as *const AttachmentDescription,
            subpass_count: 1,
            p_subpasses: &[SubpassDescription {
                flags: SubpassDescriptionFlags::empty(),
                pipeline_bind_point: PipelineBindPoint::GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: null(),
                color_attachment_count: 1,
                p_color_attachments: &[AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }] as *const AttachmentReference,
                p_resolve_attachments: null(),
                p_depth_stencil_attachment: null(),
                preserve_attachment_count: vk::FALSE,
                p_preserve_attachments: null(),
            }] as *const SubpassDescription,
            dependency_count: 0,
            p_dependencies: null(),
        };
        let render_pass = unsafe { logical_device.create_render_pass(&pass_info, None)? };
        let present_extent = Extent2D {
            width: configuration.window.outer_size().width,
            height: configuration.window.outer_size().height,
        };

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
            window: configuration.window,
            extension_surface,
            swapchain_extension,

            supported_present_modes,
            supported_presentation_formats: supported_formats,
            device_capabilities,

            present_mode: PresentModeKHR::FIFO,
            khr_surface,
            render_pass,
            present_extent,
            present_format: swapchain_format,
            swapchain_image_count: NonZeroU32::new(2).unwrap(),
            current_swapchain: SwapchainKHR::null(),
            current_swapchain_images: vec![],
            current_swapchain_image_views: vec![],
            current_swapchain_framebuffers: vec![],
            next_image_fence,
            in_flight_fence,
            render_finished_semaphore,
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

        let device =
            unsafe { instance.create_device(selected_device.physical_device, &create_info, None) };
        device
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

    pub fn presentation_surface(&self) -> SurfaceKHR {
        self.khr_surface
    }

    pub fn swapchain_format(&self) -> Format {
        self.present_format.format
    }

    pub(crate) fn in_flight_fence(&self) -> vk::Fence {
        self.in_flight_fence
    }

    pub fn render_finished_semaphore(&self) -> &Semaphore {
        &self.render_finished_semaphore
    }

    fn log_supported_features(&self) {
        info!("Device supports the following present modes:");
        for present_mode in &self.supported_present_modes {
            info!("\t{}", util::stringify_present_mode(*present_mode));
        }

        info!("Device supports the following presentation formats:");
        for presentation_format in &self.supported_presentation_formats {
            info!(
                "\t{}",
                util::stringify_presentation_format(*presentation_format)
            );
        }

        info!("Device has the folowing limits:");
        info!(
            "\tMin/Max swapchain images: {}/{}",
            &self.device_capabilities.min_image_count, self.device_capabilities.max_image_count
        );
        info!(
            "\tMin/Max swapchain extents: {}x{}/{}x{}",
            self.device_capabilities.min_image_extent.width,
            self.device_capabilities.min_image_extent.height,
            self.device_capabilities.max_image_extent.width,
            self.device_capabilities.max_image_extent.height
        );
    }

    pub fn select_present_mode(&mut self, present_mode: PresentModeKHR) -> VkResult<()> {
        self.present_mode = present_mode;
        self.recreate_swapchain()
    }

    pub fn get_next_swapchain_image(&mut self) -> VkResult<(u32, Framebuffer, ImageView)> {
        loop {
            let (next_image, suboptimal) = unsafe {
                self.swapchain_extension.acquire_next_image(
                    self.current_swapchain,
                    200000,
                    Semaphore::null(),
                    self.next_image_fence,
                )
            }?;
            unsafe {
                self.logical_device
                    .wait_for_fences(&[self.next_image_fence], true, 200000)?;
                self.logical_device.reset_fences(&[self.next_image_fence])?;
            }
            if !suboptimal {
                let image_view = self.current_swapchain_image_views.get(next_image as usize);
                let framebuffer = self.current_swapchain_framebuffers[next_image as usize];
                return Ok((next_image, framebuffer, *image_view.unwrap()));
            }
            self.recreate_swapchain()?;
        }
    }

    pub(crate) fn render_pass(&self) -> &vk::RenderPass {
        &self.render_pass
    }

    pub(crate) fn extents(&self) -> Extent2D {
        self.present_extent.clone()
    }

    pub(crate) fn present(&self, index: u32) -> VkResult<bool> {
        unsafe {
            let mut result = ash::vk::Result::SUCCESS;
            self.swapchain_extension.queue_present(
                self.graphics_queue,
                &PresentInfoKHR {
                    s_type: StructureType::PRESENT_INFO_KHR,
                    p_next: null(),
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &self.render_finished_semaphore as *const Semaphore,
                    swapchain_count: 1,
                    p_swapchains: &self.current_swapchain as *const SwapchainKHR,
                    p_image_indices: &index as *const u32,
                    p_results: &mut result as *mut ash::vk::Result,
                },
            )
        }
    }

    fn recreate_swapchain(&mut self) -> VkResult<()> {
        unsafe {
            self.swapchain_extension
                .destroy_swapchain(self.current_swapchain, None);
        }

        let khr_surface = unsafe {
            self.extension_surface
                .destroy_surface(self.khr_surface, None);
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                self.window.raw_display_handle(),
                self.window.raw_window_handle(),
                None,
            )
        }
        .unwrap();
        self.khr_surface = khr_surface;

        let (supported_formats, device_capabilities, supported_present_modes) = unsafe {
            let supported_formats = self
                .extension_surface
                .get_physical_device_surface_formats(self.physical_device, khr_surface)?;
            let device_capabilities = self
                .extension_surface
                .get_physical_device_surface_capabilities(self.physical_device, khr_surface)?;
            let supported_present_modes = self
                .extension_surface
                .get_physical_device_surface_present_modes(self.physical_device, khr_surface)?;
            (
                supported_formats,
                device_capabilities,
                supported_present_modes,
            )
        };

        self.supported_presentation_formats = supported_formats;
        self.device_capabilities = device_capabilities;
        self.supported_present_modes = supported_present_modes;
        self.present_format = self.supported_presentation_formats[0];
        self.validate_selected_swapchain_settings();

        let swapchain_creation_info = self.make_swapchain_creation_info();
        let swapchain = unsafe {
            self.swapchain_extension
                .create_swapchain(&swapchain_creation_info, None)?
        };
        self.current_swapchain = swapchain;
        trace!(
            "Created a new swapchain with present format {:?}, present mode {:?} and present extents {:?}",
            &self.present_format, &self.present_mode, &self.present_extent
        );

        self.recreate_swapchain_images()?;
        self.recreate_swapchain_image_views()?;
        self.recreate_swapchain_framebuffers()?;

        Ok(())
    }

    fn recreate_swapchain_images(&mut self) -> VkResult<()> {
        let images = unsafe {
            self.swapchain_extension
                .get_swapchain_images(self.current_swapchain)
        }?;
        self.current_swapchain_images = images;
        Ok(())
    }

    fn recreate_swapchain_image_views(&mut self) -> VkResult<()> {
        self.current_swapchain_image_views
            .resize(self.current_swapchain_images.len(), ImageView::null());
        for (i, image) in self.current_swapchain_images.iter().enumerate() {
            let view_info = ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: null(),
                flags: ImageViewCreateFlags::empty(),
                image: *image,
                view_type: ImageViewType::TYPE_2D,
                format: self.present_format.format,
                components: ComponentMapping {
                    r: ComponentSwizzle::R,
                    g: ComponentSwizzle::G,
                    b: ComponentSwizzle::B,
                    a: ComponentSwizzle::A,
                },
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let view: ImageView =
                unsafe { self.logical_device.create_image_view(&view_info, None) }?;
            self.current_swapchain_image_views[i] = view;
        }

        Ok(())
    }

    fn recreate_swapchain_framebuffers(&mut self) -> VkResult<()> {
        self.current_swapchain_framebuffers
            .resize(self.current_swapchain_images.len(), Framebuffer::null());
        for (i, image_view) in self.current_swapchain_image_views.iter().enumerate() {
            let create_info = FramebufferCreateInfo {
                s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: null(),
                flags: FramebufferCreateFlags::empty(),
                render_pass: self.render_pass,
                attachment_count: 1,
                p_attachments: image_view as *const ImageView,
                width: self.present_extent.width,
                height: self.present_extent.height,
                layers: 1,
            };

            let fb = unsafe { self.logical_device.create_framebuffer(&create_info, None) }?;
            self.current_swapchain_framebuffers[i] = fb;
        }

        Ok(())
    }

    fn make_swapchain_creation_info(&self) -> SwapchainCreateInfoKHR {
        let swapchain_creation_info = SwapchainCreateInfoKHR {
            s_type: StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: null(),
            flags: SwapchainCreateFlagsKHR::default(),
            surface: self.khr_surface,
            min_image_count: self.swapchain_image_count.get(),
            image_format: self.present_format.format,
            image_color_space: self.present_format.color_space,
            image_extent: self.present_extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: null(),
            pre_transform: self.device_capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: self.present_mode,
            clipped: vk::TRUE,
            old_swapchain: SwapchainKHR::null(),
        };
        swapchain_creation_info
    }

    fn validate_selected_swapchain_settings(&mut self) {
        if !self.supported_present_modes.contains(&self.present_mode) {
            warn!(
                "Device does not support extension mode {:?}, selecting FIFO, which must be supported as per specification",
                self.present_mode
            );
            self.present_mode = PresentModeKHR::FIFO
        };

        if !self
            .supported_presentation_formats
            .contains(&self.present_format)
        {
            warn!(
                "Device does not support present format {:?}, selecting the first available one",
                &self.present_format
            );
            self.present_format = self.supported_presentation_formats[0];
        }

        if self.swapchain_image_count.get() < self.device_capabilities.min_image_count
            || self.swapchain_image_count.get() > self.device_capabilities.max_image_count
        {
            warn!(
                "Device does not support less than {} / more than {} swapchain images! Clamping",
                self.device_capabilities.min_image_count, self.device_capabilities.max_image_count
            );
            self.swapchain_image_count = self.swapchain_image_count.clamp(
                NonZeroU32::new(self.device_capabilities.min_image_count).unwrap(),
                NonZeroU32::new(self.device_capabilities.max_image_count).unwrap(),
            );
        }

        let min_exent = self.device_capabilities.min_image_extent;
        let max_exent = self.device_capabilities.max_image_extent;
        let current_extent = self.present_extent;
        if current_extent.width < min_exent.width
            || current_extent.height < min_exent.height
            || current_extent.width > max_exent.width
            || current_extent.height > max_exent.height
        {
            warn!(
                "Device does not support extents smaller than {:?} / greather than {:?}! Clamping",
                self.device_capabilities.min_image_extent,
                self.device_capabilities.max_image_extent
            );

            self.present_extent = Extent2D {
                width: self.present_extent.width.clamp(
                    self.device_capabilities.min_image_extent.width,
                    self.device_capabilities.max_image_extent.width,
                ),
                height: self.present_extent.height.clamp(
                    self.device_capabilities.min_image_extent.height,
                    self.device_capabilities.max_image_extent.height,
                ),
            }
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl AsRef<Device> for Gpu {
    fn as_ref(&self) -> &Device {
        &self.logical_device
    }
}
