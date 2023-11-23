use std::sync::{Arc, Mutex};

use egui::{Context, FontDefinitions, FullOutput};
use egui_winit::EventResponse;
use egui_winit_ash_integration::Integration;
use gpu::{VkCommandBuffer, VkGpu, VkSwapchain};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::{event::WindowEvent, window::Window};

type EguiAllocator = Arc<Mutex<Allocator>>;

pub struct EguiSupport {
    integration: Integration<EguiAllocator>,
}

impl EguiSupport {
    pub fn new(window: &Window, gpu: &VkGpu, swapchain: &VkSwapchain) -> anyhow::Result<Self> {
        let device = gpu.vk_logical_device();
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: gpu.instance().clone(),
            device,
            physical_device: gpu.vk_physical_device(),
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        let allocator = Arc::new(Mutex::new(allocator));
        let integration = Integration::new(
            window,
            window.inner_size().width,
            window.inner_size().height,
            window.scale_factor(),
            FontDefinitions::default(),
            egui::Style::default(),
            gpu.vk_logical_device(),
            allocator,
            gpu.graphics_queue_family_index(),
            gpu.graphics_queue(),
            swapchain.swapchain_extension.clone(),
            swapchain.current_swapchain.clone(),
            swapchain.present_format,
        );
        Ok(Self { integration })
    }

    pub fn handle_event(&mut self, winit_event: &WindowEvent) -> EventResponse {
        self.integration.handle_event(winit_event)
    }

    pub fn swapchain_updated(&mut self, swapchain: &VkSwapchain) {
        self.integration.update_swapchain(
            swapchain.extents().width,
            swapchain.extents().height,
            swapchain.current_swapchain.clone(),
            swapchain.present_format,
        );
    }

    pub fn begin_frame(&mut self, window: &Window) {
        self.integration.begin_frame(window);
    }

    pub fn end_frame(&mut self, window: &Window) -> FullOutput {
        self.integration.end_frame(window)
    }

    pub fn paint_frame(
        &mut self,
        output: FullOutput,
        swapchain: &VkSwapchain,
        command_buffer: &VkCommandBuffer,
    ) {
        let clipped = self.integration.context().tessellate(output.shapes);
        let image_index = swapchain.current_swapchain_index.get().try_into().unwrap();
        self.integration.paint(
            command_buffer.inner(),
            image_index,
            clipped,
            output.textures_delta,
        );
    }

    pub fn create_context(&self) -> Context {
        self.integration.context()
    }
}