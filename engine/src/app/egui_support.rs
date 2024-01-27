use std::{
    sync::{Arc, Mutex},
    time::SystemTime,
};

use egui::{pos2, vec2, Context, FontDefinitions, FullOutput, Rect, Sense, Ui};
use egui_winit::EventResponse;
use egui_winit_ash_integration::Integration;
use gpu::{CommandBuffer, Gpu, Swapchain, VkCommandBuffer, VkGpu, VkSwapchain};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::{event::WindowEvent, window::Window};

use crate::CvarManager;

use super::Console;

type EguiAllocator = Arc<Mutex<gpu_allocator::vulkan::Allocator>>;

pub struct EguiSupport {
    integration: Integration<EguiAllocator>,
}

impl EguiSupport {
    pub fn new(window: &Window, gpu: &Arc<dyn Gpu>, swapchain: &Swapchain) -> anyhow::Result<Self> {
        let gpu = gpu.clone().as_any_arc();
        let gpu = gpu.downcast_ref::<VkGpu>().unwrap();
        let device = gpu.vk_logical_device();

        let swapchain = swapchain
            .pimpl
            .as_any()
            .downcast_ref::<VkSwapchain>()
            .unwrap();

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
            swapchain.current_swapchain,
            swapchain.present_format,
        );
        Ok(Self { integration })
    }

    pub fn paint_console(&mut self, console: &mut Console, cvar_manager: &mut CvarManager) {
        if console.show {
            let ctx = self.create_context();
            let screen_size = ctx.screen_rect();
            let console_size = vec2(screen_size.width(), screen_size.height() * 0.33);
            const CONSOLE_INPUT_HEIGHT: f32 = 8.0;
            const CONSOLE_INPUT_PADDING: f32 = 2.0;
            egui::Window::new("Console Window")
                .resizable(false)
                .title_bar(false)
                .fixed_rect(Rect {
                    min: pos2(0.0, 0.0),
                    max: pos2(console_size.x, console_size.y),
                })
                .collapsible(false)
                .show(&ctx, |ui| {
                    ui.add(|ui: &mut Ui| {
                        egui::ScrollArea::new([false, true])
                            .vertical_scroll_offset(1.0)
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                            )
                            .show(ui, |ui: &mut Ui| {
                                let response =
                                    ui.allocate_response(vec2(0.0, 0.0), Sense::click_and_drag());
                                egui::Grid::new("messages").show(ui, |ui| {
                                    for msg in &console.messages {
                                        ui.label(format!(
                                            "@{:?} - ",
                                            msg.timestamp
                                                .duration_since(SystemTime::UNIX_EPOCH)
                                                .expect("Time went backwards since UNIX_EPOCH")
                                                .as_millis()
                                        ));
                                        ui.label(&msg.content);
                                        ui.end_row();
                                    }
                                });
                                response
                            })
                            .inner
                    });

                    ui.put(
                        Rect {
                            min: pos2(CONSOLE_INPUT_PADDING, console_size.y - CONSOLE_INPUT_HEIGHT),
                            max: pos2(console_size.x - CONSOLE_INPUT_PADDING, console_size.y),
                        },
                        |ui: &mut Ui| {
                            let resp = egui::TextEdit::singleline(&mut console.pending_input)
                                .min_size(vec2(0.0, CONSOLE_INPUT_HEIGHT))
                                .show(ui)
                                .response;
                            if resp.lost_focus() && ui.ctx().input(|i| i.key_down(egui::Key::Enter))
                            {
                                let message = std::mem::take(&mut console.pending_input);
                                console.add_message(message.clone());
                                console.handle_cvar_command(message, cvar_manager)
                            }
                            resp
                        },
                    );
                });
        }
    }

    pub fn handle_event(&mut self, winit_event: &WindowEvent) -> EventResponse {
        self.integration.handle_event(winit_event)
    }

    pub fn swapchain_updated(&mut self, swapchain: &Swapchain) {
        let swapchain = swapchain
            .pimpl
            .as_any()
            .downcast_ref::<VkSwapchain>()
            .unwrap();
        self.integration.update_swapchain(
            swapchain.extents().width,
            swapchain.extents().height,
            swapchain.current_swapchain,
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
        swapchain: &Swapchain,
        command_buffer: &CommandBuffer,
    ) {
        let swapchain = swapchain
            .pimpl
            .as_any()
            .downcast_ref::<VkSwapchain>()
            .unwrap();
        let clipped = self.integration.context().tessellate(output.shapes);
        let image_index = swapchain.current_swapchain_index.get().try_into().unwrap();
        let command_buffer = command_buffer.pimpl();
        let command_buffer = command_buffer.as_any();
        let command_buffer = command_buffer
            .downcast_ref::<VkCommandBuffer>()
            .unwrap()
            .vk_command_buffer();
        self.integration
            .paint(command_buffer, image_index, clipped, output.textures_delta);
    }

    pub fn create_context(&self) -> Context {
        self.integration.context()
    }
}
