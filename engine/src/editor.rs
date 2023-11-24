use bevy_ecs::{system::Resource, world::World};
use egui::{Context, FullOutput};
use gpu::VkCommandBuffer;

use crate::{
    app::{
        app_state::{app_state, AppState},
        egui_support::EguiSupport,
    },
    components::EngineWindow,
    Plugin,
};

pub struct EditorPlugin {
    egui_support: EguiSupport,
    output: Option<FullOutput>,
}

#[derive(Resource)]
pub struct EguiContext(Context);

impl std::ops::Deref for EguiContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for EguiContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Plugin for EditorPlugin {
    fn construct(app: &mut crate::BevyEcsApp) -> Self {
        let window = app.world.get_resource::<EngineWindow>().unwrap();

        let egui_support =
            EguiSupport::new(window, &app_state().gpu, &app_state().swapchain).unwrap();

        let context = egui_support.create_context();
        let context = EguiContext(context);

        app.world.insert_resource(context);

        Self {
            egui_support,
            output: None,
        }
    }

    fn on_resize(
        &mut self,
        _world: &mut World,
        app_state: &AppState,
        _new_size: winit::dpi::PhysicalSize<u32>,
    ) {
        self.egui_support.swapchain_updated(&app_state.swapchain)
    }
    fn on_event(&mut self, _world: &mut World, event: &winit::event::Event<()>) {
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                let _ = self.egui_support.handle_event(event);
            }
            _ => {}
        }
    }
    fn pre_update(&mut self, world: &mut World) {
        let window = world.get_resource::<EngineWindow>().unwrap();
        self.egui_support.begin_frame(window);
        let context = self.egui_support.create_context();
        egui::Window::new("Important message!").show(&context, |ui| {
            ui.label("Hiiiiii");
        });
    }
    fn post_update(&mut self, world: &mut World) {
        let window = world.get_resource::<EngineWindow>().unwrap();
        self.output = Some(self.egui_support.end_frame(window));
    }
    fn draw(
        &mut self,
        _world: &mut World,
        app_state: &mut AppState,
        command_buffer: &mut VkCommandBuffer,
    ) {
        if let Some(output) = self.output.take() {
            self.egui_support
                .paint_frame(output, &app_state.swapchain, command_buffer)
        }
    }
}
