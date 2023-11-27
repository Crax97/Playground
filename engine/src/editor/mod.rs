mod editor_ui;
mod entity_outliner;

pub mod ui_extension;

use std::any::TypeId;

use bevy_reflect::TypeRegistry;
pub use editor_ui::EditorUi;

use bevy_ecs::{reflect::ReflectComponent, system::Resource, world::World};
use egui::{Context, FullOutput};
use gpu::VkCommandBuffer;

use crate::{
    app::{
        app_state::{app_state, AppState},
        egui_support::EguiSupport,
    },
    components::{EngineWindow, TestComponent},
    Plugin,
};

use self::entity_outliner::EntityOutliner;

pub struct EditorPlugin {
    egui_support: EguiSupport,
    outliner: EntityOutliner,
    ui_context: Context,
    output: Option<FullOutput>,
    type_registry: TypeRegistry,
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
impl EditorPlugin {
    pub fn new(app: &mut crate::BevyEcsApp) -> Self {
        let window = app.world.get_resource::<EngineWindow>().unwrap();

        let egui_support =
            EguiSupport::new(window, &app_state().gpu, &app_state().swapchain).unwrap();

        let context = egui_support.create_context();
        let egui_app_context = EguiContext(context.clone());

        app.world.insert_resource(egui_app_context);
        let mut type_registry = TypeRegistry::new();
        type_registry.register::<TestComponent>();
        assert!(type_registry
            .get(TypeId::of::<TestComponent>())
            .unwrap()
            .data::<ReflectComponent>()
            .is_some());
        Self {
            egui_support,
            ui_context: context,
            outliner: EntityOutliner::default(),
            output: None,
            type_registry,
        }
    }
}
impl Plugin for EditorPlugin {
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
        world: &mut World,
        app_state: &mut AppState,
        command_buffer: &mut VkCommandBuffer,
    ) {
        egui::SidePanel::new(egui::panel::Side::Right, "Outliner Panel").show(
            &self.ui_context,
            |ui| {
                self.outliner.draw(world, ui, &self.type_registry);
            },
        );
        if let Some(output) = self.output.take() {
            self.egui_support
                .paint_frame(output, &app_state.swapchain, command_buffer)
        }
    }
}
