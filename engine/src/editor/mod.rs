mod editor_ui;
mod egui_scene_editor;
mod entity_outliner;

pub mod ui_extension;
pub use egui_scene_editor::*;

use std::any::TypeId;

pub use editor_ui::EditorUi;

use bevy_ecs::{
    component::Component,
    system::Resource,
    world::{EntityWorldMut, World},
};
use egui::{Context, FullOutput, Ui};
use gpu::{CommandBuffer, Gpu};
use std::collections::HashMap;

use crate::{
    app::{app_state::AppState, egui_support::EguiSupport},
    components::{DebugName, EngineWindow, Transform2D},
    physics::{Collider2DHandle, RigidBody2DHandle},
    AppTypeRegistry, Backbuffer, BevyEcsApp, Plugin, Time,
};

use self::entity_outliner::EntityOutliner;

pub struct EditorPluginBuilder {
    with_custom_editor: HashMap<TypeId, Box<dyn FnMut(&mut EntityWorldMut, &mut Ui)>>,
}

pub struct EditorPlugin {
    egui_support: EguiSupport,
    outliner: EntityOutliner,
    ui_context: Context,
    output: Option<FullOutput>,
    with_custom_editor: HashMap<TypeId, Box<dyn FnMut(&mut EntityWorldMut, &mut Ui)>>,
    test: String,
    type_registry: AppTypeRegistry,
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

impl EditorPluginBuilder {
    pub fn new() -> Self {
        let mut builder = Self {
            with_custom_editor: HashMap::new(),
        };
        builder
            .add_custom_editor::<Transform2D>()
            .add_custom_editor::<RigidBody2DHandle>()
            .add_custom_editor::<Collider2DHandle>();
        builder
    }

    pub fn add_custom_editor<T: Component + EditorUi>(&mut self) -> &mut Self {
        let func = |entity: &mut EntityWorldMut, ui: &mut Ui| {
            let world_ptr = unsafe { entity.world_mut() } as *mut World;
            if let Some(mut component) = entity.get_mut::<T>() {
                component.ui(unsafe { world_ptr.as_mut().unwrap() }, ui);
            }
        };
        self.with_custom_editor
            .insert(TypeId::of::<T>(), Box::new(func));
        self
    }

    pub fn build(self, gpu: &dyn Gpu, app: &mut BevyEcsApp) -> EditorPlugin {
        EditorPlugin::new(gpu, app, self)
    }
}

impl EditorPlugin {
    fn new(gpu: &dyn Gpu, app: &mut crate::BevyEcsApp, builder: EditorPluginBuilder) -> Self {
        let window = app.world.get_resource::<EngineWindow>().unwrap();

        let egui_support = EguiSupport::new(window, gpu).unwrap();

        let context = egui_support.create_context();
        let egui_app_context = EguiContext(context.clone());

        app.world.insert_resource(egui_app_context);
        app.register_type::<DebugName>();
        Self {
            egui_support,
            ui_context: context,
            outliner: EntityOutliner::default(),
            output: None,
            type_registry: app.type_registry().clone(),
            with_custom_editor: builder.with_custom_editor,
            test: String::default(),
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
    fn on_event(
        &mut self,
        app_state: &AppState,
        _world: &mut World,
        event: &winit::event::Event<()>,
    ) {
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                let _ = self
                    .egui_support
                    .handle_window_event(&app_state.window, event);
            }
            _ => {}
        }
    }
    fn pre_update(&mut self, world: &mut World) {
        let window = world.get_resource::<EngineWindow>().unwrap();
        let time = world.get_resource::<Time>().unwrap();
        self.egui_support.begin_frame(window, time);
        egui::Window::new("Important message!").show(&self.ui_context, |ui| {
            ui.label("Hiiiiii");
            ui.text_edit_singleline(&mut self.test);
        });
    }
    fn post_update(&mut self, world: &mut World) {
        egui::SidePanel::new(egui::panel::Side::Right, "Outliner Panel").show(
            &self.ui_context,
            |ui| {
                self.outliner.draw(
                    &mut self.with_custom_editor,
                    world,
                    ui,
                    &self.type_registry.read(),
                );
            },
        );
    }
    fn draw(
        &mut self,
        world: &mut World,
        app_state: &mut AppState,
        backbuffer: &Backbuffer,
        command_buffer: &mut CommandBuffer,
    ) -> anyhow::Result<()> {
        let window = world.get_resource::<EngineWindow>().unwrap();
        self.output = Some(self.egui_support.end_frame(window));
        if let Some(output) = self.output.take() {
            self.egui_support.paint_frame(
                app_state.gpu(),
                command_buffer,
                backbuffer,
                output.textures_delta,
                output.shapes,
            )?;

            self.egui_support
                .handle_platform_output(window, output.platform_output);
        }

        Ok(())
    }
}
