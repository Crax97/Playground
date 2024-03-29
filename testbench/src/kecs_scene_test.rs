mod fps_camera;
mod gltf_loader;
mod utils;

use std::borrow::Cow;
use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use engine::app::egui_support::EguiSupport;
use engine::app::{app_state::*, bootstrap, App, Console};
use engine::editor::ui_extension::UiExtension;
use engine::editor::{EguiSceneEditor, TypeEditor};
use engine::kecs_app::KecsApp;
use engine::{egui, Light, LightType, ShadowConfiguration, Time};

use engine::input::InputState;
use engine::post_process_pass::TonemapPass;
use engine_macros::glsl;
use fps_camera::FpsCamera;
use gpu::{
    CommandBuffer, Extent2D, ImageFormat, Offset2D, PresentMode, Rect2D, ShaderModuleCreateInfo,
    ShaderModuleHandle, ShaderStage,
};
use winit::dpi::{PhysicalPosition, Position};

use crate::gltf_loader::{GltfLoadOptions, GltfLoader};
use engine::input::key::Key;
use engine::{
    post_process_pass::FxaaPass, Backbuffer, CvarManager, DeferredRenderingPipeline, LightHandle,
    MaterialInstance, RenderingPipeline, ResourceMap, TextureInput,
};
use nalgebra::*;
use winit::event::MouseButton;
use winit::event_loop::EventLoop;

use clap::Parser;

const DEPTH_DRAW: &[u32] = glsl!(
    path = "src/shaders/depth_draw.frag",
    kind = fragment,
    entry_point = "main"
);

#[derive(Parser)]
pub struct GltfViewerArgs {
    #[arg(value_name = "FILE")]
    gltf_file: String,

    #[arg(long, default_value_t = false)]
    no_use_bvh: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexData {
    pub position: Vector2<f32>,
    pub color: Vector3<f32>,
    pub uv: Vector2<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthDrawConstants {
    near: f32,
    far: f32,
}

pub struct KecsSceneTest {
    camera: FpsCamera,
    scene_renderer: DeferredRenderingPipeline,
    gltf_loader: GltfLoader,
    camera_light: Option<LightHandle>,

    input: InputState,
    console: Console,
    cvar_manager: CvarManager,
    resource_map: ResourceMap,
    time: Time,
    egui_support: EguiSupport,

    depth_draw: ShaderModuleHandle,
}

#[derive(Default)]
pub struct TestComponent {
    text: String,
}

pub struct TestComponentEditor;
impl TypeEditor for TestComponentEditor {
    type EditedType = TestComponent;

    fn show_ui(&self, value: &mut Self::EditedType, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Text");
            ui.text_edit_singleline(&mut value.text);
        });
    }
}

fn main() -> anyhow::Result<()> {
    let (mut app, evt_loop, state) = KecsApp::create()?;

    let mut editor = EguiSceneEditor::new(&state.window, state.gpu())?;
    editor.register_type(&mut app.world, TestComponentEditor);

    app.add_plugin(editor);

    engine::app::run(app, evt_loop, state)
}
