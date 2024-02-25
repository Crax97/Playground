use engine::{
    app,
    game_framework::{
        component::{Component, ComponentStartParams},
        systems::{
            InputAxisDefinition, InputAxisEvent, InputBindingsDefinitions, InputSystem,
            RenderingSystem,
        },
        world::{UpdateEvent, WorldBuilder, WorldEventContext},
    },
    game_scene::{Scene, SceneNodeId},
    input::Key,
    CvarManager, EngineApp, ResourceMap,
};
use nalgebra::Vector3;
use winit::dpi::PhysicalSize;

#[derive(Default)]
struct CharacterComponent {
    scene_node: SceneNodeId,

    input: Vector3<f32>,
}

impl CharacterComponent {
    fn on_any_axis_input(&mut self, input: InputAxisEvent, _: &WorldEventContext) {
        if input.axis_name == "forward" {
            self.input.z = input.value;
        }
        if input.axis_name == "side" {
            self.input.x = input.value;
        }
    }

    fn on_update(&mut self, update: UpdateEvent, context: &WorldEventContext) {
        const SPEED: f32 = 100.0;
        let mut scene = context.resources.get_mut::<Scene>();
        let movement = self.input * SPEED * update.delta_seconds;
        let old_position = scene
            .get_position(self.scene_node, engine::game_scene::TransformSpace::World)
            .unwrap();
        let new_position = old_position + movement;
        println!("New position is {new_position}");
        scene.set_position(
            self.scene_node,
            new_position,
            engine::game_scene::TransformSpace::World,
        );
    }
}

impl Component for CharacterComponent {
    fn start(&mut self, params: ComponentStartParams) {
        params.world.add_event_listener(Self::on_any_axis_input);
        params.world.add_event_listener(Self::on_update);
        self.scene_node = params
            .world
            .resources()
            .get::<Scene>()
            .find_by_tag("Player")
            .unwrap();
    }
}

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::default()
        .with_inner_size(PhysicalSize {
            width: 1920,
            height: 1080,
        })
        .with_title("Winit App")
        .build(&event_loop)
        .unwrap();

    let app_state = crate::app::app_state::init("Winit App", window).unwrap();

    let mut world = WorldBuilder::default();
    world
        .add_system(RenderingSystem::new(app_state.gpu.clone()))
        .add_system(InputSystem::new(create_input_bindings()))
        .add_resource(ResourceMap::new(app_state.gpu.clone()))
        .add_resource(CvarManager::default());
    let mut world = world.build();

    {
        let mut scene = world.resources().get_mut::<Scene>();
        scene.add_node().with_tags(["Player"]).build();
    }

    let mut entity_builder = world.add_entity();
    entity_builder.component(CharacterComponent::default());
    entity_builder.build();

    let app = EngineApp::new(world);
    crate::app::run(app, event_loop, app_state).unwrap();
}

fn create_input_bindings() -> InputBindingsDefinitions {
    // InputBindingsDefinitions implement Deserialize, so you could read them from a json
    let mut bindings = InputBindingsDefinitions::default();
    bindings
        .define_axis_bindings("forward", InputAxisDefinition::opposite(Key::W, Key::S))
        .define_axis_bindings("side", InputAxisDefinition::opposite(Key::A, Key::D));

    bindings
}
