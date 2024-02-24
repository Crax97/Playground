use engine::{
    game_framework::{
        component::Component,
        systems::{
            InputActionDefinition, InputActionEvent, InputActionState, InputAxisDefinition,
            InputAxisEvent, InputBindingsDefinitions, InputSystem,
        },
        world::WorldBuilder,
    },
    input::Key,
};
use winit::event_loop;

struct OnAnyInputComponent {}

impl OnAnyInputComponent {
    fn on_any_action_input(&mut self, input: InputActionEvent) {
        if input.state != InputActionState::Pressed {
            return;
        }
        if input.action_name == "shoot" {
            println!("Bang!!!");
        }
        if input.action_name == "jump" {
            println!("Jumping!!!");
        }
    }

    fn on_any_axis_input(&mut self, input: InputAxisEvent) {
        if input.value.abs() < 0.2 {
            return;
        }
        if input.axis_name == "forward" {
            println!(
                "Advancing {}",
                if input.value > 0.0 {
                    "forward"
                } else {
                    "backwards"
                }
            );
        }
        if input.axis_name == "up" {
            println!("Going {}", if input.value > 0.0 { "up" } else { "down" });
        }
    }
}

impl Component for OnAnyInputComponent {
    fn start(&mut self, world: &mut engine::game_framework::world::World) {
        world.add_event_listener(Self::on_any_action_input);
        world.add_event_listener(Self::on_any_axis_input);
    }
}

fn main() {
    let event_loop = event_loop::EventLoop::new().unwrap();

    // This is needed to make the window survive for the application's duration
    #[allow(unused_variables)]
    let window = winit::window::Window::new(&event_loop).unwrap();

    let bindings = create_input_bindings();

    let input_system = InputSystem::new(bindings);

    let mut world_builder = WorldBuilder::default();
    world_builder.add_system(input_system);

    let mut world = world_builder.build();

    let _ = world.add_entity().component(OnAnyInputComponent {}).build();

    world.start();

    event_loop
        .run(|event, target| {
            world.on_os_event(&event);
            world.update(1.0 / 60.0);

            match event {
                winit::event::Event::WindowEvent { event, .. } => match event {
                    winit::event::WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .unwrap();
}

fn create_input_bindings() -> InputBindingsDefinitions {
    // InputBindingsDefinitions implement Deserialize, so you could read them from a json
    let mut bindings = InputBindingsDefinitions::default();
    bindings
        .define_action_bindings(
            "shoot",
            [
                InputActionDefinition::simple(Key::Return),
                InputActionDefinition::simple(Key::P),
            ],
        )
        .define_action_bindings("jump", Key::Space)
        .define_axis_bindings("forward", InputAxisDefinition::opposite(Key::A, Key::D))
        .define_axis_bindings("up", InputAxisDefinition::opposite(Key::W, Key::S));

    bindings
}
