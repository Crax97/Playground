use engine::{
    game_framework::{
        component::{Component, ComponentStartParams},
        event_queue::EventQueue,
        systems::{
            InputActionDefinition, InputActionEvent, InputActionState, InputAxisDefinition,
            InputAxisEvent, InputBindingsDefinitions, InputSystem,
        },
        world::{DestroyEntity, Entity, SpawnEntityEvent, UpdateEvent, WorldBuilder},
    },
    input::Key,
};
use nalgebra::{vector, Vector2};
use winit::event_loop;

#[derive(Default)]
struct CharacterComponent {
    location: Vector2<f32>,
    event_queue: EventQueue,
    counter: u32,
}

pub struct BulletComponent {
    pub starting_position: Vector2<f32>,
    pub direction: Vector2<f32>,
    pub lifetime: u32,
    pub my_entity_id: Entity,
    pub event_queue: EventQueue,
}

impl BulletComponent {
    fn update(&mut self, update: UpdateEvent) {
        self.starting_position += self.direction * update.delta_seconds;

        if self.lifetime == 0 {
            self.event_queue
                .push_event(DestroyEntity(self.my_entity_id));
            println!("Bullet {:?} exhausted its life!", self.my_entity_id);
        } else {
            self.lifetime -= 1;
        }
    }
}

impl Component for BulletComponent {
    fn start(&mut self, params: ComponentStartParams) {
        println!("Bang!!!");
        params.world.add_event_listener(Self::update);
    }
}

impl CharacterComponent {
    fn on_any_action_input(&mut self, input: InputActionEvent) {
        if input.state != InputActionState::Pressed {
            return;
        }
        if input.action_name == "shoot" {
            let mut spawn = SpawnEntityEvent::new();
            let bullet_id = spawn.entity();
            spawn.add_component(BulletComponent {
                starting_position: self.location,
                direction: vector![1.0, 0.0],
                lifetime: 60000,
                my_entity_id: bullet_id,
                event_queue: self.event_queue.clone(),
            });
            self.event_queue.push_event(spawn);
        }
        if input.action_name == "jump" {
            println!("Jumping {}!!! ", self.counter);
            self.counter += 1;
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

impl Component for CharacterComponent {
    fn start(&mut self, params: ComponentStartParams) {
        params.world.add_event_listener(Self::on_any_action_input);
        params.world.add_event_listener(Self::on_any_axis_input);

        self.event_queue = params.world.get_event_queue();
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

    let mut entity = world.add_entity();
    entity.component(CharacterComponent::default());
    entity.build();

    world.start();

    event_loop
        .run(|event, target| {
            world.on_os_event(&event);

            if let winit::event::Event::WindowEvent { event, .. } = event {
                match event {
                    winit::event::WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        world.update(1.0 / 60.0);
                        window.request_redraw();
                    }
                    _ => {}
                }
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
