use super::super::{
    event_queue::EventQueue,
    resources::{Resources, ResourcesBuilder},
};

pub struct SystemStartup<'w> {
    pub event_queue: &'w EventQueue,
    pub resources: &'w mut Resources,
}

pub trait System: 'static {
    // Called before the world's initialization, use it to setup any resources shared between systems
    fn setup_resources(&self, resource_builder: &mut ResourcesBuilder);

    // Called on any os events
    fn on_os_event(&mut self, _event: &winit::event::Event<()>) {}

    // Called on the world's first frame
    fn startup(&mut self, startup: SystemStartup);

    // Called before the event loop has started
    fn update(&mut self, delta_seconds: f32);

    // Called after the event loop has ended
    fn post_update(&mut self, delta_seconds: f32);
}

#[derive(Default)]
pub struct Systems {
    systems: Vec<Box<dyn System>>,
}

impl Systems {
    pub fn add_system<S: System>(&mut self, system: S) -> &mut Self {
        self.systems.push(Box::new(system));
        self
    }

    pub(crate) fn for_each<F: FnMut(&mut dyn System)>(&mut self, mut fun: F) {
        self.systems.iter_mut().for_each(|s| fun(s.as_mut()));
    }
}
