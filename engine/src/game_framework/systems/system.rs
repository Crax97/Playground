use super::super::{
    event_queue::EventQueue,
    resources::{Resources, ResourcesBuilder},
};

pub struct SystemStartup<'w> {
    pub event_queue: &'w EventQueue,
    pub resources: &'w mut Resources,
}

pub struct SystemBeginFrameParams<'w> {
    pub delta_seconds: f32,
    pub event_queue: &'w EventQueue,
    pub resources: &'w mut Resources,
}

pub struct SystemEndFrameParams<'w> {
    pub delta_seconds: f32,
    pub event_queue: &'w EventQueue,
    pub resources: &'w mut Resources,
}

pub struct SystemOnOsEvent<'w> {
    pub event: &'w winit::event::Event<()>,
    pub event_queue: &'w EventQueue,
    pub resources: &'w mut Resources,
}
pub trait System: 'static {
    // Called before the world's initialization, use it to setup any resources shared between systems
    fn setup_resources(&self, _resource_builder: &mut ResourcesBuilder) {}

    // Called on any os events
    fn on_os_event(&mut self, _on_os_event: SystemOnOsEvent) {}
    // Called on the world's first frame
    fn startup(&mut self, _startup: SystemStartup) {}

    // Called before the event loop has started
    fn begin_frame(&mut self, _update: SystemBeginFrameParams) {}

    // Called after the event loop has ended
    fn end_frame(&mut self, _post_update: SystemEndFrameParams) {}
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
