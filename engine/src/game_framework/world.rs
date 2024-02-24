use std::{
    any::TypeId,
    collections::HashMap,
    sync::{atomic::AtomicU32, Arc, RwLock},
};

use rhai::{CustomType, TypeBuilder};

use super::{
    component::{AnyComponent, Component},
    event_queue::{Event, EventBase, EventQueue},
    resources::{Resource, Resources, ResourcesBuilder},
    systems::{
        System, SystemBeginFrameParams, SystemEndFrameParams, SystemOnOsEvent, SystemStartup,
        Systems,
    },
};

static CURRENT_ENTITY_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, CustomType)]
pub struct BeginFrameEvent;
#[derive(Clone, Copy, CustomType)]
pub struct UpdateEvent(f32);

#[derive(Clone, Copy, CustomType)]
pub struct EndFrameEvent;

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash, Debug)]
pub struct Entity {
    id: u32,
}

pub struct EntityBuilder<'world> {
    world: &'world mut World,
    entity: Entity,
    components: HashMap<TypeId, OwnedComponent>,
}

pub struct World {
    entities: Vec<Entity>,
    components: HashMap<TypeId, Arc<RwLock<Vec<OwnedComponent>>>>,
    systems: Systems,
    event_mappings: HashMap<TypeId, Vec<ComponentEventDispatcher>>,
    resources: Resources,

    event_queue: EventQueue,
}

#[derive(Default)]
pub struct WorldBuilder {
    resource_builder: ResourcesBuilder,
    systems: Systems,
}

impl WorldBuilder {
    pub fn add_resource<R: Resource>(&mut self, res: R) -> &mut Self {
        self.resource_builder.add_resource(res);
        self
    }

    pub fn add_system<S: System>(&mut self, system: S) -> &mut Self {
        system.setup_resources(&mut self.resource_builder);
        self.systems.add_system(system);
        self
    }

    pub fn build(self) -> World {
        World {
            entities: Default::default(),
            components: Default::default(),
            systems: self.systems,
            event_mappings: Default::default(),
            resources: self.resource_builder.build(),
            event_queue: EventQueue::default(),
        }
    }
}

impl<'world> EntityBuilder<'world> {
    pub fn new(world: &'world mut World) -> Self {
        Self {
            world,
            entity: Entity {
                id: CURRENT_ENTITY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            },
            components: Default::default(),
        }
    }

    pub fn component<C: Component>(mut self, component: C) -> Self {
        self.components.insert(
            TypeId::of::<C>(),
            OwnedComponent {
                owner: self.entity,
                component: Box::new(component),
            },
        );
        self
    }

    pub fn build(self) -> Entity {
        self.world.entities.push(self.entity);
        for (ty, mut component) in self.components {
            component.component.start(self.world);
            self.world
                .components
                .entry(ty)
                .or_default()
                .write()
                .unwrap()
                .push(component);
        }

        self.entity
    }
}
impl World {
    pub fn add_entity(&mut self) -> EntityBuilder {
        EntityBuilder::new(self)
    }

    pub fn get_event_queue(&self) -> EventQueue {
        self.event_queue.clone()
    }

    pub fn add_event_listener_dynamic<
        'inst,
        'event,
        S: Component,
        F: FnMut(&mut S, &Event) + 'static,
    >(
        &mut self,
        event_type: TypeId,
        mut method: F,
    ) where
        'inst: 'event,
    {
        let binding_function = move |inst: &mut dyn AnyComponent, event: &Event| {
            let inst = inst.as_any_mut().downcast_mut::<S>().unwrap();

            method(inst, event)
        };
        let dispatchers = self.event_mappings.entry(event_type).or_default();
        dispatchers.push(ComponentEventDispatcher {
            ty: TypeId::of::<S>(),
            method: Box::new(binding_function),
        });
    }
    pub fn add_event_listener<
        'inst,
        'event,
        S: Component,
        E: EventBase + Clone,
        F: FnMut(&mut S, E) + 'static,
    >(
        &mut self,
        mut method: F,
    ) where
        'inst: 'event,
    {
        self.add_event_listener_dynamic(TypeId::of::<E>(), move |inst: &mut S, event: &Event| {
            let event = event.downcast::<E>();
            let event = event.expect("Failed to get event");

            method(inst, event)
        });
    }

    pub fn start(&mut self) {
        self.systems.for_each(|s| {
            s.startup(SystemStartup {
                event_queue: &self.event_queue,
                resources: &mut self.resources,
            });
        });
    }

    /*
    1. For each system call it's `begin_frame` method
    2. Dispatch the `BeginFrameEvent` event and run the event loop
    3. Dispatch the `UpdateEvent` event and run the event loop
    4. Dispatch the `EndFrameEvent` event and run the event loop
    5. For each system call it's `end_frame` method
    */
    pub fn update(&mut self, delta_seconds: f32) {
        self.systems.for_each(|s| {
            s.begin_frame(SystemBeginFrameParams {
                delta_seconds,
                event_queue: &self.event_queue,
                resources: &mut self.resources,
            });
        });

        self.event_queue.push_event(BeginFrameEvent);
        self.pump_events();

        self.event_queue.push_event(UpdateEvent(delta_seconds));
        self.pump_events();

        self.event_queue.push_event(EndFrameEvent);
        self.pump_events();

        self.systems.for_each(|s| {
            s.end_frame(SystemEndFrameParams {
                delta_seconds,
                event_queue: &self.event_queue,
                resources: &mut self.resources,
            });
        });
    }

    fn pump_events(&mut self) {
        while let Some(event) = self.event_queue.get_event() {
            let ty = event.get_type();

            if let Some(dispatchers) = self.event_mappings.get_mut(&ty) {
                for dispatcher in dispatchers {
                    let component_list = self.components.get_mut(&dispatcher.ty).unwrap().clone();
                    let mut component_list = component_list.write().unwrap();
                    for component_list in component_list.iter_mut() {
                        (dispatcher.method)(component_list.component.as_mut(), &event);
                    }
                }
            }
        }
    }

    pub fn on_os_event(&mut self, event: &winit::event::Event<()>) {
        self.systems.for_each(|s| {
            s.on_os_event(SystemOnOsEvent {
                event,
                event_queue: &self.event_queue,
                resources: &mut self.resources,
            })
        })
    }
}

type DispatchMethod = dyn FnMut(&mut dyn AnyComponent, &Event);

struct ComponentEventDispatcher {
    ty: TypeId,
    method: Box<DispatchMethod>,
}

struct OwnedComponent {
    owner: Entity,
    component: Box<dyn AnyComponent>,
}
