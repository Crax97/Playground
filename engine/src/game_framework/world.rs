use std::{
    any::TypeId,
    collections::HashMap,
    marker::PhantomData,
    sync::{atomic::AtomicU32, Arc, RwLock},
};

use gpu::CommandBuffer;
use rhai::{CustomType, TypeBuilder};

use crate::Backbuffer;

use super::{
    component::{AnyComponent, Component, ComponentStartParams},
    event_queue::{Event, EventBase, EventQueue},
    resources::{Resource, Resources, ResourcesBuilder},
    systems::{
        System, SystemBeginFrameParams, SystemDrawParams, SystemEndFrameParams, SystemOnOsEvent,
        SystemShutdownParams, SystemStartupParams, Systems,
    },
};

static CURRENT_ENTITY_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, CustomType)]
pub struct BeginFrameEvent;
#[derive(Clone, Copy, CustomType)]
pub struct UpdateEvent {
    pub delta_seconds: f32,
    _marker: PhantomData<()>,
}

#[derive(Clone, Copy, CustomType)]
pub struct EndFrameEvent;

#[derive(Clone)]
pub struct SpawnEntityEvent {
    entity: Entity,
    components: Arc<Vec<Arc<Box<dyn AnyComponent>>>>,
}

#[derive(Clone, CustomType)]
pub struct DestroyEntity(pub Entity);

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
    event_mappings: HashMap<TypeId, HashMap<TypeId, Box<DispatchMethod>>>,
    resources: Resources,

    event_queue: EventQueue,

    entities_to_destroy: Vec<Entity>,
    entities_to_spawn: Vec<SpawnEntityEvent>,
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
            entities_to_destroy: vec![],
            entities_to_spawn: vec![],
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

    fn explicit(world: &'world mut World, entity: Entity) -> Self {
        Self {
            world,
            entity,
            components: Default::default(),
        }
    }

    pub fn component<C: Component>(&mut self, component: C) -> &mut Self {
        self.components.insert(
            TypeId::of::<C>(),
            OwnedComponent {
                owner: self.entity,
                component: Box::new(component),
            },
        );
        self
    }

    pub fn component_dyn(&mut self, component: Box<dyn AnyComponent>) -> &mut Self {
        self.components.insert(
            component.as_ref().as_any().type_id(),
            OwnedComponent {
                owner: self.entity,
                component,
            },
        );
        self
    }

    pub fn build(self) -> Entity {
        self.world.entities.push(self.entity);
        for (ty, mut component) in self.components {
            component.component.start(ComponentStartParams {
                entity: self.entity,
                world: self.world,
            });
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
        dispatchers
            .entry(TypeId::of::<S>())
            .or_insert(Box::new(binding_function));
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

    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    pub fn start(&mut self) {
        self.systems.for_each(|s| {
            s.startup(SystemStartupParams {
                event_queue: &self.event_queue,
                resources: &mut self.resources,
            });
        });
    }

    pub fn shutdown(&mut self) {
        self.systems.for_each(|s| {
            s.shutdown(SystemShutdownParams {
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
        for entity in std::mem::take(&mut self.entities_to_spawn) {
            self.spawn_entity(entity.entity, entity.components);
        }
        for entity in std::mem::take(&mut self.entities_to_destroy) {
            self.destroy_entity(entity);
        }

        self.systems.for_each(|s| {
            s.begin_frame(SystemBeginFrameParams {
                delta_seconds,
                event_queue: &self.event_queue,
                resources: &mut self.resources,
            });
        });

        self.event_queue.push_event(BeginFrameEvent);
        self.pump_events();

        self.event_queue.push_event(UpdateEvent {
            delta_seconds,
            _marker: PhantomData,
        });
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

    pub fn draw(&mut self, command_buffer: &mut CommandBuffer, backbuffer: &Backbuffer) {
        self.systems.for_each(|s| {
            s.draw(SystemDrawParams {
                resources: &mut self.resources,
                command_buffer,
                backbuffer,
            })
        })
    }

    fn pump_events(&mut self) {
        while let Some(mut event) = self.event_queue.get_event() {
            if event.try_match::<SpawnEntityEvent>(|spawn: SpawnEntityEvent| {
                self.entities_to_spawn.push(spawn)
            }) || event.try_match(|DestroyEntity(entity)| self.entities_to_destroy.push(entity))
            {
                continue;
            }

            let ty = event.get_type();

            if let Some(dispatchers) = self.event_mappings.get_mut(&ty) {
                for (&component_ty, method) in dispatchers {
                    let mut component_list = self
                        .components
                        .entry(component_ty)
                        .or_default()
                        .write()
                        .unwrap();
                    for component in component_list.iter_mut() {
                        (method)(component.component.as_mut(), &event);
                    }
                }
            }
        }
    }

    fn spawn_entity(
        &mut self,
        entity: Entity,
        mut component_funcs: Arc<Vec<Arc<Box<dyn AnyComponent>>>>,
    ) {
        let mut builder = EntityBuilder::explicit(self, entity);

        let components = Arc::get_mut(&mut component_funcs).unwrap();
        let components = std::mem::take(components);
        for component in components {
            let component = Arc::try_unwrap(component)
                .unwrap_or_else(|_| panic!("Failed to take unique ownership of new component"));
            builder.component_dyn(component);
        }
        builder.build();
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

    fn destroy_entity(&mut self, entity: Entity) {
        self.entities.retain(|&e| e != entity);

        self.components.iter_mut().for_each(|(_, c)| {
            c.write()
                .expect("Failed to lock component vector")
                .retain(|c| c.owner != entity);
        });
    }
}

impl SpawnEntityEvent {
    pub fn new() -> Self {
        Self {
            entity: Entity {
                id: CURRENT_ENTITY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            },
            components: Default::default(),
        }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn add_component<C: Component>(&mut self, component: C) -> &mut Self {
        Arc::get_mut(&mut self.components)
            .unwrap()
            .push(Arc::new(Box::new(component)));
        self
    }
}

impl Default for SpawnEntityEvent {
    fn default() -> Self {
        Self::new()
    }
}

type DispatchMethod = dyn FnMut(&mut dyn AnyComponent, &Event);
struct OwnedComponent {
    owner: Entity,
    component: Box<dyn AnyComponent>,
}
