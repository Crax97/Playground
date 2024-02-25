use std::{
    any::TypeId,
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use gpu::CommandBuffer;
use rhai::{CustomType, TypeBuilder};
use thunderdome::Arena;

use crate::Backbuffer;

use super::{
    component::{AnyComponent, Component, ComponentDestroyParams, ComponentStartParams},
    event_queue::{Event, EventBase, EventQueue},
    resources::{Resource, Resources, ResourcesBuilder},
    systems::{
        System, SystemBeginFrameParams, SystemDrawParams, SystemEndFrameParams, SystemOnOsEvent,
        SystemShutdownParams, SystemStartupParams, Systems,
    },
};

#[derive(Clone, Copy, CustomType)]
pub struct BeginFrameEvent;
#[derive(Clone, Copy, CustomType)]
pub struct UpdateEvent {
    pub delta_seconds: f32,
    _marker: PhantomData<()>,
}

#[derive(Clone, Copy, CustomType)]
pub struct EndFrameEvent;

pub struct SpawnEntityEventConstructParams<'w> {
    context: &'w WorldEventContext<'w>,
    components: Arc<Vec<Arc<Box<dyn AnyComponent>>>>,
}

impl<'w> SpawnEntityEventConstructParams<'w> {
    pub fn with_component<C: Component>(&mut self, component: C) -> &mut Self {
        Arc::get_mut(&mut self.components)
            .unwrap()
            .push(Arc::new(Box::new(component)));
        self
    }

    pub fn commit(&mut self) -> Entity {
        let entity_id = self
            .context
            .new_entities
            .write()
            .expect("Failed to lock entities")
            .insert(EntityInfo::default());
        let entity = Entity { id: entity_id };
        let event = SpawnEntityEvent {
            entity,
            components: self.components.clone(),
        };

        self.context.event_queue.push_event(event);

        entity
    }
}

#[derive(Clone)]
pub struct SpawnEntityEvent {
    entity: Entity,
    components: Arc<Vec<Arc<Box<dyn AnyComponent>>>>,
}

#[derive(Clone, CustomType)]
pub struct DestroyEntity(pub Entity);

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash, Debug)]
pub struct Entity {
    id: thunderdome::Index,
}

#[derive(Default)]
struct EntityInfo {
    components: HashMap<TypeId, thunderdome::Index>,
}

pub struct EntityBuilder<'world> {
    world: &'world mut World,
    entity: Entity,
    components: HashMap<TypeId, OwnedComponent>,
}

pub struct World {
    entities: Arc<RwLock<Arena<EntityInfo>>>,
    components: HashMap<TypeId, Arc<RwLock<Arena<OwnedComponent>>>>,
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

pub struct WorldEventContext<'w> {
    pub event_queue: &'w EventQueue,
    pub self_entity: Entity,

    new_entities: &'w Arc<RwLock<Arena<EntityInfo>>>,
}

impl<'w> WorldEventContext<'w> {
    pub fn begin_spawn_event(&self) -> SpawnEntityEventConstructParams {
        SpawnEntityEventConstructParams {
            context: self,
            components: Default::default(),
        }
    }
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
        let id = world
            .entities
            .write()
            .expect("Failed to take entities")
            .insert(EntityInfo::default());
        Self {
            world,
            entity: Entity { id },
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

    pub fn build(mut self) -> Entity {
        self.components.iter_mut().for_each(|(_, component)| {
            component.component.start(ComponentStartParams {
                entity: self.entity,
                world: self.world,
            });
        });
        let mut info = self
            .world
            .entities
            .write()
            .expect("Failed to take entities");
        let info = info.get_mut(self.entity.id).unwrap();
        for (ty, component) in self.components {
            let component_index = self
                .world
                .components
                .entry(ty)
                .or_default()
                .write()
                .unwrap()
                .insert(component);

            info.components.insert(ty, component_index);
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
        F: FnMut(&mut S, &Event, &WorldEventContext) + Send + 'static,
    >(
        &mut self,
        event_type: TypeId,
        mut method: F,
    ) where
        'inst: 'event,
    {
        let binding_function =
            move |inst: &mut dyn AnyComponent, event: &Event, ctx: &WorldEventContext| {
                let inst = inst.as_any_mut().downcast_mut::<S>().unwrap();

                method(inst, event, ctx)
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
        F: FnMut(&mut S, E, &WorldEventContext) + Send + 'static,
    >(
        &mut self,
        mut method: F,
    ) where
        'inst: 'event,
    {
        self.add_event_listener_dynamic(
            TypeId::of::<E>(),
            move |inst: &mut S, event: &Event, ctx: &WorldEventContext| {
                let event = event.downcast::<E>();
                let event = event.expect("Failed to get event");

                method(inst, event, ctx)
            },
        );
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
                    if component_list.is_empty() {
                        return;
                    }
                    for (_, component) in component_list.iter_mut() {
                        // context.self_entity = component
                        (method)(
                            component.component.as_mut(),
                            &event,
                            &WorldEventContext {
                                event_queue: &self.event_queue,
                                self_entity: component.owner,
                                new_entities: &self.entities,
                            },
                        );
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
        let info = self
            .entities
            .write()
            .expect("Failed to take entities")
            .remove(entity.id)
            .expect("Failed to find component");
        for (ty, index) in info.components {
            let comp_list = self.components.get(&ty).unwrap();
            let mut comp_list = comp_list.write().expect("Failed to lock component list");
            let mut component = comp_list.remove(index).expect("Failed to find component");
            component.component.destroy(ComponentDestroyParams {});
        }
    }
}

impl SpawnEntityEvent {
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

type DispatchMethod = dyn FnMut(&mut dyn AnyComponent, &Event, &WorldEventContext) + Send;
struct OwnedComponent {
    owner: Entity,
    component: Box<dyn AnyComponent>,
}
