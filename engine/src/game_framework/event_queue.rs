use std::{
    any::{Any, TypeId},
    borrow::Cow,
    sync::{Arc, RwLock},
};

#[derive(Clone)]
pub struct Event {
    ty: TypeId,
    ty_name: Cow<'static, str>,
    ptr: Arc<dyn EventBase>,
}

#[derive(Default, Clone)]
pub struct EventQueue {
    context: Arc<RwLock<EventContext>>,
}

impl EventQueue {
    pub fn push_event<E: EventBase>(&self, event: E) {
        self.push_event_dynamic(Event::new(event))
    }

    pub fn push_event_dynamic(&self, event: Event) {
        let mut queue = self.context.write().unwrap();
        queue.events.push(event);
    }

    pub fn get_event(&mut self) -> Option<Event> {
        let mut queue = self.context.write().unwrap();
        queue.events.pop()
    }
}

impl Event {
    pub fn new<E: EventBase>(evt: E) -> Self {
        Event {
            ty: TypeId::of::<E>(),
            ty_name: Cow::Borrowed(std::any::type_name::<E>()),
            ptr: Arc::new(evt),
        }
    }

    pub fn get_type(&self) -> TypeId {
        self.ty
    }

    pub fn type_name(&self) -> Cow<'static, str> {
        self.ty_name.clone()
    }
    pub fn downcast<E: EventBase + Clone>(&self) -> Option<E> {
        self.ptr.as_ref().as_any().downcast_ref::<E>().cloned()
    }
}

pub trait EventBase: Any + 'static {
    fn as_any(&self) -> &dyn Any;
}

impl<E: Clone + 'static> EventBase for E {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Default)]
struct EventContext {
    events: Vec<Event>,
}
