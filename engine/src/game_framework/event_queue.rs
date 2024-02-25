use std::{
    any::{Any, TypeId},
    borrow::Cow,
    sync::{Arc, RwLock},
};

#[derive(Clone)]
pub struct Event {
    ty: TypeId,
    ty_name: Cow<'static, str>,
    ptr: Option<Arc<dyn EventBase>>,
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
            ptr: Some(Arc::new(evt)),
        }
    }

    pub fn get_type(&self) -> TypeId {
        self.ty
    }

    pub fn type_name(&self) -> Cow<'static, str> {
        self.ty_name.clone()
    }
    pub fn downcast<E: EventBase + Clone>(&self) -> Option<E> {
        self.ptr
            .as_ref()
            .and_then(|e| e.as_ref().as_any().downcast_ref::<E>())
            .cloned()
    }

    pub fn try_match<E: EventBase>(&mut self, mut func: impl FnMut(&mut E)) -> bool {
        if self.ptr.is_none() {
            return false;
        }
        if self.ty == TypeId::of::<E>() {
            let taken = self.ptr.take().unwrap();
            let cast = taken.as_ref();
            let cast = cast as *const dyn EventBase;
            let cast = cast as *mut dyn EventBase;
            let cast = unsafe { cast.as_mut().unwrap() };
            let cast = cast.as_any_mut().downcast_mut::<E>().unwrap();
            func(cast);
            true
        } else {
            false
        }
    }
}

pub trait EventBase: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<E: 'static> EventBase for E {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Default)]
struct EventContext {
    events: Vec<Event>,
}
