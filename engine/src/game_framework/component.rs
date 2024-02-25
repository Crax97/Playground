use std::any::Any;

use super::world::{Entity, World};

pub struct ComponentStartParams<'w> {
    pub entity: Entity,
    pub world: &'w mut World,
}

pub struct ComponentDestroyParams {}

pub trait Component: Sync + Send + 'static {
    fn start(&mut self, _params: ComponentStartParams) {}
    fn destroy(&mut self, _params: ComponentDestroyParams) {}
}

pub trait AnyComponent: Any + Component + Sync + Send {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
impl<T: Any + Component + Sync + Send> AnyComponent for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
