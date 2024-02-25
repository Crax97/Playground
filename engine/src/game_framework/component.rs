use std::any::Any;

use super::world::{Entity, World};

pub struct ComponentStartParams<'w> {
    pub entity: Entity,
    pub world: &'w mut World,
}

pub trait Component: 'static {
    fn start(&mut self, _params: ComponentStartParams) {}
}

pub trait AnyComponent: Any + Component {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
impl<T: Any + Component> AnyComponent for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
