use std::any::Any;

use super::world::World;

pub trait Component: 'static {
    fn start(&mut self, _world: &mut World) {}
}

pub(super) trait AnyComponent: Any + Component {
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
