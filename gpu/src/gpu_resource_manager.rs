use std::{
    any::Any,
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
};

use crate::{Handle, HandleType};

pub trait HasAssociatedHandle {
    type AssociatedHandle: Handle;
}

pub mod utils {
    macro_rules! associate_to_handle {
        ($concrete:ty, $handle:ty) => {
            impl HasAssociatedHandle for $concrete {
                type AssociatedHandle = $handle;
            }
        };
    }
    pub(crate) use associate_to_handle;
}

pub struct AllocatedResourceMap<T>(HashMap<u64, T>);

impl<T: HasAssociatedHandle + Clone> AllocatedResourceMap<T> {
    pub fn resolve(&self, handle: T::AssociatedHandle) -> T {
        let res = self
            .0
            .get(&handle.id())
            .expect("Failed to resolve resource");
        res.clone()
    }

    pub fn insert(&mut self, handle: T::AssociatedHandle, res: T) {
        self.0.insert(handle.id(), res);
    }

    pub fn new() -> AllocatedResourceMap<T> {
        Self(HashMap::new())
    }
}

pub struct GpuResourceMap {
    maps: HashMap<HandleType, Box<dyn Any>>,
}

impl GpuResourceMap {
    pub fn new() -> Self {
        Self {
            maps: HashMap::new(),
        }
    }
    pub fn insert<T: HasAssociatedHandle + Clone + 'static>(
        &mut self,
        handle: T::AssociatedHandle,
        resource: T,
    ) {
        if !self.maps.contains_key(&T::AssociatedHandle::handle_type()) {
            self.maps.insert(
                T::AssociatedHandle::handle_type(),
                Box::new(RefCell::new(AllocatedResourceMap::<T>::new())),
            );
        }
        self.get_map().insert(handle, resource)
    }
    pub fn resolve<T: HasAssociatedHandle + Clone + 'static>(
        &self,
        handle: T::AssociatedHandle,
    ) -> T {
        self.get_map().resolve(handle)
    }

    pub fn get_map<T: HasAssociatedHandle + Clone + 'static>(
        &self,
    ) -> RefMut<AllocatedResourceMap<T>> {
        let map = self
            .maps
            .get(&T::AssociatedHandle::handle_type())
            .unwrap()
            .downcast_ref::<RefCell<AllocatedResourceMap<T>>>()
            .unwrap();

        map.borrow_mut()
    }
}
