use std::{
    any::Any,
    cell::{RefCell, RefMut},
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

impl<T: HasAssociatedHandle> AllocatedResourceMap<T> {
    pub fn resolve(&self, handle: T::AssociatedHandle) -> &T {
        let res = self
            .0
            .get(&handle.id())
            .expect("Failed to resolve resource");
        res
    }

    pub fn resolve_mut(&mut self, handle: T::AssociatedHandle) -> &mut T {
        let res = self
            .0
            .get_mut(&handle.id())
            .expect("Failed to resolve resource");
        res
    }

    pub fn insert(&mut self, handle: T::AssociatedHandle, res: T) {
        self.0.insert(handle.id(), res);
    }

    pub fn new() -> AllocatedResourceMap<T> {
        Self(HashMap::new())
    }
}

pub struct GpuResourceMap {
    maps: RefCell<HashMap<HandleType, Box<dyn Any>>>,
}

impl GpuResourceMap {
    pub fn new() -> Self {
        Self {
            maps: RefCell::new(HashMap::new()),
        }
    }
    pub fn insert<T: HasAssociatedHandle + 'static>(
        &self,
        handle: T::AssociatedHandle,
        resource: T,
    ) {
        self.get_map().insert(handle, resource)
    }
    pub fn resolve<T: HasAssociatedHandle + 'static>(&self, handle: T::AssociatedHandle) -> &T {
        self.get_map().resolve(handle)
    }

    pub fn resolve_mut<T: HasAssociatedHandle + 'static>(
        &mut self,
        handle: T::AssociatedHandle,
    ) -> &mut T {
        self.get_map().resolve_mut(handle)
    }

    pub fn get_map<T: HasAssociatedHandle + 'static>(&self) -> &mut AllocatedResourceMap<T> {
        {
            let mut map = self.maps.borrow_mut();
            map.entry(T::AssociatedHandle::handle_type())
                .or_insert(Box::new(RefCell::new(AllocatedResourceMap::<T>::new())));
        }

        let maps = self.maps.borrow();
        let map = maps
            .get(&T::AssociatedHandle::handle_type())
            .unwrap()
            .downcast_ref::<RefCell<AllocatedResourceMap<T>>>()
            .unwrap();

        let map = map.as_ptr();
        unsafe { &mut *map }
    }
}
