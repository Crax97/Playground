use std::{
    any::Any,
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    sync::atomic::{AtomicU32, Ordering},
};

use log::debug;

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

struct RefCounted<T> {
    resource: T,
    ref_count: AtomicU32,
}

pub trait Context {
    fn increment_resource_refcount(&self, id: u64, resource_type: HandleType);
    fn decrement_resource_refcount(&self, id: u64, resource_type: HandleType);
}

pub struct AllocatedResourceMap<T>(HashMap<u64, RefCounted<T>>);

impl<T: HasAssociatedHandle + Clone> AllocatedResourceMap<T> {
    pub fn new() -> AllocatedResourceMap<T> {
        Self(HashMap::new())
    }

    pub fn resolve(&self, handle: &T::AssociatedHandle) -> T {
        let res = self
            .0
            .get(&handle.id())
            .expect("Failed to resolve resource");
        res.resource.clone()
    }

    pub fn insert(&mut self, handle: &T::AssociatedHandle, resource: T) {
        self.0.insert(
            handle.id(),
            RefCounted {
                resource,
                ref_count: 1.into(),
            },
        );
    }

    pub fn increment_resource_count(&mut self, id: u64) {
        self.0
            .get_mut(&id)
            .expect("Failed to resolve resource")
            .ref_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn decrement_resource_count(&mut self, id: u64) -> Option<T> {
        let old_refcount = self
            .0
            .get_mut(&id)
            .expect("Failed to resolve resource")
            .ref_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

        if old_refcount == 1 {
            // Resource lost last reference
            debug!(
                "Resource {:?} - {id} lost it's last reference",
                &T::AssociatedHandle::handle_type()
            );
            self.0.remove(&id).map(|v| v.resource)
        } else {
            None
        }
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
        handle: &T::AssociatedHandle,
        resource: T,
    ) {
        assert!(!handle.is_null());
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
        handle: &T::AssociatedHandle,
    ) -> T {
        assert!(!handle.is_null());
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
