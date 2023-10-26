use std::{
    any::Any,
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    hash::Hasher,
    ops::DerefMut,
    sync::atomic::{AtomicU32, Ordering},
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
    pub fn decrement_resource_count(&mut self, id: u64) {
        self.0
            .get_mut(&id)
            .expect("Failed to resolve resource")
            .ref_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn update(&mut self) -> Vec<T> {
        let (alive, dead) = std::mem::take(&mut self.0)
            .into_iter()
            .partition(|(_, v)| v.ref_count.load(Ordering::Relaxed) > 0);
        self.0 = alive;
        dead.values().map(|v| v.resource.clone()).collect()
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
        self.get_map_mut().insert(handle, resource)
    }
    pub fn resolve<T: HasAssociatedHandle + Clone + 'static>(
        &self,
        handle: &T::AssociatedHandle,
    ) -> T {
        assert!(!handle.is_null());
        self.get_map().resolve(handle)
    }
    pub fn get_map_mut<T: HasAssociatedHandle + Clone + 'static>(
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
    pub fn get_map<T: HasAssociatedHandle + Clone + 'static>(
        &self,
    ) -> Ref<AllocatedResourceMap<T>> {
        let map = self
            .maps
            .get(&T::AssociatedHandle::handle_type())
            .unwrap()
            .downcast_ref::<RefCell<AllocatedResourceMap<T>>>()
            .unwrap();

        map.borrow()
    }
}

struct Lifetimed<T: Sized> {
    resource: T,
    current_lifetime: u32,
}

pub struct LifetimedCache<T: Sized> {
    map: RefCell<HashMap<u64, Lifetimed<T>>>,
    resource_lifetime: u32,
}

pub mod lifetime_cache_constants {
    pub const NEVER_DEALLOCATE: u32 = u32::MAX;
}

pub fn quick_hash<H: std::hash::Hash>(hashable: &H) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hashable.hash(&mut hasher);
    hasher.finish()
}

impl<T: Sized> LifetimedCache<T> {
    pub fn new(resource_lifetime: u32) -> Self {
        Self {
            map: RefCell::new(HashMap::new()),
            resource_lifetime,
        }
    }

    pub fn ensure_existing<C: FnOnce() -> T>(&self, hash: u64, creation_func: C) {
        let mut map = self.map.borrow_mut();
        if map.contains_key(&hash) {
            map.get_mut(&hash)
                .map(|l| l.current_lifetime = self.resource_lifetime);
        } else {
            let resource = creation_func();
            map.insert(
                hash,
                Lifetimed {
                    resource,
                    current_lifetime: self.resource_lifetime,
                },
            );
        }
    }

    pub fn get_ref<D: std::hash::Hash, C: FnOnce() -> T>(
        &self,
        description: &D,
        creation_func: C,
    ) -> Ref<T> {
        let hash = quick_hash(description);

        self.ensure_existing(hash, creation_func);

        Ref::map(self.map.borrow(), |m: &HashMap<u64, Lifetimed<T>>| {
            &m.get(&hash).unwrap().resource
        })
    }

    pub fn get_ref_mut<D: std::hash::Hash, C: FnOnce() -> T>(
        &self,
        description: &D,
        creation_func: C,
    ) -> RefMut<T> {
        let hash = quick_hash(description);

        self.ensure_existing(hash, creation_func);

        RefMut::map(
            self.map.borrow_mut(),
            |m: &mut HashMap<u64, Lifetimed<T>>| &mut m.get_mut(&hash).unwrap().resource,
        )
    }

    pub fn update<D: Fn(T)>(&self, destroy_func: D) {
        if self.resource_lifetime == lifetime_cache_constants::NEVER_DEALLOCATE {
            return;
        }
        let mut map = self.map.borrow_mut();
        map.values_mut().for_each(|v| v.current_lifetime -= 1);

        let map = std::mem::take(map.deref_mut());
        let (alive, dead) = map.into_iter().partition(|(_, v)| v.current_lifetime > 0);

        self.map.replace(alive);
        for resource in dead {
            destroy_func(resource.1.resource);
        }
    }
}

impl<T: Sized + Clone> LifetimedCache<T> {
    pub fn get<D: std::hash::Hash, C: FnOnce() -> T>(
        &self,
        description: &D,
        creation_func: C,
    ) -> T {
        self.get_ref(description, creation_func).clone()
    }
}
