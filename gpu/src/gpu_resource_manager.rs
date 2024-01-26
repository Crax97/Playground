use std::{
    any::Any,
    collections::HashMap,
    hash::Hasher,
    ops::DerefMut,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
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
    pub fn new() -> AllocatedResourceMap<T> {
        Self(HashMap::new())
    }

    pub fn resolve(&self, handle: &T::AssociatedHandle) -> T {
        let res = self
            .0
            .get(&handle.id())
            .expect("Failed to resolve resource");
        res.clone()
    }

    pub fn insert(&mut self, handle: &T::AssociatedHandle, resource: T) {
        self.0.insert(handle.id(), resource);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn for_each<F: FnMut(&T)>(&self, f: F) {
        self.0.values().for_each(f)
    }
}

pub struct GpuResourceMap {
    maps: HashMap<HandleType, Box<dyn Any + Send + Sync + 'static>>,
}

impl GpuResourceMap {
    pub fn new() -> Self {
        Self {
            maps: HashMap::new(),
        }
    }
    pub fn insert<T: HasAssociatedHandle + Clone + Send + Sync + 'static>(
        &mut self,
        handle: &T::AssociatedHandle,
        resource: T,
    ) {
        assert!(!handle.is_null());
        if !self.maps.contains_key(&T::AssociatedHandle::handle_type()) {
            self.maps.insert(
                T::AssociatedHandle::handle_type(),
                Box::new(RwLock::new(AllocatedResourceMap::<T>::new())),
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
    ) -> RwLockWriteGuard<AllocatedResourceMap<T>> {
        let map = self
            .maps
            .get(&T::AssociatedHandle::handle_type())
            .unwrap()
            .downcast_ref::<RwLock<AllocatedResourceMap<T>>>()
            .unwrap();

        map.write().unwrap()
    }
    pub fn get_map<T: HasAssociatedHandle + Clone + 'static>(
        &self,
    ) -> RwLockReadGuard<AllocatedResourceMap<T>> {
        let map = self
            .maps
            .get(&T::AssociatedHandle::handle_type())
            .unwrap()
            .downcast_ref::<RwLock<AllocatedResourceMap<T>>>()
            .unwrap();

        map.read().unwrap()
    }
}

struct Lifetimed<T: Sized> {
    resource: T,
    current_lifetime: u32,
}

pub struct LifetimedCache<T: Sized + Send + Sync> {
    map: RwLock<HashMap<u64, Lifetimed<T>>>,
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

impl<T: Sized + Send + Sync + 'static> LifetimedCache<T> {
    pub fn new(resource_lifetime: u32) -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            resource_lifetime,
        }
    }

    pub fn ensure_existing<C: FnOnce() -> T>(&self, hash: u64, creation_func: C) {
        let mut map = self.map.write().unwrap();
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

    pub fn use_ref<D: std::hash::Hash, R, U: FnOnce(&T) -> R, C: FnOnce() -> T>(
        &self,
        description: &D,
        use_fun: U,
        creation_func: C,
    ) -> R {
        let hash = quick_hash(description);

        self.ensure_existing(hash, creation_func);
        unsafe { self.use_ref_raw(&hash, use_fun) }
    }

    pub fn use_ref_mut<D: std::hash::Hash, R, U: FnMut(&mut T) -> R, C: FnOnce() -> T>(
        &self,
        description: &D,
        use_fun: U,
        creation_func: C,
    ) -> R {
        let hash = quick_hash(description);

        self.ensure_existing(hash, creation_func);
        unsafe { self.use_ref_mut_raw(&hash, use_fun) }
    }

    /// # Safety
    /// The inner map must have a valid item with hash 'hash'
    /// Otherwise, this function will panic
    pub unsafe fn use_ref_raw<R, F: FnOnce(&T) -> R>(&self, hash: &u64, fun: F) -> R {
        let r = self.map.read().unwrap();
        let r = r.get(hash).unwrap();
        fun(&r.resource)
    }

    /// # Safety
    /// The inner map must have a valid item with hash 'hash'
    /// Otherwise, this function will panic
    pub unsafe fn use_ref_mut_raw<R, F: FnMut(&mut T) -> R>(&self, hash: &u64, mut fun: F) -> R {
        let mut r = self.map.write().unwrap();
        let r = r.get_mut(hash).unwrap();
        fun(&mut r.resource)
    }

    pub fn update<D: Fn(T)>(&self, destroy_func: D) {
        if self.resource_lifetime == lifetime_cache_constants::NEVER_DEALLOCATE {
            return;
        }
        let (alive, dead) = {
            let mut map = self.map.write().unwrap();
            map.values_mut().for_each(|v| v.current_lifetime -= 1);

            let map = std::mem::take(map.deref_mut());
            map.into_iter().partition(|(_, v)| v.current_lifetime > 0)
        };

        *self.map.write().unwrap() = alive;
        for resource in dead {
            destroy_func(resource.1.resource);
        }
    }
}

impl<T: Sized + Copy + Send + Sync + 'static> LifetimedCache<T> {
    pub fn get<D: std::hash::Hash, C: FnOnce() -> T>(
        &self,
        description: &D,
        creation_func: C,
    ) -> T {
        self.use_ref(description, |r| *r, creation_func)
    }
}

impl<T: Sized + Clone + Send + Sync + 'static> LifetimedCache<T> {
    pub fn get_clone<D: std::hash::Hash, C: FnOnce() -> T>(
        &self,
        description: &D,
        creation_func: C,
    ) -> T {
        self.use_ref(description, |r| r.clone(), creation_func)
            .clone()
    }
}
