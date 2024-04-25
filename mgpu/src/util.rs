use std::{
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    sync::atomic::AtomicBool,
};

pub(crate) static ERROR_HAPPENED: AtomicBool = AtomicBool::new(false);

/// An Handle is a generational index (8 bytes in size) that can be used to index into a
/// [`ResourceArena<T>`].
/// The first 32 bits of the handle define an index, while the last 32 define the handle's generation
#[repr(transparent)]
#[derive(Hash, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Handle<T> {
    _ph: PhantomData<T>,
    id: u64,
}

struct Lifetimed<T> {
    lifetime: usize,
    item: T,
}
/// A ResourceArena is a generational arena for items of type T
pub struct ResourceArena<S, T> {
    resources: Vec<Entry<T>>,
    reclaimed_handles: Vec<Handle<T>>,
    destruction_queue: Vec<Lifetimed<T>>,
    lifetime: usize,
    deallocate_fn: fn(&S, T) -> MgpuResult<()>,
    len: usize,
}

/// The entry type used in a ResourceArena<T>
pub struct Entry<T> {
    generation: u32,
    payload: Option<T>,
}

pub fn hash_type<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// This macro helps to define a ResourceResolver, which is a type that can be used by
/// a `[crate::hal::Hal]` implementation to resolve an Handle type (es. a `[crate::Image]`) to
/// a concrete structured used to implement the Hal.
/// Additionally, the ResourceResolver defers the destruction of every item by a certain amount of frames
/// specified at creation time (to ensure that e.g an image is not destroyed while in use by a command buffer)
/// E.g the api `Foo` uses a `foo::Buffer` for buffers and a `foo::Image` for images, therefore
/// a ResourceResolver might be defined as
/// ```
/// pub mod foo {
/// pub struct Buffer(pub String);
/// pub struct Image(pub u64);
/// }
/// ```
/// ```
/// define_resource_resolver! {
///     foo::Buffer -> foo_buffer,
///     foo::Image -> foo_image,
/// }
///
/// let mut resolver = ResourceResolver::default();
/// let foo_buffer = foo::Buffer("Hello World!");
/// let handle = resolver.add(foo_buffer);
///
/// assert_eq!(resolver.resolve_clone(handle).0, "Hello World!".to_string());
/// ```
/// For a more concrete example, take a look at `[crate::hal::vulkan::util::VulkanResolver]`
macro_rules! define_resource_resolver {
    ($self:ty, $(($resource:ty, $deallocate_fn:expr) => $map_name:ident),* $(,)?) => {
        use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
        use crate::{util::ResourceArena, MgpuResult, MgpuError};


        pub struct ResourceResolver {
            $(
                $map_name : RwLock<ResourceArena<$self, $resource>>,
            )*
        }

        pub trait GetMap<Output> {
            fn get(&self) -> RwLockReadGuard<'_, ResourceArena<$self, Output>>;
            fn get_mut(&self) -> RwLockWriteGuard<'_, ResourceArena<$self, Output>>;
        }

        impl ResourceResolver {

            // Creates a new ResourceResolver.
            // The lifetime parameter indicates after how many update cycles the destroy fn of each removed item should be called
            pub fn new(lifetime: usize) -> Self {
                Self {
                    $(
                        $map_name: RwLock::new(ResourceArena::new($deallocate_fn, lifetime))
                    ),*
                }
            }

            // Updates all the resource allocators, ensuring to destroy any item if their lifetime is 0
            pub fn update(&self, s: &$self) -> MgpuResult<()> {
                $(
                    self.get_mut::<$resource>().update(s)?;
                )*
                Ok(())
            }

            /// Gets a read-only reference to the ResourceArena for T
            pub fn get<T>(&self) -> RwLockReadGuard<'_, ResourceArena<$self, T>>
            where
                ResourceResolver: GetMap<T>,
            {
                <Self as GetMap<T>>::get(&self)
            }

            /// Gets a writeable reference to the ResourceArena for T
            pub fn get_mut<T>(&self) -> RwLockWriteGuard<'_, ResourceArena<$self, T>>
            where
                ResourceResolver: GetMap<T>,
            {
                <Self as GetMap<T>>::get_mut(&self)
            }

            /// Adds a resource to the ResourceArena for T
            pub fn add<T>(&self, resource: T) -> Handle<T> where ResourceResolver: GetMap<T> {
                self.get_mut::<T>().add(resource)
            }

            /// Applies a function to the resource associated with the given `handle`, returning the result if f succeeds
            pub fn apply<T, U>(&self, handle: impl Into<Handle<T>>, f: impl FnOnce(&T) -> MgpuResult<U>) -> MgpuResult<U> where ResourceResolver: GetMap<T> {
                if let Some(res) = self.get::<T>().resolve(handle.into()) {
                    f(res)
                } else {
                    Err(MgpuError::InvalidHandle)
                }
            }

            /// Applies a function to the resource associated with the given `handle`, returning the result if it exists
            pub fn apply_mut<T, U>(&self, handle: impl Into<Handle<T>>, f: impl FnOnce(&mut T) -> MgpuResult<U>) -> MgpuResult<U> where ResourceResolver: GetMap<T> {
                if let Some(res) = self.get_mut::<T>().resolve_mut(handle.into()) {
                    f(res)
                } else {
                    Err(MgpuError::InvalidHandle)
                }
            }

            /// Removes a resource if it exists, returning the removed resource
            pub fn remove<T>(&self, handle: impl Into<Handle<T>>) -> MgpuResult<()> where ResourceResolver: GetMap<T> {
                self.get_mut::<T>().remove(handle.into())
            }

            /// Resolves a copy of the resource if it exists
            #[allow(dead_code)]
            pub fn resolve_copy<T: Copy>(&self, handle: impl Into<Handle<T>>) -> Option<T> where ResourceResolver: GetMap<T> {
                self.get::<T>().resolve_copy(handle.into())
            }

            /// Resolves a clone of the resource if it exists
            pub fn resolve_clone<T: Clone>(&self, handle: impl Into<Handle<T>>) -> Option<T> where ResourceResolver: GetMap<T> {
                self.get::<T>().resolve_clone(handle.into())
            }
        }



        $(
            impl GetMap<$resource> for ResourceResolver {
                fn get(&self) -> RwLockReadGuard<'_, ResourceArena<$self, $resource>> {
                    self.$map_name.read().unwrap_or_else(|_|panic!("Failed to get {}", stringify!($resource)))
                }

                fn get_mut(&self) -> RwLockWriteGuard<'_, ResourceArena<$self, $resource>> {
                    self.$map_name.write().unwrap_or_else(|_| panic!("Failed to get {}", stringify!($resource)))
                }
            }
        )*
    };
}
pub(crate) use define_resource_resolver;

impl<S, T> ResourceArena<S, T> {
    pub fn new(deallocate_fn: fn(&S, T) -> MgpuResult<()>, lifetime: usize) -> Self {
        Self {
            resources: Default::default(),
            reclaimed_handles: Default::default(),
            destruction_queue: Default::default(),
            lifetime,
            deallocate_fn,
            len: 0,
        }
    }

    pub fn add(&mut self, resource: T) -> Handle<T> {
        let handle = if let Some(handle) = self.reclaimed_handles.pop() {
            let index = handle.index();
            let entry = &mut self.resources[index as usize];
            assert!(entry.payload.is_none());
            entry.payload = Some(resource);
            handle
        } else {
            let index = self.resources.len() as u32;
            self.resources.push(Entry {
                generation: 0,
                payload: Some(resource),
            });
            Handle::from_index_generation(index, 0)
        };
        self.len += 1;
        handle
    }

    pub fn update(&mut self, s: &S) -> MgpuResult<()> {
        let mut i = 0;
        while i < self.destruction_queue.len() {
            self.destruction_queue[i].lifetime -= 1;
            if self.destruction_queue[i].lifetime == 0 {
                let item = self.destruction_queue.remove(i);
                (self.deallocate_fn)(s, item.item)?
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    pub fn resolve(&self, handle: impl Into<Handle<T>>) -> Option<&T> {
        let (index, generation) = handle.into().to_index_generation();

        if let Some(entry) = self.resources.get(index as usize) {
            if entry.generation == generation {
                let payload = entry
                    .payload
                    .as_ref()
                    .expect("Payload not found with index and generation correct");
                return Some(payload);
            }
        }
        None
    }

    pub fn resolve_mut(&mut self, handle: impl Into<Handle<T>>) -> Option<&mut T> {
        let (index, generation) = handle.into().to_index_generation();

        if let Some(entry) = self.resources.get_mut(index as usize) {
            if entry.generation == generation {
                let payload = entry
                    .payload
                    .as_mut()
                    .expect("Payload not found with index and generation correct");
                return Some(payload);
            }
        }
        None
    }

    pub fn remove(&mut self, handle: impl Into<Handle<T>>) -> MgpuResult<()> {
        let handle = handle.into();
        let (index, generation) = handle.to_index_generation();
        if let Some(entry) = self.resources.get_mut(index as usize) {
            if entry.generation == generation {
                let payload = entry
                    .payload
                    .take()
                    .expect("Payload not found with index and generation correct");

                entry.generation += 1;
                self.reclaimed_handles.push(handle.advance_generation());
                self.len -= 1;
                self.destruction_queue.push(Lifetimed {
                    lifetime: self.lifetime,
                    item: payload,
                });
                return Ok(());
            }
        }
        Err(crate::MgpuError::InvalidHandle)
    }

    pub fn clear(&mut self) {
        // We can't just clear everythin, otherwise in certain cases old handles would still be valid
        // Instead, we just increase the generation of everything
        self.reclaimed_handles.clear();
        self.resources.iter_mut().enumerate().for_each(|(i, e)| {
            e.generation += 1;
            e.payload = None;
            self.reclaimed_handles
                .push(Handle::from_index_generation(i as u32, e.generation));
        });
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.resources.iter().filter_map(|r| r.payload.as_ref())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.resources.iter_mut().filter_map(|r| r.payload.as_mut())
    }
}

impl<S, T: Copy> ResourceArena<S, T> {
    pub fn resolve_copy(&self, handle: Handle<T>) -> Option<T> {
        self.resolve(handle).copied()
    }
}

impl<S, T: Clone> ResourceArena<S, T> {
    pub fn resolve_clone(&self, handle: Handle<T>) -> Option<T> {
        self.resolve(handle).cloned()
    }
}

impl<S, T> std::ops::Index<Handle<T>> for ResourceArena<S, T> {
    type Output = T;

    fn index(&self, index: Handle<T>) -> &Self::Output {
        self.resolve(index).expect("Invalid handle")
    }
}

impl<S, T> std::ops::IndexMut<Handle<T>> for ResourceArena<S, T> {
    fn index_mut(&mut self, index: Handle<T>) -> &mut Self::Output {
        self.resolve_mut(index).expect("Invalid handle")
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Entry<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("generation", &self.generation)
            .field("payload", &self.payload)
            .finish()
    }
}

impl<S, T: std::fmt::Debug> std::fmt::Debug for ResourceArena<S, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceArena")
            .field("resources", &self.resources)
            .field("reclaimed_handles", &self.reclaimed_handles)
            .field("len", &self.len)
            .finish()
    }
}

pub type Filter<T> = fn(Entry<T>) -> Option<T>;
impl<S, T> IntoIterator for ResourceArena<S, T> {
    type Item = T;
    type IntoIter =
        <std::iter::FilterMap<std::vec::IntoIter<Entry<T>>, Filter<T>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.resources.into_iter().filter_map(|entry| entry.payload)
    }
}

impl<T> Handle<T> {
    pub(crate) unsafe fn from_u64(id: u64) -> Self {
        Self {
            _ph: PhantomData,
            id,
        }
    }

    pub(crate) fn to_u64(&self) -> u64 {
        self.id
    }

    pub(crate) fn from_index_generation(index: u32, generation: u32) -> Self {
        let id = index as u64 | ((generation as u64) << 32);
        unsafe { Self::from_u64(id) }
    }

    pub(crate) fn to_index_generation(&self) -> (u32, u32) {
        (self.index(), self.generation())
    }

    pub(crate) fn advance_generation(self) -> Self {
        let index = self.index();
        let generation = self.generation();
        Self::from_index_generation(index, generation + 1)
    }

    pub(crate) fn index(&self) -> u32 {
        (self.id & 0xFFFF) as u32
    }

    pub(crate) fn generation(&self) -> u32 {
        (self.id >> 32) as u32
    }
}

#[cfg(debug_assertions)]
macro_rules! check {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            crate::util::ERROR_HAPPENED.store(true, std::sync::atomic::Ordering::Relaxed);

            panic!(
                "A check condition failed: {}\nAdditional infos: {}",
                stringify!($cond),
                $msg
            );
        }
    };

    ($cond:expr, $msg:expr, $($args:expr $(,)?)*) => {
        check!($cond, format!($msg, $($args,)*))
    };
}

#[cfg(not(debug_assertions))]
macro_rules! check {
    ($cond:expr, $msg:expr) => {
        ()
    };

    ($cond:expr, $msg:expr, $($args:expr $(,)?)*) => {
        check!($cond, format!($msg, $($args,)*))
    };
}
pub(crate) use check;

use crate::MgpuResult;

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let index_gen_str = format!("{}v{}", self.index(), self.generation());
        f.debug_struct("Handle")
            .field("id", &index_gen_str)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{Handle, ResourceArena};

    #[test]
    fn generation_handling() {
        let handle = Handle::<()>::from_index_generation(42, 0);
        assert_eq!(handle.generation(), 0);
        assert_eq!(handle.index(), 42);

        let handle = handle.advance_generation();

        assert_eq!(handle.generation(), 1);
        assert_eq!(handle.index(), 42);

        let handle = handle.advance_generation();

        assert_eq!(handle.generation(), 2);
        assert_eq!(handle.index(), 42);

        let (index, generation) = handle.to_index_generation();
        let new_handle = Handle::from_index_generation(index, generation);
        assert_eq!(new_handle, handle)
    }

    #[test]
    fn resource_arena_tests() {
        let mut arena = ResourceArena::<(), u32>::new(|(), _| Ok(()), 5);
        let handle_1 = arena.add(0);
        let handle_2 = arena.add(42);
        let handle_3 = arena.add(145);
        assert_eq!(arena[handle_3], 145);
        assert_eq!(arena[handle_2], 42);
        assert_eq!(arena[handle_1], 0);

        assert_eq!(arena.len(), 3);
        assert_eq!(arena.len(), 2);

        assert!(arena.resolve(handle_2).is_none());
        let (index, gen) = handle_2.to_index_generation();
        let new_handle_2 = arena.add(53);
        assert_eq!(new_handle_2.index(), index);
        assert_eq!(new_handle_2.generation(), gen + 1);
        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());

        let new_handle_1 = arena.add(0);
        assert!(!arena.is_empty());
        assert_eq!(arena.len(), 1);

        assert_ne!(new_handle_1, handle_1);
    }
}
