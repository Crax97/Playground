use std::marker::PhantomData;

/// An Handle is a generational index (8 bytes in size) that can be used to index into a
/// [`ResourceArena<T>`].
/// The first 32 bits of the handle define an index, while the last 32 define the handle's generation
#[repr(transparent)]
#[derive(Hash, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Handle<T> {
    _ph: PhantomData<T>,
    id: u64,
}

/// A ResourceArena is a generational arena for items of type T
pub struct ResourceArena<T> {
    resources: Vec<Entry<T>>,
    reclaimed_handles: Vec<Handle<T>>,
    len: usize,
}

/// The entry type used in a ResourceArena<T>
pub struct Entry<T> {
    generation: u32,
    payload: Option<T>,
}

/// This macro helps to define a ResourceResolver, which is a type that can be used by
/// a `[crate::hal::Hal]` implementation to resolve an Handle type (es. a `[crate::Image]`) to
/// a concrete structured used to implement the Hal.
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
    ($($resource:ty => $map_name:ident),*) => {
        use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
        use crate::util::ResourceArena;


        #[derive(Default)]
        pub struct ResourceResolver {
            $(
                $map_name : RwLock<ResourceArena<$resource>>,
            )*
        }

        pub trait GetMap<Output> {
            fn get(&self) -> RwLockReadGuard<'_, ResourceArena<Output>>;
            fn get_mut(&self) -> RwLockWriteGuard<'_, ResourceArena<Output>>;
        }

        impl ResourceResolver {
            pub fn get<T>(&self) -> RwLockReadGuard<'_, ResourceArena<T>>
            where
                ResourceResolver: GetMap<T>,
            {
                <Self as GetMap<T>>::get(&self)
            }

            pub fn get_mut<T>(&self) -> RwLockWriteGuard<'_, ResourceArena<T>>
            where
                ResourceResolver: GetMap<T>,
            {
                <Self as GetMap<T>>::get_mut(&self)
            }

            pub fn add<T>(&self, resource: T) -> Handle<T> where ResourceResolver: GetMap<T> {
                self.get_mut::<T>().add(resource)
            }

            pub fn apply<T, U>(&self, handle: impl Into<Handle<T>>, f: impl FnOnce(&T) -> U) -> Option<U> where ResourceResolver: GetMap<T> {
                if let Some(res) = self.get::<T>().resolve(handle.into()) {
                    Some(f(res))
                } else {
                    None
                }
            }

            pub fn remove<T>(&self, handle: impl Into<Handle<T>>) -> Option<T> where ResourceResolver: GetMap<T> {
                self.get_mut::<T>().remove(handle.into())
            }

            pub fn resolve<T: Copy>(&self, handle: impl Into<Handle<T>>) -> Option<T> where ResourceResolver: GetMap<T> {
                self.get::<T>().resolve_copy(handle.into())
            }

            pub fn resolve_clone<T: Copy>(&self, handle: impl Into<Handle<T>>) -> Option<T> where ResourceResolver: GetMap<T> {
                self.get::<T>().resolve_clone(handle.into())
            }
        }



        $(
            impl GetMap<$resource> for ResourceResolver {
                fn get(&self) -> RwLockReadGuard<'_, ResourceArena<$resource>> {
                    self.$map_name.read().unwrap_or_else(|_|panic!("Failed to get {}", stringify!($resource)))
                }

                fn get_mut(&self) -> RwLockWriteGuard<'_, ResourceArena<$resource>> {
                    self.$map_name.write().unwrap_or_else(|_| panic!("Failed to get {}", stringify!($resource)))
                }
            }
        )*
    };
}
pub(crate) use define_resource_resolver;

impl<T> ResourceArena<T> {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) fn add(&mut self, resource: T) -> Handle<T> {
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

    pub(crate) fn resolve(&self, handle: Handle<T>) -> Option<&T> {
        let (index, generation) = handle.to_index_generation();

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

    pub(crate) fn resolve_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let (index, generation) = handle.to_index_generation();

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

    pub(crate) fn remove(&mut self, handle: Handle<T>) -> Option<T> {
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
                return Some(payload);
            }
        }
        None
    }

    pub(crate) fn clear(&mut self) {
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

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.resources.iter().filter_map(|r| r.payload.as_ref())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.resources.iter_mut().filter_map(|r| r.payload.as_mut())
    }
}

impl<T: Copy> ResourceArena<T> {
    pub(crate) fn resolve_copy(&self, handle: Handle<T>) -> Option<T> {
        self.resolve(handle).copied()
    }
}

impl<T: Clone> ResourceArena<T> {
    pub(crate) fn resolve_clone(&self, handle: Handle<T>) -> Option<T> {
        self.resolve(handle).cloned()
    }
}

impl<T> Default for ResourceArena<T> {
    fn default() -> Self {
        Self {
            resources: Default::default(),
            reclaimed_handles: Default::default(),
            len: 0,
        }
    }
}

impl<T> std::ops::Index<Handle<T>> for ResourceArena<T> {
    type Output = T;

    fn index(&self, index: Handle<T>) -> &Self::Output {
        self.resolve(index).expect("Invalid handle")
    }
}

impl<T> std::ops::IndexMut<Handle<T>> for ResourceArena<T> {
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

impl<T: std::fmt::Debug> std::fmt::Debug for ResourceArena<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceArena")
            .field("resources", &self.resources)
            .field("reclaimed_handles", &self.reclaimed_handles)
            .field("len", &self.len)
            .finish()
    }
}

pub type Filter<T> = fn(Entry<T>) -> Option<T>;
impl<T> IntoIterator for ResourceArena<T> {
    type Item = T;
    type IntoIter =
        <std::iter::FilterMap<std::vec::IntoIter<Entry<T>>, Filter<T>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.resources.into_iter().filter_map(|entry| entry.payload)
    }
}

impl<T> Handle<T> {
    pub(crate) fn new() -> Self {
        Self {
            _ph: PhantomData,
            id: 0,
        }
    }

    pub(crate) unsafe fn from_u64(id: u64) -> Self {
        Self {
            _ph: PhantomData,
            id,
        }
    }

    pub(crate) fn to_u64(self) -> u64 {
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
        let mut arena = ResourceArena::<u32>::default();
        let handle_1 = arena.add(0);
        let handle_2 = arena.add(42);
        let handle_3 = arena.add(145);
        assert_eq!(arena[handle_3], 145);
        assert_eq!(arena[handle_2], 42);
        assert_eq!(arena[handle_1], 0);

        assert_eq!(arena.len(), 3);
        assert_eq!(arena.remove(handle_2).unwrap(), 42);
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
