use std::fmt::Formatter;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};
use thunderdome::{Arena, Index};

pub trait Resource {
    fn get_description(&self) -> &str;
}

#[repr(transparent)]
#[derive(Clone, Copy, Hash)]
pub(crate) struct ResourceId {
    pub(crate) id: Index,
}

pub struct ResourceMap {
    map: RefCell<anymap::AnyMap>,
}

impl Default for ResourceMap {
    fn default() -> Self {
        Self {
            map: RefCell::new(anymap::AnyMap::new()),
        }
    }
}

pub struct ResourceHandle<R>
where
    R: Resource + 'static,
{
    _marker: PhantomData<R>,
    pub(crate) id: ResourceId,
    pub(crate) reference_counter: Rc<RefCell<u32>>,

    owner_arena: Rc<RefCell<Arena<R>>>,
}

impl<R: Resource + 'static> std::fmt::Debug for ResourceHandle<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.id.id.fmt(f)
    }
}

impl<R: Resource + 'static> ResourceHandle<R> {
    fn inc_ref_count(&self) -> u32 {
        let mut counter = self.reference_counter.borrow_mut();
        *counter += 1;
        *counter
    }

    fn dec_ref_count(&self) -> u32 {
        let mut counter = self.reference_counter.borrow_mut();
        *counter -= 1;
        *counter
    }
}

impl<R: Resource + 'static> PartialEq for ResourceHandle<R> {
    fn eq(&self, other: &Self) -> bool {
        self.id.id == other.id.id
    }
}

impl<R: Resource + 'static> Eq for ResourceHandle<R> {}

impl<R: Resource + 'static> std::hash::Hash for ResourceHandle<R> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
impl<R: Resource + 'static> Clone for ResourceHandle<R> {
    fn clone(&self) -> Self {
        self.inc_ref_count();
        Self {
            _marker: self._marker,
            id: self.id,
            reference_counter: self.reference_counter.clone(),
            owner_arena: self.owner_arena.clone(),
        }
    }
}

impl<R: Resource + 'static> Drop for ResourceHandle<R> {
    fn drop(&mut self) {
        let ref_count = self.dec_ref_count();
        if ref_count == 0 {
            let arena = self.owner_arena.clone();
            arena
                .borrow_mut()
                .remove(self.id.id)
                .expect("Failed to remove resource");
        }
    }
}

impl ResourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add<R: Resource + 'static>(&mut self, resource: R) -> ResourceHandle<R> {
        let handle = self.get_arena_mut::<R>();
        let id = handle.insert(resource);
        ResourceHandle {
            _marker: PhantomData,
            id: ResourceId { id },
            reference_counter: Rc::new(RefCell::new(1)),
            owner_arena: self.get_arena_handle::<R>(),
        }
    }

    fn get_arena_handle<R: Resource + 'static>(&self) -> Rc<RefCell<Arena<R>>> {
        self.map
            .borrow_mut()
            .entry::<Rc<RefCell<Arena<R>>>>()
            .or_insert_with(|| Rc::new(RefCell::new(Arena::new())))
            .clone()
    }

    fn get_arena<R: Resource + 'static>(&self) -> &Arena<R> {
        let handle = self.get_arena_handle::<R>();
        let ptr = handle.as_ptr();
        unsafe { &*ptr }
    }

    fn get_arena_mut<R: Resource + 'static>(&mut self) -> &mut Arena<R> {
        let handle = self.get_arena_handle::<R>();
        let ptr = handle.as_ptr();
        unsafe { &mut *ptr }
    }

    pub fn get<R: Resource + 'static>(&self, id: &ResourceHandle<R>) -> &R {
        self.try_get(id).unwrap()
    }

    pub fn get_mut<R: Resource + 'static>(&mut self, id: &ResourceHandle<R>) -> &mut R {
        self.try_get_mut(id).unwrap()
    }

    pub fn try_get<R: Resource + 'static>(&self, id: &ResourceHandle<R>) -> Option<&R> {
        let arena_handle = self.get_arena();
        arena_handle.get(id.id.id)
    }

    pub fn try_get_mut<R: Resource + 'static>(&mut self, id: &ResourceHandle<R>) -> Option<&mut R> {
        let arena_handle = self.get_arena_mut();
        arena_handle.get_mut(id.id.id)
    }

    pub fn len<R: Resource + 'static>(&self) -> usize {
        self.get_arena_handle::<R>().borrow().len()
    }

    pub fn is_empty<R: Resource + 'static>(&self) -> bool {
        self.get_arena_handle::<R>().borrow().is_empty()
    }
}

#[cfg(test)]
mod test {
    use super::{Resource, ResourceMap};
    use crate::ResourceHandle;

    struct TestResource {
        val: u32,
    }

    impl Resource for TestResource {
        fn get_description(&self) -> &str {
            "test resource"
        }
    }
    struct TestResource2 {
        val2: u32,
    }

    impl Resource for TestResource2 {
        fn get_description(&self) -> &str {
            "test resource 2"
        }
    }

    #[test]
    fn test_get() {
        let mut map: ResourceMap = ResourceMap::new();
        let id: ResourceHandle<TestResource> = map.add(TestResource { val: 10 });

        assert_eq!(map.get(&id).val, 10);
    }

    #[test]
    fn test_drop() {
        let mut map = ResourceMap::new();
        let id_2 = map.add(TestResource { val: 14 });
        let id_3 = map.add(TestResource2 { val2: 142 });
        {
            let id = map.add(TestResource { val: 10 });
            assert_eq!(map.get(&id).val, 10);
            assert_eq!(map.get(&id_2).val, 14);

            assert_eq!(map.len::<TestResource>(), 2);
        }

        assert_eq!(map.len::<TestResource>(), 1);
        assert_eq!(map.len::<TestResource2>(), 1);
        assert_eq!(map.get(&id_2).val, 14);
        assert_eq!(map.get(&id_3).val2, 142);
    }

    #[test]
    fn test_shuffle_memory() {
        let (mut map, id_2) = {
            let mut map = ResourceMap::new();
            let id_2 = map.add(TestResource { val: 14 });
            (map, id_2)
        };

        {
            let id = map.add(TestResource { val: 10 });

            let do_checks = |map: ResourceMap| {
                assert_eq!(map.get(&id).val, 10);
                assert_eq!(map.get(&id_2).val, 14);

                assert_eq!(map.len::<TestResource>(), 2);
                map
            };

            map = do_checks(map);
        }

        assert_eq!(map.len::<TestResource>(), 1);
        assert_eq!(map.get(&id_2).val, 14);
    }

    #[test]
    fn test_other_map() {
        let mut map_1 = ResourceMap::new();
        let id_1 = map_1.add(TestResource { val: 1 });
        drop(map_1);

        let map_2 = ResourceMap::new();
        let value = map_2.try_get(&id_1);
        assert!(value.is_none());
    }

    #[test]
    fn nested_resources() {
        struct B;
        impl Resource for B {
            fn get_description(&self) -> &str {
                "B"
            }
        }

        struct A {
            handle_1: ResourceHandle<B>,
            handle_2: ResourceHandle<B>,
        }
        impl Resource for A {
            fn get_description(&self) -> &str {
                "A"
            }
        }

        let mut map = ResourceMap::new();
        let h1 = map.add(B);
        let h2 = map.add(B);

        let ha = map.add(A {
            handle_1: h1.clone(),
            handle_2: h2.clone(),
        });

        assert_eq!(map.get(&ha).handle_1, h1);
        assert_eq!(map.get(&ha).handle_2, h2);

        drop(ha);
        drop(h1);
        drop(h2);
        assert!(map.get_arena::<A>().is_empty());
        assert!(map.get_arena::<B>().is_empty());
    }
}
