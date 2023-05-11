use std::{
    cell::RefCell,
    marker::PhantomData,
    sync::{Arc, Weak},
};
use thunderdome::{Arena, Index};

pub trait Resource {}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct ResourceId {
    pub(crate) id: Index,
}

pub struct ResourceMapState {
    types_map: anymap::AnyMap,
    resources: usize,
}

pub struct ResourceMap {
    map: Arc<RefCell<ResourceMapState>>,
}

pub struct ResourceHandle<R>
where
    R: Resource + 'static,
{
    _marker: PhantomData<R>,
    pub(crate) id: ResourceId,
    pub(crate) reference_counter: u32,

    resource_map: Weak<RefCell<ResourceMapState>>,
}

impl<R: Resource + 'static> Clone for ResourceHandle<R> {
    fn clone(&self) -> Self {
        Self {
            _marker: self._marker,
            id: self.id.clone(),
            reference_counter: self.reference_counter.clone() + 1,
            resource_map: self.resource_map.clone(),
        }
    }
}

impl<R: Resource + 'static> Drop for ResourceHandle<R> {
    fn drop(&mut self) {
        self.reference_counter -= 1;

        if self.reference_counter == 0 {
            let map = self.resource_map.upgrade().unwrap();
            let mut map = map.borrow_mut();
            map.resources -= 1;
            let arena = map.types_map.get_mut::<Arena<R>>().unwrap();
            arena.remove(self.id.id);
        }
    }
}

impl ResourceMap {
    pub(crate) fn new() -> Self {
        let state = ResourceMapState {
            types_map: anymap::AnyMap::new(),
            resources: 0,
        };

        Self {
            map: Arc::new(RefCell::new(state)),
        }
    }

    pub(crate) fn add<R: Resource + 'static>(&mut self, resource: R) -> ResourceHandle<R> {
        self.map.borrow_mut().resources += 1;
        let store = self.get_arena::<R>();
        let id = store.insert(resource);
        ResourceHandle {
            _marker: PhantomData,
            id: ResourceId { id },
            reference_counter: 1,
            resource_map: Arc::downgrade(&self.map),
        }
    }

    pub(crate) fn get<R: Resource + 'static>(&self, id: &ResourceHandle<R>) -> &R {
        let arena_ref = self.get_arena_unchecked();
        arena_ref.get(id.id.id).unwrap()
    }

    pub(crate) fn get_mut<R: Resource + 'static>(&mut self, id: &ResourceHandle<R>) -> &mut R {
        let arena_ref = self.get_arena();
        arena_ref.get_mut(id.id.id).unwrap()
    }

    pub(crate) fn len_total(&self) -> usize {
        self.map.borrow().resources
    }

    pub(crate) fn len<R: Resource + 'static>(&self) -> usize {
        self.get_arena_unchecked::<R>().len()
    }

    fn get_arena<R: Resource + 'static>(&mut self) -> &mut Arena<R> {
        let map: *mut ResourceMapState = self.map.as_ptr();
        let map =
            unsafe { std::mem::transmute::<*mut ResourceMapState, &mut ResourceMapState>(map) };

        if !map.types_map.contains::<Arena<R>>() {
            map.types_map.insert(Arena::<R>::new());
        }

        map.types_map.get_mut::<Arena<R>>().unwrap()
    }

    fn get_arena_unchecked<R: Resource + 'static>(&self) -> &Arena<R> {
        let map: *mut ResourceMapState = self.map.as_ptr();
        let map =
            unsafe { std::mem::transmute::<*mut ResourceMapState, &mut ResourceMapState>(map) };
        map.types_map.get::<Arena<R>>().unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::{Resource, ResourceMap};

    struct TestResource {
        val: u32,
    }

    impl Resource for TestResource {}
    struct TestResource2 {
        val2: u32,
    }

    impl Resource for TestResource2 {}

    #[test]
    fn test_get() {
        let mut map = ResourceMap::new();
        let id = map.add(TestResource { val: 10 });

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
}
