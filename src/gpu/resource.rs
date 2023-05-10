use std::marker::PhantomData;
use thunderdome::{Arena, Index};

pub trait Resource {}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct ResourceId {
    pub(crate) id: Index,
}

pub struct ResourceHandle<R>
where
    R: Resource + 'static,
{
    _marker: PhantomData<R>,
    pub(crate) id: ResourceId,
    pub(crate) reference_counter: u32,

    resource_map: *mut ResourceMap,
}

impl<R: Resource + 'static> Clone for ResourceHandle<R> {
    fn clone(&self) -> Self {
        Self {
            _marker: self._marker,
            id: self.id.clone(),
            reference_counter: self.reference_counter.clone() + 1,
            resource_map: self.resource_map,
        }
    }
}

impl<R: Resource + 'static> Drop for ResourceHandle<R> {
    fn drop(&mut self) {
        self.reference_counter -= 1;

        if self.reference_counter == 0 {
            unsafe { std::mem::transmute::<*mut ResourceMap, &mut ResourceMap>(self.resource_map) }
                .drop_resource::<R>(&self.id);
        }
    }
}

pub struct ResourceMap {
    types_map: anymap::AnyMap,
    resources: usize,
}

impl ResourceMap {
    pub(crate) fn new() -> Self {
        Self {
            types_map: anymap::AnyMap::new(),
            resources: 0,
        }
    }

    pub(crate) fn add<R: Resource + 'static>(&mut self, resource: R) -> ResourceHandle<R> {
        let store = self.get_arena::<R>();
        let id = store.insert(resource);
        self.resources += 1;
        ResourceHandle {
            _marker: PhantomData,
            id: ResourceId { id },
            reference_counter: 1,
            resource_map: self as *mut Self,
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
        self.resources
    }

    pub(crate) fn len<R: Resource + 'static>(&self) -> usize {
        self.get_arena_unchecked::<R>().len()
    }

    fn get_arena<R: Resource + 'static>(&mut self) -> &mut Arena<R> {
        if !self.types_map.contains::<Arena<R>>() {
            self.types_map.insert(Arena::<R>::new());
        }

        self.types_map.get_mut::<Arena<R>>().unwrap()
    }

    fn get_arena_unchecked<R: Resource + 'static>(&self) -> &Arena<R> {
        self.types_map.get::<Arena<R>>().unwrap()
    }

    fn drop_resource<R: Resource + 'static>(&mut self, id: &ResourceId) {
        self.get_arena::<R>().remove(id.id).unwrap();
        self.resources -= 1;
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
        let mut map = ResourceMap::new();

        let id_2 = map.add(TestResource { val: 14 });
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
