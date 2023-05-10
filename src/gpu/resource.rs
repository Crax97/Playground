use std::{
    cell::{Ref, RefCell},
    marker::PhantomData,
    rc::Rc,
};
use thunderdome::{Arena, Index};

pub trait Resource {}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct ResourceId {
    pub(crate) id: Index,
}

pub struct ResourceStore {}

impl ResourceStore {}
pub struct ResourceHandle<R>
where
    R: Resource,
{
    _marker: PhantomData<R>,
    pub(crate) id: ResourceId,
    pub(crate) reference_counter: u32,
}

impl<R: Resource> Clone for ResourceHandle<R> {
    fn clone(&self) -> Self {
        Self {
            _marker: self._marker,
            id: self.id.clone(),
            reference_counter: self.reference_counter.clone() + 1,
            // store: self.store.clone(),
        }
    }
}

impl<R: Resource> Drop for ResourceHandle<R> {
    fn drop(&mut self) {
        self.reference_counter -= 1;

        if self.reference_counter == 0 {
            // self.store.borrow_mut().remove(self.id.id);
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

    pub(crate) fn len(&self) -> usize {
        self.resources
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
}

#[cfg(test)]
mod test {
    use super::{Resource, ResourceMap};

    struct TestResource {
        val: u32,
    }

    impl Resource for TestResource {}

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
        {
            let id = map.add(TestResource { val: 10 });
            assert_eq!(map.get(&id).val, 10);
            assert_eq!(map.get(&id_2).val, 14);

            assert_eq!(map.len(), 2);
        }

        assert_eq!(map.len(), 1);
    }
}
