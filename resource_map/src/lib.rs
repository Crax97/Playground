use crossbeam::channel::{Receiver, Sender};
use log::{error, info};
use std::any::type_name;
use std::fmt::Formatter;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};
use thunderdome::{Arena, Index};

pub trait Resource: Send + Sync + 'static {
    fn get_description(&self) -> &str;
}

#[repr(transparent)]
#[derive(Clone, Copy, Hash)]
pub(crate) struct ResourceId {
    pub(crate) id: Index,
}

pub struct RefCounted<R: Resource> {
    resource: R,
    ref_count: u32,
}

impl<R: Resource> RefCounted<R> {
    fn new(resource: R) -> Self {
        Self {
            resource,
            ref_count: 1,
        }
    }
}

pub struct ResourceMap {
    map: RefCell<anymap::AnyMap>,
    operations_receiver: Receiver<Box<dyn ResourceMapOperation>>,
    operations_sender: Sender<Box<dyn ResourceMapOperation>>,
}

impl Default for ResourceMap {
    fn default() -> Self {
        let (operations_sender, operations_receiver) = crossbeam::channel::unbounded();
        Self {
            map: RefCell::new(anymap::AnyMap::new()),
            operations_receiver,
            operations_sender,
        }
    }
}

trait ResourceMapOperation: Send + Sync + 'static {
    fn execute(&self, map: &mut ResourceMap);
}

pub struct ResourceHandle<R>
where
    R: Resource + 'static,
{
    _marker: PhantomData<R>,
    pub(crate) id: ResourceId,
    pub(crate) operation_sender: Sender<Box<dyn ResourceMapOperation>>,
}

impl<R: Resource + 'static> std::fmt::Debug for ResourceHandle<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.id.id.fmt(f)
    }
}

impl<R: Resource + 'static> ResourceHandle<R> {
    fn inc_ref_count(&self) {
        self.operation_sender
            .send(self.inc_ref_count_operation())
            .unwrap_or_else(|err| error!("Failed to send increment operation: {err}"));
    }

    fn dec_ref_count(&self) {
        self.operation_sender
            .send(self.dec_ref_count_operation())
            .unwrap_or_else(|err| error!("Failed to send decrement operation: {err}"));
    }

    fn inc_ref_count_operation(&self) -> Box<dyn ResourceMapOperation> {
        struct IncResourceMapOperation<R: Resource>(ResourceId, PhantomData<R>);

        impl<R: Resource + 'static> ResourceMapOperation for IncResourceMapOperation<R> {
            fn execute(&self, map: &mut ResourceMap) {
                map.increment_resource_ref_count::<R>(self.0)
            }
        }
        Box::new(IncResourceMapOperation::<R>(
            self.id,
            PhantomData::default(),
        ))
    }
    fn dec_ref_count_operation(&self) -> Box<dyn ResourceMapOperation> {
        struct DecResourceMapOperation<R: Resource>(ResourceId, PhantomData<R>);

        impl<R: Resource + 'static> ResourceMapOperation for DecResourceMapOperation<R> {
            fn execute(&self, map: &mut ResourceMap) {
                map.decrement_resource_ref_count::<R>(self.0)
            }
        }
        Box::new(DecResourceMapOperation::<R>(
            self.id,
            PhantomData::default(),
        ))
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
            operation_sender: self.operation_sender.clone(),
        }
    }
}

impl<R: Resource + 'static> Drop for ResourceHandle<R> {
    fn drop(&mut self) {
        self.dec_ref_count()
    }
}

impl ResourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add<R: Resource + 'static>(&mut self, resource: R) -> ResourceHandle<R> {
        let handle = self.get_arena_mut::<R>();
        let id = handle.insert(RefCounted::new(resource));
        ResourceHandle {
            _marker: PhantomData,
            id: ResourceId { id },
            operation_sender: self.operations_sender.clone(),
        }
    }

    fn get_arena_handle<R: Resource + 'static>(&self) -> Rc<RefCell<Arena<RefCounted<R>>>> {
        self.map
            .borrow_mut()
            .entry::<Rc<RefCell<Arena<RefCounted<R>>>>>()
            .or_insert_with(|| Rc::new(RefCell::new(Arena::new())))
            .clone()
    }

    fn get_arena<R: Resource + 'static>(&self) -> &Arena<RefCounted<R>> {
        let handle = self.get_arena_handle::<R>();
        let ptr = handle.as_ptr();
        unsafe { &*ptr }
    }

    fn get_arena_mut<R: Resource + 'static>(&mut self) -> &mut Arena<RefCounted<R>> {
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
        arena_handle.get(id.id.id).map(|r| &r.resource)
    }

    pub fn try_get_mut<R: Resource + 'static>(&mut self, id: &ResourceHandle<R>) -> Option<&mut R> {
        let arena_handle = self.get_arena_mut::<R>();
        arena_handle.get_mut(id.id.id).map(|r| &mut r.resource)
    }

    pub fn len<R: Resource + 'static>(&self) -> usize {
        self.get_arena_handle::<R>().borrow().len()
    }

    pub fn is_empty<R: Resource + 'static>(&self) -> bool {
        self.get_arena_handle::<R>().borrow().is_empty()
    }

    /* Call this on each frame, to correctly destroy unreferenced resources.
    Please note that if a Resource A references another resource B, and B is only referenced by A
    when A is destroyed on an update() call, B is going to be destroyed on the next update() call
    */
    pub fn update(&mut self) {
        let operations = self.operations_receiver.try_iter().collect::<Vec<_>>();
        for op in operations {
            op.execute(self)
        }
    }

    fn increment_resource_ref_count<R: Resource + 'static>(&mut self, id: ResourceId) {
        let arena_handle = self.get_arena_mut::<R>();
        let resource_mut = arena_handle
            .get_mut(id.id)
            .unwrap_or_else(|| panic!("Failed to fetch resource of type {}", type_name::<R>()));
        resource_mut.ref_count += 1;
    }
    fn decrement_resource_ref_count<R: Resource + 'static>(&mut self, id: ResourceId) {
        let arena_handle = self.get_arena_mut::<R>();
        let ref_count = {
            let resource_mut = arena_handle
                .get_mut(id.id)
                .unwrap_or_else(|| panic!("Failed to fetch resource of type {}", type_name::<R>()));
            resource_mut.ref_count -= 1;

            resource_mut.ref_count
        };

        if ref_count == 0 {
            let removed_resource = arena_handle.remove(id.id).unwrap_or_else(|| {
                panic!("Failed to remove resource of type {}", type_name::<R>())
            });

            info!(
                "Deleted resource {} of type {}",
                removed_resource.resource.get_description(),
                type_name::<R>()
            );
        }
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

        map.update();

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

            map.update();
            let do_checks = |map: ResourceMap| {
                assert_eq!(map.get(&id).val, 10);
                assert_eq!(map.get(&id_2).val, 14);

                assert_eq!(map.len::<TestResource>(), 2);
                map
            };

            map.update();
            map = do_checks(map);
        }

        map.update();
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
        map.update();
        // Need to update again because B's are released after A's, so they're destroyed on the
        // next update call
        map.update();
        assert!(map.get_arena::<A>().is_empty());
        assert!(map.get_arena::<B>().is_empty());
    }
}
