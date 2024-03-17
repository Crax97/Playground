use bevy_ecs::system::Resource as BevyResource;
use crossbeam::channel::{Receiver, Sender};
use gpu::Gpu;
use log::{error, info};
use std::any::{type_name, Any, TypeId};
use std::collections::HashMap;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use thunderdome::{Arena, Index};

use super::hot_reload_server::HotReloadServer;

pub trait Asset: Send + Sync + 'static {
    fn get_description(&self) -> &str;
    fn destroyed(&mut self, gpu: &dyn Gpu);
}

#[repr(transparent)]
#[derive(Clone, Copy, Hash)]
pub struct AssetId {
    pub(crate) id: Option<Index>,
}

pub trait ResourceLoader: Send + Sync + 'static {
    type LoadedResource: Asset;

    fn load(&self, path: &Path) -> anyhow::Result<Self::LoadedResource>;
}

#[derive(BevyResource)]
pub struct AssetMap {
    loaded_resources: LoadedResources,
    gpu: Arc<dyn Gpu>,
    operations_receiver: Receiver<Box<dyn ResourceMapOperation>>,
    operations_sender: Sender<Box<dyn ResourceMapOperation>>,

    resource_loaders: HashMap<TypeId, Box<dyn ErasedResourceLoader>>,

    hot_reload_server: Option<HotReloadServer>,
}

#[derive(Clone)]
pub struct AssetMetadata {
    pub name: String,
}

impl AssetMap {
    pub fn new(gpu: Arc<dyn Gpu>) -> Self {
        let (operations_sender, operations_receiver) = crossbeam::channel::unbounded();
        Self {
            loaded_resources: LoadedResources::default(),
            operations_receiver,
            operations_sender,
            resource_loaders: HashMap::new(),
            gpu,
            hot_reload_server: Some(HotReloadServer::new()),
        }
    }

    pub fn add<R: Asset>(&mut self, resource: R) -> AssetHandle<R> {
        let id = {
            let mut handle = self.get_or_insert_arena_mut::<R>();
            handle.insert(RefCountedAsset::new(resource, None))
        };
        AssetHandle {
            _marker: PhantomData,
            id: AssetId { id: Some(id) },
            operation_sender: Some(self.operations_sender.clone()),
        }
    }

    pub fn iter_ids<R: Asset>(&mut self, mut fun: impl FnMut(AssetId, &AssetMetadata)) {
        let map = self.get_or_insert_arena_mut::<R>();
        map.iter()
            .for_each(|(idx, asset)| fun(AssetId { id: Some(idx) }, &asset.metadata))
    }

    pub fn upcast_index<R: Asset>(&self, id: AssetId) -> AssetHandle<R> {
        let handle = AssetHandle {
            _marker: PhantomData,
            id,
            operation_sender: Some(self.operations_sender.clone()),
        };

        handle.inc_ref_count();

        self.get(&handle);

        handle
    }
    pub fn install_resource_loader<L: ResourceLoader>(&mut self, loader: L) {
        let old = self
            .resource_loaders
            .insert(TypeId::of::<L::LoadedResource>(), Box::new(loader));

        if let Some(old) = old {
            panic!(
                "A resource loader for resource type {:?} has already been installed of type {:?}",
                TypeId::of::<L::LoadedResource>(),
                old.type_id()
            )
        }
    }

    pub fn load<R: Asset>(&mut self, path: impl AsRef<Path>) -> anyhow::Result<AssetHandle<R>> {
        let resource_type = TypeId::of::<R>();
        if let Some(loader) = self.resource_loaders.get(&resource_type) {
            let loaded_resource = loader.load_erased(path.as_ref())?;

            let id = {
                let mut handle = self.get_or_insert_arena_mut::<R>();
                handle.insert(RefCountedAsset::wrap(
                    loaded_resource,
                    Some(path.as_ref().to_str().unwrap().to_string()),
                ))
            };
            let id = AssetId { id: Some(id) };

            if let Some(server) = &mut self.hot_reload_server {
                server.watch(path.as_ref(), resource_type, id)?;
            }

            Ok(AssetHandle {
                _marker: PhantomData,
                id,
                operation_sender: Some(self.operations_sender.clone()),
            })
        } else {
            Err(anyhow::format_err!(
                "No loader for resource type {:?}",
                &TypeId::of::<R>()
            ))
        }
    }

    fn get_arena_handle<R: Asset>(&self) -> RwLockReadGuard<Resources> {
        self.loaded_resources
            .resources
            .get(&TypeId::of::<R>())
            .unwrap()
            .read()
            .unwrap()
    }
    fn get_or_insert_arena_mut<R: Asset>(&mut self) -> RwLockWriteGuard<Resources> {
        self.loaded_resources
            .resources
            .entry(TypeId::of::<R>())
            .or_insert_with(|| RwLock::new(Arena::new()))
            .write()
            .unwrap()
    }
    fn get_arena_handle_mut<R: Asset>(&self) -> RwLockWriteGuard<Resources> {
        self.loaded_resources
            .resources
            .get(&TypeId::of::<R>())
            .unwrap()
            .write()
            .unwrap()
    }

    pub fn get<R: Asset>(&self, id: &AssetHandle<R>) -> &R {
        self.try_get(id).unwrap()
    }

    pub fn get_mut<R: Asset>(&mut self, id: &AssetHandle<R>) -> &mut R {
        self.try_get_mut(id).unwrap()
    }

    pub fn try_get<'a, R: Asset>(&'a self, id: &AssetHandle<R>) -> Option<&'a R> {
        assert!(id.is_not_null(), "Tried to get a null resource");
        let arena_handle = self.get_arena_handle::<R>();
        let object_arc = arena_handle
            .get(id.id.id.unwrap())
            .map(|r| r.resource.clone());

        object_arc
            .map(|a| Arc::as_ptr(&a))
            .map(|a| unsafe { a.as_ref() }.unwrap().as_any())
            .map(|obj| obj.downcast_ref::<R>().unwrap())
    }
    pub fn try_get_mut<'a, R: Asset>(&'a self, id: &AssetHandle<R>) -> Option<&'a mut R> {
        assert!(id.is_not_null(), "Tried to get_mut a null resource");
        let arena_handle = self.get_arena_handle::<R>();
        let object_arc = arena_handle
            .get(id.id.id.unwrap())
            .map(|r| r.resource.clone());
        object_arc
            .map(|a| Arc::as_ptr(&a).cast_mut())
            .map(|a| unsafe { a.as_mut() }.unwrap().as_any_mut())
            .map(|obj| obj.downcast_mut::<R>().unwrap())
    }

    pub fn len<R: Asset>(&self) -> usize {
        self.get_arena_handle_mut::<R>().len()
    }

    pub fn is_empty<R: Asset>(&self) -> bool {
        self.get_arena_handle_mut::<R>().is_empty()
    }

    /* Call this on each frame, to correctly destroy unreferenced resources.
    Please note that if a Resource A references another resource B, and B is only referenced by A
    when A is destroyed on an update() call, B is going to be destroyed on the next update() call
    */
    pub fn update(&mut self) {
        self.update_hot_reload()
            .expect("Failed to update hot reload server");

        let operations = self.operations_receiver.try_iter().collect::<Vec<_>>();
        for op in operations {
            op.execute(self)
        }
    }

    fn update_hot_reload(&mut self) -> anyhow::Result<()> {
        if let Some(server) = &mut self.hot_reload_server {
            server.update(
                &self.resource_loaders,
                &mut self.loaded_resources,
                &self.gpu,
            )?;
        }

        Ok(())
    }

    fn increment_resource_ref_count<R: Asset>(&mut self, id: AssetId) {
        let mut arena_handle = self.get_arena_handle_mut::<R>();
        let resource_mut = arena_handle
            .get_mut(
                id.id
                    .expect("Tried to increment the refcount of a null() resource"),
            )
            .unwrap_or_else(|| panic!("Failed to fetch resource of type {}", type_name::<R>()));
        resource_mut.ref_count += 1;
    }
    fn decrement_resource_ref_count<R: Asset>(&mut self, id: AssetId) {
        let mut arena_handle = self.get_arena_handle_mut::<R>();
        let ref_count = {
            let resource_mut = arena_handle
                .get_mut(
                    id.id
                        .expect("Tried to decrement the refcount of null() resource"),
                )
                .unwrap_or_else(|| panic!("Failed to fetch resource of type {}", type_name::<R>()));
            resource_mut.ref_count -= 1;

            resource_mut.ref_count
        };

        if ref_count == 0 {
            let mut removed_resource = arena_handle.remove(id.id.unwrap()).unwrap_or_else(|| {
                panic!("Failed to remove resource of type {}", type_name::<R>())
            });

            let resource = Arc::get_mut(&mut removed_resource.resource)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<R>()
                .expect("Failed to downcast to resource");
            resource.destroyed(self.gpu.as_ref());

            info!(
                "Deleted resource {} of type {}",
                resource.get_description(),
                type_name::<R>()
            );
        }
    }

    pub fn asset_metadata<T: crate::asset_map::Asset>(
        &self,
        handle: &mut AssetHandle<T>,
    ) -> AssetMetadata {
        let arena = self.get_arena_handle::<T>();
        let asset = arena
            .get(handle.id.id.expect("handle is null!"))
            .expect("Asset id is invalid");
        asset.metadata.clone()
    }
}

impl Drop for AssetMap {
    fn drop(&mut self) {
        // Update any pending resources
        self.update();
    }
}

pub(super) trait AnyResource: Any + Asset {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any + Asset + 'static> AnyResource for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
pub(super) type ResourcePtr = Arc<dyn AnyResource>;

pub(super) struct RefCountedAsset {
    pub resource: ResourcePtr,
    pub metadata: AssetMetadata,
    ref_count: u32,
}
impl RefCountedAsset {
    fn new(resource: impl Asset, name: Option<String>) -> Self {
        let name = name.unwrap_or_else(|| resource.get_description().to_string());
        Self {
            resource: Arc::new(resource),
            ref_count: 1,
            metadata: AssetMetadata { name },
        }
    }

    fn wrap(resource: ResourcePtr, name: Option<String>) -> Self {
        let name = name.unwrap_or_else(|| resource.get_description().to_string());
        Self {
            resource,
            ref_count: 1,
            metadata: AssetMetadata { name },
        }
    }
}

pub(super) trait ErasedResourceLoader: Send + Sync + 'static {
    fn load_erased(&self, path: &Path) -> anyhow::Result<ResourcePtr>;
}

impl<T: ResourceLoader> ErasedResourceLoader for T
where
    T: Send + Sync + 'static,
{
    fn load_erased(&self, path: &Path) -> anyhow::Result<ResourcePtr> {
        let inner = self.load(path)?;

        Ok(Arc::new(inner))
    }
}

type Resources = Arena<RefCountedAsset>;

#[derive(Default)]
pub(super) struct LoadedResources {
    pub resources: HashMap<TypeId, RwLock<Resources>>,
}

pub(crate) trait ResourceMapOperation: Send + Sync + 'static {
    fn execute(&self, map: &mut AssetMap);
}

pub struct AssetHandle<R>
where
    R: Asset,
{
    _marker: PhantomData<R>,
    pub(crate) id: AssetId,
    pub(crate) operation_sender: Option<Sender<Box<dyn ResourceMapOperation>>>,
}

impl<R: Asset> std::fmt::Debug for AssetHandle<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.id.id.fmt(f)
    }
}

impl<R: Asset> Default for AssetHandle<R> {
    fn default() -> Self {
        Self::null()
    }
}

impl<R: Asset> AssetHandle<R> {
    pub fn null() -> Self {
        Self {
            _marker: PhantomData,
            id: AssetId { id: None },
            operation_sender: None,
        }
    }

    pub fn is_null(&self) -> bool {
        self.id.id.is_none()
    }

    pub fn is_not_null(&self) -> bool {
        !self.is_null()
    }

    fn inc_ref_count(&self) {
        if self.is_null() {
            return;
        }
        self.operation_sender
            .as_ref()
            .expect("Handle is null()")
            .send(self.inc_ref_count_operation())
            .unwrap_or_else(|err| error!("Failed to send increment operation: {err}"));
    }

    fn dec_ref_count(&self) {
        if self.is_null() {
            return;
        }
        self.operation_sender
            .as_ref()
            .expect("Handle is null()")
            .send(self.dec_ref_count_operation())
            .unwrap_or_else(|err| error!("Failed to send decrement operation: {err}"));
    }

    fn inc_ref_count_operation(&self) -> Box<dyn ResourceMapOperation> {
        struct IncResourceMapOperation<R: Asset>(AssetId, PhantomData<R>);

        impl<R: Asset> ResourceMapOperation for IncResourceMapOperation<R> {
            fn execute(&self, map: &mut AssetMap) {
                map.increment_resource_ref_count::<R>(self.0)
            }
        }
        Box::new(IncResourceMapOperation::<R>(self.id, PhantomData))
    }
    fn dec_ref_count_operation(&self) -> Box<dyn ResourceMapOperation> {
        struct DecResourceMapOperation<R: Asset>(AssetId, PhantomData<R>);

        impl<R: Asset> ResourceMapOperation for DecResourceMapOperation<R> {
            fn execute(&self, map: &mut AssetMap) {
                map.decrement_resource_ref_count::<R>(self.0)
            }
        }
        Box::new(DecResourceMapOperation::<R>(self.id, PhantomData))
    }
}

impl<R: Asset> PartialEq for AssetHandle<R> {
    fn eq(&self, other: &Self) -> bool {
        self.id.id == other.id.id
    }
}

impl<R: Asset> Eq for AssetHandle<R> {}

impl<R: Asset> std::hash::Hash for AssetHandle<R> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
impl<R: Asset> Clone for AssetHandle<R> {
    fn clone(&self) -> Self {
        self.inc_ref_count();
        Self {
            _marker: self._marker,
            id: self.id,
            operation_sender: self.operation_sender.clone(),
        }
    }
}

impl<R: Asset> Drop for AssetHandle<R> {
    fn drop(&mut self) {
        self.dec_ref_count()
    }
}

// #[cfg(test)]
// mod test {
//     use super::{Resource, ResourceMap};
//     use crate::ResourceHandle;

//     struct TestResource {
//         val: u32,
//     }

//     impl Resource for TestResource {
//         fn get_description(&self) -> &str {
//             "test resource"
//         }
//         fn destroyed(&mut self) {
//             println!("TestResource destroyed");
//         }
//     }
//     struct TestResource2 {
//         val2: u32,
//     }

//     impl Resource for TestResource2 {
//         fn get_description(&self) -> &str {
//             "test resource 2"
//         }
//         fn destroyed(&mut self) {
//             println!("TestResource2 destroyed");
//         }
//     }

//     #[test]
//     fn test_get() {
//         let mut map: ResourceMap = ResourceMap::new();
//         let id: ResourceHandle<TestResource> = map.add(TestResource { val: 10 });

//         assert_eq!(map.get(&id).val, 10);
//     }

//     #[test]
//     fn test_drop() {
//         let mut map = ResourceMap::new();
//         let id_2 = map.add(TestResource { val: 14 });
//         let id_3 = map.add(TestResource2 { val2: 142 });
//         {
//             let id = map.add(TestResource { val: 10 });
//             assert_eq!(map.get(&id).val, 10);
//             assert_eq!(map.get(&id_2).val, 14);

//             assert_eq!(map.len::<TestResource>(), 2);
//         }

//         map.update();

//         assert_eq!(map.len::<TestResource>(), 1);
//         assert_eq!(map.len::<TestResource2>(), 1);
//         assert_eq!(map.get(&id_2).val, 14);
//         assert_eq!(map.get(&id_3).val2, 142);
//     }

//     #[test]
//     fn test_shuffle_memory() {
//         let (mut map, id_2) = {
//             let mut map = ResourceMap::new();
//             let id_2 = map.add(TestResource { val: 14 });
//             (map, id_2)
//         };

//         {
//             let id = map.add(TestResource { val: 10 });

//             map.update();
//             let do_checks = |map: ResourceMap| {
//                 assert_eq!(map.get(&id).val, 10);
//                 assert_eq!(map.get(&id_2).val, 14);

//                 assert_eq!(map.len::<TestResource>(), 2);
//                 map
//             };

//             map.update();
//             map = do_checks(map);
//         }

//         map.update();
//         assert_eq!(map.len::<TestResource>(), 1);
//         assert_eq!(map.get(&id_2).val, 14);
//     }

//     #[test]
//     fn test_other_map() {
//         let mut map_1 = ResourceMap::new();
//         let id_1 = map_1.add(TestResource { val: 1 });
//         drop(map_1);

//         let map_2 = ResourceMap::new();
//         let value = map_2.try_get(&id_1);
//         assert!(value.is_none());
//     }

//     #[test]
//     fn nested_resources() {
//         struct B;
//         impl Resource for B {
//             fn get_description(&self) -> &str {
//                 "B"
//             }
//             fn destroyed(&mut self) {
//                 println!("B destroyed");
//             }
//         }

//         struct A {
//             handle_1: ResourceHandle<B>,
//             handle_2: ResourceHandle<B>,
//         }
//         impl Resource for A {
//             fn get_description(&self) -> &str {
//                 "A"
//             }
//             fn destroyed(&mut self) {
//                 println!("A destroyed");
//             }
//         }

//         let mut map = ResourceMap::new();
//         let h1 = map.add(B);
//         let h2 = map.add(B);

//         let ha = map.add(A {
//             handle_1: h1.clone(),
//             handle_2: h2.clone(),
//         });

//         assert_eq!(map.get(&ha).handle_1, h1);
//         assert_eq!(map.get(&ha).handle_2, h2);

//         drop(ha);
//         drop(h1);
//         drop(h2);
//         map.update();
//         // Need to update again because B's are released after A's, so they're destroyed on the
//         // next update call
//         map.update();
//         assert!(map.get_arena_handle::<A>().is_empty());
//         assert!(map.get_arena_handle::<B>().is_empty());
//     }
// }
