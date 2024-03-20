use bevy_ecs::system::Resource as BevyResource;
use crossbeam::channel::{Receiver, Sender};
use gpu::Gpu;
use log::{error, info, warn};
use std::any::{type_name, Any, TypeId};
use std::collections::HashMap;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use thunderdome::{Arena, Index};
use uuid::Uuid;

use crate::immutable_string::ImmutableString;

use super::hot_reload_server::HotReloadServer;

pub trait Asset: Send + Sync + 'static {
    fn get_description(&self) -> &str;
    fn destroyed(&mut self, gpu: &dyn Gpu);
}

pub struct AssetHandle<R>
where
    R: Asset,
{
    _marker: PhantomData<R>,
    pub(crate) name: ImmutableString,
    pub(crate) operation_sender: Option<Sender<Box<dyn ResourceMapOperation>>>,
}

impl<R: Asset> AssetHandle<R> {
    pub(crate) fn new(
        name: ImmutableString,
        operation_sender: Option<Sender<Box<dyn ResourceMapOperation>>>,
    ) -> Self {
        let me = Self {
            _marker: PhantomData,
            name,
            operation_sender,
        };
        me.inc_ref_count();

        me
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct AssetId {
    pub(crate) id: Index,
}

pub trait ResourceLoader: Send + Sync + 'static {
    type LoadedResource: Asset;

    fn load(&self, path: &Path) -> anyhow::Result<Self::LoadedResource>;
    fn accepts_extension(&self, extension: &str) -> bool;
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

#[derive(Clone, Copy)]
pub enum AssetSource {
    Memory,
    Disk,
}
#[derive(Clone)]
pub struct AssetMetadata {
    pub name: String,
    pub source: AssetSource,
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

    pub fn install(&mut self, base_path: impl AsRef<Path>) -> anyhow::Result<()> {
        let base = std::fs::read_dir(base_path)?;
        for entry in base {
            match entry {
                Ok(entry) => {
                    let ty = entry.file_type()?;
                    if ty.is_file() {
                        self.try_load_resource_metadata(entry.path())?;
                    } else {
                        self.install(entry.path())?;
                    }
                }
                Err(e) => warn!("Could not open file: {e:?}"),
            }
        }

        Ok(())
    }

    pub fn add<R: Asset>(&mut self, resource: R, name: Option<impl AsRef<str>>) -> AssetHandle<R> {
        let name: Option<&str> = name.as_ref().map(|s| s.as_ref());
        let name = name.unwrap_or(resource.get_description());
        let unique_name = format!("{}-{}", name, Uuid::new_v4());

        let sender = self.operations_sender.clone();
        let mut handle = self.get_or_insert_arena_mut::<R>();
        let id = handle
            .resources
            .insert(RefCountedAsset::new(resource, None, AssetSource::Memory));
        let name = ImmutableString::from(unique_name);
        handle
            .resource_statuses
            .insert(name.clone(), ResourceStatus::Loaded(AssetId { id }));
        AssetHandle::new(name, Some(sender))
    }

    pub fn load<R: Asset>(
        &mut self,
        name: impl Into<ImmutableString>,
    ) -> anyhow::Result<AssetHandle<R>> {
        let name = name.into();
        let contained = {
            let handle = self.get_or_insert_arena_mut::<R>();
            handle.resource_statuses.contains_key(&name)
        };
        if contained {
            let sender = self.operations_sender.clone();
            Ok(AssetHandle::new(name, Some(sender)))
        } else {
            self.reload(name.deref())
        }
    }

    pub fn reload<R: Asset>(&mut self, path: impl AsRef<Path>) -> anyhow::Result<AssetHandle<R>> {
        let resource_type = TypeId::of::<R>();
        if let Some(loader) = self.resource_loaders.get(&resource_type) {
            let sender = self.operations_sender.clone();
            let loaded_resource = loader.load_erased(path.as_ref())?;

            let (id, name) = {
                let mut handle = self.get_or_insert_arena_mut::<R>();
                let id = handle.resources.insert(RefCountedAsset::wrap(
                    loaded_resource,
                    Some(path.as_ref().to_str().unwrap().to_string()),
                    AssetSource::Disk,
                ));
                let id = AssetId { id };

                let name = ImmutableString::from(path.as_ref().to_str().unwrap());
                handle
                    .resource_statuses
                    .insert(name.clone(), ResourceStatus::Loaded(id));
                (id, name)
            };

            if let Some(server) = &mut self.hot_reload_server {
                server.watch(path.as_ref(), resource_type, id)?;
            }

            Ok(AssetHandle::new(name, Some(sender)))
        } else {
            Err(anyhow::format_err!(
                "No loader for resource type {:?}",
                &TypeId::of::<R>()
            ))
        }
    }

    pub fn iter_ids<R: Asset>(&mut self, mut fun: impl FnMut(ImmutableString)) {
        let map = self.get_or_insert_arena_mut::<R>();
        map.resource_statuses
            .keys()
            .for_each(|name| fun(name.clone()))
    }

    pub fn upcast_index<R: Asset>(&self, id: ImmutableString) -> AssetHandle<R> {
        let handle = AssetHandle::new(id, Some(self.operations_sender.clone()));
        handle.inc_ref_count();
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

    fn get_arena_handle<R: Asset>(&self) -> RwLockReadGuard<Resources> {
        self.loaded_resources
            .resources
            .get(&TypeId::of::<R>())
            .unwrap()
            .read()
            .unwrap()
    }
    fn get_or_insert_arena_mut<R: Asset>(&mut self) -> RwLockWriteGuard<Resources> {
        self.get_or_insert_arena_mut_erased(TypeId::of::<R>())
    }
    fn get_arena_handle_mut<R: Asset>(&self) -> RwLockWriteGuard<Resources> {
        self.get_arena_handle_mut_erased(TypeId::of::<R>())
    }
    fn get_arena_handle_mut_erased(&self, ty: TypeId) -> RwLockWriteGuard<Resources> {
        self.loaded_resources
            .resources
            .get(&ty)
            .unwrap()
            .write()
            .unwrap()
    }
    fn get_or_insert_arena_mut_erased(&mut self, ty: TypeId) -> RwLockWriteGuard<Resources> {
        self.loaded_resources
            .resources
            .entry(ty)
            .or_default()
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
        let status = arena_handle.resource_statuses.get(&id.name)?;
        match status {
            ResourceStatus::Loaded(id) => {
                let object_arc = arena_handle
                    .resources
                    .get(id.id)
                    .map(|r| r.resource.clone());

                object_arc
                    .map(|a| Arc::as_ptr(&a))
                    .map(|a| unsafe { a.as_ref() }.unwrap().as_any())
                    .map(|obj| obj.downcast_ref::<R>().unwrap())
            }
            ResourceStatus::Unloaded => {
                panic!("Resource is not loaded")
            }
        }
    }
    pub fn try_get_mut<'a, R: Asset>(&'a self, id: &AssetHandle<R>) -> Option<&'a mut R> {
        assert!(id.is_not_null(), "Tried to get_mut a null resource");
        let arena_handle = self.get_arena_handle::<R>();
        let status = arena_handle.resource_statuses.get(&id.name)?;
        match status {
            ResourceStatus::Loaded(id) => {
                let object_arc = arena_handle
                    .resources
                    .get(id.id)
                    .map(|r| r.resource.clone());
                object_arc
                    .map(|a| Arc::as_ptr(&a).cast_mut())
                    .map(|a| unsafe { a.as_mut() }.unwrap().as_any_mut())
                    .map(|obj| obj.downcast_mut::<R>().unwrap())
            }
            ResourceStatus::Unloaded => {
                panic!("Resource is not loaded");
            }
        }
    }

    pub fn len<R: Asset>(&self) -> usize {
        self.get_arena_handle_mut::<R>().resources.len()
    }

    pub fn is_empty<R: Asset>(&self) -> bool {
        self.get_arena_handle_mut::<R>().resources.is_empty()
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

    fn increment_resource_ref_count<R: Asset>(&mut self, name: ImmutableString) {
        let mut arena_handle = self.get_arena_handle_mut::<R>();
        if let Some(status) = arena_handle.resource_statuses.get(&name).copied() {
            match status {
                ResourceStatus::Loaded(id) => {
                    let resource_mut = arena_handle.resources.get_mut(id.id).unwrap_or_else(|| {
                        panic!("Failed to fetch resource of type {}", type_name::<R>())
                    });
                    resource_mut.ref_count += 1;
                }
                ResourceStatus::Unloaded => {
                    drop(arena_handle);
                    self.reload::<R>(name.deref())
                        .expect("Failed to reload asset");
                }
            }
        } else {
            drop(arena_handle);
            self.reload::<R>(name.deref())
                .expect("Failed to reload asset");
        }
    }
    fn decrement_resource_ref_count<R: Asset>(&mut self, name: ImmutableString) {
        let mut arena_handle = self.get_arena_handle_mut::<R>();

        let status = *arena_handle
            .resource_statuses
            .get(&name)
            .expect("Failed to resolve asset id");

        let id = match status {
            ResourceStatus::Loaded(id) => id,
            ResourceStatus::Unloaded => {
                panic!("Resource is already unloaded")
            }
        };

        let ref_count = {
            let resource_mut = arena_handle
                .resources
                .get_mut(id.id)
                .unwrap_or_else(|| panic!("Failed to fetch resource of type {}", type_name::<R>()));
            resource_mut.ref_count -= 1;

            resource_mut.ref_count
        };

        if ref_count == 0 {
            let mut removed_resource = arena_handle.resources.remove(id.id).unwrap_or_else(|| {
                panic!("Failed to remove resource of type {}", type_name::<R>())
            });

            let resource = Arc::get_mut(&mut removed_resource.resource)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<R>()
                .expect("Failed to downcast to resource");
            resource.destroyed(self.gpu.as_ref());

            arena_handle
                .resource_statuses
                .insert(name.clone(), ResourceStatus::Unloaded);

            info!("Deleted resource {name} of type {}", type_name::<R>());
        }
    }

    pub fn asset_metadata<T: crate::asset_map::Asset>(
        &self,
        handle: &mut AssetHandle<T>,
    ) -> AssetMetadata {
        let arena = self.get_arena_handle::<T>();
        let status = arena
            .resource_statuses
            .get(&handle.name)
            .expect("Failed to resolve asset id");
        let id = match status {
            ResourceStatus::Loaded(id) => id,
            ResourceStatus::Unloaded => panic!("Resource not loaded"),
        };
        let asset = arena.resources.get(id.id).expect("Asset id is invalid");
        asset.metadata.clone()
    }

    fn try_load_resource_metadata(&mut self, path: std::path::PathBuf) -> anyhow::Result<()> {
        let extension = path
            .extension()
            .ok_or(anyhow::format_err!("Could not get asset extension"))?;
        let extension = extension.to_str().ok_or(anyhow::format_err!(
            "Could not convert OsStr extension to str"
        ))?;
        let path = path
            .to_str()
            .ok_or(anyhow::format_err!("Failed to convert OsStr path to str"))?;

        let mut resource_ty = None;
        for (ty, loader) in &self.resource_loaders {
            if loader.accepts_extension_erased(extension) {
                resource_ty = Some(ty);
            }
        }

        if let Some(ty) = resource_ty {
            let mut map = self.get_or_insert_arena_mut_erased(*ty);
            let name = ImmutableString::from(path);
            info!("Added resource metadata for {name}");
            map.resource_statuses.insert(name, ResourceStatus::Unloaded);
        } else {
            info!("No loader found for {path}: extension {extension}");
        }

        Ok(())
    }
}

impl Drop for AssetMap {
    fn drop(&mut self) {
        // Update any pending resources
        self.update();
    }
}

pub(crate) trait AnyResource: Any + Asset {
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

pub(crate) struct RefCountedAsset {
    pub(crate) resource: ResourcePtr,
    pub metadata: AssetMetadata,
    ref_count: u32,
}
impl RefCountedAsset {
    fn new(resource: impl Asset, name: Option<String>, source: AssetSource) -> Self {
        let name = name.unwrap_or_else(|| resource.get_description().to_string());
        Self {
            resource: Arc::new(resource),
            ref_count: 0,
            metadata: AssetMetadata { name, source },
        }
    }

    fn wrap(resource: ResourcePtr, name: Option<String>, source: AssetSource) -> Self {
        let name = name.unwrap_or_else(|| resource.get_description().to_string());
        Self {
            resource,
            ref_count: 0,
            metadata: AssetMetadata { name, source },
        }
    }
}

pub(super) trait ErasedResourceLoader: Send + Sync + 'static {
    fn load_erased(&self, path: &Path) -> anyhow::Result<ResourcePtr>;
    fn accepts_extension_erased(&self, extension: &str) -> bool;
}

impl<T: ResourceLoader> ErasedResourceLoader for T
where
    T: Send + Sync + 'static,
{
    fn load_erased(&self, path: &Path) -> anyhow::Result<ResourcePtr> {
        let inner = self.load(path)?;

        Ok(Arc::new(inner))
    }

    fn accepts_extension_erased(&self, extension: &str) -> bool {
        self.accepts_extension(extension)
    }
}

#[derive(Default)]
pub(crate) struct Resources {
    pub(crate) resources: Arena<RefCountedAsset>,
    pub(crate) resource_statuses: HashMap<ImmutableString, ResourceStatus>,
}

#[derive(Clone, Copy)]
pub enum ResourceStatus {
    Loaded(AssetId),
    Unloaded,
}

#[derive(Default)]
pub(super) struct LoadedResources {
    pub(crate) resources: HashMap<TypeId, RwLock<Resources>>,
}

pub(crate) trait ResourceMapOperation: Send + Sync + 'static {
    fn execute(&self, map: &mut AssetMap);
}

impl<R: Asset> std::fmt::Debug for AssetHandle<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.name.fmt(f)
    }
}

impl<R: Asset> std::fmt::Display for AssetHandle<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_not_null() {
            self.name.fmt(f)
        } else {
            f.write_str("None")
        }
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
            name: ImmutableString::EMPTY,
            operation_sender: None,
        }
    }

    pub fn is_null(&self) -> bool {
        self.name.is_empty()
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
        struct IncResourceMapOperation<R: Asset>(ImmutableString, PhantomData<R>);

        impl<R: Asset> ResourceMapOperation for IncResourceMapOperation<R> {
            fn execute(&self, map: &mut AssetMap) {
                map.increment_resource_ref_count::<R>(self.0.clone())
            }
        }
        Box::new(IncResourceMapOperation::<R>(self.name.clone(), PhantomData))
    }
    fn dec_ref_count_operation(&self) -> Box<dyn ResourceMapOperation> {
        struct DecResourceMapOperation<R: Asset>(ImmutableString, PhantomData<R>);

        impl<R: Asset> ResourceMapOperation for DecResourceMapOperation<R> {
            fn execute(&self, map: &mut AssetMap) {
                map.decrement_resource_ref_count::<R>(self.0.clone())
            }
        }
        Box::new(DecResourceMapOperation::<R>(self.name.clone(), PhantomData))
    }
}

impl<R: Asset> PartialEq for AssetHandle<R> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<R: Asset> Eq for AssetHandle<R> {}

impl<R: Asset> std::hash::Hash for AssetHandle<R> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}
impl<R: Asset> Clone for AssetHandle<R> {
    fn clone(&self) -> Self {
        self.inc_ref_count();
        Self {
            _marker: self._marker,
            name: self.name.clone(),
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
