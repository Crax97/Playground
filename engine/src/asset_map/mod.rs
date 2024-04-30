mod hot_reload_server;

use crossbeam::channel::{Receiver, Sender};
use log::{error, info, warn};
use mgpu::Device;
use serde::{Deserialize, Serialize};
use std::any::{type_name, Any, TypeId};
use std::collections::HashMap;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use thunderdome::{Arena, Index};
use uuid::Uuid;

use crate::immutable_string::ImmutableString;

use hot_reload_server::HotReloadServer;

pub(crate) type OperationSender = Sender<Box<dyn ResourceMapOperation>>;

pub trait Asset: Send + Sync + 'static {
    fn get_description(&self) -> &str;
    fn destroyed(&self, device: &Device);
}

pub struct AssetHandle<R>
where
    R: Asset,
{
    _marker: PhantomData<R>,
    pub(crate) name: ImmutableString,
    pub(crate) ty: Arc<OnceLock<OperationSender>>,
}

impl<R: Asset> AssetHandle<R> {
    pub(crate) fn new(name: ImmutableString, operation_sender: OperationSender) -> Self {
        let lock = OnceLock::new();
        lock.set(operation_sender)
            .expect("Failed to set an empty OnceLock");
        let me = Self {
            _marker: PhantomData,
            name,
            ty: Arc::new(lock),
        };
        me.inc_ref_count();

        me
    }

    fn try_init(&self, operations_sender: &OperationSender) {
        let mut was_init = true;
        self.ty.get_or_init(|| {
            was_init = false;
            operations_sender.clone()
        });
        if !was_init {
            self.inc_ref_count();
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct AssetId {
    pub(crate) id: Index,
}

pub trait AssetLoader: Send + Sync + 'static {
    type LoadedAsset: Asset;

    fn load(&self, path: &Path) -> anyhow::Result<Self::LoadedAsset>;
    fn accepts_extension(&self, extension: &str) -> bool;
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
    pub status: AssetStatus,
}

pub struct AssetMap {
    loaded_resources: RwLock<LoadedResources>,
    gpu: Device,
    operations_receiver: Receiver<Box<dyn ResourceMapOperation>>,
    operations_sender: Sender<Box<dyn ResourceMapOperation>>,

    resource_loaders: HashMap<TypeId, Box<dyn ErasedResourceLoader>>,

    hot_reload_server: Option<HotReloadServer>,
}
impl AssetMap {
    pub fn new(gpu: Device, enable_hot_reload: bool) -> Self {
        let (operations_sender, operations_receiver) = crossbeam::channel::unbounded();

        Self {
            loaded_resources: RwLock::default(),
            operations_receiver,
            operations_sender,
            resource_loaders: HashMap::new(),
            gpu,
            hot_reload_server: enable_hot_reload.then(HotReloadServer::new),
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
        let id = handle.resources.insert(RefCountedAsset::new(resource));
        let name = ImmutableString::from(&unique_name);
        handle.asset_metadata.insert(
            name.clone(),
            AssetMetadata {
                name: unique_name,
                source: AssetSource::Memory,
                status: AssetStatus::Loaded {
                    id: AssetId { id },
                    ref_count: 0,
                },
            },
        );
        AssetHandle::new(name, sender)
    }

    pub fn load<R: Asset>(
        &mut self,
        name: impl Into<ImmutableString>,
    ) -> anyhow::Result<AssetHandle<R>> {
        let name = name.into();
        let contained = {
            let handle = self.get_or_insert_arena_mut::<R>();
            handle.asset_metadata.contains_key(&name)
        };
        if contained {
            let sender = self.operations_sender.clone();
            Ok(AssetHandle::new(name, sender))
        } else {
            self.reload(name.deref())
        }
    }

    pub fn reload<R: Asset>(&mut self, path: impl AsRef<Path>) -> anyhow::Result<AssetHandle<R>> {
        self.reload_inner(path)
    }

    fn reload_inner<R: Asset>(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<AssetHandle<R>, anyhow::Error> {
        let resource_type = TypeId::of::<R>();
        if let Some(loader) = self.resource_loaders.get(&resource_type) {
            let sender = self.operations_sender.clone();
            let loaded_resource = loader.load_erased(path.as_ref())?;

            let (id, name) = {
                let mut handle = self.get_or_insert_arena_mut::<R>();
                let id = handle
                    .resources
                    .insert(RefCountedAsset::wrap(loaded_resource));
                let id = AssetId { id };

                let name = ImmutableString::from(path.as_ref().to_str().unwrap());
                handle.asset_metadata.insert(
                    name.clone(),
                    AssetMetadata {
                        name: name.to_string(),
                        source: AssetSource::Disk,
                        status: AssetStatus::Loaded { id, ref_count: 0 },
                    },
                );
                (id, name)
            };

            if self.hot_reload_server.is_some() {
                self.operations_sender.send(self.watch_path_operation(
                    path.as_ref().to_owned(),
                    TypeId::of::<R>(),
                    id,
                ))?;
            }

            Ok(AssetHandle::new(name, sender))
        } else {
            Err(anyhow::format_err!(
                "No loader for resource type {:?}",
                &TypeId::of::<R>()
            ))
        }
    }

    pub fn iter_ids<R: Asset>(&mut self, mut fun: impl FnMut(ImmutableString)) {
        let map = self.get_or_insert_arena_mut::<R>();
        map.asset_metadata.keys().for_each(|name| fun(name.clone()))
    }

    pub fn upcast_index<R: Asset>(&self, id: ImmutableString) -> AssetHandle<R> {
        let handle = AssetHandle::new(id, self.operations_sender.clone());
        handle.inc_ref_count();
        handle
    }
    pub fn install_resource_loader<L: AssetLoader>(&mut self, loader: L) {
        let old = self
            .resource_loaders
            .insert(TypeId::of::<L::LoadedAsset>(), Box::new(loader));

        if let Some(old) = old {
            panic!(
                "A resource loader for resource type {:?} has already been installed of type {:?}",
                TypeId::of::<L::LoadedAsset>(),
                old.type_id()
            )
        }
    }

    fn get_arena_handle<R: Asset>(&self) -> RwLockReadGuard<Resources> {
        let loaded_resources = self.loaded_resources.read().unwrap();

        let guard = loaded_resources
            .resources
            .get(&TypeId::of::<R>())
            .unwrap()
            .read()
            .unwrap();
        unsafe { std::mem::transmute(guard) }
    }
    fn get_or_insert_arena_mut<R: Asset>(&self) -> RwLockWriteGuard<Resources> {
        self.get_or_insert_arena_mut_erased(TypeId::of::<R>())
    }
    fn get_arena_handle_mut<R: Asset>(&self) -> RwLockWriteGuard<Resources> {
        self.get_arena_handle_mut_erased(TypeId::of::<R>())
    }
    fn get_arena_handle_mut_erased(&self, ty: TypeId) -> RwLockWriteGuard<Resources> {
        let loaded_resources = self.loaded_resources.write().unwrap();

        let guard = loaded_resources
            .resources
            .get(&ty)
            .unwrap()
            .write()
            .unwrap();

        unsafe { std::mem::transmute(guard) }
    }
    fn get_or_insert_arena_mut_erased(&self, ty: TypeId) -> RwLockWriteGuard<Resources> {
        let mut loaded_resources = self.loaded_resources.write().unwrap();

        let guard = loaded_resources
            .resources
            .entry(ty)
            .or_default()
            .write()
            .unwrap();
        unsafe { std::mem::transmute(guard) }
    }

    pub fn get<R: Asset>(&self, id: &AssetHandle<R>) -> &R {
        self.try_get(id).unwrap()
    }

    pub fn get_mut<R: Asset>(&mut self, id: &AssetHandle<R>) -> &mut R {
        self.try_get_mut(id).unwrap()
    }

    pub fn try_get<'a, R: Asset>(&'a self, id: &AssetHandle<R>) -> Option<&'a R> {
        assert!(id.is_not_null(), "Tried to get a null resource");
        id.try_init(&self.operations_sender);
        let arena_handle = self.get_arena_handle::<R>();
        let metadata = arena_handle.asset_metadata.get(&id.name)?;
        match metadata.status {
            AssetStatus::Loaded { id, .. } => {
                let object_arc = arena_handle
                    .resources
                    .get(id.id)
                    .map(|r| r.resource.clone());

                object_arc
                    .map(|a| Arc::as_ptr(&a))
                    .map(|a| unsafe { a.as_ref() }.unwrap().as_any())
                    .map(|obj| obj.downcast_ref::<R>().unwrap())
            }
            AssetStatus::Unloaded => {
                self.reload_inner::<R>(id.name.to_string()).ok()?;
                self.try_get(id)
            }
        }
    }
    pub fn try_get_mut<'a, R: Asset>(&'a self, id: &AssetHandle<R>) -> Option<&'a mut R> {
        assert!(id.is_not_null(), "Tried to get_mut a null resource");
        id.try_init(&self.operations_sender);
        let arena_handle = self.get_arena_handle::<R>();
        let metadata = arena_handle.asset_metadata.get(&id.name)?;
        match metadata.status {
            AssetStatus::Loaded { id, .. } => {
                let object_arc = arena_handle
                    .resources
                    .get(id.id)
                    .map(|r| r.resource.clone());
                object_arc
                    .map(|a| Arc::as_ptr(&a).cast_mut())
                    .map(|a| unsafe { a.as_mut() }.unwrap().as_any_mut())
                    .map(|obj| obj.downcast_mut::<R>().unwrap())
            }
            AssetStatus::Unloaded => {
                self.reload_inner::<R>(id.name.to_string()).ok()?;
                self.try_get_mut(id)
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
            let mut loaded_resources = self.loaded_resources.write().unwrap();
            server.update(&self.resource_loaders, &mut loaded_resources, &self.gpu)?;
        }

        Ok(())
    }

    fn increment_resource_ref_count<R: Asset>(&mut self, name: ImmutableString) {
        let mut arena_handle = self.get_arena_handle_mut::<R>();
        if let Some(metadata) = arena_handle.asset_metadata.get_mut(&name) {
            match &mut metadata.status {
                AssetStatus::Loaded { ref_count, .. } => {
                    (*ref_count) += 1;
                }
                AssetStatus::Unloaded => {
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

        let metadata = arena_handle
            .asset_metadata
            .get_mut(&name)
            .expect("Failed to resolve asset id");

        let (id, ref_count) = match &mut metadata.status {
            AssetStatus::Loaded { id, ref_count } => {
                (*ref_count) -= 1;
                (*id, *ref_count)
            }
            AssetStatus::Unloaded => {
                panic!("Resource is already unloaded")
            }
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
            resource.destroyed(&self.gpu);

            arena_handle.asset_metadata.get_mut(&name).unwrap().status = AssetStatus::Unloaded;

            info!("Deleted resource {name} of type {}", type_name::<R>());
        }
    }

    pub fn asset_metadata<T: crate::asset_map::Asset>(
        &self,
        handle: &AssetHandle<T>,
    ) -> AssetMetadata {
        let arena = self.get_arena_handle::<T>();
        let metadata = arena
            .asset_metadata
            .get(&handle.name)
            .expect("Failed to resolve asset id");
        metadata.clone()
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
            map.asset_metadata.insert(
                name,
                AssetMetadata {
                    name: path.to_string(),
                    source: AssetSource::Disk,
                    status: AssetStatus::Unloaded,
                },
            );
        } else {
            info!("No loader found for {path}: extension {extension}");
        }

        Ok(())
    }

    fn watch_path_operation(
        &self,
        path: PathBuf,
        resource_type: TypeId,
        asset_id: AssetId,
    ) -> Box<dyn ResourceMapOperation> {
        struct WatchPathOperation {
            path: PathBuf,
            resource_type: TypeId,
            asset_id: AssetId,
        }

        impl ResourceMapOperation for WatchPathOperation {
            fn execute(&self, map: &mut AssetMap) {
                let server = map
                    .hot_reload_server
                    .as_mut()
                    .expect("Failed to get hot reload server");
                server
                    .watch(&self.path, self.resource_type, self.asset_id)
                    .expect("CHANGE Me");
            }
        }

        Box::new(WatchPathOperation {
            path,
            resource_type,
            asset_id,
        })
    }
}

impl Drop for AssetMap {
    fn drop(&mut self) {
        let resources = self.loaded_resources.read().unwrap();
        for res_map in resources.resources.values() {
            let mut res_map = res_map.write().unwrap();
            for (_, asset) in res_map.resources.iter_mut() {
                asset.resource.destroyed(&self.gpu)
            }
        }
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
}
impl RefCountedAsset {
    fn new(resource: impl Asset) -> Self {
        Self {
            resource: Arc::new(resource),
        }
    }

    fn wrap(resource: ResourcePtr) -> Self {
        Self { resource }
    }
}

pub(super) trait ErasedResourceLoader: Send + Sync + 'static {
    fn load_erased(&self, path: &Path) -> anyhow::Result<ResourcePtr>;
    fn accepts_extension_erased(&self, extension: &str) -> bool;
}

impl<T: AssetLoader> ErasedResourceLoader for T
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
    pub(crate) asset_metadata: HashMap<ImmutableString, AssetMetadata>,
}

#[derive(Clone, Copy)]
pub enum AssetStatus {
    Loaded { id: AssetId, ref_count: u32 },
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
            ty: Arc::new(OnceLock::new()),
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

        if let Some(sender) = self.ty.get() {
            sender
                .send(self.inc_ref_count_operation())
                .unwrap_or_else(|err| error!("Failed to send increment operation: {err}"));
        }
    }

    fn dec_ref_count(&self) {
        if self.is_null() {
            return;
        }
        if let Some(sender) = self.ty.get() {
            sender
                .send(self.dec_ref_count_operation())
                .unwrap_or_else(|err| error!("Failed to send increment operation: {err}"));
        }
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
            ty: self.ty.clone(),
        }
    }
}

impl<R: Asset> Drop for AssetHandle<R> {
    fn drop(&mut self) {
        self.dec_ref_count()
    }
}

impl<R: Asset> Serialize for AssetHandle<R> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.name)
    }
}

impl<'de, A: Asset> Deserialize<'de> for AssetHandle<A>
where
    Self: Sized,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = ImmutableString::deserialize(deserializer)?;

        Ok(AssetHandle {
            _marker: PhantomData,
            name,
            ty: Arc::new(OnceLock::new()),
        })
    }
}

#[cfg(test)]
mod test {
    use mgpu::{Device, DeviceConfiguration, DeviceFeatures, DevicePreference};

    use super::{Asset, AssetMap};
    use crate::asset_map::{AssetHandle, AssetLoader, AssetStatus};

    struct TestAsset {
        val: u32,
    }

    impl Asset for TestAsset {
        fn get_description(&self) -> &str {
            "test resource"
        }
        fn destroyed(&self, _gpu: &Device) {
            println!("TestAsset destroyed");
        }
    }

    #[test]
    fn asset_create() {
        let gpu = Device::new(DeviceConfiguration {
            app_name: Some("test"),
            features: DeviceFeatures::default(),
            device_preference: Some(DevicePreference::HighPerformance),
            desired_frames_in_flight: 3,
            display_handle: None,
        })
        .unwrap();
        let mut map: AssetMap = AssetMap::new(gpu, false);
        let id: AssetHandle<TestAsset> = map.add(TestAsset { val: 10 }, None::<&str>);

        assert_eq!(map.get(&id).val, 10);
    }

    #[test]
    fn asset_serde() {
        let gpu = Device::new(DeviceConfiguration {
            app_name: Some("test"),
            features: DeviceFeatures::default(),
            device_preference: Some(DevicePreference::HighPerformance),
            desired_frames_in_flight: 3,
            display_handle: None,
        })
        .unwrap();
        struct TestAssetLoader;

        impl AssetLoader for TestAssetLoader {
            type LoadedAsset = TestAsset;

            fn load(&self, path: &std::path::Path) -> anyhow::Result<Self::LoadedAsset> {
                if path.to_str().unwrap() == "a" {
                    Ok(TestAsset { val: 10 })
                } else {
                    Ok(TestAsset { val: 15 })
                }
            }

            fn accepts_extension(&self, _extension: &str) -> bool {
                true
            }
        }
        let mut map: AssetMap = AssetMap::new(gpu, false);
        map.install_resource_loader(TestAssetLoader);
        let id: AssetHandle<TestAsset> = map.load("a").unwrap();

        let st = ron::to_string(&id).unwrap();
        println!("{}", st);

        map.update();

        let id_2 = ron::from_str::<AssetHandle<TestAsset>>(&st).unwrap();

        let meta = map.asset_metadata(&id_2);

        assert_eq!(map.get(&id_2).val, 10);

        map.update();

        assert!(matches!(
            meta.status,
            AssetStatus::Loaded {
                id: _,
                ref_count: 1
            }
        ));
    }
}
