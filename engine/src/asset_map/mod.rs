use std::{
    any::{type_name, TypeId},
    collections::HashMap,
    marker::PhantomData,
    ptr::NonNull,
};

use log::{info, warn};
use mgpu::Device;
use serde::{Deserialize, Serialize};

use crate::immutable_string::ImmutableString;

use crate::utils::erased_arena::{ErasedArena, Index};

pub struct AssetMap {
    device: Device,
    registrations: HashMap<TypeId, AssetRegistration>,
    loaded_assets: HashMap<ImmutableString, Index>,
}

pub trait Asset: 'static {
    fn identifier() -> &'static str;
    fn dispose(&self, device: &Device) {
        let _ = device;
    }
}

pub trait AssetLoader: 'static {
    type LoadedAsset: Asset;

    fn accepts_identifier(&self, identifier: &str) -> bool;
    fn load(&mut self, identifier: &str) -> anyhow::Result<Self::LoadedAsset>;
}

pub struct AssetHandle<A: Asset> {
    _phantom_data: PhantomData<A>,
    pub(crate) identifier: ImmutableString,
}

struct LoadedAsset<A: Asset> {
    asset: A,
    ref_count: usize,
}

struct AssetRegistration {
    identifier: &'static str,
    dispose_fn: unsafe fn(NonNull<u8>, device: &Device),
    loader: Box<dyn UnsafeAssetLoader>,
    arena: ErasedArena,
}

impl AssetMap {
    pub(crate) fn new(device: Device) -> Self {
        Self {
            device,
            registrations: Default::default(),
            loaded_assets: Default::default(),
        }
    }

    pub fn register<A: Asset>(&mut self, loader: impl Into<Box<dyn UnsafeAssetLoader>>) {
        let loader = loader.into();
        assert!(loader.loaded_asset_type_id() == TypeId::of::<A>());
        let old_registration = self.registrations.insert(
            TypeId::of::<A>(),
            AssetRegistration {
                identifier: A::identifier(),
                dispose_fn: Self::dispose_fn::<A>,
                arena: ErasedArena::new::<LoadedAsset<A>>(),
                loader,
            },
        );

        debug_assert!(old_registration.is_none());
    }

    pub fn add<A: Asset>(
        &mut self,
        asset: A,
        identifier: impl Into<ImmutableString>,
    ) -> AssetHandle<A> {
        let identifier = identifier.into();
        let asset_ty = TypeId::of::<A>();
        let map = self
            .registrations
            .get_mut(&asset_ty)
            .unwrap_or_else(|| panic!("Asset type {} was not registered!", type_name::<A>()));
        let map = &mut map.arena;

        let index = map.add(LoadedAsset {
            asset,
            ref_count: 1,
        });
        let old = self.loaded_assets.insert(identifier.clone(), index);
        debug_assert!(
            old.is_none(),
            "Another asset with identifier '{}' was already defined! identifiers must be unique!",
            &identifier
        );
        AssetHandle {
            _phantom_data: PhantomData,
            identifier,
        }
    }

    pub fn increment_reference<A: Asset>(
        &mut self,
        identifier: &AssetHandle<A>,
    ) -> anyhow::Result<()> {
        let asset_ty = TypeId::of::<A>();
        let identifier = &identifier.identifier;
        let registration = self
            .registrations
            .get_mut(&asset_ty)
            .unwrap_or_else(|| panic!("Asset type {} was not registered!", type_name::<A>()));
        let map = &mut registration.arena;

        if let Some(existing_entry) = self
            .loaded_assets
            .get(identifier)
            .and_then(|index| map.get_mut::<LoadedAsset<A>>(*index))
        {
            existing_entry.ref_count += 1;
            return Ok(());
        }

        info!("Asset {:?} was not loaded, trying to load it", identifier);
        let loader = &mut registration.loader;

        let (index, ptr) = unsafe { map.preallocate_entry() };
        let asset_entry = unsafe { &mut *ptr.cast::<LoadedAsset<A>>() };
        asset_entry.ref_count = 1;

        unsafe { loader.load_asset(identifier, (&mut asset_entry.asset as *mut A).cast::<u8>())? };
        self.loaded_assets.insert(identifier.clone(), index);
        Ok(())
    }

    pub fn get<A: Asset>(&self, handle: &AssetHandle<A>) -> Option<&A> {
        let index = self.loaded_assets.get(&handle.identifier).copied()?;
        self.registrations
            .get(&TypeId::of::<A>())
            .and_then(|map| map.arena.get::<LoadedAsset<A>>(index))
            .map(|entry| &entry.asset)
    }

    pub fn get_mut<A: Asset>(&mut self, handle: &AssetHandle<A>) -> Option<&mut A> {
        let index = self.loaded_assets.get(&handle.identifier).copied()?;
        self.registrations
            .get_mut(&TypeId::of::<A>())
            .and_then(|map| map.arena.get_mut::<LoadedAsset<A>>(index))
            .map(|entry| &mut entry.asset)
    }

    pub fn decrement_reference<A: Asset>(&mut self, handle: impl Into<AssetHandle<A>>) {
        let handle = handle.into();
        let Some(index) = self.loaded_assets.get(&handle.identifier).copied() else {
            return;
        };
        let Some(registration) = self.registrations.get_mut(&TypeId::of::<A>()) else {
            return;
        };
        let Some(entry) = registration.arena.get_mut::<LoadedAsset<A>>(index) else {
            return;
        };
        entry.ref_count -= 1;

        if entry.ref_count == 0 {
            let removed_asset = registration.arena.remove::<LoadedAsset<A>>(index).unwrap();
            info!("Released asset {handle:?}");
            removed_asset.asset.dispose(&self.device);
            self.loaded_assets.remove(handle.identifier());
        }
    }

    pub fn unload_all(&mut self, device: &Device) {
        for registration in self.registrations.values_mut() {
            unsafe {
                for entry in registration.arena.iter_ptr() {
                    (registration.dispose_fn)(entry, device)
                }
            }
        }
        self.loaded_assets.clear();
    }

    unsafe fn dispose_fn<A: Asset>(ptr: NonNull<u8>, device: &Device) {
        let asset_ref = ptr.cast::<A>().as_ref();

        asset_ref.dispose(device)
    }
}

impl<A: Asset, S: AsRef<str>> From<S> for AssetHandle<A> {
    fn from(value: S) -> Self {
        Self {
            _phantom_data: PhantomData,
            identifier: ImmutableString::new_dynamic(value.as_ref()),
        }
    }
}

/// # Safety
/// This trait is safely implemented for all the types that implement AssetLoader
/// It is marked as unsafe because [`UnsafeAssteLoader::load_asset`] directly writes to the payload
/// pointer of an [`ErasedArena`] entry payload, which is an [`Option<T>`]
unsafe trait UnsafeAssetLoader: 'static {
    fn loaded_asset_type_id(&self) -> TypeId;
    fn accepts_identifier(&self, identifier: &str) -> bool;
    unsafe fn load_asset(
        &mut self,
        identifier: &str,
        backing_memory: *mut u8,
    ) -> anyhow::Result<()>;
}

unsafe impl<L: AssetLoader> UnsafeAssetLoader for L {
    fn loaded_asset_type_id(&self) -> TypeId {
        TypeId::of::<L::LoadedAsset>()
    }
    unsafe fn load_asset(
        &mut self,
        identifier: &str,
        backing_memory: *mut u8,
    ) -> anyhow::Result<()> {
        let asset = <Self as AssetLoader>::load(self, identifier)?;
        unsafe {
            // The backing memory points at the payload of the Entry<T> for this item
            // which is an Option<A>
            backing_memory
                .cast::<Option<L::LoadedAsset>>()
                .write(Some(asset));
        }
        Ok(())
    }

    fn accepts_identifier(&self, identifier: &str) -> bool {
        <Self as AssetLoader>::accepts_identifier(self, identifier)
    }
}
impl<A: Asset> std::fmt::Debug for AssetHandle<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "AssetHandle<{}>(\"{}\")",
            type_name::<A>(),
            &self.identifier
        ))
    }
}

impl<A: Asset> Eq for AssetHandle<A> {}

impl<A: Asset> PartialEq for AssetHandle<A> {
    fn eq(&self, other: &Self) -> bool {
        self._phantom_data == other._phantom_data && self.identifier == other.identifier
    }
}

impl<A: Asset> Ord for AssetHandle<A> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.identifier.cmp(&other.identifier)
    }
}

impl<A: Asset> PartialOrd for AssetHandle<A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Asset> std::hash::Hash for AssetHandle<A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self._phantom_data.hash(state);
        self.identifier.hash(state);
    }
}

impl<A: Asset> Clone for AssetHandle<A> {
    fn clone(&self) -> Self {
        Self {
            _phantom_data: PhantomData,
            identifier: self.identifier.clone(),
        }
    }
}

impl<A: Asset> Serialize for AssetHandle<A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.identifier.serialize(serializer)
    }
}

impl<'de, A: Asset> Deserialize<'de> for AssetHandle<A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            _phantom_data: PhantomData,
            identifier: ImmutableString::deserialize(deserializer)?,
        })
    }
}

impl<A: Asset> AssetHandle<A> {
    pub fn new(identifier: impl Into<ImmutableString>) -> Self {
        Self {
            _phantom_data: PhantomData,
            identifier: identifier.into(),
        }
    }

    pub const fn new_const(identifier: ImmutableString) -> Self {
        Self {
            _phantom_data: PhantomData,
            identifier,
        }
    }

    pub fn identifier(&self) -> &ImmutableString {
        &self.identifier
    }
}

impl<L: AssetLoader> From<L> for Box<dyn UnsafeAssetLoader> {
    fn from(value: L) -> Self {
        Box::new(value)
    }
}

#[cfg(test)]
mod tests {

    use mgpu::Device;

    use crate::asset_map::{Asset, AssetHandle, AssetLoader, AssetMap};

    #[test]
    fn basic_operations() {
        struct StringAsset {
            content: String,
        }

        impl Asset for StringAsset {
            fn dispose(&self, _device: &mgpu::Device) {
                println!("Destroyed string asset {}", self.content)
            }
        }

        struct StringAssetLoader;
        impl AssetLoader for StringAssetLoader {
            type LoadedAsset = StringAsset;

            fn accepts_identifier(&self, _identifier: &str) -> bool {
                true
            }

            fn load(&mut self, identifier: &str) -> anyhow::Result<Self::LoadedAsset> {
                Ok(StringAsset {
                    content: identifier.to_string(),
                })
            }
        }

        let mut asset_map = AssetMap::new(Device::dummy());
        let string_one = AssetHandle::<StringAsset>::new("Hello");
        let string_two = AssetHandle::<StringAsset>::new("World");
        asset_map.register::<StringAsset>(StringAssetLoader);
        asset_map
            .increment_reference::<StringAsset>(&string_one)
            .unwrap();
        asset_map
            .increment_reference::<StringAsset>(&string_two)
            .unwrap();
        assert_eq!(
            asset_map
                .get::<StringAsset>(&string_one)
                .unwrap()
                .content
                .as_str(),
            "Hello"
        );
        assert_eq!(
            asset_map
                .get::<StringAsset>(&string_two)
                .unwrap()
                .content
                .as_str(),
            "World"
        );
        asset_map.decrement_reference::<StringAsset>("Hello");
        assert!(asset_map.get::<StringAsset>(&string_one).is_none());

        asset_map
            .get_mut::<StringAsset>(&string_two)
            .unwrap()
            .content = "Pippo".to_string();
        assert_eq!(
            asset_map
                .get::<StringAsset>(&string_two)
                .unwrap()
                .content
                .as_str(),
            "Pippo"
        );
    }
}
