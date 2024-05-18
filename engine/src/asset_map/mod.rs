use std::{
    any::{type_name, TypeId},
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use log::warn;
use serde::{Deserialize, Serialize};

use crate::{assets::texture::Texture, immutable_string::ImmutableString};

use crate::utils::erased_arena::{ErasedArena, Index};

pub trait Asset: 'static {}

pub trait AssetLoader: 'static {
    type LoadedAsset: Asset;

    fn accepts_identifier(&self, identifier: &str) -> bool;
    fn load(&mut self, identifier: &str) -> anyhow::Result<Self::LoadedAsset>;
}
#[derive(Debug)]
pub struct AssetHandle<A: Asset> {
    _phantom_data: PhantomData<A>,
    pub(crate) identifier: ImmutableString,
}

pub struct AssetMap {
    arenas: HashMap<TypeId, ErasedArena>,
    loaded_assets: HashMap<ImmutableString, Index>,
    loaders: HashMap<TypeId, Box<dyn UnsafeAssetLoader>>,
}

struct LoadedAsset<A: Asset> {
    asset: A,
    ref_count: usize,
}

impl AssetMap {
    pub(crate) fn new() -> Self {
        Self {
            arenas: Default::default(),
            loaded_assets: Default::default(),
            loaders: Default::default(),
        }
    }
    pub fn add_loader<L: AssetLoader>(&mut self, loader: L) {
        let unsafe_loader: Box<dyn UnsafeAssetLoader> = Box::new(loader);
        let old_loader = self
            .loaders
            .insert(TypeId::of::<L::LoadedAsset>(), unsafe_loader);
        if old_loader.is_some() {
            warn!(
                "Loader for {} replaced with {}",
                type_name::<L::LoadedAsset>(),
                type_name::<L>()
            );
        }
    }

    pub fn add<A: Asset>(
        &mut self,
        asset: A,
        identifier: impl Into<ImmutableString>,
    ) -> AssetHandle<A> {
        let identifier = identifier.into();
        let asset_ty = TypeId::of::<A>();
        let map = self
            .arenas
            .entry(asset_ty)
            .or_insert_with(ErasedArena::new::<LoadedAsset<A>>);

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

    pub fn load<A: Asset>(&mut self, identifier: &AssetHandle<A>) -> anyhow::Result<()> {
        let asset_ty = TypeId::of::<A>();
        let identifier = &identifier.identifier;
        let map = self
            .arenas
            .entry(asset_ty)
            .or_insert_with(ErasedArena::new::<LoadedAsset<A>>);

        if let Some(existing_entry) = self
            .loaded_assets
            .get(identifier)
            .and_then(|index| map.get_mut::<LoadedAsset<A>>(*index))
        {
            existing_entry.ref_count += 1;
            return Ok(());
        }

        let loader = if let Some(loader) = self.loaders.get_mut(&asset_ty) {
            loader
        } else {
            anyhow::bail!("No loader for type {}", type_name::<A>());
        };
        let map = self
            .arenas
            .entry(asset_ty)
            .or_insert_with(ErasedArena::new::<LoadedAsset<A>>);

        let (index, ptr) = unsafe { map.preallocate_entry() };
        let asset_entry = unsafe { &mut *ptr.cast::<LoadedAsset<A>>() };
        asset_entry.ref_count = 1;

        unsafe { loader.load_asset(identifier, (&mut asset_entry.asset as *mut A).cast::<u8>())? };
        self.loaded_assets.insert(identifier.clone(), index);
        Ok(())
    }

    pub fn get<A: Asset>(&self, handle: &AssetHandle<A>) -> Option<&A> {
        let index = self.loaded_assets.get(&handle.identifier).copied()?;
        self.arenas
            .get(&TypeId::of::<A>())
            .and_then(|map| map.get::<LoadedAsset<A>>(index))
            .map(|entry| &entry.asset)
    }

    pub fn get_mut<A: Asset>(&mut self, handle: &AssetHandle<A>) -> Option<&mut A> {
        let index = self.loaded_assets.get(&handle.identifier).copied()?;
        self.arenas
            .get_mut(&TypeId::of::<A>())
            .and_then(|map| map.get_mut::<LoadedAsset<A>>(index))
            .map(|entry| &mut entry.asset)
    }

    pub fn decrement_reference<A: Asset>(&mut self, handle: impl Into<AssetHandle<A>>) {
        let handle = handle.into();
        let Some(index) = self.loaded_assets.remove(&handle.identifier) else {
            return;
        };
        let Some(arena) = self.arenas.get_mut(&TypeId::of::<A>()) else {
            return;
        };
        let Some(entry) = arena.get_mut::<LoadedAsset<A>>(index) else {
            return;
        };
        entry.ref_count -= 1;

        if entry.ref_count == 0 {
            arena.remove::<LoadedAsset<A>>(index);
        }
    }

    pub fn unload_all(&mut self) {
        for map in self.arenas.values_mut() {
            map.clear()
        }
        self.loaded_assets.clear();
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
    fn accepts_identifier(&self, identifier: &str) -> bool;
    unsafe fn load_asset(
        &mut self,
        identifier: &str,
        backing_memory: *mut u8,
    ) -> anyhow::Result<()>;
}

unsafe impl<L: AssetLoader> UnsafeAssetLoader for L {
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
        self.identifier.partial_cmp(&other.identifier)
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
#[cfg(test)]
mod tests {

    use crate::asset_map::{Asset, AssetHandle, AssetLoader, AssetMap};

    #[test]
    fn basic_operations() {
        struct StringAsset {
            content: String,
        }

        impl Asset for StringAsset {}

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

        let mut asset_map = AssetMap::new();
        let string_one = AssetHandle::<StringAsset>::new("Hello");
        let string_two = AssetHandle::<StringAsset>::new("World");
        asset_map.add_loader(StringAssetLoader);
        asset_map.load::<StringAsset>(&string_one).unwrap();
        asset_map.load::<StringAsset>(&string_two).unwrap();
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

        asset_map.unload_all();
        assert!(asset_map.get::<StringAsset>(&string_two).is_none());
    }
}
