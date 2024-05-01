use std::{
    any::{type_name, TypeId},
    collections::HashMap,
    marker::PhantomData,
};

use log::warn;

use crate::immutable_string::ImmutableString;

use self::erased_arena::{ErasedArena, Index};

mod erased_arena;

pub trait Asset: 'static {}

pub trait AssetLoader: 'static {
    type LoadedAsset: Asset;

    fn accepts_identifier(&self, identifier: &str) -> bool;
    fn load(&mut self, identifier: &str) -> anyhow::Result<Self::LoadedAsset>;
}
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetHandle<A: Asset> {
    _phantom_data: PhantomData<A>,
    identifier: ImmutableString,
}

#[derive(Default)]
pub struct AssetMap {
    arenas: HashMap<TypeId, ErasedArena>,
    loaded_assets: HashMap<ImmutableString, Index>,
    loaders: HashMap<TypeId, Box<dyn UnsafeAssetLoader>>,
}

impl AssetMap {
    pub fn new() -> Self {
        Self::default()
    }
    fn add_loader<L: AssetLoader>(&mut self, loader: L) {
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

    fn load<A: Asset>(&mut self, identifier: &str) -> anyhow::Result<AssetHandle<A>> {
        let asset_ty = TypeId::of::<A>();
        let loader = if let Some(loader) = self.loaders.get_mut(&asset_ty) {
            loader
        } else {
            anyhow::bail!("No loader for type {}", type_name::<A>());
        };
        let map = self
            .arenas
            .entry(asset_ty)
            .or_insert_with(|| loader.create_arena());

        let (index, ptr) = unsafe { map.preallocate_entry() };
        unsafe { loader.load_asset(identifier, ptr)? };
        let identifier = ImmutableString::new_dynamic(identifier);
        self.loaded_assets.insert(identifier.clone(), index);
        Ok(AssetHandle {
            _phantom_data: PhantomData,
            identifier,
        })
    }

    fn get<A: Asset>(&self, handle: &AssetHandle<A>) -> Option<&A> {
        let index = self.loaded_assets.get(&handle.identifier).copied()?;
        self.arenas
            .get(&TypeId::of::<A>())
            .and_then(|map| map.get(index))
    }

    fn get_mut<A: Asset>(&mut self, handle: &AssetHandle<A>) -> Option<&mut A> {
        let index = self.loaded_assets.get(&handle.identifier).copied()?;
        self.arenas
            .get_mut(&TypeId::of::<A>())
            .and_then(|map| map.get_mut(index))
    }

    fn unload<A: Asset>(&mut self, handle: &AssetHandle<A>) {
        let index = if let Some(index) = self.loaded_assets.get(&handle.identifier).copied() {
            index
        } else {
            return;
        };
        self.arenas
            .get_mut(&TypeId::of::<A>())
            .and_then(|map| map.remove::<A>(index));
    }

    fn unload_all(&mut self) {
        for map in self.arenas.values_mut() {
            map.clear()
        }
        self.loaded_assets.clear();
    }
}

/// # Safety
/// This trait is safely implemented for all the types that implement AssetLoader
unsafe trait UnsafeAssetLoader: 'static {
    fn accepts_identifier(&self, identifier: &str) -> bool;
    unsafe fn load_asset(
        &mut self,
        identifier: &str,
        backing_memory: *mut u8,
    ) -> anyhow::Result<()>;
    fn create_arena(&self) -> ErasedArena;
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

    fn create_arena(&self) -> ErasedArena {
        ErasedArena::new::<L::LoadedAsset>()
    }

    fn accepts_identifier(&self, identifier: &str) -> bool {
        <Self as AssetLoader>::accepts_identifier(self, identifier)
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

#[cfg(test)]
mod tests {

    use crate::asset_map::{Asset, AssetLoader, AssetMap};

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
        asset_map.add_loader(StringAssetLoader);
        let string_one = asset_map.load::<StringAsset>("Hello").unwrap();
        let string_two = asset_map.load::<StringAsset>("World").unwrap();
        assert_eq!(
            asset_map.get(&string_one).unwrap().content.as_str(),
            "Hello"
        );
        assert_eq!(
            asset_map.get(&string_two).unwrap().content.as_str(),
            "World"
        );
        asset_map.unload(&string_one);
        assert!(asset_map.get(&string_one).is_none());

        asset_map.get_mut(&string_two).unwrap().content = "Pippo".to_string();
        assert_eq!(
            asset_map.get(&string_two).unwrap().content.as_str(),
            "Pippo"
        );

        asset_map.unload_all();
        assert!(asset_map.get(&string_two).is_none());
    }
}
