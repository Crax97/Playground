use std::{
    any::{type_name, TypeId},
    collections::HashMap,
    marker::PhantomData,
    ptr::NonNull,
};

use log::info;
use mgpu::Device;
use serde::{Deserialize, Serialize};

use crate::{
    immutable_string::ImmutableString, sampler_allocator::SamplerAllocator,
    shader_cache::ShaderCache,
};

use crate::utils::erased_arena::{ErasedArena, Index};

pub struct AssetMap {
    device: Device,
    shader_cache: ShaderCache,
    sampler_allocator: SamplerAllocator,
    registrations: HashMap<TypeId, AssetRegistration>,
    loaded_assets: HashMap<ImmutableString, Index>,
}

pub struct LoadContext<'a> {
    pub device: &'a Device,
    pub shader_cache: &'a ShaderCache,
    pub sampler_allocator: &'a SamplerAllocator,
}

pub trait Asset: 'static {
    type Metadata: 'static + Serialize + for<'a> Deserialize<'a>;
    fn asset_type_name() -> &'static str;
    fn load(metadata: Self::Metadata, context: &LoadContext) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn dispose(&self, device: &Device) {
        let _ = device;
    }
}

pub struct AssetHandle<A: Asset> {
    _phantom_data: PhantomData<A>,
    pub(crate) identifier: ImmutableString,
}

struct LoadedAsset<A: Asset> {
    asset: A,
    ref_count: usize,
}

#[derive(Serialize, Deserialize)]
struct AssetSpecifier<A: Asset> {
    asset_type: String,
    metadata: A::Metadata,
}

struct AssetRegistration {
    asset_type_name: &'static str,
    dispose_fn: unsafe fn(NonNull<u8>, device: &Device),
    load_fn: unsafe fn(&str, &mut ErasedArena, &LoadContext) -> Index,
    arena: ErasedArena,
}

impl AssetMap {
    pub(crate) fn new(
        device: Device,
        shader_cache: ShaderCache,
        sampler_allocator: SamplerAllocator,
    ) -> Self {
        Self {
            device,
            registrations: Default::default(),
            loaded_assets: Default::default(),
            sampler_allocator,
            shader_cache,
        }
    }

    pub fn register<A: Asset>(&mut self) {
        let old_registration = self.registrations.insert(
            TypeId::of::<A>(),
            AssetRegistration {
                asset_type_name: A::asset_type_name(),
                dispose_fn: Self::dispose_fn::<A>,
                load_fn: Self::load_fn::<A>,
                arena: ErasedArena::new::<LoadedAsset<A>>(),
            },
        );

        debug_assert!(old_registration.is_none());
    }

    pub fn preload(&mut self, identifier: &str) {
        let toml_specifier = std::fs::read_to_string(identifier)
            .unwrap_or_else(|e| panic!("Failed to load asset {identifier}! {e:?}"));
        let toml = toml::from_str::<toml::Table>(&toml_specifier)
            .unwrap_or_else(|e| panic!("Failed to parse {identifier} as toml! {e:?}"));
        let asset_type: &str = toml["asset_type"]
            .as_str()
            .expect("Asset has no 'asset_type' field!");

        for registration in self.registrations.values_mut() {
            if asset_type == registration.asset_type_name {
                let index = unsafe {
                    (registration.load_fn)(
                        identifier,
                        &mut registration.arena,
                        &LoadContext {
                            device: &self.device,
                            shader_cache: &self.shader_cache,
                            sampler_allocator: &self.sampler_allocator,
                        },
                    )
                };

                self.loaded_assets
                    .insert(ImmutableString::new_dynamic(identifier), index);

                return;
            }
        }

        panic!("Asset of type '{}' was not registered!", asset_type);
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
        self.preload(identifier);
        Ok(())
    }

    // Gets a reference to an asset, loading it if it is not loaded
    pub fn load<A: Asset>(&mut self, handle: &AssetHandle<A>) -> &A {
        if !self.loaded_assets.contains_key(&handle.identifier) {
            self.preload(&handle.identifier);
        }
        self.get(handle)
    }

    // Gets a mutable reference to an asset, loading it if it is not loaded
    pub fn load_mut<A: Asset>(&mut self, handle: &AssetHandle<A>) -> &mut A {
        if !self.loaded_assets.contains_key(&handle.identifier) {
            self.preload(&handle.identifier);
        }
        self.get_mut(handle)
    }

    /// Gets a reference to the given asset, panicking if it is not loaded
    pub fn get<A: Asset>(&self, handle: &AssetHandle<A>) -> &A {
        let index = self
            .loaded_assets
            .get(&handle.identifier)
            .copied()
            .expect("Asset is not loaded! Use load if you're unsure wether the asset might be loaded or not");
        self.registrations
            .get(&TypeId::of::<A>())
            .and_then(|map| map.arena.get::<LoadedAsset<A>>(index))
            .map(|entry| &entry.asset)
            .unwrap()
    }

    /// Gets a mutable reference to the given asset, panicking if it is not loaded
    pub fn get_mut<A: Asset>(&mut self, handle: &AssetHandle<A>) -> &mut A {
        let index = self
            .loaded_assets
            .get(&handle.identifier)
            .copied()
            .expect("Asset is not loaded! Use load if you're unsure wether the asset might be loaded or not");
        self.registrations
            .get_mut(&TypeId::of::<A>())
            .and_then(|map| map.arena.get_mut::<LoadedAsset<A>>(index))
            .map(|entry| &mut entry.asset)
            .unwrap()
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

    // Loads dynamically an asset into an erased arena by first reading it's specifier
    // then creating the asset through the specifier's metadata
    unsafe fn load_fn<A: Asset>(
        path: &str,
        erased_arena: &mut ErasedArena,
        context: &LoadContext,
    ) -> Index {
        let (index, option_ptr) = erased_arena.preallocate_entry();
        let file_content = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!(
                "Failed to read specifier file for {}: {e:?}",
                A::asset_type_name()
            )
        });
        let specifier = toml::from_str::<AssetSpecifier<A>>(&file_content).unwrap_or_else(|e| {
            panic!(
                "Failed to parse specifier for asset {}: {e:?}",
                A::asset_type_name()
            )
        });

        let asset = A::load(specifier.metadata, context).unwrap_or_else(|e| {
            panic!(
                "Failed to load asset {} from metadata: {e:?}",
                A::asset_type_name()
            )
        });

        option_ptr
            .cast::<Option<LoadedAsset<A>>>()
            .write(Some(LoadedAsset {
                asset,
                ref_count: 1,
            }));
        index
    }

    pub fn shader_cache(&self) -> &ShaderCache {
        &self.shader_cache
    }
}

impl<A: Asset> AssetSpecifier<A> {
    fn new(metadata: A::Metadata) -> Self {
        Self {
            asset_type: A::asset_type_name().to_owned(),
            metadata,
        }
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

#[cfg(test)]
mod tests {
    use mgpu::Device;
    use serde::{Deserialize, Serialize};

    use crate::{
        asset_map::{Asset, AssetHandle, AssetMap, AssetSpecifier, LoadContext},
        assets::{
            texture::{Texture, TextureSamplerConfiguration},
            TextureMetadata,
        },
        sampler_allocator::SamplerAllocator,
        shader_cache::ShaderCache,
    };

    #[test]
    fn basic_operations() {
        struct StringAsset {
            content: String,
        }

        impl Asset for StringAsset {
            type Metadata = String;
            fn dispose(&self, _device: &mgpu::Device) {
                println!("Destroyed string asset {}", self.content)
            }

            fn asset_type_name() -> &'static str {
                "StringAsset"
            }

            fn load(metadata: Self::Metadata, _ctx: &LoadContext) -> anyhow::Result<Self>
            where
                Self: Sized,
            {
                Ok(StringAsset { content: metadata })
            }
        }

        let mut asset_map = AssetMap::new(
            Device::dummy(),
            ShaderCache::new(),
            SamplerAllocator::default(),
        );
        let string_one = AssetHandle::<StringAsset>::new("Hello");
        let string_two = AssetHandle::<StringAsset>::new("World");
        asset_map.register::<StringAsset>();
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

    #[test]
    fn specifier() {
        struct CharacterStats {
            health: u32,
            attack: u32,
            defense: u32,
        }

        #[derive(Serialize, Deserialize)]
        struct CharacterStatsMetadata {
            health: u32,
            attack: u32,
            defense: u32,
        }

        impl Asset for CharacterStats {
            type Metadata = CharacterStatsMetadata;

            fn asset_type_name() -> &'static str {
                "CharacterStats"
            }

            fn load(metadata: Self::Metadata, _ctx: &LoadContext) -> anyhow::Result<Self>
            where
                Self: Sized,
            {
                Ok(Self {
                    health: metadata.health,
                    attack: metadata.attack,
                    defense: metadata.defense,
                })
            }
        }

        let metadata = CharacterStatsMetadata {
            health: 10,
            attack: 15,
            defense: 5,
        };
        let specifier = AssetSpecifier::<CharacterStats>::new(metadata);
        let serialized_specifier = toml::to_string(&specifier).unwrap();
        println!("Specifier\n{serialized_specifier}");
        let specifier =
            toml::from_str::<AssetSpecifier<CharacterStats>>(&serialized_specifier).unwrap();

        // The LoadContext is not used, we should be fine
        let asset =
            CharacterStats::load(specifier.metadata, unsafe { std::mem::zeroed() }).unwrap();
        assert_eq!(asset.health, 10);
        assert_eq!(asset.attack, 15);
        assert_eq!(asset.defense, 5);
    }

    #[test]
    fn poo() {
        let specifier = AssetSpecifier::<Texture>::new(TextureMetadata {
            source_path: "poo".into(),
            sampler_configuration: TextureSamplerConfiguration::default(),
        });
        println!("{}", toml::to_string(&specifier).unwrap());
    }
}
