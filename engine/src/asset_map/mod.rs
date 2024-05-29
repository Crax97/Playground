use std::{
    any::{type_name, TypeId},
    collections::HashMap,
    marker::PhantomData,
    path::PathBuf,
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
    known_assets: HashMap<ImmutableString, AssetInfo>,
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
    pub(crate) identifier: Option<ImmutableString>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum AssetStatus {
    Loaded(Index),
    Unloaded,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum AssetStorage {
    Dynamic,
    Disk,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct AssetInfo {
    status: AssetStatus,
    asset_storage: AssetStorage,
    asset_ty: TypeId,
}

impl AssetStatus {
    fn assume_loaded(&self) -> Index {
        match self {
            AssetStatus::Loaded(index) => *index,
            AssetStatus::Unloaded => panic!("Asset wasn't loaded! Call load/load_mut if you're unsure wether the asset might be loaded or not"),
        }
    }
}

struct LoadedAsset<A: Asset> {
    asset: A,
    ref_count: usize,
    identifier: ImmutableString,
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
            known_assets: Default::default(),
            sampler_allocator,
            shader_cache,
        }
    }

    pub fn discover_assets(&mut self, base_path: impl Into<PathBuf>) {
        let base_path = base_path.into();

        for entry in walkdir::WalkDir::new(base_path)
            .into_iter()
            .filter_map(|s| s.ok())
            .filter(|s| {
                s.file_type().is_file() && s.path().extension().is_some_and(|e| e == "toml")
            })
        {
            let entry = entry.path();
            let toml_specifier = std::fs::read_to_string(entry)
                .unwrap_or_else(|e| panic!("Failed to load asset {entry:?}! {e:?}"));
            let toml = toml::from_str::<toml::Table>(&toml_specifier)
                .unwrap_or_else(|e| panic!("Failed to parse {entry:?} as toml! {e:?}"));
            let asset_type: &str = toml["asset_type"]
                .as_str()
                .expect("Asset has no 'asset_type' field!");
            let mut found_map = false;
            for (ty, registration) in &mut self.registrations {
                if asset_type == registration.asset_type_name {
                    let entry_name = entry.with_extension("");
                    let entry_name = entry_name.to_str().unwrap();
                    let entry_name = entry_name.replace('\\', "/");
                    self.known_assets.insert(
                        ImmutableString::new_dynamic(entry_name),
                        AssetInfo {
                            status: AssetStatus::Unloaded,
                            asset_storage: AssetStorage::Disk,
                            asset_ty: *ty,
                        },
                    );
                    found_map = true;
                    break;
                }
            }

            if !found_map {
                panic!("Asset of type '{}' was not registered!", asset_type);
            }
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
        let info: &mut AssetInfo = self
            .known_assets
            .get_mut(&ImmutableString::new_dynamic(identifier))
            .expect("Asset isn't known!");
        let ty = info.asset_ty;
        let registration = self.registrations.get_mut(&ty).unwrap();
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
        assert!(
            info.asset_storage == AssetStorage::Disk,
            "Only disk assets can be preloaded!"
        );

        debug_assert!(info.status == AssetStatus::Unloaded);
        info.status = AssetStatus::Loaded(index);
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
            identifier: identifier.clone(),
        });

        let old = self.known_assets.insert(
            identifier.clone(),
            AssetInfo {
                status: AssetStatus::Loaded(index),
                asset_storage: AssetStorage::Dynamic,
                asset_ty,
            },
        );
        debug_assert!(
            old.is_none(),
            "Another asset with identifier '{}' was already defined! identifiers must be unique!",
            &identifier
        );
        AssetHandle {
            _phantom_data: PhantomData,
            identifier: Some(identifier),
        }
    }

    pub fn increment_reference<A: Asset>(
        &mut self,
        identifier: &AssetHandle<A>,
    ) -> anyhow::Result<()> {
        let asset_ty = TypeId::of::<A>();
        let identifier = identifier
            .identifier
            .as_ref()
            .expect("AssetHandle is null!");
        let registration = self
            .registrations
            .get_mut(&asset_ty)
            .unwrap_or_else(|| panic!("Asset type {} was not registered!", type_name::<A>()));
        let map = &mut registration.arena;
        let asset_info = self
            .known_assets
            .get(identifier)
            .copied()
            .expect("Asset is unknown!");

        match asset_info.status {
            AssetStatus::Loaded(index) => {
                map.get_mut::<LoadedAsset<A>>(index).unwrap().ref_count += 1;
            }
            AssetStatus::Unloaded => {
                info!("Asset {:?} was not loaded, trying to load it", identifier);
                self.preload(identifier)
            }
        }
        Ok(())
    }

    /// Gets an iterator of all the assets of a certain type.
    /// Panics if the asset type was not registered
    pub fn iter_ids<A: Asset>(&self) -> impl Iterator<Item = &ImmutableString> {
        let registration = self
            .registrations
            .get(&TypeId::of::<A>())
            .unwrap_or_else(|| {
                panic!(
                    "Asset of type '{}' was not registered!",
                    A::asset_type_name(),
                )
            });
        registration
            .arena
            .iter::<LoadedAsset<A>>()
            .map(|loaded_asset| &loaded_asset.identifier)
    }

    // Gets a reference to an asset, loading it if it is not loaded
    pub fn load<A: Asset>(&mut self, handle: &AssetHandle<A>) -> &A {
        let identifier = handle.identifier.as_ref().expect("AssetHandle is null!");
        if self
            .known_assets
            .get(identifier)
            .copied()
            .expect("Asset isn't known!")
            .status
            == AssetStatus::Unloaded
        {
            self.preload(identifier);
        }
        self.get(handle).unwrap()
    }

    // Gets a mutable reference to an asset, loading it if it is not loaded
    pub fn load_mut<A: Asset>(&mut self, handle: &AssetHandle<A>) -> &mut A {
        let identifier = handle.identifier.as_ref().expect("AssetHandle is null!");
        if self
            .known_assets
            .get(identifier)
            .copied()
            .expect("Asset isn't known")
            .status
            == AssetStatus::Unloaded
        {
            self.preload(identifier);
        }
        self.get_mut(handle).unwrap()
    }

    /// Gets a reference to the given asset, panicking if it is not loaded
    /// Returns None only when the handle is null
    pub fn get<A: Asset>(&self, handle: &AssetHandle<A>) -> Option<&A> {
        let Some(identifier) = handle.identifier.as_ref() else {
            return None;
        };
        let index = self
            .known_assets
            .get(identifier)
            .copied()
            .unwrap_or_else(|| panic!("Asset '{}' isn't known!", identifier))
            .status
            .assume_loaded();
        Some(
            self.registrations
                .get(&TypeId::of::<A>())
                .and_then(|map| map.arena.get::<LoadedAsset<A>>(index))
                .map(|entry| &entry.asset)
                .unwrap(),
        )
    }

    /// Gets a mutable reference to the given asset, panicking if it is not loaded
    /// Returns None only when the handle is null
    pub fn get_mut<A: Asset>(&mut self, handle: &AssetHandle<A>) -> Option<&mut A> {
        let Some(identifier) = handle.identifier.as_ref() else {
            return None;
        };
        let index = self
            .known_assets
            .get(identifier)
            .copied()
            .unwrap_or_else(|| panic!("Asset '{}' isn't known!", identifier))
            .status
            .assume_loaded();
        Some(
            self.registrations
                .get_mut(&TypeId::of::<A>())
                .and_then(|map| map.arena.get_mut::<LoadedAsset<A>>(index))
                .map(|entry| &mut entry.asset)
                .unwrap(),
        )
    }

    pub fn decrement_reference<A: Asset>(&mut self, handle: impl Into<AssetHandle<A>>) {
        let handle = handle.into();
        let asset_ty = TypeId::of::<A>();
        let identifier = handle.identifier.as_ref().expect("AssetHandle is null!");
        let registration = self
            .registrations
            .get_mut(&asset_ty)
            .unwrap_or_else(|| panic!("Asset type {} was not registered!", type_name::<A>()));
        let map = &mut registration.arena;
        let asset_entry = self
            .known_assets
            .get_mut(identifier)
            .expect("Asset is unknown!");

        let (ref_count, index) = match asset_entry.status {
            AssetStatus::Loaded(index) => {
                let count = &mut map.get_mut::<LoadedAsset<A>>(index).unwrap().ref_count;
                *count -= 1;
                (*count, index)
            }
            AssetStatus::Unloaded => {
                panic!("Asset {:?} was not loaded", identifier);
            }
        };

        if ref_count == 0 {
            let removed_asset = registration.arena.remove::<LoadedAsset<A>>(index).unwrap();
            info!("Released asset {handle:?}");
            removed_asset.asset.dispose(&self.device);
            asset_entry.status = AssetStatus::Unloaded;
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
        self.known_assets.clear();
    }

    unsafe fn dispose_fn<A: Asset>(ptr: NonNull<u8>, device: &Device) {
        let asset_ref = ptr.cast::<LoadedAsset<A>>().as_ref();
        asset_ref.asset.dispose(device)
    }

    // Loads dynamically an asset into an erased arena by first reading it's specifier
    // then creating the asset through the specifier's metadata
    unsafe fn load_fn<A: Asset>(
        path: &str,
        erased_arena: &mut ErasedArena,
        context: &LoadContext,
    ) -> Index {
        let (index, option_ptr) = erased_arena.preallocate_entry();
        let file_content =
            std::fs::read_to_string(std::path::Path::new(path).with_extension("toml"))
                .unwrap_or_else(|e| {
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
                identifier: path.into(),
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
            identifier: Some(ImmutableString::new_dynamic(value.as_ref())),
        }
    }
}
impl<A: Asset> std::fmt::Debug for AssetHandle<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "AssetHandle<{}>(\"{}\")",
            type_name::<A>(),
            self.identifier.as_deref().unwrap_or("Null")
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
            identifier: Some(ImmutableString::deserialize(deserializer)?),
        })
    }
}

impl<A: Asset> AssetHandle<A> {
    pub fn new(identifier: impl Into<ImmutableString>) -> Self {
        Self {
            _phantom_data: PhantomData,
            identifier: Some(identifier.into()),
        }
    }

    pub const fn new_const(identifier: ImmutableString) -> Self {
        Self {
            _phantom_data: PhantomData,
            identifier: Some(identifier),
        }
    }

    pub fn identifier(&self) -> Option<&ImmutableString> {
        self.identifier.as_ref()
    }

    pub fn null() -> AssetHandle<A> {
        Self {
            _phantom_data: PhantomData,
            identifier: None,
        }
    }

    pub fn is_null(&self) -> bool {
        self.identifier.is_none()
    }
}

#[cfg(test)]
mod tests {
    use mgpu::Device;
    use serde::{Deserialize, Serialize};

    use crate::{
        asset_map::{Asset, AssetSpecifier, LoadContext},
        assets::{
            texture::{Texture, TextureSamplerConfiguration},
            TextureMetadata,
        },
        sampler_allocator::SamplerAllocator,
        shader_cache::ShaderCache,
    };

    #[test]
    fn basic_operations() {
        // TODO: Rewrite tests to use new filesystem based implementation
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

        let device = Device::dummy();
        let shader_cache = ShaderCache::new();
        let sampler_allocator = SamplerAllocator::default();
        // The LoadContext is not used, we should be fine
        let asset = CharacterStats::load(
            specifier.metadata,
            &LoadContext {
                device: &device,
                shader_cache: &shader_cache,
                sampler_allocator: &sampler_allocator,
            },
        )
        .unwrap();
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
