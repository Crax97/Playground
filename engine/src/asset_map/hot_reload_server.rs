use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::{any::TypeId, sync::Arc};

use gpu::Gpu;
use log::{error, info};
use notify::event::ModifyKind;
use notify::{RecommendedWatcher, Watcher};

use super::{ErasedResourceLoader, LoadedResources, ResourcePtr};
use crate::AssetId;
use crossbeam::channel::{unbounded, Receiver};

pub(super) struct ReloadedResource {
    pub id: AssetId,
    pub new_resource: ResourcePtr,
}
pub(super) struct ReloadedResources {
    pub type_id: TypeId,
    pub resources: Vec<ReloadedResource>,
}

struct WatchedResource {
    pub resource_type: TypeId,
    pub resource_id: AssetId,
    pub original_path: PathBuf,
}

pub(super) struct HotReloadServer {
    watcher: RecommendedWatcher,
    watched_paths: HashMap<PathBuf, WatchedResource>,
    changed_paths_channel: Receiver<PathBuf>,
}

impl HotReloadServer {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self {
            watcher: RecommendedWatcher::new(
                move |result: Result<notify::Event, notify::Error>| {
                    if let Ok(event) = result {
                        let kind = event.kind;
                        let paths = event.paths;
                        match kind {
                            notify::EventKind::Modify(ModifyKind::Any) => {
                                info!("File {:?} changed, sending reload event", &paths[0]);
                                sender.send(paths[0].clone()).unwrap();
                            }
                            _ => {}
                        }
                    }
                },
                notify::Config::default(),
            )
            .expect("Failed to create watcher"),
            watched_paths: Default::default(),
            changed_paths_channel: receiver,
        }
    }
    pub fn watch(
        &mut self,
        path: &Path,
        resource_type: TypeId,
        resource_id: AssetId,
    ) -> anyhow::Result<()> {
        let original_path = path.to_path_buf();
        let path = path.canonicalize()?;
        info!("Watching path {path:?} for hot reload");
        self.watched_paths.insert(
            path.clone(),
            WatchedResource {
                resource_type,
                resource_id,
                original_path: original_path.clone(),
            },
        );

        self.watcher
            .watch(&path, notify::RecursiveMode::NonRecursive)?;
        Ok(())
    }

    pub fn update(
        &mut self,
        resource_loaders: &HashMap<TypeId, Box<dyn ErasedResourceLoader>>,
        loaded_resources: &mut LoadedResources,
        gpu: &Arc<dyn Gpu>,
    ) -> anyhow::Result<()> {
        let new_resources = self.get_new_resources(resource_loaders)?;
        for resources in new_resources {
            let resource_map = &loaded_resources.resources[&resources.type_id];
            let mut resource_map = resource_map.write().expect("Poison error on resource map");

            for resource in resources.resources {
                let resource_slot = resource_map
                    .resources
                    .get_mut(resource.id.id.unwrap())
                    .expect("Could not find resource");
                let mut old_resource =
                    std::mem::replace(&mut resource_slot.resource, resource.new_resource);
                Arc::get_mut(&mut old_resource)
                    .unwrap()
                    .destroyed(gpu.as_ref());
            }
        }
        Ok(())
    }

    fn get_new_resources(
        &mut self,
        resource_loaders: &HashMap<TypeId, Box<dyn ErasedResourceLoader>>,
    ) -> anyhow::Result<Vec<ReloadedResources>> {
        let paths = self
            .changed_paths_channel
            .try_iter()
            .collect::<HashSet<_>>();

        let mut resources: HashMap<TypeId, Vec<ReloadedResource>> = HashMap::new();
        for path in paths {
            let path = path.canonicalize()?;
            let resource_info = &self.watched_paths[&path];
            info!("Reloading file {:?}", resource_info.original_path);

            let resource_loader = &resource_loaders[&resource_info.resource_type];
            let new_resource = resource_loader.load_erased(&resource_info.original_path);
            match new_resource {
                Ok(new_resource) => {
                    resources
                        .entry(resource_info.resource_type)
                        .or_default()
                        .push(ReloadedResource {
                            new_resource,
                            id: resource_info.resource_id,
                        });
                }
                Err(e) => {
                    error!("Failed to reload resource {:?} because of {:?}", path, e);
                }
            };
        }

        let reloaded = resources
            .into_iter()
            .map(|(ty, res)| ReloadedResources {
                type_id: ty,
                resources: res,
            })
            .collect();
        Ok(reloaded)
    }
}
