use std::collections::HashMap;

use crate::Handle;

pub struct GpuResourceMap<T>(HashMap<u64, T>);

pub trait HasAssociatedHandle {
    type AssociatedHandle: Handle;
}

pub mod utils {
    macro_rules! associate_to_handle {
        ($concrete:ty, $handle:ty) => {
            impl HasAssociatedHandle for $concrete {
                type AssociatedHandle = $handle;
            }
        };
    }
    pub(crate) use associate_to_handle;
}
impl<T: HasAssociatedHandle> GpuResourceMap<T> {
    pub fn resolve(&self, handle: T::AssociatedHandle) -> &T {
        let res = self
            .0
            .get(&handle.id())
            .expect("Failed to resolve resource");
        res
    }

    pub fn insert(&mut self, handle: T::AssociatedHandle, res: T) {
        self.0.insert(handle.id(), res);
    }

    pub fn new() -> GpuResourceMap<T> {
        Self(HashMap::new())
    }
}
