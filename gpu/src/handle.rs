use crate::Context;
use std::sync::{atomic::AtomicU64, Arc};

fn next_unique_id() -> u64 {
    static ATOMIC_COUNTER: AtomicU64 = AtomicU64::new(1);
    ATOMIC_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HandleType {
    Buffer,
    ShaderModule,
    Image,
    ImageView,
    Sampler,
}

pub trait Handle: std::fmt::Debug {
    fn new(context: Arc<dyn Context>) -> Self;
    fn null() -> Self;
    fn is_null(&self) -> bool;
    fn is_valid(&self) -> bool {
        !self.is_null()
    }
    fn id(&self) -> u64;
    fn handle_type() -> HandleType;
}

macro_rules! define_handle {
    ($st_name:ident, $ty:expr) => {
        pub struct $st_name {
            pub(crate) id: u64,
            pub(crate) context: Option<Arc<dyn Context>>,
        }

        impl std::hash::Hash for $st_name {
            fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
                self.id.hash(hasher)
            }
        }

        impl PartialEq for $st_name {
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }

        impl Eq for $st_name {}

        impl PartialOrd for $st_name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.id.partial_cmp(&other.id)
            }
        }

        impl Ord for $st_name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.id.cmp(&other.id)
            }
        }

        impl std::fmt::Debug for $st_name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_fmt(format_args!("{:?} - id {}", &Self::handle_type(), self.id))
            }
        }

        impl Handle for $st_name {
            fn new(context: Arc<dyn Context>) -> Self {
                Self {
                    id: next_unique_id(),
                    context: Some(context),
                }
            }

            fn null() -> Self {
                Self {
                    id: 0,
                    context: None,
                }
            }

            fn is_null(&self) -> bool {
                *self == Self::null()
            }

            fn id(&self) -> u64 {
                self.id
            }

            fn handle_type() -> HandleType {
                $ty
            }
        }

        impl Clone for $st_name {
            fn clone(&self) -> Self {
                match &self.context {
                    Some(context) => {
                        context.increment_resource_refcount(self.id, Self::handle_type());
                        Self {
                            id: self.id,
                            context: self.context.clone(),
                        }
                    }
                    None => Self::null(),
                }
            }
        }
        impl Drop for $st_name {
            fn drop(&mut self) {
                if let Some(context) = &self.context {
                    context.decrement_resource_refcount(self.id, Self::handle_type());
                }
            }
        }
    };
}

define_handle!(BufferHandle, HandleType::Buffer);
define_handle!(ShaderModuleHandle, HandleType::ShaderModule);
define_handle!(ImageHandle, HandleType::Image);
define_handle!(ImageViewHandle, HandleType::ImageView);
define_handle!(SamplerHandle, HandleType::Sampler);
