use std::sync::atomic::AtomicU64;

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
    Semaphore,
    Fence,
}

pub trait Handle: std::fmt::Debug + Send + Sync + 'static {
    fn new() -> Self;
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
        #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Hash)]
        pub struct $st_name {
            id: u64,
        }

        impl std::fmt::Debug for $st_name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_fmt(format_args!("{:?} - id {}", &Self::handle_type(), self.id))
            }
        }

        impl Handle for $st_name {
            fn new() -> Self {
                Self {
                    id: next_unique_id(),
                }
            }

            fn null() -> Self {
                Self { id: 0 }
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

        impl Default for $st_name {
            fn default() -> Self {
                Self::null()
            }
        }
    };
}

define_handle!(BufferHandle, HandleType::Buffer);
define_handle!(ShaderModuleHandle, HandleType::ShaderModule);
define_handle!(ImageHandle, HandleType::Image);
define_handle!(ImageViewHandle, HandleType::ImageView);
define_handle!(SamplerHandle, HandleType::Sampler);
define_handle!(SemaphoreHandle, HandleType::Semaphore);
define_handle!(FenceHandle, HandleType::Fence);
