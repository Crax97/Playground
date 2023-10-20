#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HandleType {
    Buffer,
    ShaderModule,
    Image,
    ImageView,
}

pub trait Handle {
    fn new(id: u64) -> Self;
    fn null() -> Self;
    fn is_null(&self) -> bool;
    fn id(&self) -> u64;
    fn handle_type(&self) -> HandleType;
}

macro_rules! define_handle {
    ($st_name:ident, $ty:expr) => {
        #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
        pub struct $st_name {
            pub(crate) id: u64,
        }

        impl Handle for $st_name {
            fn new(id: u64) -> Self {
                assert!(id != 0, "ID 0 is reserved for null handles!");
                Self { id }
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

            fn handle_type(&self) -> HandleType {
                $ty
            }
        }
    };
}

define_handle!(BufferHandle, HandleType::Buffer);
define_handle!(ShaderModuleHandle, HandleType::ShaderModule);
define_handle!(ImageHandle, HandleType::Image);
define_handle!(ImageViewHandle, HandleType::ImageView);
