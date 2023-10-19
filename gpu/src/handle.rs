#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HandleType {
    Buffer,
    ShaderModule,
}

pub trait Handle {
    fn new(id: u64) -> Self;
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
                Self {
                    id
                }
            }

            fn id(&self) -> u64 {
                self.id
            }

            fn handle_type(&self) -> HandleType {
                $ty
            }
        }

    }
}

define_handle!(BufferHandle, HandleType::Buffer);
define_handle!(ShaderModuleHandle, HandleType::ShaderModule);