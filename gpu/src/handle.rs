#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HandleType {
    Buffer,
}

trait Handle {
    fn id(&self) -> u64;
    fn handle_type(&self) -> HandleType;
}

macro_rules! define_handle {
    ($st_name:ident, $ty:expr) => {
        #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
        pub struct $st_name {
            id: u64,
        }

        impl Handle for $st_name {
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