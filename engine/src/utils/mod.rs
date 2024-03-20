use erased_serde::Deserializer;
pub mod immutable_string;
mod texture_packer;

pub use texture_packer::*;

// A resource that may optionally need something stored in the World to be deserialized
pub trait WorldDeserialize<'de> {
    fn deserialize(
        deserializer: Box<dyn Deserializer<'de>>,
        world: &kecs::World,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl<'de, T: serde::Deserialize<'de>> WorldDeserialize<'de> for T {
    fn deserialize(
        deserializer: Box<dyn Deserializer<'de>>,
        _world: &kecs::World,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let me = Self::deserialize(deserializer)?;
        Ok(me)
    }
}
