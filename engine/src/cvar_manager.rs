use bitflags::bitflags;
use std::{collections::HashMap, error::Error, fmt::Display, str::FromStr};

use bevy_ecs::prelude::Resource;

#[derive(Copy, Clone)]
pub struct CvarId(usize);

#[derive(Copy, Clone)]
pub enum CvarType {
    Integer(i32),
    Float(f32),
}

bitflags! {
    #[derive(Clone, Copy)]
    pub struct CvarFlags: u32 {

    }
}

#[derive(Debug)]
pub enum CvarError {
    CvarNotFound(String),
    GenericError(Box<dyn Error + 'static>),
}

#[allow(dead_code)]
pub struct Cvar {
    ty: CvarType,
    flags: CvarFlags,
}

#[derive(Default, Resource)]
pub struct CvarManager {
    cvars: Vec<Cvar>,
    cvar_id_map: HashMap<&'static str, CvarId>,
}

impl std::fmt::Display for CvarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CvarError::CvarNotFound(name) => f.write_fmt(format_args!("Cvar not found: {}", &name)),
            CvarError::GenericError(e) => e.fmt(f),
        }
    }
}

fn setter_helper<T: std::str::FromStr + std::fmt::Debug>(
    n: &mut T,
    value: &str,
) -> Result<(), Box<dyn Error>>
where
    <T as FromStr>::Err: std::fmt::Debug + std::error::Error + Send + Sync + 'static,
{
    let value = value.parse::<T>()?;
    *n = value;
    Ok(())
}

impl CvarType {
    fn set_from_str(&mut self, value: &str) -> Result<(), Box<dyn Error>> {
        match self {
            CvarType::Integer(n) => setter_helper(n, value),
            CvarType::Float(n) => setter_helper(n, value),
        }
    }
}

pub trait CvarSetFromBase {
    fn set(base: &mut CvarType, base: Self);
}

macro_rules! impl_into_cvar_type {
    ($base:ty, $wrapper:path) => {
        impl From<$base> for CvarType {
            fn from(value: $base) -> CvarType {
                $wrapper(value)
            }
        }
        impl<'a> From<&'a CvarType> for &'a $base {
            fn from(value: &CvarType) -> &$base {
                match value {
                    $wrapper(n) => n,
                    _ => panic!("Invalid Cvar type"),
                }
            }
        }
        impl<'a> From<&'a mut CvarType> for &'a mut $base {
            fn from(value: &'a mut CvarType) -> &'a mut $base {
                match value {
                    $wrapper(n) => n,
                    _ => panic!("Invalid Cvar type"),
                }
            }
        }
    };
}

macro_rules! impl_from_cvar_type {
    ($base:ty) => {
        impl From<CvarType> for $base {
            fn from(value: CvarType) -> $base {
                match value {
                    CvarType::Integer(n) => n as $base,
                    CvarType::Float(n) => n as $base,
                }
            }
        }
    };
}

impl_into_cvar_type!(i32, CvarType::Integer);
impl_into_cvar_type!(f32, CvarType::Float);
impl_from_cvar_type!(i32);
impl_from_cvar_type!(f32);

impl From<CvarType> for String {
    fn from(value: CvarType) -> String {
        match value {
            CvarType::Integer(n) => n.to_string(),
            CvarType::Float(n) => n.to_string(),
        }
    }
}

impl CvarManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_cvar<T: Into<CvarType>>(
        &mut self,
        name: &'static str,
        default_value: T,
        flags: CvarFlags,
    ) -> CvarId {
        let ty = default_value.into();
        let cvar = Cvar { ty, flags };

        let id = self.cvars.len();
        let id = CvarId(id);
        self.cvar_id_map.insert(name, id);
        self.cvars.push(cvar);

        id
    }

    pub fn set_named<T: Display>(&mut self, name: &str, value: T) -> Result<(), CvarError> {
        let id = self
            .cvar_id_map
            .get(name)
            .ok_or_else(|| CvarError::CvarNotFound(name.to_owned()))?;
        self.set(*id, value)
    }

    pub fn set<T: Display>(&mut self, id: CvarId, value: T) -> Result<(), CvarError> {
        self.cvars
            .get_mut(id.0)
            .expect("Cvar ID not valid! This could be a bug with the CvarManager")
            .ty
            .set_from_str(&value.to_string())
            .map_err(CvarError::GenericError)?;
        Ok(())
    }

    pub fn get_named<T: From<CvarType>>(&self, name: &str) -> Result<T, CvarError> {
        let id = self
            .cvar_id_map
            .get(name)
            .ok_or(CvarError::CvarNotFound(name.to_owned()))?;
        Ok(self.get(*id))
    }

    pub fn get<T: From<CvarType>>(&self, id: CvarId) -> T {
        self.cvars
            .get(id.0)
            .expect("CvarId not valid! This is probably a bug with the CvarManager")
            .ty
            .into()
    }

    pub fn get_named_ref_mut<'a, T>(&'a mut self, name: &str) -> Result<&'a mut T, CvarError>
    where
        &'a mut T: From<&'a mut CvarType>,
    {
        let id = self
            .cvar_id_map
            .get(name)
            .ok_or(CvarError::CvarNotFound(name.to_owned()))?;
        self.get_ref_mut(*id)
    }

    fn get_ref_mut<'a, T>(&'a mut self, id: CvarId) -> Result<&'a mut T, CvarError>
    where
        &'a mut T: From<&'a mut CvarType>,
    {
        Ok(self
            .cvars
            .get_mut(id.0)
            .map(|cv| From::from(&mut cv.ty))
            .expect("CvarId not valid! This is probably a bug with the CvarManager"))
    }

    pub fn get_named_ref<'a, T>(&'a self, name: &str) -> Result<&'a T, CvarError>
    where
        &'a T: From<&'a CvarType>,
    {
        let id = self
            .cvar_id_map
            .get(name)
            .ok_or(CvarError::CvarNotFound(name.to_owned()))?;
        self.get_ref(*id)
    }

    fn get_ref<'a, T>(&'a self, id: CvarId) -> Result<&'a T, CvarError>
    where
        &'a T: From<&'a CvarType>,
    {
        Ok(self
            .cvars
            .get(id.0)
            .map(|cv| From::from(&cv.ty))
            .expect("CvarId not valid! This is probably a bug with the CvarManager"))
    }

    pub fn cvar_names(&self) -> impl Iterator<Item = &str> {
        self.cvar_id_map.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::cvar_manager::CvarFlags;

    use super::CvarManager;

    #[test]
    fn test_it_works() {
        let mut manager = CvarManager::new();
        let _float_id = manager.register_cvar("g_float_val", 10.0, CvarFlags::empty());
        let _int_id = manager.register_cvar("int_id", 42, CvarFlags::empty());

        assert_eq!(manager.get_named::<f32>("g_float_val").unwrap(), 10.0);
        assert_eq!(manager.get_named::<i32>("int_id").unwrap(), 42);

        manager.set_named("g_float_val", 15.0).unwrap();
        manager.set_named("int_id", -123).unwrap();

        assert_eq!(manager.get_named::<f32>("g_float_val").unwrap(), 15.0);
        assert_eq!(manager.get_named::<i32>("int_id").unwrap(), -123);
    }
}
