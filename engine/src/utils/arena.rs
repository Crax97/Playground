use std::{fmt::Debug, marker::PhantomData};

use serde::{Deserialize, Serialize};

use crate::erased_arena::{self, ErasedArena};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub struct Index(erased_arena::Index);

pub struct Arena<T: 'static> {
    _ph: PhantomData<T>,
    arena: ErasedArena,
}

impl<T: 'static> Arena<T> {
    pub fn new() -> Self {
        Self {
            _ph: PhantomData,
            arena: ErasedArena::new::<T>(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.arena.iter()
    }

    pub fn iter_with_index(&self) -> impl Iterator<Item = (Index, &T)> {
        self.arena.iter_with_index().map(|(i, t)| (Index(i), t))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.arena.iter_mut()
    }

    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    pub fn add(&mut self, value: T) -> Index {
        let index = self.arena.add(value);
        Index(index)
    }

    pub fn get(&self, index: Index) -> Option<&T> {
        self.arena.get(index.0)
    }

    pub fn get_mut(&mut self, index: Index) -> Option<&mut T> {
        self.arena.get_mut(index.0)
    }

    pub fn remove(&mut self, index: Index) -> Option<T> {
        self.arena.remove(index.0)
    }

    pub fn clear(&mut self) {
        self.arena.clear()
    }
}

impl<T: 'static + Debug> Debug for Arena<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for entry in self.iter() {
            list.entry(entry);
        }
        list.finish()
    }
}
impl<T: 'static> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}
