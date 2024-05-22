use std::{alloc::Layout, fmt::Debug, ptr::NonNull};

#[cfg(debug_assertions)]
use std::any::TypeId;

use serde::{Deserialize, Serialize};

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Index {
    index: u32,
    generation: u32,
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy)]
struct DebugOnlyTypeId {
    #[cfg(debug_assertions)]
    id: TypeId,
}

impl DebugOnlyTypeId {
    fn of<T: 'static>() -> DebugOnlyTypeId {
        DebugOnlyTypeId {
            #[cfg(debug_assertions)]
            id: TypeId::of::<T>(),
        }
    }
}

pub struct Entry<T: 'static> {
    payload: Option<T>,
    generation: u32,
}
pub struct ErasedArena {
    // An erased dynamic array of ErasedEntry<T>
    entries: NonNull<u8>,

    // How many entries are stored at the moment in the Arena
    len: usize,

    // How many entries the Arena can hold
    capacity: usize,

    // Layout of a single Entry<T>
    entry_layout: Layout,

    // Reusable indices for the arena: when an item is removed, it goes here, and when
    // a new item is added if there's an index here, it's used for the new item
    freed_indices: Vec<Index>,

    // Tries to drop the entry at the pointer, returning the new generation if the operation succeeds
    drop_fn: unsafe fn(*mut u8) -> Option<u32>,

    // The second arg is an offset, the third is a count
    // Function responsibile for setting the entries in (ptr)[offset..count] to their default value
    default_fn: unsafe fn(*mut u8, usize, usize),

    // Used to enforce that we're dealing with the correct type id, the checks actually only happens
    // in debug builds
    stored_type_id: DebugOnlyTypeId,
}

impl ErasedArena {
    pub fn new<T: 'static>() -> Self {
        ErasedArena {
            entries: NonNull::<T>::dangling().cast(),
            len: 0,
            capacity: 0,
            drop_fn: Self::drop_fn::<T>,
            default_fn: Self::default_fn::<T>,
            entry_layout: Self::entry_layout::<T>(),
            freed_indices: vec![],
            stored_type_id: DebugOnlyTypeId::of::<T>(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter<T: 'static>(&self) -> impl Iterator<Item = &T> {
        debug_assert!(DebugOnlyTypeId::of::<T>() == self.stored_type_id);
        (0..self.capacity).flat_map(|index| {
            let entry = self.entry_ptr_at(index);
            let entry_ref = unsafe { entry.as_ref().unwrap() };
            entry_ref.payload.as_ref()
        })
    }

    pub fn iter_with_index<T: 'static>(&self) -> impl Iterator<Item = (Index, &T)> {
        debug_assert!(DebugOnlyTypeId::of::<T>() == self.stored_type_id);
        (0..self.capacity).flat_map(|index| {
            let entry = self.entry_ptr_at(index);
            let entry_ref = unsafe { entry.as_ref().unwrap() };
            entry_ref.payload.as_ref().map(|e| {
                (
                    Index {
                        index: index as u32,
                        generation: entry_ref.generation,
                    },
                    e,
                )
            })
        })
    }

    pub fn iter_mut<T: 'static>(&mut self) -> impl Iterator<Item = &mut T> {
        debug_assert!(DebugOnlyTypeId::of::<T>() == self.stored_type_id);
        (0..self.capacity).flat_map(|index| {
            let entry = self.entry_mut_ptr_at(index);
            let entry_ref = unsafe { entry.as_mut().unwrap() };
            entry_ref.payload.as_mut()
        })
    }

    pub fn add<T: 'static>(&mut self, value: T) -> Index {
        debug_assert!(DebugOnlyTypeId::of::<T>() == self.stored_type_id);
        let index = unsafe { self.allocate_index() };

        let entry_ptr = self.entry_mut_ptr_at::<T>(index.index as usize);
        let entry_slot = unsafe { entry_ptr.as_mut().unwrap() };
        debug_assert!(entry_slot.payload.is_none());
        entry_slot.payload = Some(value);

        index
    }

    pub fn get<T: 'static>(&self, index: Index) -> Option<&T> {
        debug_assert!(DebugOnlyTypeId::of::<T>() == self.stored_type_id);
        if index.index as usize >= self.capacity() {
            return None;
        }

        let entry = unsafe { self.entry_ptr_at(index.index as usize).as_ref().unwrap() };
        if entry.generation == index.generation {
            debug_assert!(entry.payload.is_some());
            entry.payload.as_ref()
        } else {
            None
        }
    }

    pub fn get_mut<T: 'static>(&mut self, index: Index) -> Option<&mut T> {
        debug_assert!(DebugOnlyTypeId::of::<T>() == self.stored_type_id);
        if index.index as usize >= self.capacity() {
            return None;
        }

        let entry = unsafe {
            self.entry_mut_ptr_at(index.index as usize)
                .as_mut()
                .unwrap()
        };
        if entry.generation == index.generation {
            debug_assert!(entry.payload.is_some());
            entry.payload.as_mut()
        } else {
            None
        }
    }

    pub fn remove<T: 'static>(&mut self, index: Index) -> Option<T> {
        debug_assert!(DebugOnlyTypeId::of::<T>() == self.stored_type_id);
        if index.index as usize >= self.capacity() {
            return None;
        }

        let entry = unsafe {
            self.entry_mut_ptr_at(index.index as usize)
                .as_mut()
                .unwrap()
        };

        if entry.generation == index.generation {
            debug_assert!(entry.payload.is_some());
            let payload = entry.payload.take().unwrap();
            entry.generation += 1;
            self.freed_indices.push(Index {
                index: index.index,
                generation: entry.generation,
            });
            self.len -= 1;
            Some(payload)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        // Count up to capacity because when calling remove len is decreased
        // thus we might miss the elements at the end of the arena
        for index in 0..self.capacity {
            let payload = self.erased_payload_mut_ptr_at(index);
            if let Some(generation) = unsafe { (self.drop_fn)(payload) } {
                self.freed_indices.push(Index {
                    index: index as u32,
                    generation,
                })
            }
        }
        self.len = 0;
    }

    /// # Safety
    /// The entry pointed by index must be written with a valid value, before any read operations
    pub unsafe fn allocate_index(&mut self) -> Index {
        const GROWTH_FACTOR: f32 = 1.5;

        let index = if let Some(index) = self.freed_indices.pop() {
            index
        } else {
            if self.capacity() - self.len() == 0 {
                let new_capacity = if self.capacity == 0 {
                    2
                } else {
                    (self.capacity as f32 * GROWTH_FACTOR).ceil() as usize
                };
                let difference = new_capacity - self.capacity;
                self.grow(difference);
            }

            let index: Index = Index {
                index: self.len as u32,
                generation: 0,
            };
            index
        };
        self.len += 1;

        index
    }

    /// # Safety
    /// The entry pointed by index must be written with a valid value, before any read operations.
    /// The returned pointer is a pointer to the Option<T> payload of the entry
    pub unsafe fn preallocate_entry(&mut self) -> (Index, *mut u8) {
        let index = self.allocate_index();
        let ptr = self.erased_payload_mut_ptr_at(index.index as usize);
        (index, ptr)
    }

    fn erased_payload_mut_ptr_at(&mut self, index: usize) -> *mut u8 {
        unsafe { self.entries.as_ptr().add(index * self.entry_layout.size()) }
    }

    fn entry_mut_ptr_at<T: 'static>(&mut self, index: usize) -> *mut Entry<T> {
        unsafe { self.entries.cast::<Entry<T>>().as_ptr().add(index) }
    }

    fn entry_ptr_at<T: 'static>(&self, index: usize) -> *const Entry<T> {
        unsafe {
            self.entries
                .cast::<Entry<T>>()
                .as_ptr()
                .add(index)
                .cast_const()
        }
    }

    fn grow(&mut self, count: usize) {
        let new_capacity = self.capacity + count;
        let array_layout = Self::array_layout(self.entry_layout, new_capacity);
        let old_array_layout = Self::array_layout(self.entry_layout, self.capacity);

        let new_ptr = if self.capacity > 0 {
            unsafe {
                std::alloc::realloc(self.entries.as_ptr(), old_array_layout, array_layout.size())
            }
        } else {
            unsafe { std::alloc::alloc(array_layout) }
        };
        if new_ptr.is_null() {
            std::alloc::handle_alloc_error(array_layout);
        }
        self.entries = unsafe { NonNull::new_unchecked(new_ptr) };

        // Initialize memory
        unsafe {
            (self.default_fn)(new_ptr, self.capacity, count);
        }

        self.capacity = new_capacity;
    }

    fn array_layout(element_layout: Layout, capacity: usize) -> Layout {
        Layout::from_size_align(element_layout.size() * capacity, element_layout.align())
            .expect("Failed to create layout")
    }
    fn entry_layout<T: 'static>() -> Layout {
        Layout::new::<Entry<T>>()
    }

    fn drop_fn<T: 'static>(data: *mut u8) -> Option<u32> {
        let ptr_t = data.cast::<Entry<T>>();
        let ref_t = unsafe { ptr_t.as_mut().unwrap_unchecked() };
        // Take takes care of actually dropping the value
        if ref_t.payload.take().is_some() {
            ref_t.generation += 1;
            Some(ref_t.generation)
        } else {
            None
        }
    }
    fn default_fn<T: 'static>(data: *mut u8, offset: usize, count: usize) {
        let data = unsafe { data.cast::<Entry<T>>().add(offset) };
        for i in 0..count {
            let elem_at = unsafe { data.add(i) };
            unsafe { elem_at.write(Default::default()) };
        }
    }
}

// ErasedArena is Send + Sync because the only way to modify it is through &mut refs
unsafe impl Send for ErasedArena {}
unsafe impl Sync for ErasedArena {}

impl Drop for ErasedArena {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }
        self.clear();

        unsafe {
            std::alloc::dealloc(
                self.entries.as_ptr(),
                Self::array_layout(self.entry_layout, self.capacity),
            );
        }
    }
}

impl<T: 'static> Default for Entry<T> {
    fn default() -> Self {
        Self {
            payload: Default::default(),
            generation: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        alloc::Layout,
        sync::{atomic::AtomicU32, Arc},
    };

    use super::ErasedArena;

    #[cfg(not(debug_assertions))]
    const _: [u8; 8] = [0; std::mem::size_of::<super::Index>()];

    #[test]
    fn layout_tests() {
        let array_layout = ErasedArena::array_layout(Layout::new::<u32>(), 10);
        assert_eq!(array_layout, Layout::new::<[u32; 10]>());

        let array_layout = ErasedArena::array_layout(Layout::new::<u32>(), 0);
        assert_eq!(array_layout, Layout::new::<[u32; 0]>());
    }

    #[test]
    fn empty_arena() {
        let _ = ErasedArena::new::<u32>();
    }

    #[test]
    fn erased_arena_operations() {
        let mut erased_arena = ErasedArena::new::<u32>();
        let index_of_10 = erased_arena.add::<u32>(10);
        let index_of_42 = erased_arena.add::<u32>(42);
        let index_of_69 = erased_arena.add::<u32>(69);
        assert_eq!(erased_arena.len(), 3);
        assert_eq!(erased_arena.capacity(), 3);

        assert_eq!(erased_arena.get::<u32>(index_of_10).copied().unwrap(), 10);
        assert_eq!(erased_arena.get::<u32>(index_of_42).copied().unwrap(), 42);
        assert_eq!(erased_arena.get::<u32>(index_of_69).copied().unwrap(), 69);

        *erased_arena.get_mut::<u32>(index_of_42).unwrap() = 120;
        assert_eq!(erased_arena.get::<u32>(index_of_42).copied().unwrap(), 120);

        let removed_1 = erased_arena.remove::<u32>(index_of_10).unwrap();
        assert_eq!(removed_1, 10);

        assert_eq!(erased_arena.len(), 2);
        assert_eq!(erased_arena.capacity(), 3);

        assert!(erased_arena.get::<u32>(index_of_10).is_none());

        let new_element = erased_arena.add::<u32>(150);
        assert_eq!(new_element.index, index_of_10.index);
        assert_eq!(new_element.generation, index_of_10.generation + 1);

        assert!(erased_arena.get::<u32>(index_of_10).is_none());
        assert!(erased_arena
            .get::<u32>(new_element)
            .is_some_and(|&v| v == 150));

        assert_eq!(erased_arena.len(), 3);
        assert_eq!(erased_arena.capacity(), 3);

        erased_arena.clear();

        assert!(erased_arena.is_empty());
        assert_eq!(erased_arena.capacity(), 3);

        assert!(erased_arena.get::<u32>(index_of_10).is_none());
        assert!(erased_arena.get::<u32>(index_of_42).is_none());
        assert!(erased_arena.get::<u32>(index_of_69).is_none());
        assert!(erased_arena.get::<u32>(new_element).is_none());
    }

    #[test]
    fn empty_align() {
        #[repr(align(16))]
        struct WeirdAlign;
        let mut erased_arena = ErasedArena::new::<WeirdAlign>();
        let index_of_first = erased_arena.add::<WeirdAlign>(WeirdAlign);
        let index_of_second = erased_arena.add::<WeirdAlign>(WeirdAlign);
        let index_of_third = erased_arena.add::<WeirdAlign>(WeirdAlign);
        assert_eq!(erased_arena.len(), 3);
        assert_eq!(erased_arena.capacity(), 3);

        erased_arena.clear();

        assert!(erased_arena.is_empty());
        assert_eq!(erased_arena.capacity(), 3);

        assert!(erased_arena.get::<WeirdAlign>(index_of_first).is_none());
        assert!(erased_arena.get::<WeirdAlign>(index_of_second).is_none());
        assert!(erased_arena.get::<WeirdAlign>(index_of_third).is_none());
    }

    #[test]
    fn drop() {
        struct TypeWithCustomDrop {
            counter: Arc<AtomicU32>,
        }

        impl Clone for TypeWithCustomDrop {
            fn clone(&self) -> Self {
                self.counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Self {
                    counter: self.counter.clone(),
                }
            }
        }

        impl Drop for TypeWithCustomDrop {
            fn drop(&mut self) {
                self.counter
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let mut erased_arena = ErasedArena::new::<TypeWithCustomDrop>();
        let counter: Arc<AtomicU32> = Default::default();
        let value = TypeWithCustomDrop {
            counter: counter.clone(),
        };
        let index_of_first = erased_arena.add(value.clone());
        let _ = erased_arena.add(value.clone());
        let _ = erased_arena.add(value.clone());
        assert_eq!(erased_arena.len(), 3);
        assert_eq!(erased_arena.capacity(), 3);

        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 3);

        erased_arena
            .remove::<TypeWithCustomDrop>(index_of_first)
            .take();

        erased_arena.clear();

        assert!(erased_arena.is_empty());
        assert_eq!(erased_arena.capacity(), 3);

        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}
