pub mod immutable_string;
mod texture_packer;

pub use texture_packer::*;

pub fn ensure_vec_length<T: Default + Clone>(vec: &mut Vec<T>, index: usize) {
    if vec.len() < index {
        let diff = index - vec.len();
        vec.extend(std::iter::repeat(T::default()).take(diff))
    }
}
