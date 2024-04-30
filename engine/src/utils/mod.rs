pub mod immutable_string;
mod tick;

pub use tick::Tick;

pub fn ensure_vec_length<T: Default + Clone>(vec: &mut Vec<T>, index: usize) {
    if vec.len() <= index {
        let diff = index - vec.len() + 1;
        vec.extend(std::iter::repeat(T::default()).take(diff))
    }
}
