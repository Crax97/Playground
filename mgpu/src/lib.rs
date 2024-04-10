mod device;
mod swapchain;

pub use device::*;
pub use swapchain::*;


#[derive(Debug)]
pub enum MgpuError {

}

pub type MgpuResult<T> = Result<T, MgpuError>;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
