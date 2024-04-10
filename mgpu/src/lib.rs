mod device;
mod hal;
mod swapchain;

pub use device::*;
pub use swapchain::*;

#[derive(Debug)]
pub enum MgpuError {
    Dynamic(String),
}

pub type MgpuResult<T> = Result<T, MgpuError>;

impl std::fmt::Display for MgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MgpuError::Dynamic(msg) => f.write_str(msg),
        }
    }
}
impl std::error::Error for MgpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.source()
    }
}
