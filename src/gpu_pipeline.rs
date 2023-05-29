use gpu::Pipeline;
use resource_map::Resource;

pub struct GpuPipeline(pub Pipeline);

impl Resource for GpuPipeline {
    fn get_description(&self) -> &str {
        "GpuPipeline"
    }
}
