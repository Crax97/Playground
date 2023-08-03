use gpu::GraphicsPipeline;
use resource_map::Resource;

pub struct GpuPipeline(pub GraphicsPipeline);

impl Resource for GpuPipeline {
    fn get_description(&self) -> &str {
        "GpuPipeline"
    }
}
