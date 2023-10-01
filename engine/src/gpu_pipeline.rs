use gpu::VkGraphicsPipeline;
use resource_map::Resource;

pub struct GpuPipeline(pub VkGraphicsPipeline);

impl Resource for GpuPipeline {
    fn get_description(&self) -> &str {
        "GpuPipeline"
    }
}
