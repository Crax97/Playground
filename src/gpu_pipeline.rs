use gpu::{Pipeline, RenderPass};
use resource_map::Resource;

pub struct GpuPipeline(pub Pipeline, pub RenderPass);

impl Resource for GpuPipeline {
    fn get_description(&self) -> &str {
        "GpuPipeline"
    }
}
