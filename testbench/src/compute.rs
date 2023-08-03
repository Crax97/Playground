use gpu::{Gpu, GpuConfiguration};

fn main() -> anyhow::Result<()> {
    
    
    let gpu = Gpu::new(GpuConfiguration {
        app_name: "compute sample",
        engine_name: "compute engine",
        pipeline_cache_path: None,
        enable_debug_utilities: true,
        window: todo!(),
    });
    
    /*
        let buffer = create_buffer();
    */
    
    
    Ok(())
}