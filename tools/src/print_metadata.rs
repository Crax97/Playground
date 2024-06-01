use clap::Parser;
use engine::{sampler_allocator::SamplerAllocator, shader_cache::ShaderCache};
use mgpu::Device;

#[derive(Parser, Debug)]
#[command(version, about)]
struct GltfViewerArgs {
    #[arg()]
    asset_name: String,
}

fn main() -> anyhow::Result<()> {
    let device = Device::new(mgpu::DeviceConfiguration {
        app_name: None,
        features: Default::default(),
        device_preference: None,
        desired_frames_in_flight: 1,
        display_handle: None,
    })?;
    let shader_cache = ShaderCache::new();
    let sampler_allocator = SamplerAllocator::default();

    let asset_map =
        engine::app::asset_map_with_defaults(&device, &sampler_allocator, shader_cache)?;

    let args = GltfViewerArgs::parse();

    for registration in asset_map.registrations() {
        if registration.asset_type_name == &args.asset_name {
            let specifier = (registration.specifier_fn)();
            println!("{}", specifier);
            return Ok(());
        }
    }

    eprintln!("Asset {} is not registered", args.asset_name);
    Ok(())
}
