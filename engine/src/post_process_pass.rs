use bytemuck::{Pod, Zeroable};
use engine_macros::glsl;
use gpu::{
    render_pass_2::RenderPass2, Binding2, Extent2D, Gpu, ImageViewHandle, SamplerHandle,
    ShaderModuleCreateInfo, ShaderModuleHandle, ShaderStage,
};
use nalgebra::{vector, Vector2};

use crate::{CvarFlags, CvarManager};

pub struct PostProcessResources<'a> {
    pub screen_quad: &'a ShaderModuleHandle,
    pub previous_pass_result: &'a ImageViewHandle,
    pub sampler: &'a SamplerHandle,
    pub cvar_manager: &'a CvarManager,
    pub render_size: Extent2D,
}

pub trait PostProcessPass: Send + Sync + 'static {
    fn name(&self) -> String;
    fn apply(
        &self,
        post_process_pass: &mut RenderPass2,
        resources: &PostProcessResources,
    ) -> anyhow::Result<()>;
    fn destroy(&self, gpu: &dyn Gpu);
}

pub struct TonemapPass {
    shader_handle: ShaderModuleHandle,
}

pub struct FxaaPass {
    fxaa_vs: ShaderModuleHandle,
    fxaa_fs: ShaderModuleHandle,
}

impl TonemapPass {
    pub fn new(gpu: &dyn Gpu) -> anyhow::Result<Self> {
        const TONEMAP: &[u32] = glsl!(
            kind = fragment,
            path = "src/shaders/tonemap.frag",
            entry_point = "main"
        );
        Ok(Self {
            shader_handle: gpu.make_shader_module(&gpu::ShaderModuleCreateInfo {
                label: Some("Tonemap shader"),
                code: bytemuck::cast_slice(TONEMAP),
            })?,
        })
    }
}

impl PostProcessPass for TonemapPass {
    fn name(&self) -> String {
        "Tonemap".to_owned()
    }
    fn apply(
        &self,
        post_process_pass: &mut RenderPass2,
        resources: &PostProcessResources,
    ) -> anyhow::Result<()> {
        post_process_pass.bind_resources_2(
            0,
            &[Binding2 {
                ty: gpu::DescriptorBindingType2::ImageView {
                    image_view_handle: *resources.previous_pass_result,
                    sampler_handle: *resources.sampler,
                },
                write: false,
            }],
        )?;
        post_process_pass.set_vertex_shader(*resources.screen_quad);
        post_process_pass.set_fragment_shader(self.shader_handle);
        post_process_pass.draw(4, 1, 0, 0)
    }

    fn destroy(&self, gpu: &dyn Gpu) {
        gpu.destroy_shader_module(self.shader_handle);
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
struct FxaaShaderParams {
    rcp_frame: Vector2<f32>,
    fxaa_quality_subpix: f32,
    fxaa_quality_edge_threshold: f32,
    fxaa_quality_edge_threshold_min: f32,
    iterations: u32,
}

unsafe impl Pod for FxaaShaderParams {}
unsafe impl Zeroable for FxaaShaderParams {}

const FXAA_FS: &[u32] = glsl!(
    kind = fragment,
    path = "src/shaders/fxaa_fs.frag",
    entry_point = "main"
);

const FXAA_VS: &[u32] = glsl!(
    kind = vertex,
    path = "src/shaders/fxaa_vs.vert",
    entry_point = "main"
);

impl FxaaPass {
    pub const FXAA_ITERATIONS_CVAR_NAME: &'static str = "fxaa.iterations";
    pub const FXAA_SUBPIX_CVAR_NAME: &'static str = "fxaa.subpix";
    pub const FXAA_EDGE_THRESHOLD_CVAR_NAME: &'static str = "fxaa.edge_threshold";
    pub const FXAA_EDGE_THRESHOLD_MIN_CVAR_NAME: &'static str = "fxaa.edge_threshold_min";

    pub fn new(gpu: &dyn Gpu, cvar_manager: &mut CvarManager) -> anyhow::Result<Self> {
        cvar_manager.register_cvar(Self::FXAA_ITERATIONS_CVAR_NAME, 12, CvarFlags::empty());
        cvar_manager.register_cvar(Self::FXAA_SUBPIX_CVAR_NAME, 0.75, CvarFlags::empty());
        cvar_manager.register_cvar(
            Self::FXAA_EDGE_THRESHOLD_CVAR_NAME,
            0.166,
            CvarFlags::empty(),
        );
        cvar_manager.register_cvar(
            Self::FXAA_EDGE_THRESHOLD_MIN_CVAR_NAME,
            0.0833,
            CvarFlags::empty(),
        );

        let fxaa_vs = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("FXAA VS"),
            code: bytemuck::cast_slice(FXAA_VS),
        })?;
        let fxaa_fs = gpu.make_shader_module(&ShaderModuleCreateInfo {
            label: Some("FXAA FS"),
            code: bytemuck::cast_slice(FXAA_FS),
        })?;
        Ok(Self { fxaa_vs, fxaa_fs })
    }
}

impl PostProcessPass for FxaaPass {
    fn name(&self) -> String {
        "FXAA".to_owned()
    }

    fn apply(
        &self,
        post_process_pass: &mut RenderPass2,
        resources: &PostProcessResources,
    ) -> anyhow::Result<()> {
        post_process_pass.bind_resources_2(
            0,
            &[Binding2 {
                ty: gpu::DescriptorBindingType2::ImageView {
                    image_view_handle: *resources.previous_pass_result,
                    sampler_handle: *resources.sampler,
                },
                write: false,
            }],
        )?;
        let rcp_frame = vector![
            resources.render_size.width as f32,
            resources.render_size.height as f32
        ];
        let rcp_frame = vector![1.0 / rcp_frame.x, 1.0 / rcp_frame.y];

        let iterations = resources
            .cvar_manager
            .get_named::<i32>(Self::FXAA_ITERATIONS_CVAR_NAME)
            .expect("Fxaa Cvar not found");
        let fxaa_quality_subpix = resources
            .cvar_manager
            .get_named::<f32>(Self::FXAA_SUBPIX_CVAR_NAME)
            .unwrap();
        let fxaa_quality_edge_threshold = resources
            .cvar_manager
            .get_named::<f32>(Self::FXAA_EDGE_THRESHOLD_CVAR_NAME)
            .unwrap();
        let fxaa_quality_edge_threshold_min = resources
            .cvar_manager
            .get_named::<f32>(Self::FXAA_EDGE_THRESHOLD_MIN_CVAR_NAME)
            .unwrap();
        let params = FxaaShaderParams {
            rcp_frame,
            fxaa_quality_subpix,
            fxaa_quality_edge_threshold,
            fxaa_quality_edge_threshold_min,
            iterations: iterations as _,
        };

        post_process_pass.set_vertex_shader(self.fxaa_vs);
        post_process_pass.set_fragment_shader(self.fxaa_fs);
        post_process_pass.push_constants(
            0,
            0,
            bytemuck::cast_slice(&[params]),
            ShaderStage::ALL_GRAPHICS,
        );
        post_process_pass.draw(3, 1, 0, 0)
    }

    fn destroy(&self, gpu: &dyn Gpu) {
        gpu.destroy_shader_module(self.fxaa_vs);
        gpu.destroy_shader_module(self.fxaa_fs);
    }
}
