use std::sync::Arc;

use gpu::{Gpu, Offset2D, Rect2D};
use log::{error, warn};

use crate::{
    game_scene::Scene, Camera, CvarManager, DeferredRenderingPipeline, RenderScene,
    RenderingPipeline, ResourceMap,
};

use super::System;

pub struct RenderingSystem {
    gpu: Arc<dyn Gpu>,
}

impl RenderingSystem {
    pub fn new(gpu: Arc<dyn Gpu>) -> Self {
        Self { gpu }
    }
}

impl System for RenderingSystem {
    fn setup_resources(
        &self,
        resource_builder: &mut crate::game_framework::resources::ResourcesBuilder,
    ) {
        resource_builder.add_resource(Scene::new());
        let combine_shader =
            DeferredRenderingPipeline::make_3d_combine_shader(self.gpu.as_ref()).unwrap();
        resource_builder.add_resource(
            DeferredRenderingPipeline::new(self.gpu.as_ref(), combine_shader).unwrap(),
        );
    }

    fn draw(&mut self, params: super::SystemDrawParams) {
        let resource_map = params.resources.try_get::<ResourceMap>();
        let resource_map = if let Some(rm) = resource_map {
            rm
        } else {
            error!("Rendering System: no resource map");
            return;
        };

        let cvar_manager = if let Some(cvar_manager) = params.resources.try_get::<CvarManager>() {
            cvar_manager
        } else {
            error!("Rendering System: no cvar_manager");
            return;
        };

        let camera = if let Some(cam) = params.resources.try_get::<Camera>() {
            *cam
        } else {
            warn!("Rendering System: using default camera");
            Camera::default()
        };

        if let Some(scene) = params.resources.try_get::<RenderScene>() {
            let mut rendering_pipeline = params.resources.get_mut::<DeferredRenderingPipeline>();

            let mut graphics_command_buffer = self
                .gpu
                .start_command_buffer(gpu::QueueType::Graphics)
                .unwrap();
            let final_render = rendering_pipeline
                .render(
                    self.gpu.as_ref(),
                    &mut graphics_command_buffer,
                    &camera,
                    &scene,
                    &resource_map,
                    &cvar_manager,
                )
                .unwrap();
            rendering_pipeline
                .draw_textured_quad(
                    &mut graphics_command_buffer,
                    &params.backbuffer.image_view,
                    &final_render,
                    Rect2D {
                        offset: Offset2D::default(),
                        extent: params.backbuffer.size,
                    },
                    true,
                    None,
                )
                .unwrap();
        }
    }

    fn shutdown(&mut self, params: super::SystemShutdownParams) {
        let mut rendering_pipeline = params.resources.get_mut::<DeferredRenderingPipeline>();
        rendering_pipeline.destroy(self.gpu.as_ref());
    }
}
