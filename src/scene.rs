use std::rc::Rc;

use ash::vk::{
    ClearColorValue, ClearDepthStencilValue, ClearValue, Extent2D, IndexType, Offset2D,
    PipelineBindPoint, PipelineStageFlags, Rect2D,
};
use gpu::{BeginRenderPassInfo, Gpu, GpuFramebuffer, RenderPass, Swapchain};
use nalgebra::{point, vector, Matrix4};
use resource_map::{ResourceHandle, ResourceMap};

use crate::{material::Material, mesh::Mesh, PerObjectData};

pub struct ScenePrimitive {
    pub mesh: ResourceHandle<Mesh>,
    pub material: ResourceHandle<Material>,
    pub transform: Matrix4<f32>,
}

pub struct Scene {
    primitives: Vec<ScenePrimitive>,
}

impl Scene {
    pub fn new() -> Self {
        Self { primitives: vec![] }
    }

    pub fn add(&mut self, primitive: ScenePrimitive) -> usize {
        let idx = self.primitives.len();
        self.primitives.push(primitive);
        idx
    }

    pub fn edit(&mut self, idx: usize) -> &mut ScenePrimitive {
        &mut self.primitives[idx]
    }

    pub fn all_primitives(&self) -> &[ScenePrimitive] {
        &self.primitives
    }
    pub fn edit_all_primitives(&mut self) -> &mut [ScenePrimitive] {
        &mut self.primitives
    }
}

pub trait SceneRenderer {
    fn render(
        &mut self,
        gpu: &Gpu,
        scene: &Scene,
        framebuffer: &GpuFramebuffer,
        swapchain: &mut Swapchain,
    );
}

pub struct ForwardNaiveRenderer<'a> {
    resource_map: Rc<ResourceMap>,
    extents: Extent2D,
    render_pass: &'a RenderPass,
}
impl<'a> ForwardNaiveRenderer<'a> {
    pub fn new(
        resource_map: Rc<ResourceMap>,
        extents: Extent2D,
        render_pass: &'a RenderPass,
    ) -> Self {
        Self {
            resource_map,
            extents,
            render_pass,
        }
    }
}

impl<'a> SceneRenderer for ForwardNaiveRenderer<'a> {
    fn render(
        &mut self,
        gpu: &Gpu,
        scene: &Scene,
        framebuffer: &GpuFramebuffer,
        swapchain: &mut Swapchain,
    ) {
        let mut command_buffer = gpu::CommandBuffer::new(gpu, gpu::QueueType::Graphics).unwrap();

        {
            let mut render_pass = command_buffer.begin_render_pass(&BeginRenderPassInfo {
                framebuffer,
                render_pass: &self.render_pass,
                clear_color_values: &[
                    ClearValue {
                        color: ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    ClearValue {
                        depth_stencil: ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ],
                render_area: Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: self.extents,
                },
            });
            for primitive in scene.all_primitives() {
                let transform = primitive.transform;
                let mesh = self.resource_map.get(&primitive.mesh);
                let material = self.resource_map.get(&primitive.material);
                let pipeline = self.resource_map.get(&material.pipeline);
                gpu.write_buffer_data(
                    &material.uniform_buffers[0],
                    &[PerObjectData {
                        model: transform,
                        view: nalgebra::Matrix4::look_at_rh(
                            &point![2.0, 2.0, 2.0],
                            &point![0.0, 0.0, 0.0],
                            &vector![0.0, 0.0, -1.0],
                        ),
                        projection: nalgebra::Matrix4::new_perspective(
                            1240.0 / 720.0,
                            45.0,
                            0.1,
                            10.0,
                        ),
                    }],
                )
                .unwrap();
                render_pass.bind_pipeline(&pipeline.0);
                render_pass.bind_descriptor_sets(
                    PipelineBindPoint::GRAPHICS,
                    &pipeline.0,
                    0,
                    &[&material.resources_descriptor_set],
                );

                render_pass.bind_index_buffer(&mesh.index_buffer, 0, IndexType::UINT32);
                render_pass.bind_vertex_buffer(
                    0,
                    &[
                        &mesh.position_component,
                        &mesh.color_component,
                        &mesh.normal_component,
                        &mesh.tangent_component,
                        &mesh.uv_component,
                    ],
                    &[0, 0, 0, 0, 0],
                );
                render_pass.draw_indexed(6, 1, 0, 0, 0);
            }
        }

        command_buffer
            .submit(&gpu::CommandBufferSubmitInfo {
                wait_semaphores: &[&swapchain.image_available_semaphore],
                wait_stages: &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                signal_semaphores: &[&swapchain.render_finished_semaphore],
                fence: Some(&swapchain.in_flight_fence),
            })
            .unwrap();
    }
}
