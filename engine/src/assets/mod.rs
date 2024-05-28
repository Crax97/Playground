use crate::asset_map::Asset;

use self::{material::Material, mesh::Mesh, texture::Texture};

pub mod loaders;

pub mod material;
pub mod mesh;
pub mod shader;
pub mod texture;

impl Asset for Mesh {
    fn identifier() -> &'static str {
        "Mesh"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device.destroy_buffer(self.index_buffer).unwrap();
        device.destroy_buffer(self.position_component).unwrap();
        device.destroy_buffer(self.normal_component).unwrap();
        device.destroy_buffer(self.tangent_component).unwrap();
        device.destroy_buffer(self.uv_component).unwrap();
        device.destroy_buffer(self.color_component).unwrap();
    }
}
impl Asset for Texture {
    fn identifier() -> &'static str {
        "Texture"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device.destroy_image_view(self.view).unwrap();
        device.destroy_image(self.image).unwrap();
    }
}
impl Asset for Material {
    fn identifier() -> &'static str {
        "Material"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device
            .destroy_binding_set(self.binding_set.clone())
            .unwrap();
    }
}
