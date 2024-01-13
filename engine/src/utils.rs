use gpu::Gpu;
use nalgebra::vector;

use crate::{Mesh, MeshPrimitiveCreateInfo, ResourceHandle, ResourceMap};

pub fn to_u8_slice<T>(vals: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, std::mem::size_of_val(vals)) }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TiledTexture2DSection {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
impl TiledTexture2DSection {
    fn resize_splitted_with_rest(
        self,
        width: u32,
        height: u32,
    ) -> (TiledTexture2DSection, Vec<TiledTexture2DSection>) {
        if self.width == width && self.height == height {
            (self, vec![])
        } else {
            let s1 = TiledTexture2DSection {
                x: self.x,
                y: self.y,
                width,
                height,
            };
            let mut remaining = vec![];
            let w_remain = self.width - width;
            let h_remain = self.height - height;

            if w_remain > 0 {
                remaining.push(Self {
                    x: self.x + width,
                    y: self.y,
                    width: w_remain,
                    height,
                });
            }
            if h_remain > 0 {
                remaining.push(Self {
                    x: self.x,
                    y: self.y + height,
                    width: self.width,
                    height: h_remain,
                })
            }

            (s1, remaining)
        }
    }
}

pub struct TiledTexture2DPacker {
    tile_size: u32,

    sections: Vec<TiledTexture2DSection>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TexturePackingError {
    InvalidWidthOrHeight {
        tile_size: u32,
        width: u32,
        height: u32,
    },
    NoSpaceLeft,
}

impl TiledTexture2DPacker {
    pub fn new(tile_size: u32, width: u32, height: u32) -> Result<Self, TexturePackingError> {
        if width == 0 || height == 0 || width % tile_size != 0 || height % tile_size != 0 {
            return Err(TexturePackingError::InvalidWidthOrHeight {
                tile_size,

                width,
                height,
            });
        }

        return Ok(Self {
            tile_size,
            sections: vec![TiledTexture2DSection {
                x: 0,
                y: 0,
                width,
                height,
            }],
        });
    }

    pub fn allocate(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<TiledTexture2DSection, TexturePackingError> {
        // adjust for tile_size
        let width = if width % self.tile_size > 0 {
            width + (self.tile_size - width % self.tile_size)
        } else {
            width
        };

        let height = if height % self.tile_size > 0 {
            height + (self.tile_size - height % self.tile_size)
        } else {
            height
        };

        let s = self.find_min_containing_section(width, height)?;
        let (s, r) = s.resize_splitted_with_rest(width, height);
        if !r.is_empty() {
            self.sections.extend(r.into_iter());
            self.sections.sort();
        }

        Ok(s)
    }

    fn find_min_containing_section(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<TiledTexture2DSection, TexturePackingError> {
        let s = self
            .sections
            .iter()
            .enumerate()
            .find(|(_, s)| s.width >= width && s.height >= height);
        if let Some((i, s)) = s {
            let s = *s;
            self.sections.remove(i);
            Ok(s)
        } else {
            Err(TexturePackingError::NoSpaceLeft)
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn invalid_packer() {
        let packer = super::TiledTexture2DPacker::new(16, 0, 0);
        assert!(packer.is_err());
    }

    #[test]
    fn no_space_when_filled() {
        let mut packer = super::TiledTexture2DPacker::new(16, 32, 32).unwrap();
        assert!(packer.allocate(16, 16).is_ok());
        assert!(packer.allocate(16, 16).is_ok());
        assert!(packer.allocate(16, 16).is_ok());
        assert!(packer.allocate(16, 16).is_ok());

        assert!(packer.allocate(16, 16).is_err());
    }

    #[test]
    fn no_space_overflow() {
        let mut packer = super::TiledTexture2DPacker::new(16, 32, 32).unwrap();
        assert!(packer.allocate(16, 16).is_ok());
        assert!(packer.allocate(32, 16).is_ok());
        assert!(packer.allocate(32, 32).is_err());
    }
}

pub fn load_cube_to_resource_map(
    gpu: &dyn Gpu,
    resource_map: &mut ResourceMap,
) -> anyhow::Result<ResourceHandle<crate::Mesh>> {
    let mesh_create_info = crate::MeshCreateInfo {
        label: Some("cube"),
        primitives: &[MeshPrimitiveCreateInfo {
            indices: vec![
                0, 1, 2, 3, 1, 0, //Bottom
                6, 5, 4, 4, 5, 7, // Front
                10, 9, 8, 8, 9, 11, // Left
                12, 13, 14, 15, 13, 12, // Right
                16, 17, 18, 19, 17, 16, // Up
                22, 21, 20, 20, 21, 23, // Down
            ],
            positions: vec![
                // Back
                vector![-1.0, -1.0, 1.0],
                vector![1.0, 1.0, 1.0],
                vector![-1.0, 1.0, 1.0],
                vector![1.0, -1.0, 1.0],
                // Front
                vector![-1.0, -1.0, -1.0],
                vector![1.0, 1.0, -1.0],
                vector![-1.0, 1.0, -1.0],
                vector![1.0, -1.0, -1.0],
                // Left
                vector![1.0, -1.0, -1.0],
                vector![1.0, 1.0, 1.0],
                vector![1.0, 1.0, -1.0],
                vector![1.0, -1.0, 1.0],
                // Right
                vector![-1.0, -1.0, -1.0],
                vector![-1.0, 1.0, 1.0],
                vector![-1.0, 1.0, -1.0],
                vector![-1.0, -1.0, 1.0],
                // Up
                vector![-1.0, 1.0, -1.0],
                vector![1.0, 1.0, 1.0],
                vector![1.0, 1.0, -1.0],
                vector![-1.0, 1.0, 1.0],
                // Down
                vector![-1.0, -1.0, -1.0],
                vector![1.0, -1.0, 1.0],
                vector![1.0, -1.0, -1.0],
                vector![-1.0, -1.0, 1.0],
            ],
            colors: vec![],
            normals: vec![
                // Back
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                vector![0.0, 0.0, 1.0],
                // Front
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                vector![0.0, 0.0, -1.0],
                // Left
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
                // Right
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                vector![-1.0, 0.0, 0.0],
                // Up
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                vector![0.0, 1.0, 0.0],
                // Down
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
                vector![0.0, -1.0, 0.0],
            ],
            tangents: vec![],
            uvs: vec![
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
                vector![0.0, 0.0],
                vector![1.0, 1.0],
                vector![0.0, 1.0],
                vector![1.0, 0.0],
            ],
        }],
    };
    let cube_mesh = Mesh::new(gpu, &mesh_create_info)?;
    let cube_mesh_handle = resource_map.add(cube_mesh);
    Ok(cube_mesh_handle)
}
