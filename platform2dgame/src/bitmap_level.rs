use engine::{Resource, ResourceLoader};
use image::Rgb;

pub struct BitmapLevelLoader;

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Debug)]
pub enum EntityType {
    Player,
    Enemy,
    Terrain,
    Grass,
    Star,
    Platform,
}

#[derive(Clone, Copy, Debug)]
pub struct Entity {
    pub x: f32,
    pub y: f32,
    pub ty: EntityType,
}

#[derive(Default)]
pub struct BitmapLevel {
    pub entities: Vec<Entity>,
}

impl Resource for BitmapLevel {
    fn get_description(&self) -> &str {
        "Bitmap level"
    }
}

impl ResourceLoader for BitmapLevelLoader {
    type LoadedResource = BitmapLevel;

    fn load(
        &self,
        _gpu: &gpu::VkGpu,
        path: &std::path::Path,
    ) -> anyhow::Result<Self::LoadedResource> {
        const TERRAIN: Rgb<u8> = Rgb([142, 65, 11]);
        const GRASS: Rgb<u8> = Rgb([108, 201, 85]);
        const PLAYER: Rgb<u8> = Rgb([255, 40, 60]);
        const STAR: Rgb<u8> = Rgb([221, 247, 49]);
        const PLATFORM: Rgb<u8> = Rgb([223, 113, 38]);
        const ENEMY: Rgb<u8> = Rgb([143, 179, 255]);
        let mut level = BitmapLevel::default();
        let file = std::fs::read(path)?;
        let bitmap_image = image::load_from_memory(&file)?;
        let bitmap_image = bitmap_image.as_rgb8().unwrap();
        for (y, row) in bitmap_image.rows().enumerate() {
            for (x, pixel) in row.enumerate() {
                let ty = match *pixel {
                    TERRAIN => EntityType::Terrain,
                    GRASS => EntityType::Grass,
                    PLAYER => EntityType::Player,
                    STAR => EntityType::Star,
                    PLATFORM => EntityType::Platform,
                    ENEMY => EntityType::Enemy,
                    _ => {
                        continue;
                    }
                };

                level.entities.push(Entity {
                    x: (bitmap_image.width() as f32 - x as f32) * 8.0,
                    y: y as f32 * -8.0,
                    ty,
                });
            }
        }
        Ok(level)
    }
}
