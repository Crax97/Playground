use engine::{bevy_ecs, GpuDevice};
use engine::{
    bevy_ecs::{
        component::Component,
        system::{Commands, Res, ResMut},
    },
    components::{DebugName, SpriteComponent, SpriteComponentDescription, Transform2D},
    physics::{
        rapier2d::{
            dynamics::RigidBodyBuilder,
            geometry::{ColliderBuilder, SharedShape},
            math::Isometry,
            pipeline::ActiveEvents,
        },
        PhysicsContext2D,
    },
    CommonResources, ResourceMap, Texture,
};
use gpu::Filter;
use nalgebra::{point, vector};

use crate::{
    bitmap_level::{self, BitmapLevel, Entity},
    character::KinematicCharacter,
    Enemy, Player,
};

#[derive(Component)]
pub struct Star;

const SPRITE_SIZE: u32 = 8;
pub fn load_level_system(
    gpu: Res<GpuDevice>,
    mut resource_map: ResMut<ResourceMap>,
    mut physics_context: ResMut<PhysicsContext2D>,
    common_resources: Res<CommonResources>,
    mut commands: Commands,
) {
    physics_context.gravity = [0.0, -15.0].into();
    let level = resource_map
        .load::<BitmapLevel>("images/levels/test_level.bmp")
        .unwrap();

    let entities = resource_map.load("images/sprites/entities.png").unwrap();

    resource_map
        .get_mut::<Texture>(&entities)
        .sampler_settings
        .min_filter = Filter::Nearest;
    resource_map
        .get_mut::<Texture>(&entities)
        .sampler_settings
        .mag_filter = Filter::Nearest;

    let level = resource_map.get(&level);

    for entity in &level.entities {
        let entity_spawned = commands.spawn((Transform2D {
            position: point![entity.x, entity.y],
            layer: 0,
            rotation: 0.0,
            scale: vector![1.0, 1.0],
        },));
        match entity.ty {
            bitmap_level::EntityType::Player => {
                spawn_player(
                    &gpu,
                    entity,
                    &common_resources,
                    entities.clone(),
                    entity_spawned,
                    physics_context.as_mut(),
                );
            }
            bitmap_level::EntityType::Enemy => {
                spawn_enemy(
                    &gpu,
                    entity,
                    &common_resources,
                    entities.clone(),
                    entity_spawned,
                    &mut physics_context,
                );
            }
            bitmap_level::EntityType::Terrain => spawn_terrain(
                &gpu,
                entity,
                &common_resources,
                entities.clone(),
                entity_spawned,
                &mut physics_context,
            ),
            bitmap_level::EntityType::Grass => {
                spawn_grass(&gpu, &common_resources, entities.clone(), entity_spawned)
            }
            bitmap_level::EntityType::Star => spawn_star(
                &gpu,
                entity,
                &common_resources,
                entities.clone(),
                entity_spawned,
                &mut physics_context,
            ),
            bitmap_level::EntityType::Platform => spawn_platform(
                &gpu,
                entity,
                &common_resources,
                entities.clone(),
                entity_spawned,
                &mut physics_context,
            ),
        };
    }
}

fn make_collider() -> ColliderBuilder {
    ColliderBuilder::new(SharedShape::cuboid(
        SPRITE_SIZE as f32 * 0.5,
        SPRITE_SIZE as f32 * 0.5,
    ))
}

fn spawn_player(
    gpu: &GpuDevice,
    entity: &Entity,
    common_resources: &CommonResources,
    entities_texture: engine::ResourceHandle<Texture>,
    mut entity_spawned: engine::bevy_ecs::system::EntityCommands<'_, '_, '_>,
    physics_context: &mut PhysicsContext2D,
) {
    let collider = ColliderBuilder::new(SharedShape::ball(SPRITE_SIZE as f32 * 0.5))
        .sensor(true)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    let rigid_body = RigidBodyBuilder::kinematic_position_based()
        .translation([entity.x, entity.y].into())
        .additional_mass(10.0)
        .build();
    let rigid_body = physics_context.add_rigidbody(rigid_body);
    let collider = physics_context.add_collider_with_parent(collider, rigid_body);
    entity_spawned.insert((
        SpriteComponent::new(
            gpu,
            SpriteComponentDescription {
                texture: entities_texture,
                material: common_resources.default_sprite_material.clone(),
                atlas_offset: [SPRITE_SIZE * 2, 0].into(),
                atlas_size: [SPRITE_SIZE, SPRITE_SIZE].into(),
                sprite_size: [SPRITE_SIZE as f32, SPRITE_SIZE as f32].into(),
                z_layer: 0,
            },
        ),
        rigid_body,
        collider,
        Player,
        KinematicCharacter::new(50.0, 0.25),
        DebugName("Player".to_owned()),
    ));
}

fn spawn_enemy(
    gpu: &GpuDevice,
    entity: &Entity,
    common_resources: &CommonResources,
    entities_texture: engine::ResourceHandle<Texture>,
    mut entity_spawned: engine::bevy_ecs::system::EntityCommands<'_, '_, '_>,
    physics_context: &mut PhysicsContext2D,
) {
    let collider = ColliderBuilder::new(SharedShape::ball(SPRITE_SIZE as f32 * 0.5)).build();
    let rigid_body = RigidBodyBuilder::kinematic_position_based()
        .translation([entity.x, entity.y].into())
        .additional_mass(10.0)
        .build();
    let rigid_body = physics_context.add_rigidbody(rigid_body);
    let collider = physics_context.add_collider_with_parent(collider, rigid_body);
    entity_spawned.insert((
        SpriteComponent::new(
            gpu,
            SpriteComponentDescription {
                texture: entities_texture,
                material: common_resources.default_sprite_material.clone(),
                atlas_offset: [SPRITE_SIZE * 2, 0].into(),
                atlas_size: [SPRITE_SIZE, SPRITE_SIZE].into(),
                sprite_size: [SPRITE_SIZE as f32, SPRITE_SIZE as f32].into(),
                z_layer: 0,
            },
        ),
        rigid_body,
        collider,
        Enemy,
        KinematicCharacter::new(50.0, 0.25),
        DebugName("Enemy".to_owned()),
    ));
}

fn spawn_terrain(
    gpu: &GpuDevice,
    entity: &Entity,
    common_resources: &CommonResources,
    entities_texture: engine::ResourceHandle<Texture>,
    mut entity_spawned: engine::bevy_ecs::system::EntityCommands<'_, '_, '_>,
    physics_context: &mut PhysicsContext2D,
) {
    let collider = make_collider()
        .position(Isometry::translation(entity.x, entity.y))
        .build();
    let collider = physics_context.add_collider(collider);
    entity_spawned.insert((
        SpriteComponent::new(
            gpu,
            SpriteComponentDescription {
                texture: entities_texture,
                material: common_resources.default_sprite_material.clone(),
                atlas_offset: [0, 0].into(),
                atlas_size: [SPRITE_SIZE, SPRITE_SIZE].into(),
                sprite_size: [SPRITE_SIZE as f32, SPRITE_SIZE as f32].into(),
                z_layer: 0,
            },
        ),
        collider,
    ));
}

fn spawn_grass(
    gpu: &GpuDevice,
    common_resources: &CommonResources,
    entities_texture: engine::ResourceHandle<Texture>,
    mut entity_spawned: engine::bevy_ecs::system::EntityCommands<'_, '_, '_>,
) {
    entity_spawned.insert((SpriteComponent::new(
        gpu,
        SpriteComponentDescription {
            texture: entities_texture,
            material: common_resources.default_sprite_material.clone(),
            atlas_offset: [SPRITE_SIZE, 0].into(),
            atlas_size: [SPRITE_SIZE, SPRITE_SIZE].into(),
            sprite_size: [SPRITE_SIZE as f32, SPRITE_SIZE as f32].into(),
            z_layer: 0,
        },
    ),));
}

fn spawn_star(
    gpu: &GpuDevice,
    entity: &Entity,
    common_resources: &CommonResources,
    entities_texture: engine::ResourceHandle<Texture>,
    mut entity_spawned: engine::bevy_ecs::system::EntityCommands<'_, '_, '_>,
    physics_context: &mut PhysicsContext2D,
) {
    let collider = make_collider()
        .position(Isometry::translation(entity.x, entity.y))
        .sensor(true)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    let collider = physics_context.add_collider(collider);
    entity_spawned.insert((
        SpriteComponent::new(
            gpu,
            SpriteComponentDescription {
                texture: entities_texture,
                material: common_resources.default_sprite_material.clone(),
                atlas_offset: [SPRITE_SIZE * 3, 0].into(),
                atlas_size: [SPRITE_SIZE, SPRITE_SIZE].into(),
                sprite_size: [SPRITE_SIZE as f32, SPRITE_SIZE as f32].into(),
                z_layer: 0,
            },
        ),
        collider,
        Star,
        DebugName("Star".to_owned()),
    ));
}

fn spawn_platform(
    gpu: &GpuDevice,
    entity: &Entity,
    common_resources: &CommonResources,
    entities_texture: engine::ResourceHandle<Texture>,
    mut entity_spawned: engine::bevy_ecs::system::EntityCommands<'_, '_, '_>,
    physics_context: &mut PhysicsContext2D,
) {
    let collider = make_collider()
        .position(Isometry::translation(entity.x, entity.y))
        .build();
    let collider = physics_context.add_collider(collider);
    entity_spawned.insert((
        SpriteComponent::new(
            gpu,
            SpriteComponentDescription {
                texture: entities_texture,
                material: common_resources.default_sprite_material.clone(),
                atlas_offset: [SPRITE_SIZE * 5, 0].into(),
                atlas_size: [SPRITE_SIZE, SPRITE_SIZE].into(),
                sprite_size: [SPRITE_SIZE as f32, SPRITE_SIZE as f32].into(),
                z_layer: 0,
            },
        ),
        collider,
    ));
}
