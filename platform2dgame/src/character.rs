use engine::bevy_ecs::component::Component;
use engine::bevy_ecs::query::With;
use engine::bevy_ecs::reflect::ReflectComponent;
use engine::bevy_ecs::system::{Query, Res};
use engine::bevy_reflect;
use engine::bevy_reflect::Reflect;
use engine::components::Transform2D;
use engine::input::{InputState, Key};
use engine::physics::rapier2d::parry::query::TOIStatus;
use engine::physics::rapier2d::pipeline::QueryFilter;
use engine::physics::{Collider2DHandle, PhysicsContext2D};
use engine::{bevy_ecs, Time};
use nalgebra::{vector, Isometry2, Unit, Vector2};

use crate::Player;

#[derive(Reflect, Clone, Copy)]
pub struct CharacterState {}

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct PlayerCharacter {
    pub player_speed: f32,
    pub jump_height: f32,
    pub min_angle_for_wall_degrees: f32,
    #[reflect(ignore)]
    pub velocity: Vector2<f32>,
    pub skin_size: f32,

    is_in_air: bool,
    is_grounded: bool,
    enabled: bool,
    reset: bool,
}

impl PlayerCharacter {
    pub fn new(player_speed: f32, jump_height: f32) -> Self {
        Self {
            player_speed,
            jump_height,
            velocity: Vector2::zeros(),
            min_angle_for_wall_degrees: 45.0,
            skin_size: 0.01,
            is_in_air: true,
            is_grounded: false,
            enabled: true,
            reset: false,
        }
    }
}

pub fn player_input_system(
    mut query: Query<(&mut PlayerCharacter, With<Player>)>,
    input: Res<InputState>,
    physics: Res<PhysicsContext2D>,
    time: Res<Time>,
) {
    let (mut character, _) = query.single_mut();

    character.velocity.y += 0.5 * physics.gravity.y * time.delta_frame();
    let mut input_x = 0.0;
    if input.is_key_pressed(Key::Left) {
        input_x += 1.0;
    }
    if input.is_key_pressed(Key::Right) {
        input_x -= 1.0;
    }
    character.velocity.x = character.player_speed * time.delta_frame() * input_x;

    if input.is_key_just_pressed(Key::Space) && character.is_grounded {
        character.is_in_air = true;
        character.is_grounded = false;
        let grav = physics.gravity.magnitude();
        let jump_impulse = (2.0 * character.jump_height * grav).sqrt();
        character.velocity.y = jump_impulse;
    }
}

pub fn player_movement_system(
    mut query: Query<(&mut Transform2D, &mut PlayerCharacter, &Collider2DHandle)>,
    physics: Res<PhysicsContext2D>,
    time: Res<Time>,
) {
    let (mut transform, mut player_character, player_collider) = query.single_mut();
    if player_character.reset {
        player_character.reset = false;
        player_character.velocity = [0.0, 0.0].into();
    }

    if !player_character.enabled {
        return;
    }
    let shape = physics
        .get_collider(player_collider)
        .unwrap()
        .shared_shape()
        .clone();

    let velocity = player_character.velocity;
    move_and_slide(
        2,
        &mut player_character,
        velocity,
        &physics,
        &mut transform,
        &shape,
        player_collider,
    );
}

fn move_and_slide(
    iterations: u32,
    player_character: &mut PlayerCharacter,
    velocity: Vector2<f32>,
    physics: &PhysicsContext2D,
    transform: &mut Transform2D,
    collision_shape: &engine::physics::rapier2d::prelude::SharedShape,
    player_collider: &Collider2DHandle,
) {
    if player_character.reset {
        player_character.reset = false;
        player_character.velocity = [0.0, 0.0].into();
    }

    if !player_character.enabled {
        return;
    }
    let translation = velocity;
    if translation.magnitude_squared() <= 0.025 {
        return;
    }
    let distance = translation.magnitude();

    if let Some((_, toi)) = physics.cast_shape(
        Isometry2::translation(transform.position.x, transform.position.y),
        translation.normalize(),
        collision_shape.0.as_ref(),
        distance,
        true,
        QueryFilter::new().exclude_collider(player_collider.as_ref().clone()),
    ) {
        if toi.status == TOIStatus::Penetrating {
            println!("Penetrating");
        }
        let normal = toi.normal1;

        let surface_angle = vector![0.0, 1.0].angle(&*normal);
        if surface_angle.to_degrees() < player_character.min_angle_for_wall_degrees {
            player_character.is_grounded = true;
            player_character.velocity.y = 0.0;
        }

        let toi = (toi.toi - player_character.skin_size) / distance;

        let doable_translation = translation * toi;
        let remaining_translation = translation * (1.0 - toi);
        let remaining_translation = slide_vector(remaining_translation, normal);
        if iterations > 0 {
            move_and_slide(
                iterations - 1,
                player_character,
                remaining_translation,
                physics,
                transform,
                collision_shape,
                player_collider,
            );
        }
        transform.position += doable_translation;
    } else {
        transform.position += translation;
    }
}

fn slide_vector(vel: Vector2<f32>, n: Unit<Vector2<f32>>) -> Vector2<f32> {
    vel - vel.dot(&n) * *n
}
