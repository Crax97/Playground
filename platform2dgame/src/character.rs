use engine::bevy_ecs::component::Component;
use engine::bevy_ecs::query::With;
use engine::bevy_ecs::system::{Query, Res};
use engine::components::Transform2D;
use engine::input::{InputState, Key};
use engine::physics::rapier2d::pipeline::QueryFilter;
use engine::physics::{Collider2DHandle, PhysicsContext2D};
use engine::{bevy_ecs, Time};
use nalgebra::{Isometry2, Unit, Vector2};

use crate::Player;

#[derive(Component, Default)]
pub struct PlayerCharacter {
    pub player_speed: f32,
    pub player_jump_impulse: f32,
    pub velocity: Vector2<f32>,
    pub skin_size: f32,

    is_in_air: bool,
    is_grounded: bool,
}

impl PlayerCharacter {
    pub fn new(player_speed: f32, player_jump_impulse: f32) -> Self {
        Self {
            player_speed,
            player_jump_impulse,
            velocity: Vector2::zeros(),
            skin_size: 0.05,
            is_in_air: true,
            is_grounded: false,
        }
    }
}

pub fn player_input_system(
    mut query: Query<(&mut PlayerCharacter, With<Player>)>,
    input: Res<InputState>,
    time: Res<Time>,
) {
    let (mut character, _) = query.single_mut();

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
        character.velocity.y = character.player_jump_impulse;
    }
}

pub fn player_movement_system(
    mut query: Query<(&mut Transform2D, &mut PlayerCharacter, &Collider2DHandle)>,
    physics: Res<PhysicsContext2D>,
    time: Res<Time>,
) {
    let (mut transform, mut player_character, player_collider) = query.single_mut();

    let shape = physics
        .get_collider(player_collider)
        .unwrap()
        .shared_shape()
        .clone();

    if !player_character.is_grounded {
        player_character.velocity.y += 0.5 * physics.gravity.y * time.delta_frame().powf(2.0);
    }

    player_character.is_grounded = false;
    player_character.is_in_air = true;
    move_and_slide(
        2,
        &mut player_character,
        &physics,
        &mut transform,
        shape,
        player_collider,
    );
}

fn move_and_slide(
    iterations: u32,
    player_character: &mut PlayerCharacter,
    physics: &PhysicsContext2D,
    transform: &mut Transform2D,
    collision_shape: engine::physics::rapier2d::prelude::SharedShape,
    player_collider: &Collider2DHandle,
) {
    let mut translation = player_character.velocity;

    if let Some((_, toi)) = physics.cast_shape(
        Isometry2::translation(transform.position.x, transform.position.y),
        translation.normalize(),
        collision_shape.0.as_ref(),
        1.0f32.min(translation.magnitude()),
        true,
        QueryFilter::new().exclude_collider(player_collider.as_ref().clone()),
    ) {
        if player_character.velocity.y < 0.0 {
            player_character.is_grounded = true;
            player_character.velocity.y = 0.0;
        }

        let normal = toi.normal1;
        let toi = toi.toi - player_character.skin_size;
        let remaining_velocity = translation * (1.0 - toi);
        let remaining_velocity = slide_vector(remaining_velocity, normal);
        player_character.velocity = remaining_velocity;
        if iterations > 0 {
            move_and_slide(
                iterations - 1,
                player_character,
                physics,
                transform,
                collision_shape,
                player_collider,
            );
        }
        translation = translation * toi;
    }
    transform.position += translation;
}

fn slide_vector(vel: Vector2<f32>, n: Unit<Vector2<f32>>) -> Vector2<f32> {
    vel - vel.dot(&n) * *n
}
