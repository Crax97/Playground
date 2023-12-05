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

#[derive(Reflect, Clone, Copy, Default, Debug)]
pub struct CharacterState {
    #[reflect(ignore)]
    pub velocity: Vector2<f32>,
    pub grounded: bool,
    pub against_wall: bool,
    pub facing_forward_x: f32,
}

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct PlayerCharacter {
    pub player_speed: f32,
    pub jump_height: f32,
    pub min_angle_for_wall_degrees: f32,
    pub skin_size: f32,
    pub surface_check_distance: f32,

    enabled: bool,
    reset: bool,

    current_state: CharacterState,
    last_state: CharacterState,
}

impl PlayerCharacter {
    pub fn new(player_speed: f32, jump_height: f32) -> Self {
        Self {
            player_speed,
            jump_height,
            min_angle_for_wall_degrees: 45.0,
            skin_size: 0.01,
            surface_check_distance: 0.08,
            current_state: CharacterState::default(),
            last_state: CharacterState::default(),
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

    character.current_state.velocity.y += physics.gravity.y * time.delta_frame();
    let mut input_x = 0.0;
    if input.is_key_pressed(Key::Left) {
        input_x += 1.0;
    }
    if input.is_key_pressed(Key::Right) {
        input_x -= 1.0;
    }
    character.current_state.velocity.x = character.player_speed * time.delta_frame() * input_x;

    if input.is_key_just_pressed(Key::Space) && character.current_state.grounded {
        character.current_state.grounded = false;
        let grav = physics.gravity.magnitude();
        let jump_impulse = (2.0 * character.jump_height * grav).sqrt();
        character.current_state.velocity.y = jump_impulse;
    }
}

pub fn player_movement_system(
    mut query: Query<(&mut Transform2D, &mut PlayerCharacter, &Collider2DHandle)>,
    physics: Res<PhysicsContext2D>,
) {
    let (mut transform, mut player_character, player_collider) = query.single_mut();
    if player_character.reset {
        player_character.reset = false;
        player_character.current_state.velocity = [0.0, 0.0].into();
    }

    if !player_character.enabled {
        return;
    }
    let shape = physics
        .get_collider(player_collider)
        .unwrap()
        .shared_shape()
        .clone();

    player_character.last_state = player_character.current_state;
    player_character.current_state.grounded = false;
    player_character.current_state.against_wall = false;
    if player_character.current_state.velocity.x.abs() > 0.0 {
        player_character.current_state.facing_forward_x =
            player_character.current_state.velocity.x.signum();
    }
    move_and_slide(
        2,
        &mut player_character,
        &physics,
        &mut transform,
        &shape,
        player_collider,
    );

    let forward = vector![player_character.current_state.facing_forward_x, 0.0];
    let surface_check_distance = player_character.surface_check_distance;

    if let Some(_) = physics.cast_shape(
        Isometry2::translation(transform.position.x, transform.position.y),
        forward,
        shape.0.as_ref(),
        surface_check_distance,
        true,
        QueryFilter::new().exclude_collider(player_collider.as_ref().clone()),
    ) {
        player_character.current_state.against_wall = true;
    }
    if let Some(_) = physics.cast_shape(
        Isometry2::translation(transform.position.x, transform.position.y),
        vector![0.0, -1.0],
        shape.0.as_ref(),
        surface_check_distance,
        true,
        QueryFilter::new().exclude_collider(player_collider.as_ref().clone()),
    ) {
        player_character.current_state.grounded = true;
    }
}

fn move_and_slide(
    iterations: u32,
    player_character: &mut PlayerCharacter,
    physics: &PhysicsContext2D,
    transform: &mut Transform2D,
    collision_shape: &engine::physics::rapier2d::prelude::SharedShape,
    player_collider: &Collider2DHandle,
) {
    if player_character.reset {
        player_character.reset = false;
        player_character.current_state.velocity = [0.0, 0.0].into();
    }

    if !player_character.enabled {
        return;
    }
    let translation = player_character.current_state.velocity;
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
            player_character.current_state.velocity.y = 0.0;
        }

        let toi = (toi.toi - player_character.skin_size) / distance;

        let doable_translation = translation * toi;
        let remaining_translation = translation * (1.0 - toi);
        let remaining_translation = slide_vector(remaining_translation, normal);
        player_character.current_state.velocity = remaining_translation;
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
        transform.position += doable_translation;
    } else {
        transform.position += translation;
    }
}

fn slide_vector(vel: Vector2<f32>, n: Unit<Vector2<f32>>) -> Vector2<f32> {
    vel - vel.dot(&n) * *n
}
