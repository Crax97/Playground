use bevy_ecs::{
    component::Component,
    query::Changed,
    system::{Query, Res, ResMut, Resource},
};
use nalgebra::{Isometry2, Vector2};
pub use rapier2d;

use rapier2d::prelude::*;

use crate::components::Transform2D;

#[derive(Resource)]
pub struct PhysicsContext2D {
    pub gravity: Vector2<f32>,

    pipeline: PhysicsPipeline,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
}

#[derive(Component)]
pub struct Collider2DHandle(ColliderHandle);
impl AsRef<ColliderHandle> for Collider2DHandle {
    fn as_ref(&self) -> &ColliderHandle {
        &self.0
    }
}

#[derive(Component)]
pub struct RigidBody2DHandle(RigidBodyHandle);
impl AsRef<RigidBodyHandle> for RigidBody2DHandle {
    fn as_ref(&self) -> &RigidBodyHandle {
        &self.0
    }
}

impl PhysicsContext2D {
    pub fn new() -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            gravity: vector![0.0, -9.81],
        }
    }

    pub fn add_rigidbody(&mut self, body: RigidBody) -> RigidBody2DHandle {
        RigidBody2DHandle(self.rigid_body_set.insert(body))
    }

    pub fn remove_rigidbody(&mut self, handle: RigidBody2DHandle) -> Option<RigidBody> {
        self.rigid_body_set.remove(
            handle.0,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            false,
        )
    }

    pub fn add_collider(&mut self, collider: Collider) -> Collider2DHandle {
        Collider2DHandle(self.collider_set.insert(collider))
    }

    pub fn remove_collider(&mut self, handle: Collider2DHandle) -> Option<Collider> {
        self.collider_set.remove(
            handle.0,
            &mut self.island_manager,
            &mut self.rigid_body_set,
            false,
        )
    }

    pub fn update(&mut self) {
        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        )
    }
}

pub fn update_physics_2d_context(mut context: ResMut<PhysicsContext2D>) {
    context.update();
}

pub fn update_positions_before_physics_system(
    mut context: ResMut<PhysicsContext2D>,
    mut query: Query<(&Transform2D, &RigidBody2DHandle)>,
) {
    for (transform, body_handle) in query.iter_mut() {
        if let Some(body) = context.rigid_body_set.get_mut(body_handle.0) {
            body.set_position(
                Isometry2::new(transform.position.coords, transform.rotation),
                true,
            );
        }
    }
}
pub fn update_positions_after_physics_system(
    context: Res<PhysicsContext2D>,
    mut query: Query<(&mut Transform2D, &RigidBody2DHandle)>,
) {
    for (mut transform, body_handle) in query.iter_mut() {
        if let Some(body) = context.rigid_body_set.get(body_handle.0) {
            transform.position = point![
                body.position().translation.vector.x,
                body.position().translation.vector.y
            ];
            transform.rotation = body.position().rotation.angle();
        }
    }
}
