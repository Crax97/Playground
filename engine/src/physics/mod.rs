use bevy_ecs::{
    component::Component,
    system::{Query, Res, ResMut, Resource},
};
use nalgebra::{Isometry2, Point2, UnitVector2, Vector2};
pub use rapier2d;

use rapier2d::prelude::*;

#[derive(Resource)]
pub struct PhysicsContext2D {
    pub gravity: Vector2<f32>,

    pub(crate) pipeline: PhysicsPipeline,
    pub(crate) rigid_body_set: RigidBodySet,
    pub(crate) collider_set: ColliderSet,
    pub(crate) integration_parameters: IntegrationParameters,
    pub(crate) island_manager: IslandManager,
    pub(crate) broad_phase: BroadPhase,
    pub(crate) narrow_phase: NarrowPhase,
    pub(crate) impulse_joint_set: ImpulseJointSet,
    pub(crate) multibody_joint_set: MultibodyJointSet,
    pub(crate) ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
}

#[derive(Component, Clone, Copy, Eq, PartialEq)]
pub struct Collider2DHandle(pub(crate) ColliderHandle);
impl AsRef<ColliderHandle> for Collider2DHandle {
    fn as_ref(&self) -> &ColliderHandle {
        &self.0
    }
}

#[derive(Component, Clone, Copy)]
pub struct RigidBody2DHandle(pub(crate) RigidBodyHandle);
impl AsRef<RigidBodyHandle> for RigidBody2DHandle {
    fn as_ref(&self) -> &RigidBodyHandle {
        &self.0
    }
}

impl PhysicsContext2D {
    pub fn new() -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            query_pipeline: QueryPipeline::new(),
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

    pub fn get_collider(&self, handle: &Collider2DHandle) -> Option<&Collider> {
        self.collider_set.get(handle.0)
    }

    pub fn get_rigidbody_mut(&mut self, body: &RigidBody2DHandle) -> Option<&mut RigidBody> {
        self.rigid_body_set.get_mut(body.0)
    }

    pub fn query_pipeline(&self) -> &QueryPipeline {
        &self.query_pipeline
    }

    pub fn cast_ray(
        &self,
        origin: Point2<f32>,
        direction: UnitVector2<f32>,
        length: f32,
        stop_at_penetration: bool,
        filter: QueryFilter,
    ) -> Option<(Collider2DHandle, f32)> {
        let ray = Ray::new(origin, *direction);
        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                length,
                stop_at_penetration,
                filter,
            )
            .map(|(h, d)| (Collider2DHandle(h), d))
    }

    pub fn cast_shape(
        &self,
        shape_pos: Isometry2<f32>,
        shape_vel: Vector2<f32>,
        shape: &dyn Shape,
        max_toi: f32,
        stop_at_penetration: bool,
        filter: QueryFilter,
    ) -> Option<(Collider2DHandle, TOI)> {
        if let Some((handle, toi)) = self.query_pipeline.cast_shape(
            &self.rigid_body_set,
            &self.collider_set,
            &shape_pos,
            &shape_vel,
            shape,
            max_toi,
            stop_at_penetration,
            filter,
        ) {
            Some((Collider2DHandle(handle), toi))
        } else {
            None
        }
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

    pub fn are_colliding(
        &self,
        collider_a: &Collider2DHandle,
        collider_b: &Collider2DHandle,
    ) -> bool {
        if let Some(true) = self
            .narrow_phase
            .intersection_pair(collider_a.0, collider_b.0)
        {
            return true;
        }

        if let Some(collision) = self.narrow_phase.contact_pair(collider_a.0, collider_b.0) {
            return collision.has_any_active_contact;
        }

        false
    }

    pub fn add_collider_with_parent(
        &mut self,
        collider: Collider,
        parent: RigidBody2DHandle,
    ) -> Collider2DHandle {
        Collider2DHandle(self.collider_set.insert_with_parent(
            collider,
            parent.0,
            &mut self.rigid_body_set,
        ))
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
        let (collision_send, collision_recv) = crossbeam::channel::unbounded();
        let (contact_force_send, contact_force_recv) = crossbeam::channel::unbounded();
        let evt_handler = ChannelEventCollector::new(collision_send, contact_force_send);
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
            &evt_handler,
        );
        self.query_pipeline
            .update(&self.rigid_body_set, &self.collider_set);
        while let Ok(collision_event) = collision_recv.try_recv() {
            // Handle the collision event.
            println!("Received collision event: {:?}", collision_event);
        }

        while let Ok(contact_force_event) = contact_force_recv.try_recv() {
            // Handle the contact force event.
            println!("Received contact force event: {:?}", contact_force_event);
        }
    }
}

impl Default for PhysicsContext2D {
    fn default() -> Self {
        Self::new()
    }
}

pub fn update_physics_2d_context(mut context: ResMut<PhysicsContext2D>) {
    context.update();
}
