use crate::input::{Axis, Input, MouseAxis};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use vek::Clamp;
use winit::keyboard::KeyCode;
use rapier3d::{control::{CharacterLength, DynamicRayCastVehicleController, KinematicCharacterController, WheelTuning}, prelude::*};


#[derive(Default)]
pub struct Player {
    pub position: vek::Vec3<f32>,
    pub last_position: vek::Vec3<f32>,
    pub interpolated_position: vek::Vec3<f32>,
    
    pub body_rotation: vek::Quaternion<f32>,
    pub head_rotation_local: vek::Quaternion<f32>,
    pub head_rotation_global: vek::Quaternion<f32>,
    
    summed_mouse: vek::Vec2<f32>,
    
    handle: RigidBodyHandle,
    character: KinematicCharacterController,
    grounded_last_frame: bool,

    local_velocity: vek::Vec2<f32>,
    jump_bufferred: bool,
}

impl Player {
    pub fn new(physics_world: &mut rapier3d::prelude::PhysicsWorld) -> Self {        
        let (ball_body_handle, _) = physics_world.insert(
            RigidBodyBuilder::kinematic_velocity_based().translation(Vector::new(0.0, 95.0, 0.0)),
            ColliderBuilder::ball(2.0).restitution(0.3),
        );
        let mut character = KinematicCharacterController::default();
        character.offset = CharacterLength::Absolute(0.05);
        
        Self {
            handle: ball_body_handle,
            character,
            ..Default::default()
        }
    }

    pub fn update_input(&mut self, input: &Input) {
        self.local_velocity = vek::Vec2::<f32>::zero();

        let sprinting = input.get_button(KeyCode::ShiftLeft).held();
        let speed = if sprinting { 50.0 } else { 2.0 };

        self.local_velocity.y = input.create_axis(KeyCode::KeyW, KeyCode::KeyS);
        self.local_velocity.x = input.create_axis(KeyCode::KeyA, KeyCode::KeyD);

        let sens = 1.0f32;
        self.summed_mouse += vek::Vec2::new(
            input.get_axis(Axis::Mouse(MouseAxis::DeltaX)) * 0.003 * sens,
            input.get_axis(Axis::Mouse(MouseAxis::DeltaY)) * -0.003 * sens,
        );
        self.summed_mouse.y = (self.summed_mouse.y).clamp(-f32::pi() * 0.5f32, f32::pi() * 0.5f32);
        self.head_rotation_local = vek::Quaternion::rotation_x(self.summed_mouse.y);
        self.body_rotation = vek::Quaternion::rotation_y(self.summed_mouse.x);
        self.head_rotation_global = self.body_rotation * self.head_rotation_local;

        self.local_velocity = self.local_velocity * speed;
    }

    pub fn update_physics(&mut self, physics_world: &mut rapier3d::prelude::PhysicsWorld, physics_delta: f32) {
        self.last_position = self.position;

        let rot = vek::Mat4::from(self.body_rotation);
        let forward = rot.mul_direction(-vek::Vec3::unit_z());
        let right = rot.mul_direction(vek::Vec3::unit_x());
        
        // velocity to apply this physics tick
        let mut velocity = (forward * self.local_velocity.y + right * self.local_velocity.x);
        // let body = physics_world.bodies.get(self.handle).unwrap();
        // let pose = body.position();
        // let oldvel = body.linvel();

        // apply gravity
        if !self.grounded_last_frame {
            velocity.y -= 20.5f32;
        }

        // jump
        if self.jump_bufferred {
            if self.grounded_last_frame {
                velocity.y = 80f32;
            }
            self.jump_bufferred = false;
        }

        
        let shape = rapier3d::parry::shape::Capsule::new(glam::Vec3::ZERO, glam::Vec3::Y, 0.2);
        let filter = QueryFilter::default()
            // Make sure the character we are trying to move isn’t considered an obstacle.
            .exclude_rigid_body(self.handle);

        let pose = physics_world.bodies.get(self.handle).unwrap().position();

        let mut collisions = vec![];
        let corrected = self.character.move_shape(
            physics_delta,
            &&physics_world.query_pipeline_with_filter(filter),
            &shape,
            pose,
            glam::vec3(velocity.x, velocity.y, velocity.z) * physics_delta,
            |collision| collisions.push(collision),
        );

        // apply translation by updating velocity
        let body = physics_world.bodies.get_mut(self.handle).unwrap();
        body.set_linvel(corrected.translation / physics_delta, true);

        // update stored position 
        let v1 = body.position().translation;
        let v2 = vek::Vec3::<f32>::new(v1.x, v1.y, v1.z);
        self.position = v2;

        self.grounded_last_frame = corrected.grounded;
    }

    pub fn update_frame(&mut self, interpolation_factor: f32) {
        self.interpolated_position = vek::Vec3::lerp(self.last_position, self.position, interpolation_factor);
    }
}
