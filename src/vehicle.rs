use crate::input::{Axis, Input, MouseAxis};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use vek::Clamp;
use winit::keyboard::KeyCode;
use rapier3d::{control::{CharacterLength, DynamicRayCastVehicleController, KinematicCharacterController, Wheel, WheelTuning}, prelude::*};

#[derive(Clone, Copy)]
pub struct WheelState {
    pub position: vek::Vec3<f32>,
    pub rotation: f32,
}

impl WheelState {
    fn interpolate(a: &WheelState, b: &WheelState, t: f32) -> WheelState {
        WheelState { position: vek::Vec3::<f32>::lerp(a.position, b.position, t), rotation: vek::Lerp::lerp(a.rotation, b.rotation, t) }
    }
}

#[derive(Clone, Default)]
pub struct VehicleState {
    pub position: vek::Vec3<f32>,
    pub rotation: vek::Quaternion<f32>,
    pub wheels: Vec<WheelState>,
}

impl VehicleState {
    fn interpolate(a: &VehicleState, b: &VehicleState, t: f32) -> VehicleState {
        VehicleState {
            position: vek::Vec3::<f32>::lerp(a.position, b.position, t), rotation: vek::Quaternion::<f32>::slerp(a.rotation, b.rotation, t),
            wheels: a.wheels.iter().zip(b.wheels.iter()).map(|(a, b)| WheelState::interpolate(a, b, t)).collect::<Vec<_>>()
        }
    }
}

#[derive(Default)]
pub struct Vehicle {
    vehicle: Option<DynamicRayCastVehicleController>,
    summed_mouse: vek::Vec2<f32>,
    pub camera_position: vek::Vec3<f32>,

    pub last_state: VehicleState,
    pub current_state: VehicleState,
    pub interpolated_state: VehicleState,
    
    
    local_velocity: vek::Vec2<f32>,
    pub head_rotation_global: vek::Quaternion<f32>,
    camera_position_offset: vek::Vec3<f32>,
    reset_pending: bool,
    camera_orbit_distance: f32,
}

impl Vehicle {
    pub fn new(physics_world: &mut rapier3d::prelude::PhysicsWorld) -> Self {       
        let hw = 1.4;
        let hh = 0.25;

        // https://rapier.rs/docs/user_guides/rust/character_controller/#vehicle-controller
        let (chassis_handle, _) = physics_world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 35.0, 0.0)),
            ColliderBuilder::cuboid(hw * 2.0 - 0.5, hh, hw - 0.5).density(100.0),
        );
                        
        // The tuning shared by the wheels: the suspension and the grip.
        let tuning = WheelTuning {
            suspension_stiffness: 50.0,
            suspension_damping: 2.0,
            ..WheelTuning::default()
        };      

        let mut vehicle = DynamicRayCastVehicleController::new(chassis_handle);
        let wheel_positions = [
            Vector::new(hw * 1.5, -hh, hw),
            Vector::new(hw * 1.5, -hh, -hw),
            Vector::new(-hw * 1.5, -hh, hw),
            Vector::new(-hw * 1.5, -hh, -hw),
        ];      

        let wheel_radius = 0.5;
        for position in wheel_positions {
            // The position of the wheel, the direction its suspension pushes along, its axle, the
            // rest length of its suspension, and its radius; all in the local frame of the chassis.
            vehicle.add_wheel(position, -Vector::Y, Vector::Z, hh, wheel_radius, &tuning);
        }
        
        Self {
            vehicle: Some(vehicle),
            camera_orbit_distance: 15f32,
            ..Default::default()
        }
    }

    pub fn update_input(&mut self, input: &Input) {
        self.local_velocity = vek::Vec2::<f32>::zero();

        /*
        let sprinting = input.get_button(KeyCode::ShiftLeft).held();
        let speed = if sprinting { 50.0 } else { 2.0 };
        */
        self.local_velocity.y = input.create_axis(KeyCode::KeyW, KeyCode::KeyS);
        self.local_velocity.x = input.create_axis(KeyCode::KeyA, KeyCode::KeyD);

        if input.get_button(KeyCode::KeyL).pressed() {
            self.reset_pending = true;
        }

        
        let sens = 1.0f32;
        self.summed_mouse += vek::Vec2::new(
            input.get_axis(Axis::Mouse(MouseAxis::DeltaX)) * 0.003 * sens,
            input.get_axis(Axis::Mouse(MouseAxis::DeltaY)) * -0.003 * sens,
        );
        self.summed_mouse.y = (self.summed_mouse.y).clamp(-f32::pi() * 0.5f32, f32::pi() * 0.5f32);
        let head_rotation_local = vek::Quaternion::rotation_x(-self.summed_mouse.y);
        let body_rotation = vek::Quaternion::rotation_y(-self.summed_mouse.x);
        self.head_rotation_global = body_rotation * head_rotation_local;

        self.camera_orbit_distance += input.get_axis(Axis::Mouse(MouseAxis::ScrollDelta));
        self.camera_orbit_distance = self.camera_orbit_distance.clamp(5f32, 500f32);

        let rot = vek::Mat4::from(self.head_rotation_global);  
        self.camera_position_offset = rot.mul_direction(vek::Vec3::unit_z()) * self.camera_orbit_distance.powf(2f32);
    }

    pub fn update_physics(&mut self, physics_world: &mut rapier3d::prelude::PhysicsWorld, physics_delta: f32) {
        let vehicle = self.vehicle.as_mut().unwrap();
        // The vehicle is driven by setting the engine force, the brake, and the steering angle of
        // its wheels. Here the two front wheels are the driving and steering ones.
        let wheels = vehicle.wheels_mut();

        let force = 2500f32;
        let braking_force = 100f32;
        
        let steering_strength = -0.4f32;

        if self.local_velocity.y > 0f32 {
            wheels[0].engine_force = force;
            wheels[1].engine_force = force;
            wheels[0].brake = 0f32;
            wheels[1].brake = 0f32;
        } else if self.local_velocity.y < 0f32 {
            wheels[0].engine_force = 0f32;
            wheels[1].engine_force = 0f32;
            wheels[0].brake = braking_force;
            wheels[1].brake = braking_force;
        } else {
            wheels[0].engine_force = 0f32;
            wheels[1].engine_force = 0f32;
            wheels[0].brake = 0f32;
            wheels[1].brake = 0f32;
        }
        
        
        wheels[0].steering = self.local_velocity.x * steering_strength;
        wheels[1].steering = self.local_velocity.x * steering_strength;

        // The wheels are ray-casted against the scene: the chassis itself, as well as every other
        // dynamic body, is generally excluded from these ray-casts.
        let queries = physics_world.broad_phase.as_query_pipeline_mut(
            physics_world.narrow_phase.query_dispatcher(),
            &mut physics_world.bodies,
            &mut physics_world.colliders,
            QueryFilter::exclude_dynamic().exclude_rigid_body(vehicle.chassis.clone()),
        );
        vehicle.update_vehicle(physics_world.integration_parameters.dt, queries);

        if self.reset_pending { 
            self.reset_pending = false;
            
            let rigidbody = physics_world.bodies.get_mut(vehicle.chassis).unwrap();
            rigidbody.set_position(rigidbody.position().append_translation(glam::vec3(0f32, 2f32, 0f32)),true);
            rigidbody.set_rotation(glam::Quat::IDENTITY, true);
            rigidbody.set_angvel(glam::Vec3::ZERO, false);
            rigidbody.set_linvel(glam::Vec3::ZERO, false);
        }

        let body = physics_world.bodies.get_mut(vehicle.chassis.clone()).unwrap();
        let v1 = body.position().translation;
        let q1 = body.position().rotation;

        let v2 = vec3_glam_to_vek(v1);
        let q2 = quat_glam_to_vek(q1);
        
        self.current_state = VehicleState {
            position: v2,
            rotation: q2,
            wheels: vehicle.wheels().iter().map(|wheel| WheelState {
                position: vec3_glam_to_vek(wheel.center()),
                rotation: wheel.rotation,
            }).collect()
        };
    }

    pub fn update_frame(&mut self, interpolation_factor: f32, delta: f32) {
        self.interpolated_state = VehicleState::interpolate(&self.last_state, &self.current_state, interpolation_factor);
        let target = self.interpolated_state.position + self.camera_position_offset;
        self.camera_position = target;
    }
}

fn vec3_glam_to_vek(vec: glam::Vec3) -> vek::Vec3<f32> {
    vek::Vec3::<f32>::from(vec.to_array())
}

fn quat_glam_to_vek(quat: glam::Quat) -> vek::Quaternion<f32> {
    vek::Quaternion::<f32>::from_vec4(vek::Vec4::from(quat.to_array()))
}