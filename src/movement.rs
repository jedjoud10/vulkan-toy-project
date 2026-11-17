use crate::input::{Axis, Input, MouseAxis};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use vek::Clamp;
use winit::keyboard::KeyCode;
use rapier3d::{control::{CharacterLength, DynamicRayCastVehicleController, KinematicCharacterController, WheelTuning}, prelude::*};

const DEFAULT_STARTING_SNAPSHOT: Option<usize> = None;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Snapshot {
    pub position: vek::Vec3<f32>,
    pub rotation: vek::Quaternion<f32>,
    
    #[serde(default = "default_fov")]
    pub fov: f32,
}

fn default_fov() -> f32 {
    80f32
}

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

#[derive(Default)]
pub struct Vehicle {
    vehicle: Option<DynamicRayCastVehicleController>,
    summed_mouse: vek::Vec2<f32>,
    pub camera_position: vek::Vec3<f32>,
    
    last_position: vek::Vec3<f32>,
    last_rotation: vek::Quaternion<f32>, 
    pub position: vek::Vec3<f32>,
    pub rotation: vek::Quaternion<f32>,
    pub interpolated_position: vek::Vec3<f32>,
    pub interpolated_rotation: vek::Quaternion<f32>,
    
    
    local_velocity: vek::Vec2<f32>,
    pub head_rotation_global: vek::Quaternion<f32>,
    camera_position_offset: vek::Vec3<f32>,
    reset_pending: bool,
}

impl Vehicle {
    pub fn new(physics_world: &mut rapier3d::prelude::PhysicsWorld) -> Self {       
        let hw = 1.4;
        let hh = 0.25;

        // https://rapier.rs/docs/user_guides/rust/character_controller/#vehicle-controller
        let (chassis_handle, _) = physics_world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 35.0, 0.0)),
            ColliderBuilder::cuboid(hw * 2.0, hh, hw).density(100.0),
        );
                        
        // The tuning shared by the wheels: the suspension and the grip.
        let tuning = WheelTuning {
            suspension_stiffness: 60.0,
            suspension_damping: 5.0,
            max_suspension_travel: 8f32,
            ..WheelTuning::default()
        };      

        let mut vehicle = DynamicRayCastVehicleController::new(chassis_handle);
        let wheel_positions = [
            Vector::new(hw * 1.5, -hh, hw),
            Vector::new(hw * 1.5, -hh, -hw),
            Vector::new(-hw * 1.5, -hh, hw),
            Vector::new(-hw * 1.5, -hh, -hw),
        ];      

        for position in wheel_positions {
            // The position of the wheel, the direction its suspension pushes along, its axle, the
            // rest length of its suspension, and its radius; all in the local frame of the chassis.
            vehicle.add_wheel(position, -Vector::Y, Vector::Z, hh, hh, &tuning);
        }
        
        Self {
            vehicle: Some(vehicle),
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

        let rot = vek::Mat4::from(self.head_rotation_global);  
        self.camera_position_offset = rot.mul_direction(vek::Vec3::unit_z()) * 15f32;
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
            rigidbody.set_position(rigidbody.position().append_translation(glam::vec3(0f32, 10f32, 0f32)),true);
            rigidbody.set_rotation(glam::Quat::IDENTITY, true);
            rigidbody.set_angvel(glam::Vec3::ZERO, false);
            rigidbody.set_linvel(glam::Vec3::ZERO, false);
        }

        self.last_position = self.position;
        self.last_rotation = self.rotation;
        /*

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
        */

        let body = physics_world.bodies.get_mut(vehicle.chassis.clone()).unwrap();
        let v1 = body.position().translation;
        let q1 = body.position().rotation;

        let v2 = vek::Vec3::<f32>::from(v1.to_array());
        let q2 = vek::Quaternion::<f32>::from_vec4(vek::Vec4::from(q1.to_array()));
        
        self.position = v2;
        self.rotation = q2;
        
        // self.head_rotation_global = q2;
    }

    pub fn update_frame(&mut self, interpolation_factor: f32, delta: f32) {
        // self.interpolated_position = vek::Vec3::lerp(self.last_position, self.position, interpolation_factor);
        // self.interpolated_rotation = vek::Quaternion::slerp(self.last_rotation, self.rotation, interpolation_factor);
        self.interpolated_position = vek::Vec3::lerp(self.interpolated_position, self.position, delta * 10f32);
        self.interpolated_rotation = vek::Quaternion::slerp(self.interpolated_rotation, self.rotation, delta * 10f32);
        
        let target = self.interpolated_position + self.camera_position_offset;
        self.camera_position = target;
    }
}

pub fn horizontal_to_vertical(hfov: f32, ratio: f32) -> f32 {
    2.0 * ((hfov.to_radians() / 2.0).tan() * (1.0 / (ratio))).atan()
}


#[derive(Default)]
pub struct Camera {
    pub position: vek::Vec3<f32>,
    pub rotation: vek::Quaternion<f32>,
    pub proj_matrix: vek::Mat4<f32>,
    pub view_matrix: vek::Mat4<f32>,
    pub camera_frustum_planes: [vek::Vec4<f32>; 6],
    pub update_frustum: bool,
    
    fov: f32,
    target_fov: f32,
    fixed_mode_snapshot_index: Option<usize>,
    
    snapshots: Vec<Snapshot>,
}

impl Camera {
    pub fn new() -> Self {
        let snapshots: Vec<Snapshot> = serde_json::from_str(include_str!("snapshots.json")).unwrap();

        Self {
            snapshots,
            fov: default_fov(),
            ..Default::default()
        }
    }

    pub fn update_camera(&mut self, ratio: f32) {
        // recalculate projection matrices
        self.proj_matrix = vek::Mat4::<f32>::perspective_rh_no(horizontal_to_vertical(self.fov.clamp(0.0001f32, 180f32), ratio), ratio, 0.1f32, 10000.0f32);
        self.view_matrix = vek::Mat4::look_at_rh(self.position, self.forward() + self.position, self.up());

        // https://github.com/jedjoud10/cflake-engine/blob/3369199f0cfa8b220edc0363a76401b50c83fada/crates/math/src/bounds/frustum.rs#L47
        if self.update_frustum {
            let columns = (self.proj_matrix * self.view_matrix).transposed().into_col_arrays();
            let columns = columns
                .into_iter()
                .map(vek::Vec4::from)
                .collect::<SmallVec<[vek::Vec4<f32>; 4]>>();

            // Magic from https://www.braynzarsoft.net/viewtutorial/q16390-34-aabb-cpu-side-frustum-culling
            // And also from https://gamedev.stackexchange.com/questions/156743/finding-the-normals-of-the-planes-of-a-view-frustum
            // YAY https://stackoverflow.com/questions/12836967/extracting-view-frustum-planes-gribb-hartmann-method
            let left = columns[3] + columns[0];
            let right = columns[3] - columns[0];
            let top = columns[3] - columns[1];
            let bottom = columns[3] + columns[1];
            let near = columns[3] + columns[2];
            let far = columns[3] - columns[2];
            
            self.camera_frustum_planes = [top, bottom, left, right, near, far].map(|x| {
                let magnitude = x.xyz().magnitude();
                let normal = x.xyz().normalized();
                let distance = x.w / magnitude;

                normal.with_w(distance)
            })
        }
    }
    
    pub fn forward(&self) -> vek::Vec3<f32> {
        vek::Mat4::from(self.rotation).mul_direction(-vek::Vec3::unit_z())
    }

    pub fn up(&self) -> vek::Vec3<f32> {
        vek::Mat4::from(self.rotation).mul_direction(vek::Vec3::unit_y())
    }
}

/*

    pub fn update(&mut self, input: &Input, delta: f32) {
        self.local_velocity = vek::Vec2::<f32>::zero();

        let boosted = input.get_button(KeyCode::ShiftLeft).held();

        let speed = if boosted {
            2f32.powf(self.boost)
        } else {
            1.0f32
        };

        if input.get_button(KeyCode::KeyW).held() {
            self.local_velocity.y = 1f32;
        } else if input.get_button(KeyCode::KeyS).held() {
            self.local_velocity.y = -1f32;
        }

        if input.get_button(KeyCode::KeyA).held() {
            self.local_velocity.x = 1f32;
        } else if input.get_button(KeyCode::KeyD).held() {
            self.local_velocity.x = -1f32;
        }

        if boosted {
            self.boost += input.get_axis(Axis::Mouse(MouseAxis::ScrollDelta)) * 0.2;
            self.boost = self.boost.clamp(-5.0, 5.0);
        }
        let sens = 1.0f32;
        let summed_mouse_target = vek::Vec2::new(
            input.get_axis(Axis::Mouse(MouseAxis::PositionX)) * 0.003 * sens,
            input.get_axis(Axis::Mouse(MouseAxis::PositionY)) * -0.003 * sens,
        );
        self.summed_mouse = vek::Vec2::lerp(
            self.summed_mouse,
            summed_mouse_target,
            (40f32 * delta).clamped01(),
        );

        if self.fixed_mode_snapshot_index.is_none() {
            self.rotation = vek::Quaternion::rotation_y(self.summed_mouse.x) * vek::Quaternion::rotation_x(self.summed_mouse.y);
        }
        

        if !boosted {
            self.target_fov -= input.get_axis(Axis::Mouse(MouseAxis::ScrollDelta)) * 5f32;
        }

        self.target_fov = self.target_fov.clamp(0.05, 179.5);
        self.fov += (self.target_fov-self.fov).clamp(-100f32, 100f32) * delta * 20f32;

        
        let rot = vek::Mat4::from(self.rotation);
        let forward = rot.mul_direction(-vek::Vec3::unit_z());
        let right = rot.mul_direction(vek::Vec3::unit_x());
        let up = rot.mul_direction(vek::Vec3::unit_y());

        let velocity = forward * self.local_velocity.y + right * self.local_velocity.x;
        self.velocity = vek::Vec3::lerp(
            self.velocity,
            velocity * 20.0f32 * speed,
            (40f32 * delta).clamped01(),
        );

        if self.fixed_mode_snapshot_index.is_none() {
            self.position += self.velocity * delta;
        }

        // take a snapshot of the movement (position + rotation) and print to console
        if input.get_button(KeyCode::KeyU).pressed() {
            let snap = Snapshot {
                position: self.position,
                rotation: self.rotation,
                fov: self.target_fov,
            };

            let str = serde_json::to_string_pretty(&snap).unwrap();
            println!("{str}");
        }

        // toggle fixed mode
        if input.get_button(KeyCode::KeyI).pressed() {
            self.fixed_mode_snapshot_index = match self.fixed_mode_snapshot_index {
                Some(_) => None,
                None => Some(0),
            };
        }

        // toggle frustum updates
        if input.get_button(KeyCode::KeyN).pressed() {
            self.update_frustum = !self.update_frustum;
        }

        // iterate over snapshots
        if let Some(ref mut idx) = self.fixed_mode_snapshot_index && input.get_button(KeyCode::KeyO).pressed()
            && !self.snapshots.is_empty() {
                *idx += 1;
                *idx %= self.snapshots.len();
            }

        // copy snapshot params to camera
        if let Some(idx) = self.fixed_mode_snapshot_index && !self.snapshots.is_empty() {
            self.position = self.snapshots[idx].position;
            self.rotation = self.snapshots[idx].rotation;
            self.fov = self.snapshots[idx].fov;
        }
    }
*/