use crate::input::{Axis, Input, MouseAxis};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use vek::Clamp;
use winit::keyboard::KeyCode;
use rapier3d::{control::{CharacterLength, KinematicCharacterController}, prelude::*};

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