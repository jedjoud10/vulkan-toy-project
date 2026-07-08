use crate::input::{Axis, Input, MouseAxis};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use vek::Clamp;
use winit::keyboard::KeyCode;


#[derive(Serialize, Deserialize)]
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
pub struct Movement {
    pub position: vek::Vec3<f32>,
    pub rotation: vek::Quaternion<f32>,
    pub proj_matrix: vek::Mat4<f32>,
    pub view_matrix: vek::Mat4<f32>,
    pub camera_frustum_planes: [vek::Vec4<f32>; 6],
    
    fov: f32,
    target_fov: f32,
    summed_mouse: vek::Vec2<f32>,
    local_velocity: vek::Vec2<f32>,
    velocity: vek::Vec3<f32>,
    boost: f32,
    fixed_mode_snapshot_index: Option<usize>,
    pub update_frustum: bool,
    snapshots: Vec<Snapshot>,
}

impl Movement {
    pub fn new() -> Self {
        let snapshots: Vec<Snapshot> = serde_json::from_str(include_str!("snapshots.json")).unwrap();

        Self {
            fov: default_fov(),
            target_fov: default_fov(),
            //position: vek::Vec3::new(40.5f32, 80f32, 40.5f32),
            //rotation : vek::Quaternion::rotation_y(-130f32.to_radians()),
            position: vek::Vec3::zero(),
            camera_frustum_planes: Default::default(),
            rotation: vek::Quaternion::identity(),
            fixed_mode_snapshot_index: None,
            update_frustum: true,
            snapshots,
            ..Default::default()
        }
    }
    pub fn update(&mut self, input: &Input, ratio: f32, delta: f32) {
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
                self.position = self.snapshots[*idx].position;
                self.rotation = self.snapshots[*idx].rotation;
                self.fov = self.snapshots[*idx].fov;
            }

            
        
        // recalculate projection matrices
        self.proj_matrix = vek::Mat4::<f32>::perspective_rh_no(horizontal_to_vertical(self.fov.clamp(0.0001f32, 180f32), ratio), ratio, 0.001f32, 1000.0f32);
        self.view_matrix = vek::Mat4::look_at_rh(self.position, forward + self.position, up);

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
}

pub fn horizontal_to_vertical(hfov: f32, ratio: f32) -> f32 {
    2.0 * ((hfov.to_radians() / 2.0).tan() * (1.0 / (ratio))).atan()
}