use std::collections::{HashMap, HashSet};

use crate::ticker;

#[derive(Debug)]
#[derive(Clone)]
pub struct Vertex {
    pub inv_mass: f32,              // w_i
    pub position: vek::Vec3<f32>,   // x_i
    pub velocity: vek::Vec3<f32>,   // v_i
}

#[derive(Debug)]
pub enum Mode {
    Equality,
    Inequality,
} 

pub struct Constraint {
    // n_j
    pub cardinality: usize,

    // C_j             
    pub function: Box<dyn Fn(&[vek::Vec3<f32>]) -> f32>,
    
    // {i_1, ..., i_n_j}
    pub indices: Vec<usize>,

    // k_j
    pub stiffness: f32,
    
    // equality OR inequality
    pub mode: Mode,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint").field("cardinality", &self.cardinality).field("indices", &self.indices).field("stiffness", &self.stiffness).field("mode", &self.mode).finish()
    }
}

#[derive(Debug)]
pub struct DynamicObject {
    pub vertices: Vec<Vertex>,
    pub constraints: Vec<Constraint>,
}


// TODO: need to somehow compute the derivative of this function on respect to each of the input points
// finite differences :eyes:
// approximate derivatives with respect to a specific point
fn finite_difference(input: &[vek::Vec3<f32>], index: usize, func: &dyn Fn(&[vek::Vec3<f32>]) -> f32) -> vek::Vec3<f32> {
    let pos = input[index];

    let mut tmp = input.to_vec();

    let eps = 0.001f32;

    tmp[index] = pos + vek::Vec3::new(-eps, 0f32, 0f32);
    let nx = func(&tmp);

    tmp[index] = pos + vek::Vec3::new(eps, 0f32, 0f32);
    let px = func(&tmp);

    
    tmp[index] = pos + vek::Vec3::new(0f32, -eps, 0f32);
    let ny = func(&tmp);

    tmp[index] = pos + vek::Vec3::new(0f32, eps, 0f32);
    let py = func(&tmp);

    
    tmp[index] = pos + vek::Vec3::new(0f32, 0f32, -eps);
    let nz = func(&tmp);

    tmp[index] = pos + vek::Vec3::new(0f32, 0f32, eps);
    let pz = func(&tmp);

    return vek::Vec3::new(px-nx, py-ny, pz-nz) / (2f32 * eps)
}

const SUBSTEP_ITERATIONS: usize = 8;
const SOLVER_ITERATIONS: usize = 4;
const GRAVITY: f32 = -20f32;
 

// gravity for now
// external FORCES
fn f_ext(p: &Vertex) -> vek::Vec3<f32> {
    let mass = 1f32 / p.inv_mass;

    // F = m * a
    vek::Vec3::new(0f32, GRAVITY, 0f32) * mass
}

pub fn simulate(dynamic_object: &mut DynamicObject, other: &[vek::Vec3<f32>], delta_t: f32) {
    // add external velocity forces
    for vertex in dynamic_object.vertices.iter_mut().filter(|v| v.inv_mass > 0f32) {
        vertex.velocity += vertex.inv_mass * f_ext(&vertex) * delta_t;
    }

    // integrate external forces to temp positions
    let mut temp_positions = Vec::<vek::Vec3<f32>>::new();
    for vertex in dynamic_object.vertices.iter_mut() {
        temp_positions.push(vertex.position + vertex.velocity * delta_t);
    }

    let mut collision_constraints = Vec::<Constraint>::new();

    for (vertex_index, vertex) in dynamic_object.vertices.iter().enumerate() {
        collision_constraints.push(Constraint {
            cardinality: 1,
            function: Box::new(|data: &[vek::Vec3<f32>]| -> f32 {
                data[0].dot(vek::Vec3::unit_y()) - 10.0
            }),
            indices: vec![vertex_index],
            stiffness: 1.0f32,
            mode: Mode::Inequality
        });

        for x in other.iter().copied() {
            if x.distance(vertex.position) < 10f32 {
                collision_constraints.push(Constraint {
                    cardinality: 1,
                    function: Box::new(move |data: &[vek::Vec3<f32>]| -> f32 {
                        (data[0] - x).magnitude() - 1.0
                    }),
                    indices: vec![vertex_index],
                    stiffness: 0.02f32,
                    mode: Mode::Inequality
                });
            }
        }

        /*
        collision_constraints.push(Constraint {
            cardinality: 1,
            function: Box::new(|data: &[vek::Vec3<f32>]| -> f32 {
                data[0].dot(vek::Vec3::new(1f32, 1f32, 1f32).normalized()) - 10.0
            }),
            indices: vec![vertex_index],
            stiffness: 1.0f32,
            mode: Mode::Inequality
        });
        */
    }
    
    
    // solver loop
    for _ in 0..SOLVER_ITERATIONS {
        for constraint in dynamic_object.constraints.iter().chain(collision_constraints.iter()) {
            let constraint_positions_looked_up = constraint.indices.iter().map(|index| temp_positions[*index]).collect::<Vec<_>>();

            let c_p1_pn = (constraint.function)(&constraint_positions_looked_up);

            let project = match (&constraint.mode, c_p1_pn < 0f32) {
                (Mode::Equality, _) => true,
                (Mode::Inequality, e) => e
            };            

            let k_prime = 1f32 - (1f32 - constraint.stiffness).powf(1f32 / SOLVER_ITERATIONS as f32);
            assert!(k_prime >= 0f32);
            assert!(k_prime <= 1f32);
            


            if project {
                //let scaling_factor = (constraint.function)(&constraint_positions_looked_up);
                // HACK: this works only because we are dealing with distance constraints for now
                // fix later...
                //let scaling_factor = (constraint.function)(&constraint_positions_looked_up) / constraint.indices.iter().map(|index| dynamic_object.vertices[*index].inv_mass).sum::<f32>();

                // scaling factor is lambda
                let scaling_factor = c_p1_pn / constraint.indices.iter().map(|index| dynamic_object.vertices[*index].inv_mass * finite_difference(&constraint_positions_looked_up, *index, &constraint.function).magnitude_squared()).sum::<f32>();
                
                for (index, index2) in constraint.indices.iter().copied().enumerate() {
                    let nabla_p_i_times_cp = finite_difference(&constraint_positions_looked_up, index, &constraint.function);
                    let w_i = dynamic_object.vertices[index2].inv_mass;


                    let delta_p_i = -scaling_factor * w_i * nabla_p_i_times_cp;

                    if w_i > 0f32 {
                        temp_positions[index2] += k_prime * delta_p_i;
                    }
                    //dbg!(delta_p_i);
                }
            }
        }
    }

    // end loop
    for (vertex, tmp_position) in dynamic_object.vertices.iter_mut().zip(temp_positions.iter()) {
        vertex.velocity = (*tmp_position - vertex.position) / delta_t;
        vertex.position = *tmp_position;
    } 
}

pub struct Physics {
    pub objects: Vec<DynamicObject>
}

impl Physics {
    pub fn tick(&mut self) {
        let delta_t = 1f32 / ticker::TICKS_PER_SECOND;
        for _ in 0..SUBSTEP_ITERATIONS {
            
            for index in 0..self.objects.len() {
                let other_verts_only = self.objects.iter().enumerate().filter(|(i, _)| *i != index).map(|(_, x)| x.vertices[0].position).collect::<Vec<_>>();
                let object = &mut self.objects[index];
                simulate(object, &other_verts_only, delta_t / (SUBSTEP_ITERATIONS as f32));
            }
        }        
    }
}