use vek::Clamp;
use super::*;
use super::utils;


pub fn compute_normals(densities: &Vec<f32>, normals_lookup: &mut [vek::Vec3<f32>]) {
    for x in 0..(CHUNK_SIZE_WITH_PADDING-1) {
        for y in 0..(CHUNK_SIZE_WITH_PADDING-1) {
            for z in 0..(CHUNK_SIZE_WITH_PADDING-1) {
                let pos = vek::Vec3::new(x,y,z);
                let mut normal = vek::Vec3::<f32>::zero();
                let d1 = densities[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)];
            
                for axis in 0..3 {
                    // next cell is the cell "forward" to the current cell based on axiss
                    let mut next_cell = pos;
                    next_cell[axis] += 1;
                    
                    let d2 = densities[utils::offset_to_index(next_cell, CHUNK_SIZE_WITH_PADDING)];
                    normal[axis] = d2 - d1;
                }

                normal.normalize();
                normals_lookup[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] = normal;
            }
        }
    }
}

pub fn core_quads(densities: &Vec<f32>, indices: &mut Vec<u32>, lookup: &Vec<u32>) {
    for x in 1..(CHUNK_SIZE_WITH_PADDING-2) {
        for y in 1..(CHUNK_SIZE_WITH_PADDING-2) {
            for z in 1..(CHUNK_SIZE_WITH_PADDING-2) {
                let pos = vek::Vec3::new(x,y,z);
                let is_set = densities[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] < 0f32;
            
                for axis in 0..3 {
                    // next cell is the cell "forward" to the current cell based on axiss
                    let mut next_cell = pos;
                    next_cell[axis] += 1;
                
                    let next_cell_is_set = densities[utils::offset_to_index(next_cell, CHUNK_SIZE_WITH_PADDING)] < 0f32;
                
                    if is_set != next_cell_is_set {
                        let mut quad_vertex_indices: vek::Vec4<u32> = vek::Vec4::broadcast(u32::MAX);
                    
                        let flip = !is_set;
                        let vertex_offsets: [vek::Vec3<usize>; 4] = utils::quad_vertex_offsets_for_axis_ccw_order_looking_from_positive_axis(axis as u32);

                        // inside quad local vertex index
                        for index in 0..4 {
                            let target = vertex_offsets[index] + next_cell - 1;

                            if let Some(looked_up_vertex_index) = utils::try_offset_to_index(target, CHUNK_SIZE_WITH_PADDING).map(|x| lookup[x]) {
                                quad_vertex_indices[index] = looked_up_vertex_index;
                            }
                        }

                        // don't do anything if quad contains invalid vertices
                        if quad_vertex_indices.iter().any(|x| *x == u32::MAX) {
                            continue;
                        }

                        // generate quad
                        if flip {
                            utils::add_cw_quad(quad_vertex_indices, indices);
                        } else {
                            utils::add_ccw_quad(quad_vertex_indices, indices);
                        }
                    }
                }
            }
        }
    }
}

pub fn core_vertices(densities: &Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32, vertices: &mut Vec<vek::Vec3<f32>>, normals: &mut Vec<vek::Vec3<f32>>, normals_lookup: &[vek::Vec3<f32>], lookup: &mut Vec<u32>) {
    for x in 0..(CHUNK_SIZE_WITH_PADDING-2) {
        for y in 0..(CHUNK_SIZE_WITH_PADDING-2) {
            for z in 0..(CHUNK_SIZE_WITH_PADDING-2) {
                let pos = vek::Vec3::new(x,y,z);

                let mut vertex = vek::Vec3::<f32>::zero();
                let mut normal = vek::Vec3::<f32>::zero();
                
                let mut count = 0;

                for edge in 0..12 {
                    let p1_offsetted = utils::EDGE_POSITIONS_0[edge];
                    let p2_offsetted = utils::EDGE_POSITIONS_1[edge];

                    let p1_offsetted_index = utils::offset_to_index(pos + p1_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
                    let p2_offsetted_index = utils::offset_to_index(pos + p2_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);

                    let d1 = densities[p1_offsetted_index];
                    let d2 = densities[p2_offsetted_index];

                    let n1 = normals_lookup[p1_offsetted_index];
                    let n2 = normals_lookup[p2_offsetted_index];

                    if (d1 < 0f32) ^ (d2 < 0f32) {
                        let factor = utils::unlerp(0f32, d1, d2);
                        vertex += vek::Vec3::lerp(p1_offsetted.as_::<f32>(), p2_offsetted.as_::<f32>(), factor);
                        normal += vek::Vec3::lerp(n1, n2, factor);
                        count += 1;
                    }
                }

                if count > 0 {
                    let inside_cell_offset = if SMOOTH {
                        vertex / count as f32
                    } else {
                        vek::Vec3::<f32>::one() * 0.5f32
                    };
                
                    let vertex_position = (inside_cell_offset + pos.as_::<f32>()) * chunk_scale + chunk_position.as_::<f32>();
                
                    let index = vertices.len();
                    lookup[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] = index as u32;
                    vertices.push(vertex_position);
                    normals.push(normal.normalized())
                }
            }
        }
    }
}