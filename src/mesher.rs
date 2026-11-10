use vek::Clamp;

use crate::{utils, voxel::{CHUNK_SIZE, CHUNK_SIZE_WITH_PADDING, CHUNK_VOLUME}};



const INDEX_QUAD_ORDER: [usize; 6] = [0, 1, 2, 2, 1, 3];
const INDEX_OPPOSITE_QUAD_ORDER: [usize; 6] = [1, 0, 2, 1, 2, 3];

const EDGE_POSITIONS_0: [vek::Vec3<u32>; 12] = [
    vek::Vec3::new(0, 0, 0),
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(1, 1, 0),
    vek::Vec3::new(0, 1, 0),
    vek::Vec3::new(0, 0, 1),
    vek::Vec3::new(1, 0, 1),
    vek::Vec3::new(1, 1, 1),
    vek::Vec3::new(0, 1, 1),
    vek::Vec3::new(0, 0, 0),
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(1, 1, 0),
    vek::Vec3::new(0, 1, 0),
];

const EDGE_POSITIONS_1: [vek::Vec3<u32>; 12] = [
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(1, 1, 0),
    vek::Vec3::new(0, 1, 0),
    vek::Vec3::new(0, 0, 0),
    vek::Vec3::new(1, 0, 1),
    vek::Vec3::new(1, 1, 1),
    vek::Vec3::new(0, 1, 1),
    vek::Vec3::new(0, 0, 1),
    vek::Vec3::new(0, 0, 1),
    vek::Vec3::new(1, 0, 1),
    vek::Vec3::new(1, 1, 1),
    vek::Vec3::new(0, 1, 1),
];


const SKIRT_2D_EDGE_POSITIONS_0: [vek::Vec2<u32>; 4] = [
    vek::Vec2::new(0, 0),
    vek::Vec2::new(0, 1),
    vek::Vec2::new(1, 1),
    vek::Vec2::new(1, 0),
];

const SKIRT_2D_EDGE_POSITIONS_1: [vek::Vec2<u32>; 4] = [
    vek::Vec2::new(0, 1),
    vek::Vec2::new(1, 1),
    vek::Vec2::new(1, 0),
    vek::Vec2::new(0, 0),
];


pub struct MeshedChunk {
    pub vertices: Vec<vek::Vec3<f32>>,
    pub indices: Vec<u32>,
}

const SMOOTH: bool = true;

// stupid and unoptimized but works for now ig
pub fn mesh_chunk(densities: Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32) -> MeshedChunk {
    let mut vertices = Vec::<vek::Vec3<f32>>::new();
    
    let mut indices = Vec::<u32>::new();
    let mut lookup = vec![u32::MAX; CHUNK_VOLUME];

    // first pass will generate vertex positions
    for x in 0..(CHUNK_SIZE_WITH_PADDING-2) {
        for y in 0..(CHUNK_SIZE_WITH_PADDING-2) {
            for z in 0..(CHUNK_SIZE_WITH_PADDING-2) {
                let pos = vek::Vec3::new(x,y,z);

                let mut vertex = vek::Vec3::<f32>::zero();
                let mut count = 0;

                for edge in 0..12 {
                    let p1_offsetted = EDGE_POSITIONS_0[edge];
                    let p2_offsetted = EDGE_POSITIONS_1[edge];

                    let p1_offsetted_index = utils::offset_to_index(pos + p1_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
                    let p2_offsetted_index = utils::offset_to_index(pos + p2_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);

                    let d1 = densities[p1_offsetted_index];
                    let d2 = densities[p2_offsetted_index];

                    if (d1 < 0f32) ^ (d2 < 0f32) {
                        let factor = utils::unlerp(0f32, d1, d2);
                        vertex += vek::Vec3::lerp(p1_offsetted.as_::<f32>(), p2_offsetted.as_::<f32>(), factor);
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
                }
            }
        }
    }

    // second pass will generate quads
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
                        let mut quad_vertex_indices: [u32; 4] = [u32::MAX; 4];
                        
                        let dir = is_set ^ (axis == 1);
                        let vertex_offsets: [vek::Vec3<usize>; 4] = utils::quad_vertex_offsets_for_axis(axis as u32);

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

                        // holy cursed
                        let quad_vertex_order = if dir { INDEX_QUAD_ORDER } else { INDEX_OPPOSITE_QUAD_ORDER };
                        for what_to_call_this_index in 0..6 {
                            indices.push(quad_vertex_indices[quad_vertex_order[what_to_call_this_index]]);
                        }
                    }
                }
            }
        }
    }

    let mut chunk = MeshedChunk {
        vertices,
        indices,
    };

    // WIP
    mesh_skirt(densities, chunk_position, chunk_scale, &mut chunk, &lookup);
    chunk
}

// this uses the +2 padding because we need to store an extra vertex per-skirt face (to handle the *literal* edge case)
const SKIRT_SIZE: usize = CHUNK_SIZE_WITH_PADDING;
const SKIRT_FACE: usize = SKIRT_SIZE*SKIRT_SIZE;


pub fn mesh_skirt(densities: Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32, MeshedChunk { vertices, indices }: &mut MeshedChunk, lookup: &[u32]) {
    
    // TODO: move face logic to own function and separate the two passes
    for face in 0..6 {
        let mut skirt_lookup = vec![u32::MAX; SKIRT_FACE];
        let direction = face % 3;
        let negative = face < 3;

        // first pass will generate vertex positions ON THE SKIRT FACE
        skirt_vertices(&densities, chunk_position, chunk_scale, vertices, &mut skirt_lookup, direction, negative, face);
        
        // second pass will generate quads BETWEEN the skirt vertices and base mesh
        skirt_quads(&densities, indices, lookup, face, skirt_lookup, direction);
    }
}

fn as_bitmask(b4: vek::Vec4<bool>) -> u32 {
    let bitmask = b4.as_::<u32>() << vek::Vec4::new(0,1,2,3);
    bitmask.reduce_bitor()
}

// there are three types of skirt vertices
// middle: vertices in the "x,y in [1..SKIRT_SIZE)" area in the middle of the skirt 
// edge: vertices on the literal edge of the skirt boundary. "x,y in [1..SKIRT_SIZE) where one of x,y is set to 0 or (SKIRT_SIZE-1)"
// corner: corner vertices of the chunk, only a maximum of 8 could exist at a time "(x,y) in { (0,0), (SKIRT_SIZE-1, 0), (0, SKIRT_SIZE-1), (SKIRT_SIZE-1, SKIRT_SIZE-1)  }" 
struct VertexToSpawn {
    use_world_position: bool,
    world_position: vek::Vec3<f32>,
    inner_offset: vek::Vec3<f32>,
}

struct VertexSpawningData<'a> {
    densities: &'a [f32]
}

// TODO: implement this corner case
fn create_corner(data: &VertexSpawningData, direction: u32, negative: bool, relative: vek::Vec2<usize>) -> Option<VertexToSpawn> {
    None
}

fn create_skirt_sn_1d(data: &VertexSpawningData, relative: vek::Vec2<usize>, min_max_mask: vek::Vec4<bool>, negative: bool, face: u32, direction: u32) -> Option<VertexToSpawn> {
    let mask = vek::Vec2::new(min_max_mask.x || min_max_mask.z, min_max_mask.y || min_max_mask.w);
    let edge_direction_index = edge_dir_face_relative(mask, direction);
    let edge_direction = utils::FORWARD[edge_direction_index as usize];
    // log::debug!("{min_max_mask} {negative} {face} {direction} {}", edge_direction);
    
    
    let mut vertex = vek::Vec3::<f32>::zero();
    let mut count = 0;
    
    let missing = if negative { 0 } else { CHUNK_SIZE_WITH_PADDING - 2 };

    let relative = relative.clamped(0, CHUNK_SIZE_WITH_PADDING - 2);
    let world_position = utils::unflatten_from_face_relative(relative.as_::<f32>(), missing as f32, direction);

    let unoffsetted = utils::unflatten_from_face_relative(relative, missing, direction);
    // log::debug!("{} {}", unoffsetted - edge_direction.as_::<usize>(), unoffsetted);
    

    let p1_offsetted_index = utils::offset_to_index(unoffsetted - edge_direction.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
    let p2_offsetted_index = utils::offset_to_index(unoffsetted, CHUNK_SIZE_WITH_PADDING as usize);
    
    let d1 = data.densities[p1_offsetted_index];
    let d2 = data.densities[p2_offsetted_index];

    log::debug!("{} {} {} {}", p1_offsetted_index, p2_offsetted_index, d1, d2);
    
    if (d1 < 0f32) ^ (d2 < 0f32) {
        let factor = utils::unlerp(0f32, d1, d2);
        vertex += vek::Vec3::lerp(-edge_direction.as_::<f32>(), vek::Vec3::zero(), factor);
        count += 1;
    }

    if count > 0 {
        log::debug!("adding edge vertex");
        let inside_cell_offset = vertex;
        // let inside_cell_offset = if SMOOTH {
        //     vertex / count as f32
        // } else {
        //     let mut t = vek::Vec3::<f32>::one() * 0.5f32;
        //     t[direction as usize] = if negative { 0f32 } else { 1f32 };
        //     t
        // };

        let inner_offset = inside_cell_offset;

        Some(VertexToSpawn {
            use_world_position: true,
            world_position,
            inner_offset
        })
        /*
        let vertex_position = (inside_cell_offset + pos.as_::<f32>()) * chunk_scale + chunk_position.as_::<f32>();
        let index = vertices.len();
        skirt_lookup[utils::offset_to_index_flat(relative, SKIRT_SIZE)] = index as u32;
        vertices.push(vertex_position);
        */
    } else {
        None
    }
}

// https://github.com/jedjoud10/VoxelTerrain/blob/e949784031a0b32104f467e3c3d2138f54286dd4/Runtime/Utils/SkirtUtils.cs#L95-L140
fn edge_dir_face_relative(mask: vek::Vec2<bool>, face_normal: u32) -> u32 {
    assert!(face_normal < 3);

    // no two can be set, otherwise that means that this is a CORNER
    assert!(mask.x ^ mask.y);

    // Need to pick the "other" value that isn't at a boundary
    let mask = !mask;

    // X AXIS
    if (face_normal == 0) {
        if (mask.x) {
            // Y
            return 1;
        } else if (mask.y) {
            // Z
            return 2;
        }

    // Y AXIS
    } else if (face_normal == 1) {
        if (mask.x) {
            // X
            return 0;
        } else if (mask.y) {
            // Z
            return 2;
        }

    // Z AXIS
    } else if (face_normal == 2) {
        if (mask.x) {
            // X
            return 0;
        } else if (mask.y) {
            // Y
            return 1;
        }
    }

    // should never happen
    unreachable!()
}

fn create_skirt_sn_2d(data: &VertexSpawningData, face: u32, negative: bool, direction: u32, relative: vek::Vec2<usize>, position: vek::Vec3<usize>) -> Option<VertexToSpawn> {
    let mut vertex = vek::Vec3::<f32>::zero();
    let mut count = 0;

    // this is INSIDE cell, so offset is 1 and not the chunk size (we add the chunk offset position afterwards)
    let missing = if negative { 0 } else { 1 } as u32;
    
    for edge in 0..4 {
        let p1_relative_offsetted = SKIRT_2D_EDGE_POSITIONS_0[edge];
        let p2_relative_offsetted = SKIRT_2D_EDGE_POSITIONS_1[edge];

        let p1_offsetted = utils::unflatten_from_face_relative(p1_relative_offsetted, missing, direction);
        let p2_offsetted = utils::unflatten_from_face_relative(p2_relative_offsetted, missing, direction);

        let p1_offsetted_index = utils::offset_to_index(position + p1_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
        let p2_offsetted_index = utils::offset_to_index(position + p2_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
        
        let d1 = data.densities[p1_offsetted_index];
        let d2 = data.densities[p2_offsetted_index];
        
        if (d1 < 0f32) ^ (d2 < 0f32) {
            let factor = utils::unlerp(0f32, d1, d2);
            vertex += vek::Vec3::lerp(p1_offsetted.as_::<f32>(), p2_offsetted.as_::<f32>(), factor);
            count += 1;
        }
    }
    
    if count > 0 {
        let inside_cell_offset = if SMOOTH {
            vertex / count as f32
        } else {
            let mut t = vek::Vec3::<f32>::one() * 0.5f32;
            t[direction as usize] = if negative { 0f32 } else { 1f32 };
            t
        };

        let inner_offset = inside_cell_offset - utils::unflatten_from_face_relative(vek::Vec2::new(1, 1), 0, direction).as_::<f32>();

        Some(VertexToSpawn {
            use_world_position: false,
            world_position: vek::Vec3::zero(),
            inner_offset
        })
        /*
        let vertex_position = (inside_cell_offset + pos.as_::<f32>()) * chunk_scale + chunk_position.as_::<f32>();
        let index = vertices.len();
        skirt_lookup[utils::offset_to_index_flat(relative, SKIRT_SIZE)] = index as u32;
        vertices.push(vertex_position);
        */
    } else {
        None
    }
}

fn skirt_vertices(densities: &Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32, vertices: &mut Vec<vek::Vec3<f32>>, skirt_lookup: &mut Vec<u32>, direction: u32, negative: bool, face: u32) {
    let missing = if negative { 0 } else { CHUNK_SIZE_WITH_PADDING - 3 }; 

    let data = VertexSpawningData {
        densities,
    };

    // this handles center vertices
    for rx in 0..SKIRT_SIZE {
        for ry in 0..SKIRT_SIZE {
            let relative = vek::Vec2::new(rx, ry);
            
            // mask to detect if we're dealing with an edge case or corner case
            let min_max_mask = vek::Vec4::new(relative.x, relative.y, relative.x, relative.y).cmpeq(&vek::Vec4::new(0, 0, SKIRT_SIZE-1, SKIRT_SIZE-1));
            let count = as_bitmask(min_max_mask).count_ones();

            let maybe_vertex = if (count == 2) {
                // corner case!!!!
                None
                // create_corner(&data, direction, negative, relative)
            } else if (count == 1) {
                // edge case!!!
                // None
                create_skirt_sn_1d(&data, relative, min_max_mask, negative, face, direction)
            } else {
                // normal face case!!!
                // it is safe to subtract 1 here because we know both coordinates of x,y are > 0 because this is the "middle" vertex, not an edge nor a corner
                let position = utils::unflatten_from_face_relative(relative - 1, missing, direction);
                create_skirt_sn_2d(&data, face, negative, direction, relative, position)
            };

            let unoffsetted = utils::unflatten_from_face_relative(relative, missing, direction);

            // Actually spawn the vertex if needed
            if let Some(VertexToSpawn { use_world_position, world_position, inner_offset }) = maybe_vertex {
                let mut local_vertex_position = if use_world_position {
                    world_position + inner_offset
                } else {
                    unoffsetted.as_::<f32>() + inner_offset
                };

                /*
                if (!negative) {
                    local_vertex_position[direction as usize] += 1.0f32;
                }
                */

                let index = vertices.len();
                
                let mut world_vertex_position = local_vertex_position * chunk_scale + chunk_position.as_::<f32>();
                
                
                vertices.push(world_vertex_position);
                
                skirt_lookup[utils::offset_to_index_flat(relative, SKIRT_SIZE)] = index as u32;
            }
        }
    }
}



fn skirt_quads(densities: &Vec<f32>, indices: &mut Vec<u32>, lookup: &[u32], face: u32, skirt_lookup: Vec<u32>, direction: u32) {

    let missing = if (face < 3) { 0 } else { SKIRT_SIZE - 2 }; 
    for rx in 0..(SKIRT_SIZE-1) {
        for ry in 0..(SKIRT_SIZE-1) {
            let relative = vek::Vec2::new(rx, ry);
            // log::debug!("{relative}");
            let pos = utils::unflatten_from_face_relative(relative, missing, direction);
            // log::debug!("{pos}");
            let is_set = densities[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] < 0f32;
            // log::debug!("{is_set}");
            for axis in 0..3 {
                // skip the axis of the face itself
                // if axis == direction { continue; }

                // TODO: implement force mode
        
                // next cell is the cell "forward" to the current cell based on axiss
                let mut next_cell = pos;
                next_cell[axis as usize] += 1;
            
                let next_cell_is_set = densities[utils::offset_to_index(next_cell, CHUNK_SIZE_WITH_PADDING)] < 0f32;
            
                if is_set != next_cell_is_set {
                    let mut quad_vertex_indices: [u32; 4] = [u32::MAX; 4];
                
                    let flip = is_set;
            
                    // are these the perpendicular offsets value?
                    let vertex_offsets: [vek::Vec3<usize>; 4] = utils::quad_vertex_offsets_for_axis_ccw_order_looking_from_positive_axis(axis as u32);
            
                    // inside quad local vertex index
                    for index in 0..4 {
                        // equivalent to (int3)((int3)unflattened + (int3)forward - math.int3(1)) + (int3)DirectionOffsetUtils.PERPENDICULAR_OFFSETS[index * 4 + i]
                        let target = vertex_offsets[index].as_::<i32>() + next_cell.as_::<i32>() - 1;
                
                        let looked_up_vertex_index = fetch_skirt_or_base_index(target, face, &lookup, &skirt_lookup);
                
                        // TODO: handle skirt vertex index lookup here
                        if let Some(looked_up_vertex_index) = looked_up_vertex_index {
                            quad_vertex_indices[index] = looked_up_vertex_index;
                        }
                    }
            
                    // triangulate mode based on missing/duplicated vertices
                    if let Some(mode) = try_calculate_quads_or_tris(quad_vertex_indices, flip) {
                        add_quads_or_tris(mode, indices);
                    }
                }
            }
        }
    }
}


/*
    int FetchIndex(int3 position, int face) {
        int direction = face % 3;
        int2 flattened = SkirtUtils.FlattenToFaceRelative(position, direction);
        int other = position[direction];

        // checks if we are dealing with a skirt vertex or a copied vertex in a particular direction
        if (other < 0 || other > VoxelUtils.SIZE - 3) {
            // since the skirt generated vertices have 2 padding vertices (for edges), we need to add an offset 
            flattened += 1;
            flattened = math.clamp(flattened, 0, VoxelUtils.SKIRT_SIZE);

            // lookup in the appropriate 2D index table
            int lookup = VoxelUtils.PosToIndex2D((uint2)flattened, VoxelUtils.SKIRT_SIZE);
            int res = skirtVertexIndicesGenerated[lookup + VoxelUtils.SKIRT_FACE * face];
            return res;
        } else {
            flattened = math.clamp(flattened, 0, VoxelUtils.SIZE);

            // lookup in the appropriate 2D index table
            int lookup = VoxelUtils.PosToIndex2D((uint2)flattened, VoxelUtils.SIZE);
            int res = skirtVertexIndicesCopied[lookup + VoxelUtils.FACE * face];
            return res;
        }
    }
*/

fn fetch_skirt_or_base_index(target: vek::Vec3<i32>, face: u32, lookup: &[u32], skirt_lookup_for_face: &[u32]) -> Option<u32> {
    log::trace!("fetch: {target}");
    
    let direction = face % 3;
    let face_relative = utils::flatten_to_face_relative(target, direction);
    let other = target[direction as usize];

    if (other < 0 || other > (CHUNK_SIZE_WITH_PADDING as i32 - 3)) {
        // because the skirt vertices contain padding vertices on the very edge to be able to handle edge vertices and corner vertices
        // the entire indexing scheme is offset by 1 in the relative x,y
        let face_relative = face_relative + 1; 
        
        let face_relative = face_relative.clamped(0, SKIRT_SIZE as i32 - 1);
        let flat_relative_index = utils::offset_to_index_flat(face_relative.as_::<usize>(), SKIRT_SIZE);
        
        log::trace!("skirt index: {}", skirt_lookup_for_face[flat_relative_index]);
        return Some(skirt_lookup_for_face[flat_relative_index])
    } else {
        let mut target = target;

        // reconstruct the target because of the SkirtCopyVertexIndicesJob. it uses missing as CHUNK_SIZE_WITH_PADDING-3 instead of CHUNK_SIZE_WITH_PADDING-2
        target[direction as usize] = if (face < 3) { 0 } else { CHUNK_SIZE_WITH_PADDING as i32 - 3 };

        let other = utils::try_offset_to_index(target.as_::<usize>(), CHUNK_SIZE_WITH_PADDING).map(|x| lookup[x]);
        log::trace!("base index: {:?}", &other);
        

        other
    }
}

// https://github.com/jedjoud10/VoxelTerrain/blob/ecs/Runtime/Mesher/Skirts/SkirtQuadJob.cs
const DEDUPE_TRIS_THING: [vek::Vec3<i32>;6] = [
    vek::Vec3::<i32>::new(0, 2, 3), // x/y, discard y
    vek::Vec3::<i32>::new(0, 1, 3), // x/z, discard z
    vek::Vec3::<i32>::new(0, 1, 2), // x/w, discard w
    vek::Vec3::<i32>::new(0, 1, 3), // y/z, discard z
    vek::Vec3::<i32>::new(0, 1, 2), // y/w, discard w
    vek::Vec3::<i32>::new(0, 1, 2), // z/w, discard w
];

const IGNORE_SPECIFIC_VALUE_TRI: [vek::Vec3<i32>;4] = [
    vek::Vec3::<i32>::new(1, 2, 3), // discard x
    vek::Vec3::<i32>::new(0, 2, 3), // discard y
    vek::Vec3::<i32>::new(0, 1, 3), // discard z
    vek::Vec3::<i32>::new(0, 1, 2), // discard w
];


/*
static  void AddQuadsOrTris(Triangulate triangulate, NativeCounter.Concurrent counter, NativeArray<int> indices) {
    int4 v = triangulate.indices;

    if (triangulate.triangle) {
        int triIndex = counter.Add(1) * 3;
        indices[triIndex + 0] = v[0];
        indices[triIndex + 1] = v[1];
        indices[triIndex + 2] = v[2];
    } else {
        int triIndex = counter.Add(2) * 3;

        indices[triIndex + 0] = v[0];
        indices[triIndex + 1] = v[1];
        indices[triIndex + 2] = v[2];

        indices[triIndex + 3] = v[2];
        indices[triIndex + 4] = v[3];
        indices[triIndex + 5] = v[0];
    }
}
*/
fn add_quads_or_tris(triangulate: Mode, indices: &mut Vec<u32>) {
    
    match triangulate {
        Mode::Quad(v) => {
            log::trace!("handle quad: {v}");
            indices.push(v[0]);
            indices.push(v[1]);
            indices.push(v[2]);

            indices.push(v[2]);
            indices.push(v[3]);
            indices.push(v[0]);
        },
        Mode::Tri(v) => {
            log::trace!("handle tri: {v}");
            indices.push(v[0]);
            indices.push(v[1]);
            indices.push(v[2]);
        },
    }
}

enum Mode {
    Quad(vek::Vec4<u32>),
    Tri(vek::Vec3<u32>),
}

/*
// Add quads/tris for stitched triangles (filling the gap between surface nets verts and copied verts)
// We NEED to have triangle fallback to handle the literal "edge" case (since there are only 3 vertices there)
static bool TryCalculateQuadOrTris(bool flip, int4 v, out Triangulate data) {
    data = default;

    // Ts gpt-ed kek
    int dupeType = 0;
    dupeType |= math.select(0, 1, v.x == v.y);
    dupeType |= math.select(0, 2, v.x == v.z);
    dupeType |= math.select(0, 4, v.x == v.w);
    dupeType |= math.select(0, 8, v.y == v.z && v.x != v.y);
    dupeType |= math.select(0, 16, v.y == v.w && v.x != v.y && v.z != v.y);
    dupeType |= math.select(0, 32, v.z == v.w && v.x != v.z && v.y != v.z);

    bool4 b4 = v == int.MaxValue;
    int bitmask = math.bitmask(b4);

    // Means that there are more than 2 duplicate verts, not possible?
    if (math.countbits(dupeType) > 1) {
        return false;
    }

    // Means that there are more than 2 invalid verts, not possible to create a tri nor a quad
    if (math.countbits(bitmask) > 1) {
        return false;
    }

    // If there's only a SINGLE invalid index, then consider it as an extra duplicate one (and create a triangle for the valid ones instead of a quad)
    if (math.countbits(bitmask) == 1) {
        int3 remapper = IGNORE_SPECIFIC_VALUE_TRI[math.tzcnt(bitmask)];
        int3 uniques = new int3(v[remapper[0]], v[remapper[1]], v[remapper[2]]);

        data.triangle = true;
        data.indices.x = uniques[flip ? 0 : 2];
        data.indices.y = uniques[1];
        data.indices.z = uniques[flip ? 2 : 0];

        return true;
    }

    if (dupeType == 0) {
        if (math.cmax(v) == int.MaxValue | math.cmin(v) < 0) {
            return false;
        }

        data.triangle = false;
        data.indices.x = v[flip ? 0 : 2];
        data.indices.y = v[1];
        data.indices.z = v[flip ? 2 : 0];
        data.indices.w = v[3];
        return true;
    } else {
        int config = math.tzcnt(dupeType);
        int3 remapper = DEDUPE_TRIS_THING[config];
        int3 uniques = new int3(v[remapper[0]], v[remapper[1]], v[remapper[2]]);

        if (math.cmax(uniques) == int.MaxValue | math.cmin(v) < 0) {
            return false;
        }

        data.triangle = true;
        data.indices.x = uniques[flip ? 0 : 2];
        data.indices.y = uniques[1];
        data.indices.z = uniques[flip ? 2 : 0];
        return true;
    }
}
*/
// Add quads/tris for stitched triangles (filling the gap between surface nets verts and copied verts)
// We NEED to have triangle fallback to handle the literal "edge" case (since there are only 3 vertices there)
fn try_calculate_quads_or_tris(quad_vertex_indices: [u32; 4], flip: bool) -> Option<Mode> {
    let v = vek::Vec4::<u32>::from(quad_vertex_indices);
    log::trace!("v = {v}");

    // Ts gpt-ed kek
    let mut dupe_type: u32 = 0;
    dupe_type |= u32::from(v.x == v.y);
    dupe_type |= u32::from(v.x == v.z) << 1;
    dupe_type |= u32::from(v.x == v.w) << 2;
    dupe_type |= u32::from(v.y == v.z && v.x != v.y) << 3;
    dupe_type |= u32::from(v.y == v.w && v.x != v.y && v.z != v.y) << 4;
    dupe_type |= u32::from(v.z == v.w && v.x != v.z && v.y != v.z) << 5;

    let b4: vek::Vec4<bool> = v.cmpeq(&vek::Vec4::broadcast(u32::MAX));
    let bitmask = as_bitmask(b4);

    // Means that there are more than 2 duplicate verts, not possible?
    if (dupe_type.count_ones() > 1) {
        return None;
    }

    // Means that there are more than 2 invalid verts, not possible to create a tri nor a quad
    if (bitmask.count_ones() > 1) {
        return None;
    }

    // If there's only a SINGLE invalid index, then consider it as an extra duplicate one (and create a triangle for the valid ones instead of a quad)
    if (bitmask.count_ones() == 1) {
        log::trace!("invalid index, fallbacking to try");
        let remapper = IGNORE_SPECIFIC_VALUE_TRI[bitmask.trailing_zeros() as usize].as_::<usize>();
        let uniques = vek::Vec3::new(v[remapper[0]], v[remapper[1]], v[remapper[2]]);

        return Some(
            Mode::Tri(vek::Vec3::new(
                uniques[if flip { 0 } else { 2 }],
                uniques[1],
                uniques[if flip { 2 } else { 0 }],
            ))
        );
    }

    if (dupe_type == 0) {
        log::trace!("no duplicate verts, quad");
        if (v.reduce_max() == u32::MAX) {
            return None;
        }

        return Some(
            Mode::Quad(vek::Vec4::new(
                v[if flip { 0 } else { 2 }],
                v[1],
                v[if flip { 2 } else { 0}],
                v[3],
            ))
        );
    } else {
        log::trace!("duplicate verts, triangle");

        let remapper = DEDUPE_TRIS_THING[dupe_type.trailing_zeros() as usize].as_::<usize>();
        let uniques = vek::Vec3::new(v[remapper[0]], v[remapper[1]], v[remapper[2]]);

        if (uniques.reduce_max() == u32::MAX) {
            return None;
        }

        return Some(
            Mode::Tri(vek::Vec3::new(
                uniques[if flip { 0 } else { 2 }],
                uniques[1],
                uniques[if flip { 2 } else { 0 }],
            ))
        );
    }
}
