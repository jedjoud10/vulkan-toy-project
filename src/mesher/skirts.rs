use vek::Clamp;
use super::*;
use super::utils;

// this uses the +2 padding because we need to store an extra vertex per-skirt face (to handle the *literal* edge case)
pub const SKIRT_SIZE: usize = CHUNK_SIZE_WITH_PADDING;
pub const SKIRT_FACE: usize = SKIRT_SIZE*SKIRT_SIZE;

// there are three types of skirt vertices
// middle: vertices in the "x,y in [1..SKIRT_SIZE)" area in the middle of the skirt 
// edge: vertices on the literal edge of the skirt boundary. "x,y in [1..SKIRT_SIZE) where one of x,y is set to 0 or (SKIRT_SIZE-1)"
// corner: corner vertices of the chunk, only a maximum of 8 could exist at a time "(x,y) in { (0,0), (SKIRT_SIZE-1, 0), (0, SKIRT_SIZE-1), (SKIRT_SIZE-1, SKIRT_SIZE-1)  }" 
struct VertexToSpawn {
    use_world_position: bool,
    world_position: vek::Vec3<f32>,
    inner_offset: vek::Vec3<f32>,
}

#[derive(Clone, Copy)]
struct VertexSpawningData<'a> {
    densities: &'a [f32],
    direction: u32,
    negative: bool,
    missing: usize,
    relative: vek::Vec2<usize>,
    min_max_mask: vek::Vec4<bool>,
}

// TODO: implement this corner case
fn create_corner(data: &VertexSpawningData) -> Option<VertexToSpawn> {
    None
}

fn create_skirt_sn_1d(data: &VertexSpawningData) -> Option<VertexToSpawn> {
    let VertexSpawningData { densities, direction, negative, relative, min_max_mask, .. } = *data;

    let mask = vek::Vec2::new(min_max_mask.x || min_max_mask.z, min_max_mask.y || min_max_mask.w);
    let edge_direction_index = edge_dir_face_relative(mask, direction);
    let edge_direction = utils::FORWARD[edge_direction_index as usize];
    // log::trace!("{min_max_mask} {negative} {face} {direction} {}", edge_direction);
    
        
    let missing = if negative { 0 } else { CHUNK_SIZE_WITH_PADDING - 2 };

    let relative = relative.clamped(0, CHUNK_SIZE_WITH_PADDING - 2);
    let world_position = utils::unflatten_from_face_relative(relative.as_::<f32>(), missing as f32, direction);

    let unoffsetted = utils::unflatten_from_face_relative(relative, missing, direction);
    // log::trace!("{} {}", unoffsetted - edge_direction.as_::<usize>(), unoffsetted);
    

    let p1_offsetted_index = utils::offset_to_index(unoffsetted - edge_direction.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
    let p2_offsetted_index = utils::offset_to_index(unoffsetted, CHUNK_SIZE_WITH_PADDING as usize);
    
    let d1 = densities[p1_offsetted_index];
    let d2 = densities[p2_offsetted_index];

    // log::trace!("{} {} {} {}", p1_offsetted_index, p2_offsetted_index, d1, d2);
    
    if (d1 < 0f32) ^ (d2 < 0f32) {
        log::trace!("adding edge vertex");

        let factor = if SMOOTH {
            utils::unlerp(0f32, d1, d2)
        } else {
            0.5f32
        };

        let inner_offset = vek::Vec3::lerp(-edge_direction.as_::<f32>(), vek::Vec3::zero(), factor);

        Some(VertexToSpawn {
            use_world_position: true,
            world_position,
            inner_offset
        })
        
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

    match (face_normal, mask.x, mask.y) {
        // X-normal plane
        (0, true, false) => 1, // Y
        (0, false, true) => 2, // Z
        
        // Y-normal plane
        (1, true, false) => 0, // X
        (1, false, true) => 2, // Z
        
        // Z-normal plane
        (2, true, false) => 0, // X
        (2, false, true) => 1, // Y

        _ => unreachable!()
    }
}

fn create_skirt_sn_2d(data: &VertexSpawningData) -> Option<VertexToSpawn> {
    let VertexSpawningData { direction, negative, relative, missing, densities, .. } = *data;

    // it is safe to subtract 1 here because we know both coordinates of x,y are > 0 because this is the "middle" vertex, not an edge nor a corner
    let position = utils::unflatten_from_face_relative(relative - 1, missing, direction);

    let mut vertex = vek::Vec3::<f32>::zero();
    let mut count = 0;

    // this is INSIDE cell, so offset is 1 and not the chunk size (we add the chunk offset position afterwards)
    let missing = if negative { 0 } else { 1 } as u32;
    
    for edge in 0..4 {
        let p1_relative_offsetted = utils::SKIRT_2D_EDGE_POSITIONS_0[edge];
        let p2_relative_offsetted = utils::SKIRT_2D_EDGE_POSITIONS_1[edge];

        let p1_offsetted = utils::unflatten_from_face_relative(p1_relative_offsetted, missing, direction);
        let p2_offsetted = utils::unflatten_from_face_relative(p2_relative_offsetted, missing, direction);

        let p1_offsetted_index = utils::offset_to_index(position + p1_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
        let p2_offsetted_index = utils::offset_to_index(position + p2_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
        
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
    } else {
        None
    }
}

pub fn skirt_vertices(densities: &Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32, vertices: &mut Vec<vek::Vec3<f32>>, skirt_lookup: &mut Vec<u32>, direction: u32, negative: bool, face: u32) {
    let missing = if negative { 0 } else { CHUNK_SIZE_WITH_PADDING - 3 }; 

    // this handles center vertices
    for rx in 0..SKIRT_SIZE {
        for ry in 0..SKIRT_SIZE {
            let relative = vek::Vec2::new(rx, ry);
            
            // mask to detect if we're dealing with an edge case or corner case
            let min_max_mask = vek::Vec4::new(relative.x, relative.y, relative.x, relative.y).cmpeq(&vek::Vec4::new(0, 0, SKIRT_SIZE-1, SKIRT_SIZE-1));
            let count = utils::as_bitmask(min_max_mask).count_ones();

            let data = VertexSpawningData {
                densities,
                direction,
                negative,
                missing,
                relative,
                min_max_mask,
            };

            let maybe_vertex = if count == 2 {
                // corner case!!!!
                create_corner(&data)
            } else if (count == 1) {
                // edge case!!!
                // None
                create_skirt_sn_1d(&data)
            } else {
                // normal face case!!!
                create_skirt_sn_2d(&data)
            };

            let unoffsetted = utils::unflatten_from_face_relative(relative, missing, direction);

            // actually spawn the vertex if needed
            if let Some(VertexToSpawn { use_world_position, world_position, inner_offset }) = maybe_vertex {
                let local_vertex_position = if use_world_position {
                    world_position + inner_offset
                } else {
                    unoffsetted.as_::<f32>() + inner_offset
                };
                
                let index = vertices.len();
                let world_vertex_position = local_vertex_position * chunk_scale + chunk_position.as_::<f32>();
                vertices.push(world_vertex_position);
                
                skirt_lookup[utils::offset_to_index_flat(relative, SKIRT_SIZE)] = index as u32;
            }
        }
    }
}



pub fn skirt_quads(densities: &Vec<f32>, indices: &mut Vec<u32>, lookup: &[u32], face: u32, skirt_lookup: Vec<u32>, direction: u32) {

    let missing = if face < 3 { 0 } else { SKIRT_SIZE - 2 }; 
    for rx in 0..(SKIRT_SIZE-1) {
        for ry in 0..(SKIRT_SIZE-1) {
            let relative = vek::Vec2::new(rx, ry);
            let pos = utils::unflatten_from_face_relative(relative, missing, direction);
            let is_set = densities[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] < 0f32;

            for axis in 0..3 {
                // skip the axis of the face itself
                // TODO: implement force mode
                // if axis == direction { continue; }

        
                check_edge(densities, indices, lookup, face, &skirt_lookup, pos, is_set, axis);
            }
        }
    }
}

// https://github.com/jedjoud10/VoxelTerrain/blob/e949784031a0b32104f467e3c3d2138f54286dd4/Runtime/Mesher/Skirts/SkirtQuadJob.cs#L58-L114
fn check_edge(densities: &Vec<f32>, indices: &mut Vec<u32>, lookup: &[u32], face: u32, skirt_lookup: &[u32], pos: vek::Vec3<usize>, is_set: bool, axis: i32) {
    // next cell is the cell "forward" to the current cell based on axiss
    let mut next_cell = pos;
    next_cell[axis as usize] += 1;
            
    let next_cell_is_set = densities[utils::offset_to_index(next_cell, CHUNK_SIZE_WITH_PADDING)] < 0f32;
            
    if is_set != next_cell_is_set {
        let mut quad_vertex_indices: vek::Vec4::<u32> = vek::Vec4::broadcast(u32::MAX);

        let flip = !is_set;
            
        // are these the perpendicular offsets value?
        let vertex_offsets: [vek::Vec3<usize>; 4] = utils::quad_vertex_offsets_for_axis_ccw_order_looking_from_positive_axis(axis as u32);
            
        // inside quad local vertex index
        for index in 0..4 {
            // equivalent to (int3)((int3)unflattened + (int3)forward - math.int3(1)) + (int3)DirectionOffsetUtils.PERPENDICULAR_OFFSETS[index * 4 + i]
            let target = vertex_offsets[index].as_::<i32>() + next_cell.as_::<i32>() - 1;

            let looked_up_vertex_index = fetch_skirt_or_base_index(target, face, &lookup, skirt_lookup);

            // TODO: handle skirt vertex index lookup here
            if let Some(looked_up_vertex_index) = looked_up_vertex_index {
                quad_vertex_indices[index] = looked_up_vertex_index;
            }
        }
            
        // triangulate mode based on missing/duplicated vertices
        if let Some(quad_or_tri) = try_calculate_quads_or_tris(quad_vertex_indices) {
            utils::add_quads_or_tris(quad_or_tri, flip, indices);
        }
    }
}

// https://github.com/jedjoud10/VoxelTerrain/blob/e949784031a0b32104f467e3c3d2138f54286dd4/Runtime/Mesher/Skirts/SkirtQuadJob.cs#L32-L55
fn fetch_skirt_or_base_index(target: vek::Vec3<i32>, face: u32, lookup: &[u32], skirt_lookup_for_face: &[u32]) -> Option<u32> {
    log::trace!("fetch: {target}");
    
    let direction = face % 3;
    let face_relative = utils::flatten_to_face_relative(target, direction);
    let other = target[direction as usize];

    if other < 0 || other > (CHUNK_SIZE_WITH_PADDING as i32 - 3) {
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

// Add quads/tris for stitched triangles (filling the gap between surface nets verts and copied verts)
// We NEED to have triangle fallback to handle the literal "edge" case (since there are only 3 vertices there)
// https://github.com/jedjoud10/VoxelTerrain/blob/e949784031a0b32104f467e3c3d2138f54286dd4/Runtime/Mesher/Skirts/SkirtQuadJob.cs#L161-L225
fn try_calculate_quads_or_tris(v: vek::Vec4::<u32>) -> Option<utils::QuadOrTri> {
    log::trace!("v = {v}");

    // this checks which vertices are duplicate to which, and stores it in a bitmask
    // the trailing zeros of this bitmask are used to index into DEDUPE_TRIS_THING were we ignore one of the duplicate indices
    // this has been generated by ChatGPT in the original implementation
    let mut dupe_type: u32 = 0;
    dupe_type |= u32::from(v.x == v.y);
    dupe_type |= u32::from(v.x == v.z) << 1;
    dupe_type |= u32::from(v.x == v.w) << 2;
    dupe_type |= u32::from(v.y == v.z && v.x != v.y) << 3;
    dupe_type |= u32::from(v.y == v.w && v.x != v.y && v.z != v.y) << 4;
    dupe_type |= u32::from(v.z == v.w && v.x != v.z && v.y != v.z) << 5;

    // create a bitmask that compares the quad indices to the invalid indices (u32::MAX)
    let b4: vek::Vec4<bool> = v.cmpeq(&vek::Vec4::broadcast(u32::MAX));
    let bitmask = utils::as_bitmask(b4);

    // means that there are more than 2 duplicate verts, not possible?
    if dupe_type.count_ones() > 1 {
        return None;
    }

    // means that there are more than 2 invalid verts, not possible to create a tri nor a quad
    if bitmask.count_ones() > 1 {
        return None;
    }

    // if there's only a SINGLE invalid index, then consider it as an extra duplicate one (and create a triangle for the valid ones instead of a quad)
    if bitmask.count_ones() == 1 {
        log::trace!("invalid index, fallbacking to try");
        let remapper = IGNORE_SPECIFIC_VALUE_TRI[bitmask.trailing_zeros() as usize].as_::<usize>();
        let uniques = vek::Vec3::new(v[remapper[0]], v[remapper[1]], v[remapper[2]]);

        return Some(
            utils::QuadOrTri::Tri(uniques)
        );
    }

    if dupe_type == 0 {
        log::trace!("no duplicate verts, quad");
        if v.reduce_max() == u32::MAX {
            return None;
        }

        return Some(
            utils::QuadOrTri::Quad(v)
        );
    } else {
        log::trace!("duplicate verts, triangle");

        let remapper = DEDUPE_TRIS_THING[dupe_type.trailing_zeros() as usize].as_::<usize>();
        let uniques = vek::Vec3::new(v[remapper[0]], v[remapper[1]], v[remapper[2]]);

        if uniques.reduce_max() == u32::MAX {
            return None;
        }

        return Some(
            utils::QuadOrTri::Tri(uniques)
        );
    }
}
