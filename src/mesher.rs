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


pub struct MeshedChunk {
    pub vertices: Vec<vek::Vec3<f32>>,
    pub indices: Vec<u32>,
}

// stupid and unoptimized but works for now ig
pub fn mesh_chunk(densities: Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32) -> MeshedChunk {
    let mut vertices = Vec::<vek::Vec3<f32>>::new();
    
    let mut indices = Vec::<u32>::new();
    let mut lookup = vec![0u32; CHUNK_VOLUME];

    // first pass will generate vertex positions
    for x in 0..(CHUNK_SIZE_WITH_PADDING-1) {
        for y in 0..(CHUNK_SIZE_WITH_PADDING-1) {
            for z in 0..(CHUNK_SIZE_WITH_PADDING-1) {
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
                    let inside_cell_offset = vertex / count as f32;
                    let vertex_position = (inside_cell_offset + pos.as_::<f32>()) * chunk_scale + chunk_position.as_::<f32>();
                    let index = vertices.len();
                    lookup[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] = index as u32;
                    vertices.push(vertex_position);
                }
            }
        }
    }

    // second pass will generate quads
    for x in 1..(CHUNK_SIZE_WITH_PADDING-1) {
        for y in 1..(CHUNK_SIZE_WITH_PADDING-1) {
            for z in 1..(CHUNK_SIZE_WITH_PADDING-1) {
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


pub fn mesh_skirt(densities: Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32, MeshedChunk { vertices, indices }: &mut MeshedChunk, lookup: &[u32]) {
    let mut skirt_lookup = vec![0u32; CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING];

    // first pass will generate vertex positions ON THE SKIRT FACE
    for x in 0..(CHUNK_SIZE_WITH_PADDING-1) {
        for y in 0..(CHUNK_SIZE_WITH_PADDING-1) {
            // assume z=0
            let z = 0;
            let pos = vek::Vec3::new(x,y,z);

            let mut vertex = vek::Vec3::<f32>::zero();
            let mut count = 0;

            // in 2D there are only 4 edges to check
            // conveniently, for the xy plane, these are the first 4
            for edge in 0..4 {
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
                let inside_cell_offset = vertex / count as f32;
                let vertex_position = (inside_cell_offset + pos.as_::<f32>()) * chunk_scale + chunk_position.as_::<f32>();
                let index = vertices.len();
                skirt_lookup[utils::offset_to_index_flat(pos.xy(), CHUNK_SIZE_WITH_PADDING)] = index as u32;
                vertices.push(vertex_position);
            }
        }
    }



    // second pass will generate quads BETWEEN the skirt vertices and base mesh
    for x in 1..(CHUNK_SIZE_WITH_PADDING-1) {
        for y in 1..(CHUNK_SIZE_WITH_PADDING-1) {
            let z = 0;

            let pos = vek::Vec3::new(x,y,z);
            let is_set = densities[utils::offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] < 0f32;
                
            for axis in 0..3 {
                // will only look in the Y and X axii, not the Z
                if axis == 2 {
                    continue;
                }

                // next cell is the cell "forward" to the current cell based on axiss
                let mut next_cell = pos;
                next_cell[axis] += 1;
                
                let next_cell_is_set = densities[utils::offset_to_index(next_cell, CHUNK_SIZE_WITH_PADDING)] < 0f32;
                
                if is_set != next_cell_is_set {
                    let mut quad_vertex_indices: [u32; 4] = [u32::MAX; 4];
                    
                    let flip = is_set ^ (axis == 1);

                    // are these the perpendicular offsets value?
                    let vertex_offsets: [vek::Vec3<usize>; 4] = utils::quad_vertex_offsets_for_axis(axis as u32);

                    // inside quad local vertex index
                    for index in 0..4 {
                        let target = vertex_offsets[index].as_::<i32>() + next_cell.as_::<i32>() - 1;

                        let looked_up_vertex_index = fetch_skirt_or_base_index(target, &lookup, &skirt_lookup);

                        // TODO: handle skirt vertex index lookup here
                        if let Some(looked_up_vertex_index) = looked_up_vertex_index {
                            quad_vertex_indices[index] = looked_up_vertex_index;
                        }
                        // if let Some(looked_up_vertex_index) = utils::try_offset_to_index(target, CHUNK_SIZE_WITH_PADDING).map(|x| lookup[x]) {
                        //     quad_vertex_indices[index] = looked_up_vertex_index;
                        // }
                    }

                    // triangulate mode based on missing/duplicated vertices
                    if let Some(mode) = try_calculate_quads_or_tris(quad_vertex_indices, flip) {
                        add_quads_or_tris(mode, indices);
                    } 

                    // // holy cursed
                    // let quad_vertex_order = if dir { INDEX_QUAD_ORDER } else { INDEX_OPPOSITE_QUAD_ORDER };
                    // for what_to_call_this_index in 0..6 {
                    //     indices.push(quad_vertex_indices[quad_vertex_order[what_to_call_this_index]]);
                    // }
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
// assume xy plane, z=0
fn fetch_skirt_or_base_index(target: vek::Vec3<i32>, lookup: &[u32], skirt_lookup: &[u32]) -> Option<u32> {
    let face_relative = target.xy();
    let other = target.z;

    if (other < 0/* || other > (CHUNK_SIZE_WITH_PADDING - 3)*/) {
        // let face_relative = face_relative + 1; 
        let face_relative = face_relative.clamped(0, CHUNK_SIZE_WITH_PADDING as i32);
        let a = utils::offset_to_index_flat(face_relative.as_::<usize>(), CHUNK_SIZE_WITH_PADDING);
        return Some(skirt_lookup[a])
    } else {
        utils::try_offset_to_index(target.as_::<usize>(), CHUNK_SIZE_WITH_PADDING).map(|x| lookup[x])
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
            indices.push(v[0]);
            indices.push(v[1]);
            indices.push(v[2]);

            indices.push(v[2]);
            indices.push(v[3]);
            indices.push(v[0]);
        },
        Mode::Tri(v) => {
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

    // Ts gpt-ed kek
    let mut dupe_type: u32 = 0;
    dupe_type |= u32::from(v.x == v.y);
    dupe_type |= u32::from(v.x == v.z) << 1;
    dupe_type |= u32::from(v.x == v.w) << 2;
    dupe_type |= u32::from(v.y == v.z && v.x != v.y) << 3;
    dupe_type |= u32::from(v.y == v.w && v.x != v.y && v.z != v.y) << 4;
    dupe_type |= u32::from(v.z == v.w && v.x != v.z && v.y != v.z) << 5;

    let b4: vek::Vec4<bool> = v.cmpeq(&vek::Vec4::broadcast(u32::MAX));
    let bitmask = b4.as_::<u32>() << vek::Vec4::new(0,1,2,3);
    let bitmask = bitmask.reduce_bitor();

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
