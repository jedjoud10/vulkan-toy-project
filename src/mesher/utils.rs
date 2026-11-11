
pub fn offset_to_index(offset: vek::Vec3<usize>, size: usize) -> usize {
    assert!(offset.cmpge(&vek::Vec3::broadcast(0)).reduce_and());
    assert!(offset.cmplt(&vek::Vec3::broadcast(size)).reduce_and());
    
    offset.x + offset.y * size + offset.z * size * size
}

pub fn index_to_offset(index: usize, size: usize) -> vek::Vec3<usize> {
    assert!(index < (size*size*size));
    
    let x: usize = index % size;
    let y = (index / size) % size;
    let z = index / (size*size);
    vek::Vec3::new(x,y,z)
}


pub fn offset_to_index_flat(offset: vek::Vec2<usize>, size: usize) -> usize {
    assert!(offset.cmpge(&vek::Vec2::broadcast(0)).reduce_and());
    assert!(offset.cmplt(&vek::Vec2::broadcast(size)).reduce_and());
    
    offset.x + offset.y * size
}

pub fn index_to_offset_flat(index: usize, size: usize) -> vek::Vec2<usize> {
    assert!(index < (size*size));
    
    let x: usize = index % size;
    let y = (index / size) % size;
    vek::Vec2::new(x,y)
}

// ok temporary solution time
pub fn quad_vertex_offsets_for_axis_ccw_order_looking_from_positive_axis(axis: u32) -> [vek::Vec3<usize>; 4] {
    match axis {
        0 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(0, 1, 0), vek::Vec3::new(0,1, 1), vek::Vec3::new(0, 0, 1)], // x
        1 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(0, 0, 1), vek::Vec3::new(1, 0, 1), vek::Vec3::new(1, 0, 0)], // y
        2 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(1, 0, 0), vek::Vec3::new(1, 1, 0), vek::Vec3::new(0, 1, 0)], // z
        _ => unreachable!()
    }
}

// https://github.com/jedjoud10/VoxelTerrain/blob/e949784031a0b32104f467e3c3d2138f54286dd4/Runtime/Utils/SkirtUtils.cs#L32-L45
pub fn flatten_to_face_relative<T>(pos: vek::Vec3<T>, direction: u32) -> vek::Vec2<T> {
    assert!(direction <= 2);

    match direction {
        0 => vek::Vec2::new(pos.y,pos.z),
        1 => vek::Vec2::new(pos.x,pos.z),
        2 => vek::Vec2::new(pos.x,pos.y),
        _ => unreachable!()
    }
}

// https://github.com/jedjoud10/VoxelTerrain/blob/e949784031a0b32104f467e3c3d2138f54286dd4/Runtime/Utils/SkirtUtils.cs#L50-L63
pub fn unflatten_from_face_relative<T>(relative: vek::Vec2<T>, missing: T, direction: u32) -> vek::Vec3<T> {
    assert!(direction <= 2);

    match direction {
        0 => vek::Vec3::new(missing, relative.x, relative.y),
        1 => vek::Vec3::new(relative.x, missing, relative.y),
        2 => vek::Vec3::new(relative.x, relative.y, missing),
        _ => unreachable!()
    }
}

pub fn try_offset_to_index(offset: vek::Vec3<usize>, size: usize) -> Option<usize> {
    if offset.cmpge(&vek::Vec3::broadcast(0)).reduce_and() && offset.cmplt(&vek::Vec3::broadcast(size)).reduce_and() {
        Some(offset.x + offset.y * size + offset.z * size * size)
    } else {
        None
    }
}

pub const FORWARD: [vek::Vec3::<i32>; 3] = [
    // vek::Vec3::new(-1, 0, 0),
    vek::Vec3::new(1, 0, 0),
    // vek::Vec3::new(0, -1, 0),
    vek::Vec3::new(0, 1, 0),
    // vek::Vec3::new(0, 0, -1),
    vek::Vec3::new(0, 0, 1),
];

pub fn unlerp(x: f32, a: f32, b: f32) -> f32 {
    return (x - a) / (b - a);
}

pub const EDGE_POSITIONS_0: [vek::Vec3<u32>; 12] = [
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

pub const EDGE_POSITIONS_1: [vek::Vec3<u32>; 12] = [
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


pub const SKIRT_2D_EDGE_POSITIONS_0: [vek::Vec2<u32>; 4] = [
    vek::Vec2::new(0, 0),
    vek::Vec2::new(0, 1),
    vek::Vec2::new(1, 1),
    vek::Vec2::new(1, 0),
];

pub const SKIRT_2D_EDGE_POSITIONS_1: [vek::Vec2<u32>; 4] = [
    vek::Vec2::new(0, 1),
    vek::Vec2::new(1, 1),
    vek::Vec2::new(1, 0),
    vek::Vec2::new(0, 0),
];


/*
0 --- 3
|     |
|     |
1 --- 2
*/
pub fn add_ccw_quad(quad_vertex_indices: vek::Vec4::<u32>, indices: &mut Vec<u32>) {
    const INDEX_QUAD_ORDER: [usize; 6] = [0, 1, 2, 0, 2, 3];
    for what_to_call_this_index in 0..6 {
        indices.push(quad_vertex_indices[INDEX_QUAD_ORDER[what_to_call_this_index]]);
    }
}

/*
0 --- 3
|     |
|     |
1 --- 2
*/
pub fn add_cw_quad(quad_vertex_indices: vek::Vec4::<u32>, indices: &mut Vec<u32>) {
    const INDEX_OPPOSITE_QUAD_ORDER: [usize; 6] = [2, 1, 0, 3, 2, 0];
    for what_to_call_this_index in 0..6 {
        indices.push(quad_vertex_indices[INDEX_OPPOSITE_QUAD_ORDER[what_to_call_this_index]]);
    }
}

pub fn add_ccw_tri(tri_vertex_indices: vek::Vec3::<u32>, indices: &mut Vec<u32>) {
    indices.push(tri_vertex_indices[0]);
    indices.push(tri_vertex_indices[1]);
    indices.push(tri_vertex_indices[2]);
}

pub fn add_cw_tri(tri_vertex_indices: vek::Vec3::<u32>, indices: &mut Vec<u32>) {
    indices.push(tri_vertex_indices[2]);
    indices.push(tri_vertex_indices[1]);
    indices.push(tri_vertex_indices[0]);
}

pub enum QuadOrTri {
    Quad(vek::Vec4<u32>),
    Tri(vek::Vec3<u32>),
}

// https://github.com/jedjoud10/VoxelTerrain/blob/e949784031a0b32104f467e3c3d2138f54286dd4/Runtime/Mesher/Skirts/SkirtQuadJob.cs#L138-L157
pub fn add_quads_or_tris(quad_or_tri: QuadOrTri, flip: bool, indices: &mut Vec<u32>) {
    match quad_or_tri {
        QuadOrTri::Quad(v) => {
            log::trace!("handle quad: {v}, flip: {flip}");

            if flip {
                add_cw_quad(v, indices);
            } else {
                add_ccw_quad(v, indices);
            }
        },

        // TODO: the only way a triangle can be generated is if we connect inner skirt vertex to edge skirt vertex and main mesh vertex
        // we should NOT generate triangles in the "inner part" of the skirt (i.e not edges or corners)
        // very easy fix for this, just have to implement it :P
        // just handle this in `try_calculate_quads_or_tris`
        QuadOrTri::Tri(v) => {
            log::trace!("handle tri: {v}, flip: {flip}");

            if flip {
                add_cw_tri(v, indices);
            } else {
                add_ccw_tri(v, indices);
            }
        },
    }
}

pub fn as_bitmask(b4: vek::Vec4<bool>) -> u32 {
    let bitmask = b4.as_::<u32>() << vek::Vec4::new(0,1,2,3);
    bitmask.reduce_bitor()
}