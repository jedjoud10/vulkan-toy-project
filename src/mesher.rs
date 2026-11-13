
pub const CHUNK_SIZE: usize = 32; // physical chunk size
pub const CHUNK_SIZE_WITH_PADDING: usize = CHUNK_SIZE + 2; // logical chunk size
pub const CHUNK_VOLUME: usize = CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING;
pub const SMOOTH: bool = true;

mod core;
mod utils;
mod skirts;

pub use core::*;
pub use skirts::*;


pub struct MeshedChunk {
    pub vertices: Vec<vek::Vec3<f32>>,
    pub normals: Vec<vek::Vec3<f32>>,
    pub indices: Vec<u32>,
}

// stupid and unoptimized but works for now ig
pub fn mesh_chunk(densities: &[f32], chunk_position: vek::Vec3<i32>, chunk_scale: f32) -> MeshedChunk {
    assert_eq!(densities.len(), CHUNK_VOLUME);
    
    let mut vertices = Vec::<vek::Vec3<f32>>::new();
    let mut normals = Vec::<vek::Vec3<f32>>::new();
    
    let mut indices = Vec::<u32>::new();
    let mut lookup = vec![u32::MAX; CHUNK_VOLUME];
    let mut normals_lookup = vec![vek::Vec3::<f32>::zero(); CHUNK_VOLUME];
    
    compute_normals(&densities, &mut normals_lookup);

    // first pass will generate vertex positions
    core_vertices(&densities, chunk_position, chunk_scale, &mut vertices, &mut normals, &normals_lookup, &mut lookup);

    // second pass will generate quads
    core_quads(&densities, &mut indices, &lookup);

    for face in 0..6 {
        let mut skirt_lookup = vec![u32::MAX; skirts::SKIRT_FACE];
        let direction = face % 3;
        let negative = face < 3;

        // first pass will generate vertex positions ON THE SKIRT FACE
        skirt_vertices(&densities, &normals_lookup, chunk_position, chunk_scale, &mut vertices, &mut normals, &mut skirt_lookup, direction, negative, face);
        
        // second pass will generate quads BETWEEN the skirt vertices and base mesh
        skirt_quads(&densities, &mut indices, &lookup, face, skirt_lookup, direction);
    }
    
    MeshedChunk {
        vertices,
        normals,
        indices,
    }

}
