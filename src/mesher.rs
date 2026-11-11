
pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_WITH_PADDING: usize = 34;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING;
pub const SMOOTH: bool = true;

mod core;
mod utils;
mod skirts;

pub use core::*;
pub use skirts::*;


pub struct MeshedChunk {
    pub vertices: Vec<vek::Vec3<f32>>,
    pub indices: Vec<u32>,
}

// stupid and unoptimized but works for now ig
pub fn mesh_chunk(densities: Vec<f32>, chunk_position: vek::Vec3<i32>, chunk_scale: f32) -> MeshedChunk {
    let mut vertices = Vec::<vek::Vec3<f32>>::new();
    
    let mut indices = Vec::<u32>::new();
    let mut lookup = vec![u32::MAX; CHUNK_VOLUME];

    // first pass will generate vertex positions
    core_vertices(&densities, chunk_position, chunk_scale, &mut vertices, &mut lookup);

    // second pass will generate quads
    core_quads(&densities, &mut indices, &lookup);

    for face in 0..6 {
        let mut skirt_lookup = vec![u32::MAX; skirts::SKIRT_FACE];
        let direction = face % 3;
        let negative = face < 3;

        // first pass will generate vertex positions ON THE SKIRT FACE
        skirt_vertices(&densities, chunk_position, chunk_scale, &mut vertices, &mut skirt_lookup, direction, negative, face);
        
        // second pass will generate quads BETWEEN the skirt vertices and base mesh
        skirt_quads(&densities, &mut indices, &lookup, face, skirt_lookup, direction);
    }
    
    MeshedChunk {
        vertices,
        indices,
    }

}
