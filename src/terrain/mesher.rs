
pub const CHUNK_SIZE: usize = super::CHUNK_SIZE; // physical chunk size
pub const CHUNK_SIZE_WITH_PADDING: usize = super::CHUNK_SIZE + 2; // logical chunk size
pub const CHUNK_VOLUME: usize = CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING;
pub const SMOOTH: bool = true;

mod core;
mod utils;
mod skirts;

pub use core::*;
pub use skirts::*;


// TODO: isolate this or use generic so that we don't need to take account the octree implementation details
pub use super::super::octree::*;

pub struct MeshingRequest {
    pub node: Node,
    pub data: Vec<f32>,
}

pub struct GeneratedChunk {
    pub node: Node,
    pub data: Vec<f32>,
    pub mesh: Option<MeshedChunk>
}

pub struct AsyncMesher {
    pub meshing_request_tx: crossbeam::channel::Sender<MeshingRequest>,
    pub meshing_result_rx: std::sync::mpsc::Receiver<GeneratedChunk>,
}

impl AsyncMesher {
    pub fn new() -> Self {
        let (result_tx, meshing_result_rx) = std::sync::mpsc::channel::<GeneratedChunk>();
        let (meshing_request_tx, request_rx) = crossbeam::channel::unbounded::<MeshingRequest>();

        let _threads = (0..4).into_iter().map(|thread_id| {
            let tx = result_tx.clone();
            let request_rx = request_rx.clone();
            let thread = std::thread::spawn(move || {
                loop {                
                    if let Ok(MeshingRequest { node, data }) = request_rx.recv() {
                        let mut positive = false;
                        let mut negative = false;
                        for index in 0..CHUNK_VOLUME  {
                            let density = data[index];
                            positive |= density >= 0f32;
                            negative |= density < 0f32;
                        }

                        let mut chunk = GeneratedChunk {
                            data,
                            mesh: None,
                            node,
                        };
                    
                        if positive == negative {
                            log::trace!("TID: {thread_id}. generating mesh for chunk");

                            
                            let chunk_position = node.position();
                            let chunk_size = node.size();
                            let chunk_scale = (chunk_size as f32 / super::CHUNK_SIZE as f32);
                    
                            let mesh = mesh_chunk(&chunk.data, chunk_position, chunk_scale);
                            log::trace!("TID: {thread_id}. vert count: {}. index count: {}", mesh.vertices.len(), mesh.indices.len());

                            if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
                                chunk.mesh = Some(mesh);
                            }
                        }

                        if let Err(err) = tx.send(chunk) {
                            log::warn!("Tx Send Error: {err}");
                            break;
                        }
                    }
                }
            });

            thread
        }).collect::<Vec<_>>();
        
        Self {
            meshing_request_tx,
            meshing_result_rx,
        }
    }
}

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
