use std::{collections::{HashMap, HashSet, VecDeque}, sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Receiver}, thread::JoinHandle};

use ash::{util::Align, vk};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{buffer, octree::{Node, Octree, OctreeDelta}, others, ray_tracing, renderer::GraphicsContext};

pub struct Chunk {    
    pub persistent_mesh_instance_key: Option<slotmap::DefaultKey>,
    pub pending_destruction: Option<super::PersistentChunkRendererInstance>,
    pub data: Option<Vec<f32>>,
}

// xyz is position, w is scale
// local to world: `p*r.w + r.xyz` where `r` is the result and `p` is the local voxel
pub fn get_chunk_world_position_and_scale(node: &Node) -> vek::Vec4<f32> {
    let chunk_position = node.position().as_::<f32>();
    let chunk_size = node.size();
    let chunk_scale = chunk_size as f32 / super::CHUNK_SIZE as f32;
    let chunk_position_and_scale = chunk_position.with_w(chunk_scale);
    chunk_position_and_scale
}
