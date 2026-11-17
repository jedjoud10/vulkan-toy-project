pub const VERTEX_STRIDE: usize = size_of::<vek::Vec3::<f32>>();
pub const INDEX_STRIDE: usize = size_of::<u32>();

pub const CHUNK_SIZE: usize = 32; // physical chunk size
pub const CHUNK_SIZE_WITH_PADDING: usize = CHUNK_SIZE + 2; // logical chunk size
pub const CHUNK_VOLUME: usize = CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING;
pub const SMOOTH: bool = true;

pub const MULTI_CHUNK_TEXTURE_SIZE: usize = CHUNK_SIZE_WITH_PADDING * 4;
pub const MULTI_CHUNK_TEXTURE_VOLUME: usize = MULTI_CHUNK_TEXTURE_SIZE*MULTI_CHUNK_TEXTURE_SIZE*MULTI_CHUNK_TEXTURE_SIZE;
pub const NUM_MULTI_CHUNKS: usize = 64;

pub const ENABLED: bool = true;
pub const FORCE_ONE_CHUNK_PER_MULTI_CHUNK: bool = false;

mod mesher;
mod other;
mod per_frame_data;
mod renderer;
mod terrain;

pub use mesher::*;
pub use other::*;
pub use per_frame_data::*;
pub use renderer::*;
pub use terrain::*;