use std::{collections::{HashMap, HashSet, VecDeque}, sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Receiver}, thread::JoinHandle};

use ash::{util::Align, vk};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{buffer, octree::{Node, Octree, OctreeDelta}, others, ray_tracing, renderer::GraphicsContext};

// contains an 4x4x4 volume of chunks
// indexed using utils::offset_to_index(offset, 4)
pub struct PerFrameChunkGenerationData {
    pub readback_voxel_image: crate::texture_3d::Texture3D,
    pub readback_voxel_buffer: buffer::Buffer,
    pub multi_chunk_buffer: buffer::Buffer,
    pub pending_chunk_node_requests: Option<Vec<crate::octree::Node>>,
}

impl PerFrameChunkGenerationData {
    pub unsafe fn create(ctx: &GraphicsContext) -> Self {        
        let extent = vek::Extent3::<u32> {
            w: super::MULTI_CHUNK_TEXTURE_SIZE as u32,
            h: super::MULTI_CHUNK_TEXTURE_SIZE as u32,
            d: super::MULTI_CHUNK_TEXTURE_SIZE as u32,
        };

        let readback_voxel_image = crate::texture_3d::create_texture_3d(ctx, extent, vk::Format::R32_SFLOAT, None, "");
        let readback_voxel_buffer = crate::buffer::create_buffer_with_location_with_flags_super(ctx, size_of::<f32>() * super::MULTI_CHUNK_TEXTURE_VOLUME as usize, "", vk::BufferUsageFlags::STORAGE_BUFFER, gpu_allocator::MemoryLocation::GpuToCpu);
        let multi_chunk_buffer = crate::buffer::create_buffer_gpu_default_flags(ctx, size_of::<vek::Vec4<f32>>() * super::NUM_MULTI_CHUNKS, "");

        Self {
            readback_voxel_image,
            readback_voxel_buffer,
            multi_chunk_buffer,
            pending_chunk_node_requests: None,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, cmd_pool: vk::CommandPool, allocator: &mut Allocator) {
        self.readback_voxel_image.destroy(device, allocator);
        self.readback_voxel_buffer.destroy(device, allocator);
        self.multi_chunk_buffer.destroy(device, allocator);
    }
}