use std::{collections::VecDeque, time::Instant};
use crate::renderer::GraphicsContext;
use crate::utils::*;
use crate::voxel::chunk::CHUNK_SIZE;
use ash::vk;
use bytemuck::bytes_of;
use gpu_allocator::vulkan::Allocator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use vek::Clamp;
use crate::buffer::{self, Buffer, ScratchBuffer};
use super::{SVO_DEPTH, TOTAL_SIZE, FULL_NODE};
use super::chunk::Chunk;

pub struct TestingStructure {
    pub buffer: Buffer,
}

impl TestingStructure {
    pub unsafe fn new(
        ctx: &mut GraphicsContext
    ) -> Self {
        let buffer = buffer::create_buffer(ctx, 2 * size_of::<u64>(), "oogaa boooga", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);

        Self {
            buffer
        }
    }

    pub fn register_chunk(&mut self, chunk: Chunk) {
    }

    pub unsafe fn rebuild(&mut self, ctx: &mut GraphicsContext, cmd: vk::CommandBuffer, scratch_buffer: &mut ScratchBuffer) {
        let bitmasks = [0xABCD_1234_5678_EFABu64, 0u64];
        let bytes = bytes_of(&bitmasks);

        buffer::write_with_scratch_buffer(ctx, cmd, scratch_buffer, bytes, self.buffer.buffer, 0);
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.buffer.destroy(device, allocator);
    }
}