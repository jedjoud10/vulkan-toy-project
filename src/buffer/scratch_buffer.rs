
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use crate::{buffer::{MIN_BUFFER_ALIGNMENT, create_buffer_with_location_with_flags_super}, renderer::GraphicsContext};
use super::{Buffer, create_buffer_gpu_with_flags_super};


pub const SCRATCH_BUFFER_SIZE: usize = 1024*1024*256; // ~256mB

pub unsafe fn create_buffer_write_with_scratch_buffer(
    ctx: &GraphicsContext,
    cmd: vk::CommandBuffer,
    scratch_buffer: &mut ScratchBuffer,
    bytes: &[u8],
    name: &str,
    flags: vk::BufferUsageFlags,
) -> Buffer {
    let buffer = create_buffer_gpu_with_flags_super(ctx, bytes.len(), name, flags);

    let written = scratch_buffer.write_bytes(bytes);

    let region = vk::BufferCopy2::default()
        .dst_offset(0)
        .size(bytes.len() as u64)
        .src_offset(written.buffer_offset_start);

    let regions = [region];
    let copy_staging_buffer_to_buffer = vk::CopyBufferInfo2::default()
        .dst_buffer(buffer.buffer)
        .regions(&regions)
        .src_buffer(scratch_buffer.buffer);

    ctx.device.cmd_copy_buffer2(cmd, &copy_staging_buffer_to_buffer);

    buffer
}

pub unsafe fn write_with_scratch_buffer(
    ctx: &GraphicsContext,
    cmd: vk::CommandBuffer,
    scratch_buffer: &mut ScratchBuffer,
    bytes: &[u8],
    buffer: vk::Buffer,
    offset: u64,
) {
    if bytes.is_empty() {
        return;
    }

    let written = scratch_buffer.write_bytes(bytes);

    let region = vk::BufferCopy2::default()
        .dst_offset(offset)
        .size(bytes.len() as u64)
        .src_offset(written.buffer_offset_start);

    let regions = [region];
    let copy_staging_buffer_to_buffer = vk::CopyBufferInfo2::default()
        .dst_buffer(buffer)
        .regions(&regions)
        .src_buffer(scratch_buffer.buffer);

    ctx.device.cmd_copy_buffer2(cmd, &copy_staging_buffer_to_buffer);
}

pub unsafe fn write_with_scratch_buffer_safer(
    ctx: &GraphicsContext,
    cmd: vk::CommandBuffer,
    scratch_buffer: &mut ScratchBuffer,
    bytes: &[u8],
    buffer: &Buffer,
    offset: u64,
) {
    if bytes.is_empty() {
        return;
    }

    assert!((bytes.len() as u64 + offset) <= buffer.size as u64, "writing {} bytes at offset {} would overrun buffer of size {}", bytes.len(), offset, buffer.size);

    write_with_scratch_buffer(ctx, cmd, scratch_buffer, bytes, buffer.buffer, offset);
}



pub unsafe fn begin_buffer_writer(ctx: &GraphicsContext) -> ScratchBuffer {
    ScratchBuffer::new(ctx)
}

pub unsafe fn end_buffer_writer(ctx: &GraphicsContext<'_>, writer: ScratchBuffer) {
    let mut lock = ctx.allocator.lock().unwrap();
    writer.destroy(ctx.device, &mut lock);
}

// TODO: scratch buffer writes don't necessarily need to take in &[u8], they could take in iterators as well
pub struct ScratchBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub base_address: u64,
    pub bytes_written: usize,
}

impl ScratchBuffer {
    pub unsafe fn new(ctx: &GraphicsContext) -> Self {
        let Buffer {
            buffer,
            allocation,
            address,
            size,
        } = create_buffer_with_location_with_flags_super(
            ctx,
            SCRATCH_BUFFER_SIZE,
            "scratch buffer",
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            gpu_allocator::MemoryLocation::CpuToGpu
        );

        ScratchBuffer {
            buffer,
            base_address: address,
            
            bytes_written: 0,
            allocation,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        allocator.free(self.allocation).unwrap();
        device.destroy_buffer(self.buffer, None);
    }

    pub unsafe fn begin_of_cmd_recording(&mut self, queue_family_index: u32, device: &ash::Device, cmd: vk::CommandBuffer) {
        self.bytes_written = 0;
        let barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let buffer_memory_barriers = [barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);
    }

    /// Returns the GPU buffer start address range of the written data  
    pub unsafe fn write_bytes(&mut self, bytes: &[u8]) -> WrittenBytes {
        assert!(bytes.len() > 0);
        assert!(bytes.len() + self.bytes_written < SCRATCH_BUFFER_SIZE, "scratch buffer overrun. bytes written already: {}    num bytes: {}", self.bytes_written, bytes.len());
        log::trace!("scratch buffer write. num bytes: {}, bytes written: {}", bytes.len(), self.bytes_written);

        let dst = self.allocation.mapped_slice_mut().unwrap();
        dst[self.bytes_written..(self.bytes_written + bytes.len())].copy_from_slice(bytes);

        let prev = self.bytes_written;
        self.bytes_written += bytes.len();
        
        WrittenBytes {
            buffer_device_address_start: self.base_address + prev as u64,
            buffer_offset_start: prev as u64,
        }
    }

    /// Returns the GPU buffer start address range of the written data. This WILL be aligned to 16 bytes for now
    pub unsafe fn write_bytes_aligned(&mut self, bytes: &[u8]) -> WrittenBytes {
        assert!(bytes.len() > 0);
        assert!(bytes.len() + self.bytes_written < SCRATCH_BUFFER_SIZE, "scratch buffer overrun. bytes written already: {}    num bytes: {}", self.bytes_written, bytes.len());
        log::trace!("scratch buffer write aligned. num bytes: {}, bytes written: {}", bytes.len(), self.bytes_written);


        // we need this because the TLAS building requires that the output buffer device address is aligned to 16 bytes
        // easiest way to implement this is to add padding bytes when we need it 
        pub const ALIGNMENT: usize = 16;
        let bytes_written_aligned = self.bytes_written.next_multiple_of(ALIGNMENT);

        let dst = self.allocation.mapped_slice_mut().unwrap();
        dst[bytes_written_aligned..(bytes_written_aligned + bytes.len())].copy_from_slice(bytes);

        let prev = bytes_written_aligned as u64;
        self.bytes_written = bytes_written_aligned + bytes.len();
        
        WrittenBytes {
            buffer_device_address_start: self.base_address + prev as u64,
            buffer_offset_start: prev as u64,
        }
    }
}

pub struct WrittenBytes {
    pub buffer_device_address_start: u64,
    pub buffer_offset_start: u64,
}
