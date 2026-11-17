use std::{collections::{HashMap, HashSet, VecDeque}, sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Receiver}, thread::JoinHandle};

use ash::{util::Align, vk};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{buffer, octree::{Node, Octree, OctreeDelta}, others, ray_tracing, renderer::GraphicsContext, terrain::{INDEX_STRIDE, VERTEX_STRIDE}};

pub struct PersistentChunkRenderer {
    pub vertex_buffer: buffer::Buffer,
    pub normals_buffer: buffer::Buffer,
    pub index_buffer: buffer::Buffer,
    
    pub instances: slotmap::SlotMap<slotmap::DefaultKey, TmpInstance>,
    pub instances_buffer: buffer::Buffer,

    pub sub_buffer_vertices_allocator: SubBufferAllocator,
    pub sub_buffer_indices_allocator: SubBufferAllocator,
}

const TOTAL_VERTICES: usize = 1024*1024*64;
const MAX_INSTANCES: usize = 1024*16;

// Euler's formula for manifold meshes
// f : v ~ 2 : 1
const TOTAL_INDICES: usize = TOTAL_VERTICES*2*3; // one face is a triangle


const VERTEX_BUFFER_BYTE_SIZE: usize = TOTAL_VERTICES * super::VERTEX_STRIDE;
const INDEX_BUFFER_BYTE_SIZE: usize = TOTAL_INDICES * super::INDEX_STRIDE;


impl PersistentChunkRenderer {
    pub unsafe fn new(ctx: &GraphicsContext<'_>) -> Self {

        let vertex_buffer = buffer::create_buffer(ctx, VERTEX_BUFFER_BYTE_SIZE, "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let normals_buffer = buffer::create_buffer(ctx, VERTEX_BUFFER_BYTE_SIZE, "normal buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let index_buffer = buffer::create_buffer(ctx, INDEX_BUFFER_BYTE_SIZE, "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let instances_buffer = buffer::create_buffer(ctx, size_of::<vk::DrawIndexedIndirectCommand>() * MAX_INSTANCES, "indirect instances draw buffer", vk::BufferUsageFlags::INDIRECT_BUFFER);
        

        Self {
            vertex_buffer,
            normals_buffer,
            index_buffer,

            instances: Default::default(),
            instances_buffer,
            
            sub_buffer_vertices_allocator: SubBufferAllocator::new(VERTEX_BUFFER_BYTE_SIZE, VERTEX_STRIDE * 256),
            sub_buffer_indices_allocator: SubBufferAllocator::new(INDEX_BUFFER_BYTE_SIZE, INDEX_STRIDE * 1024),
        }
    }

    pub unsafe fn add_instance(&mut self, ctx: &GraphicsContext, cmd: vk::CommandBuffer, scratchy: &mut buffer::ScratchBuffer, super::mesher::MeshedChunk { vertices, normals, indices  }: super::MeshedChunk) -> slotmap::DefaultKey {
        let index_buffers_allocation = self.sub_buffer_indices_allocator.allocate(indices.len() * INDEX_STRIDE).unwrap();
        buffer::write_with_scratch_buffer_safer(ctx, cmd, scratchy, cast_slice(indices.as_slice()), &self.index_buffer, index_buffers_allocation.start_byte_offset as u64);

        // FIXME (HACK): we can only do this because the stride of normals is the same as the stride for position. ideally we would compute this in a separate sub-buffer allocator
        let vertex_buffers_allocation = self.sub_buffer_vertices_allocator.allocate(vertices.len() * VERTEX_STRIDE).unwrap();
        buffer::write_with_scratch_buffer_safer(ctx, cmd, scratchy, cast_slice(&normals.as_slice()), &self.normals_buffer, vertex_buffers_allocation.start_byte_offset as u64);
        buffer::write_with_scratch_buffer_safer(ctx, cmd, scratchy, cast_slice(&vertices.as_slice()), &self.vertex_buffer, vertex_buffers_allocation.start_byte_offset as u64);

        // the start byte offset MUST be aligned to the stride, otherwie our vertex_offset and first_index calculations start at unaligned bytes
        assert!(vertex_buffers_allocation.start_byte_offset.is_multiple_of(VERTEX_STRIDE));
        assert!(index_buffers_allocation.start_byte_offset.is_multiple_of(INDEX_STRIDE));        
        
        let instance = TmpInstance {
            vertex_offset: (vertex_buffers_allocation.start_byte_offset / VERTEX_STRIDE) as i32,
            first_index: (index_buffers_allocation.start_byte_offset / INDEX_STRIDE) as u32,
            index_count: indices.len() as u32,
            vertex_buffers_allocation,
            index_buffers_allocation,
        };

        self.instances.insert(instance)
    }

    pub fn remove_instance(&mut self, key: slotmap::DefaultKey) -> TmpInstance {
        self.instances.remove(key).unwrap()
    }
    
    pub fn free_instance(&mut self, instance: TmpInstance) {
        self.sub_buffer_vertices_allocator.free(instance.vertex_buffers_allocation);
        self.sub_buffer_indices_allocator.free(instance.index_buffers_allocation);        
    }

    pub unsafe fn update_instance_buffer(&self, ctx: &GraphicsContext, cmd: vk::CommandBuffer, scratchy: &mut buffer::ScratchBuffer) {
        if self.instances.is_empty() {
            return;
        }

        let draw_indexed_indirect_commands: Vec<vk::DrawIndexedIndirectCommand> = self.instances.iter().map(|(_, tmp_instance)| {
            vk::DrawIndexedIndirectCommand {
                index_count: tmp_instance.index_count,
                instance_count: 1,
                first_index: tmp_instance.first_index,
                vertex_offset: tmp_instance.vertex_offset,
                first_instance: 0,
            }
        }).collect();

        let bytes = draw_indexed_indirect_commands.len() * size_of::<vk::AccelerationStructureInstanceKHR>();
        let ptr = draw_indexed_indirect_commands.as_ptr() as *const u8;
        let data = &*std::ptr::slice_from_raw_parts(ptr, bytes);

        buffer::write_with_scratch_buffer_safer(ctx, cmd, scratchy, data, &self.instances_buffer, 0);
    }

    pub unsafe fn render_all_chunks(&self, ctx: &GraphicsContext, cmd: vk::CommandBuffer, scratchy: &mut buffer::ScratchBuffer) {
        if self.instances.is_empty() {
            return;
        }

        #[derive(Clone, Copy, Pod, Zeroable)]
        #[repr(C)]
        struct ChunkPushConstants {
            vertex_buffer_address: vk::DeviceAddress,
            normals_buffer_address: vk::DeviceAddress,
        }

        let pc = ChunkPushConstants {
            vertex_buffer_address: self.vertex_buffer.address,
            normals_buffer_address: self.normals_buffer.address,
        };

        ctx.device.cmd_push_constants(cmd, ctx.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&pc));
        ctx.device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
        
        ctx.device.cmd_draw_indexed_indirect(cmd, self.instances_buffer.buffer, 0, self.instances.len() as u32, size_of::<vk::DrawIndexedIndirectCommand>() as u32);
    }

    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        self.index_buffer.destroy(device, allocator);
        self.normals_buffer.destroy(device, allocator);
        self.vertex_buffer.destroy(device, allocator);
        self.instances_buffer.destroy(device, allocator);
    } 
}


pub struct TmpInstance {
    vertex_offset: i32,
    first_index: u32,
    index_count: u32,

    vertex_buffers_allocation: AAllocation,
    index_buffers_allocation: AAllocation,
}

struct Page {
    in_use: bool,
}

// contains a contiguous list of pages
// when we do an allocation, we try to fit it into some contiguous "free" pages
// this suffers from both internal and external fragmentation
// always allocates in entire pages
pub struct SubBufferAllocator {
    // TODO (OPTIMIZATION): use a bitmask to speed up page retrieval
    // TODO (OPTIMIZATION): use multiple layers / page sizes to reduce fragmentation
    pages: Vec<Page>,
    page_size: usize,
    currently_allocated_bytes: usize,
    total_buffer_size: usize,
}

pub struct SubBufferAllocatorStats {
    total_buffer_size: usize,
    total_allocated_bytes: usize,
    internal_frag_bytes: usize,
    pages_allocated: usize,
    page_size: usize,
}

impl std::fmt::Display for SubBufferAllocatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = bytesize::ByteSize::b(self.total_allocated_bytes as u64);
        let wasted = bytesize::ByteSize::b(self.internal_frag_bytes as u64);
        let pages = self.pages_allocated;
        let unused = bytesize::ByteSize::b(self.total_buffer_size as u64 - self.total_allocated_bytes as u64);
        let page_size = bytesize::ByteSize::b(self.page_size as u64);
        
        write!(f, "total: {total}, wasted (internal frag): {wasted}, unused: {unused}, pages: {pages}, page size: {page_size}")
    }
}
struct AAllocation {
    start_byte_offset: usize,
    start_page: usize,
    page_count: usize,
    byte_size: usize
}


impl SubBufferAllocator {
    fn new(total_buffer_size: usize, page_size: usize) -> Self {
        log::debug!("creating sub buffer allocator for {} and page size {}", bytesize::ByteSize::b(total_buffer_size as u64), bytesize::ByteSize::b(page_size as u64));
        let num_pages = total_buffer_size.div_ceil(page_size);
        log::debug!("number of pages: {}", num_pages);
        
        
        Self {
            pages: (0..num_pages).into_iter().map(|_| Page { in_use: false }).collect(),
            page_size,
            total_buffer_size,
            currently_allocated_bytes: 0,
        }
    }

    fn allocate(&mut self, byte_size: usize) -> Option<AAllocation> {
        let num_contiguous_pages_required = byte_size.div_ceil(self.page_size);
        let num_pages = self.pages.len();

        // TODO (OPTIMIZATION): literally do anything BUT this. lmao
        for start_page in 0..num_pages {
            let end = (start_page+num_contiguous_pages_required);

            if end > num_pages {
                break;
            }

            let slice = &mut self.pages[start_page..end]; 
            let all_free = slice.iter().all(|x| !x.in_use);
            let start_byte_offset = start_page * self.page_size; 

            if all_free {
                // set as used
                slice.iter_mut().for_each(|x| x.in_use = true);

                self.currently_allocated_bytes += byte_size; 

                return Some(AAllocation {
                    start_byte_offset,
                    start_page,
                    byte_size,
                    page_count: num_contiguous_pages_required,
                });
            }
        }
    
        None
    }

    fn total_internal_fragmentation(&self) -> usize {
        let num_pages_in_use = self.pages.iter().filter(|x| x.in_use).count();
        let total_page_bytes = num_pages_in_use * self.page_size;
        return total_page_bytes - self.currently_allocated_bytes;
    }

    pub fn stats(&self) -> SubBufferAllocatorStats {
        SubBufferAllocatorStats {
            total_buffer_size: self.total_buffer_size,
            total_allocated_bytes: self.currently_allocated_bytes,
            internal_frag_bytes: self.total_internal_fragmentation(),
            pages_allocated: self.pages.iter().filter(|x| x.in_use).count(),
            page_size: self.page_size,
        }
    }

    fn free(&mut self, alloc: AAllocation) {        
        let end = (alloc.start_page+alloc.page_count).min(self.pages.len());
        let slice = &mut self.pages[(alloc.start_page)..end]; 
        slice.iter_mut().for_each(|x| x.in_use = false);
        self.currently_allocated_bytes -= alloc.byte_size;
    }
}

#[cfg(test)]
mod tests {
    pub use super::*;

    #[test]
    fn test_allocator_internal_fragmentation() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(4).unwrap();
        assert_eq!(a1.page_count, 1);
        assert_eq!(a1.start_page, 0);

        let a2 = allocator.allocate(4).unwrap();
        assert_eq!(a2.page_count, 1);
        assert_eq!(a2.start_page, 1);

        let a3 = allocator.allocate(4).unwrap();
        assert_eq!(a3.page_count, 1);
        assert_eq!(a3.start_page, 2);

        assert!(a3.start_byte_offset != a2.start_byte_offset);
        assert!(a3.start_byte_offset != a1.start_byte_offset);
        assert!(a1.start_byte_offset != a3.start_byte_offset);
    }

    #[test]
    fn test_allocator_one_page() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(32).unwrap();
        assert_eq!(a1.page_count, 1);

        let a2 = allocator.allocate(32).unwrap();
        assert_eq!(a2.page_count, 1);

        let a3 = allocator.allocate(32).unwrap();
        assert_eq!(a3.page_count, 1);

        assert!(a3.start_byte_offset != a2.start_byte_offset);
        assert!(a3.start_byte_offset != a1.start_byte_offset);
        assert!(a1.start_byte_offset != a3.start_byte_offset);
    }

    #[test]
    fn test_allocator_multiple_pages() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(34).unwrap();
        assert_eq!(a1.page_count, 2);

        let a2 = allocator.allocate(60).unwrap();
        assert_eq!(a2.page_count, 2);

        let a3 = allocator.allocate(65).unwrap();
        assert_eq!(a3.page_count, 3);

        assert!(a3.start_byte_offset != a2.start_byte_offset);
        assert!(a3.start_byte_offset != a1.start_byte_offset);
        assert!(a1.start_byte_offset != a3.start_byte_offset);
    }

    
    #[test]
    fn test_allocator_entire_allocator() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(1024).unwrap();
        assert_eq!(a1.page_count, allocator.pages.len());
        assert_eq!(a1.byte_size, 1024);
        
        let should_fail = allocator.allocate(60);
        assert!(should_fail.is_none());
    }
}
