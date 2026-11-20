use std::{collections::{HashMap, HashSet, VecDeque}, sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Receiver}, thread::JoinHandle};

use ash::{util::Align, vk};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{buffer::{self, SubBufferAllocation, SubBufferAllocator, SubBufferAllocatorStats}, octree::{Node, Octree, OctreeDelta}, others, ray_tracing, renderer::GraphicsContext, terrain::{INDEX_STRIDE, VERTEX_STRIDE}};

pub struct PersistentChunkRenderer {
    vertex_buffer: buffer::Buffer,
    normals_buffer: buffer::Buffer,
    index_buffer: buffer::Buffer,
    
    instances: slotmap::SlotMap<slotmap::DefaultKey, PersistentChunkRendererInstance>,
    instances_buffer: buffer::Buffer,

    sub_buffer_vertices_allocator: SubBufferAllocator,
    sub_buffer_indices_allocator: SubBufferAllocator,
}

const TOTAL_VERTICES: usize = 1024*1024*64;
const MAX_INSTANCES: usize = 1024*16;

// Euler's formula for manifold meshes
// f : v ~ 2 : 1
const TOTAL_INDICES: usize = TOTAL_VERTICES*2;


const VERTEX_BUFFER_BYTE_SIZE: usize = TOTAL_VERTICES * super::VERTEX_STRIDE;
const INDEX_BUFFER_BYTE_SIZE: usize = TOTAL_INDICES * super::INDEX_STRIDE;


impl PersistentChunkRenderer {
    pub unsafe fn new(ctx: &GraphicsContext<'_>) -> Self {

        let vertex_buffer = buffer::create_buffer_gpu_with_flags_super(ctx, VERTEX_BUFFER_BYTE_SIZE, "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let normals_buffer = buffer::create_buffer_gpu_with_flags_super(ctx, VERTEX_BUFFER_BYTE_SIZE, "normal buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let index_buffer = buffer::create_buffer_gpu_with_flags_super(ctx, INDEX_BUFFER_BYTE_SIZE, "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let instances_buffer = buffer::create_buffer_gpu_with_flags_super(ctx, size_of::<vk::DrawIndexedIndirectCommand>() * MAX_INSTANCES, "indirect instances draw buffer", vk::BufferUsageFlags::INDIRECT_BUFFER);
        

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
        
        let instance = PersistentChunkRendererInstance {
            vertex_offset: (vertex_buffers_allocation.start_byte_offset / VERTEX_STRIDE) as i32,
            first_index: (index_buffers_allocation.start_byte_offset / INDEX_STRIDE) as u32,
            index_count: indices.len() as u32,
            vertex_buffers_allocation,
            index_buffers_allocation,
        };
         

        /*
        let vertex_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, scratchy, cast_slice(vertices.as_slice()), "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let normals_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, scratchy, cast_slice(normals.as_slice()), "normals buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let index_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, scratchy, cast_slice(indices.as_slice()), "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        
        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(vertex_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::VERTEX_SHADER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ)
            .size(vertex_buffer.size as u64)
            .offset(0)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
        let index_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(index_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::INDEX_INPUT)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::INDEX_READ | vk::AccessFlags2::SHADER_READ)
            .size(index_buffer.size as u64)
            .offset(0)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
        let buffer_memory_barriers = [vertex_buffer_barrier, index_buffer_barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        ctx.device.cmd_pipeline_barrier2(cmd, &dep);

        let blas = ray_tracing::create_blas(ctx, cmd, ray_tracing::BlasGeometry::Triangles {
            vertex_count: vertices.len(),
            vertex_offset: 0,
            vertex_stride: VERTEX_STRIDE,
            index_count: indices.len(),
            index_offset: 0,
            index_stride: INDEX_STRIDE,
            vertex_buffer_address: vertex_buffer.address,
            index_buffer_address: index_buffer.address
        });

        let instance = ray_tracing::instantiate_blas(vek::Quaternion::identity(), vek::Vec3::zero(), vek::Vec3::one(), &blas, 0, 0xFF);

        let chunk = Chunk {
            vertex_buffer,
            index_buffer,
            normals_buffer,
            blas,
            instance,
            index_count: indices.len() as u32,
        };

        self.chunks.insert(node, Some(chunk));
        */

        self.instances.insert(instance)
    }

    pub fn remove_instance(&mut self, key: slotmap::DefaultKey) -> PersistentChunkRendererInstance {
        self.instances.remove(key).unwrap()
    }
    
    pub fn free_instance(&mut self, instance: PersistentChunkRendererInstance) {
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
        struct PushConstants {
            vertex_buffer_address: vk::DeviceAddress,
            normals_buffer_address: vk::DeviceAddress,
        }

        let pc = PushConstants {
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
    
    pub unsafe fn pre_render_barriers(&self, ctx: &GraphicsContext<'_>, cmd: vk::CommandBuffer) {
        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT | vk::PipelineStageFlags2::VERTEX_SHADER)
            .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::SHADER_READ)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
        
        let normal_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.normals_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT | vk::PipelineStageFlags2::VERTEX_SHADER)
            .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::SHADER_READ)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);

        let index_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
            .dst_access_mask(vk::AccessFlags2::INDEX_READ)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
    
        let instance_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.instances_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::TRANSFER_WRITE)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
        
        let buffer_memory_barriers = [vertex_buffer_barrier, normal_buffer_barrier, index_buffer_barrier, instance_buffer_barrier];
        ctx.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers));
    } 

    pub unsafe fn pre_instance_creation_barriers(&self, ctx: &GraphicsContext<'_>, cmd: vk::CommandBuffer) {
        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
        
        let normal_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.normals_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);

        let index_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
    
        let instance_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.instances_buffer.buffer)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .src_queue_family_index(ctx.queue_family_index)
            .dst_queue_family_index(ctx.queue_family_index);
        
        let buffer_memory_barriers = [vertex_buffer_barrier, normal_buffer_barrier, index_buffer_barrier, instance_buffer_barrier];
        ctx.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers));
    }
    
    pub(crate) fn stats(&self) -> PersistentChunkRendererStats {
        PersistentChunkRendererStats {
            vertices_sub_buffer_allocator_stats: self.sub_buffer_vertices_allocator.stats(),
            indices_sub_buffer_allocator_stats: self.sub_buffer_indices_allocator.stats()
        }
    }
}

pub struct PersistentChunkRendererStats {
    pub vertices_sub_buffer_allocator_stats: SubBufferAllocatorStats,
    pub indices_sub_buffer_allocator_stats: SubBufferAllocatorStats,
}

pub struct PersistentChunkRendererInstance {
    vertex_offset: i32,
    first_index: u32,
    index_count: u32,

    vertex_buffers_allocation: SubBufferAllocation,
    index_buffers_allocation: SubBufferAllocation,
}
