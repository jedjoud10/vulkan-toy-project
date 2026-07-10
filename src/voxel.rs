use std::{collections::HashMap, sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Receiver}, thread::JoinHandle};

use ash::{util::Align, vk};
use bytemuck::{Pod, Zeroable, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{buffer, others, ray_tracing, renderer::GraphicsContext};

pub const VERTEX_STRIDE: usize = size_of::<vek::Vec3::<f32>>();
pub const INDEX_STRIDE: usize = size_of::<u32>();

pub const CHUNK_SIZE: usize = 64;
pub const CHUNK_SIZE_WITH_PADDING: usize = 66;
pub const CHUNK_VOLUME: usize = 66*66*66;
        

struct GeneratedChunk {
    chunk_offset: vek::Vec3<i32>,
    vertices: Vec<vek::Vec3<f32>>,
    indices: Vec<u32>,
}

struct Shared {
    queue: crossbeam::queue::SegQueue<vek::Vec3<i32>>,
    fbm: noise::Fbm::<noise::Perlin>,
    extra: noise::Fbm::<noise::Billow::<noise::Simplex>>,
}

pub struct MultipleChunks {
    pub chunks: HashMap<vek::Vec3<i32>, Chunk>,
    receiver: Receiver<GeneratedChunk>,
}

impl MultipleChunks {
    pub unsafe fn create(
        ctx: &mut GraphicsContext,
    ) -> Self {
        let chunk_render_distance = 4i32;

        let (tx, rx) = std::sync::mpsc::channel::<GeneratedChunk>();
        let shared = Arc::<Shared>::new(Shared {
            queue: Default::default(),
            fbm: {
                let mut fbm = noise::Fbm::<noise::Perlin>::new(0); 
                fbm.octaves = 6;
                fbm.frequency = 0.001;
                fbm
            },
            extra: {
                let mut extra = noise::Fbm::<noise::Billow::<noise::Simplex>>::new(0); 
                extra.octaves = 3;
                extra.frequency = 0.01;
                extra
            },
        });

        let _threads = (0..0).into_iter().map(|thread_id| {
            let shared = shared.clone();
            let tx = tx.clone();
            let thread = std::thread::spawn(move || {
                loop {                
                    if let Some(chunk_offset) = shared.queue.pop() {
                        let mut densities = vec![0f32; CHUNK_VOLUME];

                        log::debug!("TID: {thread_id}. generating densities for chunk {chunk_offset}");
                        for index in 0..CHUNK_VOLUME  {
                            let local_position = index_to_offset(index, CHUNK_SIZE_WITH_PADDING).as_::<i32>();
                            let world_position = local_position + chunk_offset * (CHUNK_SIZE as i32);
                            let pos = world_position.as_::<f64>();
                        
                            let height = shared.fbm.get([pos.x, pos.z]) * 160.0f64;
                        
                            let stepped = (height / 10f64).floor() * 10f64;
                            let diff = ((height - stepped).abs() / 5.0f64) - 0.5;
                        
                            let density = pos.y - (stepped + -diff * shared.extra.get([pos.x,pos.z]) * 5.0f64);
                            densities[index] = density as f32;
                        }
                    
                        log::debug!("TID: {thread_id}. generating mesh for chunk {chunk_offset}");
                    
                        let (vertices, indices) = mesh_chunk(densities, chunk_offset);
                        
                        if !vertices.is_empty() && !indices.is_empty() {
                            let result = GeneratedChunk {
                                chunk_offset,
                                vertices,
                                indices,
                            };

                            if let Err(_) = tx.send(result) {
                                break;
                            }
                        }
                    }
                }
            });

            thread
        }).collect::<Vec<_>>();



        for x in -chunk_render_distance..chunk_render_distance {
            for y in -1..2 {
                for z in -chunk_render_distance..chunk_render_distance {
                    //shared.queue.push(vek::Vec3::new(x,y,z));
                }
            }
        }
        
        Self {
            chunks: HashMap::<vek::Vec3<i32>, Chunk>::new(),
            receiver: rx,
        }
    }

    pub unsafe fn frame(&mut self, ctx: &mut GraphicsContext, scratchy: &mut buffer::ScratchBuffer, cmd: vk::CommandBuffer) {
        for GeneratedChunk { chunk_offset, vertices, indices } in self.receiver.try_iter() {
            if indices.len() > 0 && vertices.len() > 0 {
                let vertex_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, scratchy, cast_slice(vertices.as_slice()), "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
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
                    .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::VERTEX_SHADER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ)
                    .size(index_buffer.size as u64)
                    .offset(0)
                    .src_queue_family_index(ctx.queue_family_index)
                    .dst_queue_family_index(ctx.queue_family_index);
                let buffer_memory_barriers = [vertex_buffer_barrier, index_buffer_barrier];
                let dep = vk::DependencyInfo::default()
                    .buffer_memory_barriers(&buffer_memory_barriers);
                ctx.device.cmd_pipeline_barrier2(cmd, &dep);

                let (blas, instance) = ray_tracing::create_blas(
                    ctx,
                    cmd,
                    vertices.len(),
                    0,
                    VERTEX_STRIDE,
                    indices.len(),
                    0,
                    INDEX_STRIDE,
                    &vertex_buffer,
                    &index_buffer,
                    1
                );

                self.chunks.insert(chunk_offset, Chunk {
                    vertex_buffer,
                    index_buffer,
                    blas,
                    instance,
                    index_count: indices.len() as u32,
                });
            }
        }
    }
    
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        for (_, chunk) in self.chunks {
            chunk.destroy(acceleration_structure_device, device, allocator);
        }
    }
}

pub struct Chunk {    
    pub vertex_buffer: buffer::Buffer,
    pub index_buffer: buffer::Buffer,

    pub blas: ray_tracing::AccelerationStructureData,
    pub instance: vk::AccelerationStructureInstanceKHR,
    pub index_count: u32,
}

impl Chunk {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        self.index_buffer.destroy(device, allocator);
        self.vertex_buffer.destroy(device, allocator);
        self.blas.destroy(acceleration_structure_device, device, allocator);
    } 
}


const INDEX_QUAD_ORDER: [usize; 6] = [0, 1, 2, 2, 1, 3];
const INDEX_OPPOSITE_QUAD_ORDER: [usize; 6] = [1, 0, 2, 1, 2, 3];

const EDGE_POSITIONS_0: [vek::Vec3<u32>; 12] = [
    vek::Vec3::new(0, 0, 0),
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(1, 1, 0),
    vek::Vec3::new(0, 1, 0),
    vek::Vec3::new(0, 0, 1),
    vek::Vec3::new(1, 0, 1),
    vek::Vec3::new(1, 1, 1),
    vek::Vec3::new(0, 1, 1),
    vek::Vec3::new(0, 0, 0),
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(1, 1, 0),
    vek::Vec3::new(0, 1, 0),
];

const EDGE_POSITIONS_1: [vek::Vec3<u32>; 12] = [
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(1, 1, 0),
    vek::Vec3::new(0, 1, 0),
    vek::Vec3::new(0, 0, 0),
    vek::Vec3::new(1, 0, 1),
    vek::Vec3::new(1, 1, 1),
    vek::Vec3::new(0, 1, 1),
    vek::Vec3::new(0, 0, 1),
    vek::Vec3::new(0, 0, 1),
    vek::Vec3::new(1, 0, 1),
    vek::Vec3::new(1, 1, 1),
    vek::Vec3::new(0, 1, 1),
];

fn unlerp(x: f32, a: f32, b: f32) -> f32 {
    return (x - a) / (b - a);
}

// stupid and unoptimized but works for now ig
fn mesh_chunk(densities: Vec<f32>, chunk_position: vek::Vec3<i32>) -> (Vec::<vek::Vec3<f32>>, Vec<u32>) {
    let mut vertices = Vec::<vek::Vec3<f32>>::new();
    let mut indices = Vec::<u32>::new();
    let mut lookup = vec![0u32; CHUNK_VOLUME];

    // first pass will generate vertex positions
    for x in 0..(CHUNK_SIZE_WITH_PADDING-1) {
        for y in 0..(CHUNK_SIZE_WITH_PADDING-1) {
            for z in 0..(CHUNK_SIZE_WITH_PADDING-1) {
                let pos = vek::Vec3::new(x,y,z);

                let mut vertex = vek::Vec3::<f32>::zero();
                let mut count = 0;

                for edge in 0..12 {
                    let p1_offsetted = EDGE_POSITIONS_0[edge];
                    let p2_offsetted = EDGE_POSITIONS_1[edge];

                    let p1_offsetted_index = offset_to_index(pos + p1_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);
                    let p2_offsetted_index = offset_to_index(pos + p2_offsetted.as_::<usize>(), CHUNK_SIZE_WITH_PADDING as usize);

                    let d1 = densities[p1_offsetted_index];
                    let d2 = densities[p2_offsetted_index];

                    if (d1 < 0f32) ^ (d2 < 0f32) {
                        let factor = unlerp(0f32, d1, d2);
                        vertex += vek::Vec3::lerp(p1_offsetted.as_::<f32>(), p2_offsetted.as_::<f32>(), factor);
                        count += 1;
                    }
                }

                if count > 0 {
                    let inside_cell_offset = vertex / count as f32;
                    let vertex_position = inside_cell_offset + pos.as_::<f32>() + chunk_position.as_::<f32>() * CHUNK_SIZE as f32;
                    let index = vertices.len();
                    lookup[offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] = index as u32;
                    vertices.push(vertex_position);
                }
            }
        }
    }

    // second pass will generate quads
    for x in 1..(CHUNK_SIZE_WITH_PADDING-1) {
        for y in 1..(CHUNK_SIZE_WITH_PADDING-1) {
            for z in 1..(CHUNK_SIZE_WITH_PADDING-1) {
                let pos = vek::Vec3::new(x,y,z);
                let is_set = densities[offset_to_index(pos, CHUNK_SIZE_WITH_PADDING)] < 0f32;
                
                for axis in 0..3 {
                    // next cell is the cell "forward" to the current cell based on axiss
                    let mut next_cell = pos;
                    next_cell[axis] += 1;
                    
                    let next_cell_is_set = densities[offset_to_index(next_cell, CHUNK_SIZE_WITH_PADDING)] < 0f32;
                    
                    if is_set != next_cell_is_set {
                        let mut quad_vertex_indices: [u32; 4] = [u32::MAX; 4];
                        
                        let dir = is_set ^ (axis == 1);
                        let vertex_offsets: [vek::Vec3<usize>; 4] = quad_vertex_offsets_for_axis(axis as u32);

                        // inside quad local vertex index
                        for index in 0..4 {
                            let target = vertex_offsets[index] + next_cell - 1;

                            if let Some(looked_up_vertex_index) = try_offset_to_index(target, CHUNK_SIZE_WITH_PADDING).map(|x| lookup[x]) {
                                quad_vertex_indices[index] = looked_up_vertex_index;
                            }
                        }

                        // don't do anything if quad contains invalid vertices
                        if quad_vertex_indices.iter().any(|x| *x == u32::MAX) {
                            continue;
                        }

                        // holy cursed
                        let quad_vertex_order = if dir { INDEX_QUAD_ORDER } else { INDEX_OPPOSITE_QUAD_ORDER };
                        for what_to_call_this_index in 0..6 {
                            indices.push(quad_vertex_indices[quad_vertex_order[what_to_call_this_index]]);
                        }
                    }
                }
            }
        }
    }

    return (vertices, indices);
}

fn quad_vertex_offsets_for_axis(axis: u32) -> [vek::Vec3<usize>; 4] {
    match axis {
        0 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(0, 1, 0), vek::Vec3::new(0, 0, 1), vek::Vec3::new(0, 1, 1)], // x
        1 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(1, 0, 0), vek::Vec3::new(0, 0, 1), vek::Vec3::new(1, 0, 1)], // y
        2 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(1, 0, 0), vek::Vec3::new(0, 1, 0), vek::Vec3::new(1, 1, 0)], // z
        _ => unreachable!()
    }
}


pub fn try_offset_to_index(offset: vek::Vec3<usize>, size: usize) -> Option<usize> {
    if offset.cmpge(&vek::Vec3::broadcast(0)).reduce_and() && offset.cmplt(&vek::Vec3::broadcast(size)).reduce_and() {
        Some(offset.x + offset.y * size + offset.z * size * size)
    } else {
        None
    }
}

pub fn offset_to_index(offset: vek::Vec3<usize>, size: usize) -> usize {
    assert!(offset.cmpge(&vek::Vec3::broadcast(0)).reduce_and());
    assert!(offset.cmplt(&vek::Vec3::broadcast(size)).reduce_and());
    
    offset.x + offset.y * size + offset.z * size * size
}

pub fn index_to_offset(index: usize, size: usize) -> vek::Vec3<usize> {
    assert!(index < (size*size*size));
    
    let x: usize = index % size;
    let y = (index / size) % size;
    let z = index / (size*size);
    vek::Vec3::new(x,y,z)
}

pub fn child_offset_to_child_index(offset: vek::Vec3<usize>) -> usize {
    offset_to_index(offset, 4)
}

pub fn child_index_to_child_offset(index: usize) -> vek::Vec3<usize> {
    index_to_offset(index, 4)
}



/*
use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::{buffer, ray_tracing, renderer::GraphicsContext};

pub const VERTICES_PER_CHUNK: usize = 1 << 18;
pub const TRIANGLES_PER_CHUNK: usize = 1 << 18;
pub const VERTEX_STRIDE: usize = size_of::<vek::Vec3::<f32>>();
pub const INDEX_STRIDE: usize = size_of::<u32>();

const PADDING: u32 = 2;
const SIZE: u32 = 64;
const IMAGE_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
        

pub struct MultipleChunks {
    pub voxel_texture: VoxelTexture3D,
    pub vertex_buffer: buffer::Buffer,
    pub index_buffer: buffer::Buffer,
    pub vertex_counter: buffer::Buffer,
    pub index_counter: buffer::Buffer,
    pub indirect_draw_buffer: buffer::Buffer,
    pub total_num_chunks: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

impl MultipleChunks {
    pub unsafe fn create(
        ctx: &mut GraphicsContext,
        total_num_chunks: usize,
    ) -> Self {
        
        let vertex_buffer = buffer::create_buffer(ctx, VERTEX_STRIDE * VERTICES_PER_CHUNK * total_num_chunks, "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let index_buffer = buffer::create_buffer(ctx, INDEX_STRIDE * 3 * TRIANGLES_PER_CHUNK * total_num_chunks, "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let indirect_draw_buffer = buffer::create_buffer(ctx, size_of::<DrawIndexedIndirectCommand>() * total_num_chunks, "indirect buffer", vk::BufferUsageFlags::INDIRECT_BUFFER);

        
        let arr = (0..total_num_chunks).into_iter().map(|i| DrawIndexedIndirectCommand {
            index_count: 0,
            instance_count: 1,
            first_index: (3 * TRIANGLES_PER_CHUNK * i) as u32,
            vertex_offset: (VERTICES_PER_CHUNK * i) as i32,
            first_instance: 0,
        }).collect::<Vec<_>>();
        buffer::write_to_buffer_with_offset(ctx, indirect_draw_buffer.buffer, cast_slice(&arr), 0);

        let vertex_counter = buffer::create_counter_buffer(ctx, "vertex counter");
        let index_counter = buffer::create_counter_buffer(ctx, "index counter");
        let voxel_texture = create_voxel_texture(ctx);

        Self {
            voxel_texture,
            vertex_buffer,
            index_buffer,
            vertex_counter,
            index_counter,
            indirect_draw_buffer,
            total_num_chunks
        }
    }

    
    pub unsafe fn do_sum_shi(
        &mut self,
        chunk_index: usize,
        chunk_offset: vek::Vec3<i32>,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        density_pipeline: vk::Pipeline,
        surface_generation_pipeline: vk::Pipeline,
        queue_family_index: u32
    ) {
        let groups = 16;

        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            density_pipeline,
        );

        let constants = bytemuck::bytes_of(&chunk_offset); 
        device.cmd_push_constants(cmd, pipeline_layout, vk::ShaderStageFlags::ALL, 0, constants);

        device.cmd_dispatch(cmd, groups+1, groups+1, groups+1);

        let zero = 0u32;
        device.cmd_update_buffer(cmd, self.vertex_counter.buffer, 0, bytemuck::bytes_of(&zero));
        device.cmd_update_buffer(cmd, self.index_counter.buffer, 0, bytemuck::bytes_of(&zero));

        let vertex_counter_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_counter.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let index_counter_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_counter.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let voxelize_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(self.voxel_texture.image)
            .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_array_layer(0).base_mip_level(0).layer_count(1).level_count(1));
        let index_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let buffer_memory_barriers = [vertex_counter_barrier, index_counter_barrier, index_buffer_barrier, vertex_buffer_barrier];
        let image_memory_barriers = [voxelize_image_barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers)
            .image_memory_barriers(&image_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);        
        
        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            surface_generation_pipeline,
        );

        device.cmd_dispatch(cmd, groups, groups, groups);

        let index_counter_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_counter.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let index_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let buffer_memory_barriers = [index_counter_barrier, vertex_buffer_barrier, index_buffer_barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);

        let regions = [vk::BufferCopy::default().size(size_of::<u32>() as u64).dst_offset((size_of::<DrawIndexedIndirectCommand>() * chunk_index) as u64).src_offset(0)];
        device.cmd_copy_buffer(cmd, self.index_counter.buffer, self.indirect_draw_buffer.buffer, &regions);

        let indirect_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.indirect_draw_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let buffer_memory_barriers = [indirect_buffer_barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);
    }

    pub unsafe fn create_blas(
        &mut self,
        ctx: &mut GraphicsContext,
        chunk_index: usize,
        cmd: vk::CommandBuffer,
    ) -> (ray_tracing::AccelerationStructureData, vk::AccelerationStructureInstanceKHR) {

        crate::ray_tracing::create_blas(
            ctx,
            cmd,
            VERTICES_PER_CHUNK,
            VERTICES_PER_CHUNK * chunk_index,
            VERTEX_STRIDE,
            TRIANGLES_PER_CHUNK * 3,
            TRIANGLES_PER_CHUNK * 3 * chunk_index,
            INDEX_STRIDE,
            &self.vertex_buffer,
            &self.index_buffer,
        )
    }
    
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.vertex_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
        self.vertex_counter.destroy(device, allocator);
        self.index_counter.destroy(device, allocator);
        self.voxel_texture.destroy(device, allocator);
        self.indirect_draw_buffer.destroy(device, allocator);
    } 
}

pub struct Chunk {
    pub chunk_index: usize,
    pub chunk_offset: vek::Vec3<i32>,
    pub built: bool,
    pub vertex_buffer_start_offset: usize,
    pub index_buffer_start_offset: usize,
    pub accel_structure: Option<ray_tracing::AccelerationStructureData>,
}

impl Chunk {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(accel_struct) = self.accel_structure {
            accel_struct.destroy(acceleration_structure_device, device, allocator);          
        }
    } 
}

pub struct VoxelTexture3D {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
}

impl VoxelTexture3D {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}

pub unsafe fn create_voxel_texture(
    ctx: &mut GraphicsContext,
) -> VoxelTexture3D {
    let GraphicsContext {
        device,
        pool,
        queue,
        queue_family_index,
        allocator,
        debug_marker,
        ..
    } = ctx;

    let queue_family_indices = [*queue_family_index];
    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: SIZE+PADDING,
            height: SIZE+PADDING,
            depth: SIZE+PADDING,
        })
        .format(IMAGE_FORMAT)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let image = device.create_image(&image_create_info, None).unwrap();
    crate::debug::set_object_name(image, debug_marker, "Voxel Texture");

    
    let image_requirements = device.get_image_memory_requirements(image);
    let image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Image Allocation",
            requirements: image_requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(image, image_allocation.memory(), image_allocation.offset()).unwrap();

    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(*pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .level_count(1)
        .base_array_layer(0)
        .base_mip_level(0);

    let image_layout_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::empty())
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(*queue_family_index)
        .dst_queue_family_index(*queue_family_index)
        .image(image)
        .subresource_range(image_subresource_range);
    let image_memory_barriers = [image_layout_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    // end command buffer and submit
    device.end_command_buffer(cmd).unwrap();
    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);
    device.queue_submit(*queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .format(IMAGE_FORMAT)
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .subresource_range(image_subresource_range);
    let image_view = device.create_image_view(&image_view_create_info, None).unwrap();


    VoxelTexture3D {
        image,
        image_view,
        allocation: image_allocation,
    }
}
*/