use std::{collections::{HashMap, HashSet, VecDeque}, sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Receiver}, thread::JoinHandle};

use ash::{util::Align, vk};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{buffer, octree::{Node, Octree, OctreeDelta}, others, ray_tracing, renderer::GraphicsContext};

pub const VERTEX_STRIDE: usize = size_of::<vek::Vec3::<f32>>();
pub const INDEX_STRIDE: usize = size_of::<u32>();

pub const CHUNK_SIZE: usize = crate::mesher::CHUNK_SIZE;
pub const CHUNK_SIZE_WITH_PADDING: usize = crate::mesher::CHUNK_SIZE_WITH_PADDING;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING*CHUNK_SIZE_WITH_PADDING;

pub const MULTI_CHUNK_TEXTURE_SIZE: usize = CHUNK_SIZE_WITH_PADDING * 4;
pub const MULTI_CHUNK_TEXTURE_VOLUME: usize = MULTI_CHUNK_TEXTURE_SIZE*MULTI_CHUNK_TEXTURE_SIZE*MULTI_CHUNK_TEXTURE_SIZE;
pub const NUM_MULTI_CHUNKS: usize = 64;


pub const ENABLED: bool = true;

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
            w: MULTI_CHUNK_TEXTURE_SIZE as u32,
            h: MULTI_CHUNK_TEXTURE_SIZE as u32,
            d: MULTI_CHUNK_TEXTURE_SIZE as u32,
        };

        let readback_voxel_image = crate::texture_3d::create_texture_3d(ctx, extent, vk::Format::R32_SFLOAT, None, "");
        let readback_voxel_buffer = crate::buffer::create_buffer_with_location(ctx, size_of::<f32>() * MULTI_CHUNK_TEXTURE_VOLUME as usize, "", vk::BufferUsageFlags::STORAGE_BUFFER, gpu_allocator::MemoryLocation::GpuToCpu);
        let multi_chunk_buffer = crate::buffer::create_buffer_default_flags(ctx, size_of::<f32>() * 4 * NUM_MULTI_CHUNKS, "");

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

struct MeshingRequest {
    node: Node,
    data: Vec<f32>,
}

struct GeneratedChunk {
    node: Node,
    mesh: Option<crate::mesher::MeshedChunk>
}

pub struct MultipleChunks {
    pub octree: Octree,
    pub destruction: Vec<Chunk>,
    pub chunks: HashMap<Node, Option<Chunk>>,
    meshing_request_tx: crossbeam::channel::Sender<MeshingRequest>,
    meshing_result_rx: Receiver<GeneratedChunk>,
    generate_data_chunks_queue: VecDeque<Node>,
    pub chunks_requested: usize,
    pub chunks_to_remove: Vec<Node>,
}

impl MultipleChunks {
    pub unsafe fn create(
        ctx: &GraphicsContext,
    ) -> Self {
        let (result_tx, meshing_result_rx) = std::sync::mpsc::channel::<GeneratedChunk>();
        let (meshing_request_tx, request_rx) = crossbeam::channel::unbounded::<MeshingRequest>();


        let _threads = (0..4).into_iter().map(|thread_id| {
            let tx = result_tx.clone();
            let request_rx = request_rx.clone();
            let thread = std::thread::spawn(move || {
                loop {                
                    if let Ok(MeshingRequest { node, data }) = request_rx.recv() {
                        log::trace!("TID: {thread_id}. generating densities for chunk");
                        let mut positive = false;
                        let mut negative = false;
                        for index in 0..CHUNK_VOLUME  {
                            let density = data[index];
                            positive |= density >= 0f32;
                            negative |= density < 0f32;
                        }

                        let mut chunk = GeneratedChunk {
                            mesh: None,
                            node,
                        };
                    
                        if positive == negative {
                            log::trace!("TID: {thread_id}. generating mesh for chunk");

                            
                            let chunk_position = node.position();
                            let chunk_size = node.size();
                            let chunk_scale = (chunk_size as f32 / CHUNK_SIZE as f32);
                    
                            let mesh = crate::mesher::mesh_chunk(&data, chunk_position, chunk_scale);
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
            chunks_requested: 0,
            meshing_request_tx,
            generate_data_chunks_queue: VecDeque::new(),
            meshing_result_rx,
            chunks: Default::default(),
            destruction: Default::default(),
            chunks_to_remove: Vec::new(),
            octree: Octree::new(5, CHUNK_SIZE as u32),
        }
    }

    pub fn force_recompute(&mut self) {
        // FIXME: this is hideous lmfao
        self.octree.clear();
    }

    pub fn stuff(&mut self, position: vek::Vec3<f32>) {
        if !ENABLED {
            return;
        }
	
        // terrain is still busy, don't start generating chunks yet
        if self.chunks_requested > 0 {
            return;
        }

        // when we generated all chunks and they are all visible, remove older ones that were pending for deletion
        for x in self.chunks_to_remove.drain(..) {
            if let Some(chunk) = self.chunks.remove(&x).flatten() {
                // chunk has been removed, put it pending for destruction (where we destroy its GPU data)
                self.destruction.push(chunk)
            }
        }

        let OctreeDelta { added, removed } = self.octree.compute(&[position]);

        for removed in removed {
            self.chunks_to_remove.push(removed.clone());
        }
        
        for added in added {
            if added.leaf() {
                self.generate_data_chunks_queue.push_back(added);
                self.chunks.insert(added, None);
                self.chunks_requested += 1;
            }
        }
    }

    pub unsafe fn frame(&mut self, ctx: &GraphicsContext, scratchy: &mut buffer::ScratchBuffer, cmd: vk::CommandBuffer, data: &mut PerFrameChunkGenerationData, compute_pipelines: &crate::renderer::ComputePipelines) {
        let mut allocator = ctx.allocator.lock().unwrap();
        for chunk in self.destruction.drain(..) {
            chunk.destroy(ctx.acceleration_structure_device, ctx.device, &mut allocator);
        }
        drop(allocator);

        if let Some(nodes) = data.pending_chunk_node_requests.take() {
            // fetch data from buffer which was copied into last time
            // by now, the compute and copy should have finished
            let bytes = data.readback_voxel_buffer.allocation.mapped_slice().unwrap();
            let multi_chunk_densities = cast_slice::<u8, f32>(bytes);
            assert_eq!(multi_chunk_densities.len(), MULTI_CHUNK_TEXTURE_VOLUME as usize); 

            let i = std::time::Instant::now();

            for (i, node) in nodes.into_iter().enumerate() {
                let multi_chunk_chunk_offset = crate::utils::index_to_offset(i, 4);
                let multi_chunk_texture_page_start = multi_chunk_chunk_offset * CHUNK_SIZE_WITH_PADDING;

                // the data in multi_chunk_densities is not chunk contiguous, we must remap and copy it ourselves
                let mut densities = vec![0f32; CHUNK_VOLUME];
                for j in 0..CHUNK_VOLUME {
                    let voxel_position = crate::utils::index_to_offset(j, CHUNK_SIZE_WITH_PADDING);
                    let multi_chunk_buffer_index = crate::utils::offset_to_index(multi_chunk_texture_page_start+voxel_position, MULTI_CHUNK_TEXTURE_SIZE);
                    densities[j] = multi_chunk_densities[multi_chunk_buffer_index];
                }
                
                self.meshing_request_tx.send(MeshingRequest { node, data: densities }).unwrap();
            }

            let a = std::time::Instant::now();
            // log::warn!("{}", (a-i).as_millis());            
        }

        let popped = (0..NUM_MULTI_CHUNKS).into_iter().map(|_| self.generate_data_chunks_queue.pop_front()).flat_map(|x| x).collect::<Vec<_>>();
        // let popped = if let Some(p) = self.generate_data_chunks_queue.pop_back() { vec![p] } else { Vec::new() };

        if !popped.is_empty() {
            // execute compute shader and do image->buffer copy
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute_pipelines[crate::renderer::COMPUTE_VOXELS].entry_points["main"].pipeline);
        
            let chunk_positions_and_scales = popped.iter().map(|node| {
                let chunk_position = node.position().as_::<f32>();
                let chunk_size = node.size();
                let chunk_scale = (chunk_size as f32 / CHUNK_SIZE as f32);
                let chunk_position_and_scale = chunk_position.with_w(chunk_scale);
                chunk_position_and_scale 
            }).collect::<Vec<_>>();

            let bytes = cast_slice(&chunk_positions_and_scales);
            buffer::write_with_scratch_buffer(ctx, cmd, scratchy, bytes, data.multi_chunk_buffer.buffer, 0);
            
            #[derive(Clone, Copy, Pod, Zeroable)]
            #[repr(C)]
            struct DispatchPushConstants {
                chunk_positions_and_scales_buffer_address: vk::DeviceAddress,
                num_chunks: u32,
                _padding: u32
            }

            let pc = DispatchPushConstants {
                chunk_positions_and_scales_buffer_address: data.multi_chunk_buffer.address,
                num_chunks: popped.len() as u32,
                _padding: 0,
            };

            ctx.device.cmd_push_constants(cmd, ctx.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytes_of(&pc));
            
            // TODO: we can be smart here and only dispatch the number of groups that we ACTUALLY need in the shader
            // improves the case where we only have a few chunks, we don't need to dispatch ALL groups of the compute shader for a single chunk        
            // we must compute some sort of AABB/bounds for the number of chunks dispatched (chunk_positions_and_scales.len())
            let group_count = MULTI_CHUNK_TEXTURE_SIZE.div_ceil(4) as u32;

            ctx.device.cmd_dispatch(cmd, group_count, group_count, group_count);

            let subresource_layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .mip_level(0)
                .layer_count(1);
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .level_count(1)
                .layer_count(1);
            let region = vk::BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D::default().depth(MULTI_CHUNK_TEXTURE_SIZE as u32).width(MULTI_CHUNK_TEXTURE_SIZE as u32).height(MULTI_CHUNK_TEXTURE_SIZE as u32))
                .image_subresource(subresource_layers);

            let image_barrier = vk::ImageMemoryBarrier2::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_queue_family_index(ctx.queue_family_index)
                .dst_queue_family_index(ctx.queue_family_index)
                .image(data.readback_voxel_image.image)
                .subresource_range(subresource_range);
            let image_memory_barriers = [image_barrier];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
            ctx.device.cmd_pipeline_barrier2(cmd, &dep);
            
            let regions = [region];
            let copy = vk::CopyImageToBufferInfo2::default()
                .dst_buffer(data.readback_voxel_buffer.buffer)
                .src_image(data.readback_voxel_image.image)
                .src_image_layout(vk::ImageLayout::GENERAL)
                .regions(&regions);
            ctx.device.cmd_copy_image_to_buffer2(cmd, &copy);

            data.pending_chunk_node_requests = Some(popped);
        }

        for GeneratedChunk { node, mesh } in self.meshing_result_rx.try_iter() {
            self.chunks_requested -= 1;
            
            if !self.chunks.contains_key(&node) {
                continue;
            }

            if let Some(crate::mesher::MeshedChunk { vertices, normals, indices }) = mesh {
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
            }
        }
    }
    
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        for chunk in self.chunks.into_iter().filter_map(|(_, b)| b) {
            chunk.destroy(acceleration_structure_device, device, allocator);
        }
    }
    
    pub unsafe fn render_all_chunks(&self, ctx: &GraphicsContext, cmd: vk::CommandBuffer, materials: &[crate::material::Material]) {
        let material_base_index = materials[1].base_index;
        
        #[derive(Clone, Copy, Pod, Zeroable)]
        #[repr(C)]
        struct ChunkPushConstants {
            vertex_buffer_address: vk::DeviceAddress,
            normals_buffer_address: vk::DeviceAddress,
            index_buffer_address: vk::DeviceAddress,
            
            // material_base_index: u32,
            draw_index: u32,
            triangle_count_mesh: u32,
            // _padding: u32,
        }
        

        for (i, chunk) in self.chunks.values().filter_map(|x| x.as_ref()).enumerate() {
            // self.device.cmd_bind_vertex_buffers(cmd, 0, &[chunk.vertex_buffer.buffer], &[0]);
            let pc = ChunkPushConstants {
                vertex_buffer_address: chunk.vertex_buffer.address,
                normals_buffer_address: chunk.normals_buffer.address,
                index_buffer_address: chunk.index_buffer.address,
                // material_base_index,
                draw_index: i as u32,
                triangle_count_mesh: chunk.index_count / 3,
                // _padding: 0,
            };

            ctx.device.cmd_push_constants(cmd, ctx.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&pc));
            ctx.device.cmd_bind_index_buffer(cmd, chunk.index_buffer.buffer, 0, vk::IndexType::UINT32);
            ctx.device.cmd_draw_indexed(cmd, chunk.index_count, 1, 0, 0, 0);
            // ctx.mesh_shader_device.cmd_draw_mesh_tasks(cmd, (chunk.index_count / 3).div_ceil(32), 1, 1);
        }
    }
}

pub struct Chunk {    
    pub vertex_buffer: buffer::Buffer,
    pub normals_buffer: buffer::Buffer,
    pub index_buffer: buffer::Buffer,

    pub blas: ray_tracing::AccelerationStructureData,
    pub instance: vk::AccelerationStructureInstanceKHR,
    pub index_count: u32,
}

impl Chunk {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        self.index_buffer.destroy(device, allocator);
        self.normals_buffer.destroy(device, allocator);
        self.vertex_buffer.destroy(device, allocator);
        self.blas.destroy(acceleration_structure_device, device, allocator);
    } 
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
        ctx: &GraphicsContext,
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
        ctx: &GraphicsContext,
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
    ctx: &GraphicsContext,
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
