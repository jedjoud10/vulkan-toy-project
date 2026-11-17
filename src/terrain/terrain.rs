use std::{collections::{HashMap, HashSet, VecDeque}, sync::{Arc, Mutex, RwLock, atomic::AtomicBool, mpsc::Receiver}, thread::JoinHandle};

use ash::{util::Align, vk};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{buffer, octree::{Node, Octree, OctreeDelta}, others, ray_tracing, renderer::GraphicsContext};

use super::*;

pub struct TerrainStats {
    pub num_chunks: usize,
    pub num_nodes: usize,
    pub num_chunks_requested: usize,
    pub num_chunks_to_remove: usize,
    pub persistent_renderer_stats: PersistentChunkRendererStats,
}

pub struct Terrain {
    octree: Octree,
    destruction: Vec<Chunk>,
    chunks: HashMap<Node, Chunk>,
    async_mesher: AsyncMesher,
    generate_data_chunks_queue: VecDeque<Node>,
    chunks_requested: usize,
    chunks_to_remove: Vec<Node>,
    persistent_renderer: PersistentChunkRenderer,
}

impl Terrain {
    pub unsafe fn create(
        ctx: &GraphicsContext,
    ) -> Self {
        
        Self {
            chunks_requested: 0,
            async_mesher: AsyncMesher::new(),
            generate_data_chunks_queue: VecDeque::new(),
            chunks: Default::default(),
            destruction: Default::default(),
            chunks_to_remove: Vec::new(),
            octree: Octree::new(super::MAX_DEPTH, super::CHUNK_SIZE as u32),
            persistent_renderer: PersistentChunkRenderer::new(ctx),
        }
    }

    pub fn force_recompute(&mut self, physics_world: &mut rapier3d::prelude::PhysicsWorld) {
        // FIXME: this is hideous lmfao
        self.octree.clear();

        for (_, chunk) in self.chunks.drain() {
            if let Some(key) = chunk.persistent_mesh_instance_key {
                let instance = self.persistent_renderer.remove_instance(key);
                self.persistent_renderer.free_instance(instance);

                if let Some(handle) = chunk.collider_handle {
                    physics_world.remove_collider(handle);
                }
            }
        }
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
            if let Some(mut chunk) = self.chunks.remove(&x) {
                // chunk has been removed, put it pending for destruction (where we destroy its GPU data)
                chunk.pending_destruction = chunk.persistent_mesh_instance_key.take().map(|key| self.persistent_renderer.remove_instance(key));
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
                self.chunks.insert(added, Chunk {
                    persistent_mesh_instance_key: None,
                    pending_destruction: None,
                    data: None,
                    collider_handle: None,
                });
                self.chunks_requested += 1;
            }
        }
    }

    pub unsafe fn frame(&mut self, ctx: &GraphicsContext, physics_world: &mut rapier3d::prelude::PhysicsWorld, scratchy: &mut buffer::ScratchBuffer, cmd: vk::CommandBuffer, data: &mut PerFrameChunkGenerationData, compute_pipelines: &crate::renderer::ComputePipelines) {
        self.persistent_renderer.pre_instance_creation_barriers(ctx, cmd);

        for chunk in self.destruction.drain(..) {
            chunk.pending_destruction.map(|instance| self.persistent_renderer.free_instance(instance));
            chunk.collider_handle.map(|handle| physics_world.remove_collider(handle));
        }

        try_readback_gpu_voxels(data, &mut self.async_mesher);

        let popped = if FORCE_ONE_CHUNK_PER_MULTI_CHUNK {
            if let Some(p) = self.generate_data_chunks_queue.pop_front() { vec![p] } else { Vec::new() }
        } else {
            (0..NUM_MULTI_CHUNKS).into_iter().map(|_| self.generate_data_chunks_queue.pop_front()).flat_map(|x| x).collect::<Vec<_>>()
        };

        if !popped.is_empty() {
            dispatch_multi_chunk_compute(ctx, scratchy, cmd, data, compute_pipelines, popped);
        }


        for GeneratedChunk { node, mesh, data } in self.async_mesher.meshing_result_rx.try_iter() {
            self.chunks_requested = self.chunks_requested.checked_sub(1).expect("chunks requested zero. implies that it was not incremented when creating meshing request");
            
            if !self.chunks.contains_key(&node) {
                continue;
            }

            // must exist in the scene, it might just have no mesh
            let chunk = self.chunks.get_mut(&node).unwrap();
            
            // we temporarily "took" away the owned ref to chunk data. give it back to the chunk
            chunk.data = Some(data);

            // possible that the chunk was already generated and this is a re-mesh
            // remove old chunk mesh and free it
            // TODO: add a removal queue just for instance keys
            chunk.persistent_mesh_instance_key.take().map(|key| self.persistent_renderer.remove_instance(key)).map(|instance| self.persistent_renderer.free_instance(instance));
            chunk.collider_handle.take().map(|handle| physics_world.remove_collider(handle));

            if let Some(mesh) = mesh {
                // recreate mesh chunk collider if this chunk is at max depth (implying closest to player)
                chunk.collider_handle = (node.depth() == MAX_DEPTH).then(|| {
                    let vertices: Vec::<glam::Vec3> = mesh.vertices.iter().map(|v| glam::vec3(v.x,v.y,v.z)).collect();
                    let indices: Vec::<[u32; 3]> = mesh.indices.as_chunks::<3>().0.to_vec();
                    let builder = rapier3d::prelude::ColliderBuilder::trimesh(vertices, indices).unwrap();
                    let collider_handle = physics_world.insert_collider(builder, None);
                    collider_handle
                });

                let instance_key = self.persistent_renderer.add_instance(ctx, cmd, scratchy, mesh);
                chunk.persistent_mesh_instance_key = Some(instance_key);   
            }
        }

        
        self.persistent_renderer.update_instance_buffer(ctx, cmd, scratchy);
        self.persistent_renderer.pre_render_barriers(ctx, cmd);
    }
    
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        self.persistent_renderer.destroy(acceleration_structure_device, device, allocator);
    }
    
    pub unsafe fn render_all_chunks(&self, ctx: &GraphicsContext, cmd: vk::CommandBuffer, scratchy: &mut buffer::ScratchBuffer, materials: &[crate::material::Material]) {
        self.persistent_renderer.render_all_chunks(ctx, cmd, scratchy);
    }
}

impl Terrain {
    pub fn edit(&mut self, position: vek::Vec3<f32>) {
        let radius = 5f32;
        let mut leafs = Vec::<Node>::new();
        let editing_aabb = vek::Aabb {
            min: position - radius,
            max: position + radius,
        };

        self.octree.recurse(|node| {
            let aabb = vek::Aabb {
                min: node.position().as_::<f32>(),
                max: node.position().as_::<f32>() + node.size() as f32
            };

            if node.leaf() {
                leafs.push(node.clone());
            }

            aabb.collides_with_aabb(editing_aabb) || aabb.contains_aabb(editing_aabb)
        });

        log::debug!("{}", leafs.len());

        for leaf in leafs {
            if let Some(chunk) = self.chunks.get_mut(&leaf) {
                if let Some(mut data) = chunk.data.take() {
                    let chunk_position_scale = get_chunk_world_position_and_scale(&leaf);

                    for x in 0..CHUNK_SIZE_WITH_PADDING {
                        for y in 0..CHUNK_SIZE_WITH_PADDING {
                            for z in 0..CHUNK_SIZE_WITH_PADDING {
                                let local = vek::Vec3::new(x,y,z);
                                let i = crate::utils::offset_to_index(local, CHUNK_SIZE_WITH_PADDING);
                                let world = local.as_::<f32>() * chunk_position_scale.w + chunk_position_scale.xyz();
                                let sphere = world.distance(position) - radius;

                                if sphere < 0f32 {
                                    data[i] = -sphere;
                                }
                            }
                        }   
                    }
                    
                    self.chunks_requested += 1;
                    if let Err(err) = self.async_mesher.meshing_request_tx.send(MeshingRequest { node: leaf, data: data }) {
                        log::error!("send error: {err}");
                    }
                }
            }
        }
    }

    pub fn stats(&self) -> TerrainStats {
        TerrainStats {
            num_chunks: self.chunks.len(),
            num_nodes: self.octree.nodes().len(),
            num_chunks_requested: self.chunks_requested,
            num_chunks_to_remove: self.chunks_to_remove.len(),
            persistent_renderer_stats: self.persistent_renderer.stats(),
        }
    }
}

unsafe fn try_readback_gpu_voxels(data: &mut PerFrameChunkGenerationData, async_mesher: &mut AsyncMesher) {
    if let Some(nodes) = data.pending_chunk_node_requests.take() {
        // fetch data from buffer which was copied into last time
        // by now, the compute and copy should have finished
        let bytes = data.readback_voxel_buffer.allocation.mapped_slice().unwrap();
        let multi_chunk_densities = cast_slice::<u8, f32>(bytes);
        assert_eq!(multi_chunk_densities.len(), MULTI_CHUNK_TEXTURE_VOLUME as usize); 
        let chunk_count = nodes.len();
        let copy = multi_chunk_densities.to_owned();
        let meshing_request_tx = async_mesher.meshing_request_tx.clone();

        rayon::spawn(move || {
           
            let i = std::time::Instant::now();
            for (i, node) in nodes.into_iter().enumerate() {
                let multi_chunk_chunk_offset = crate::utils::index_to_offset(i, 4);
                let multi_chunk_texture_page_start = multi_chunk_chunk_offset * CHUNK_SIZE_WITH_PADDING;

                // the data in multi_chunk_densities is not chunk contiguous, we must remap and copy it ourselves
                let mut densities = vec![0f32; CHUNK_VOLUME];
                for j in 0..CHUNK_VOLUME {
                    let voxel_position = crate::utils::index_to_offset(j, CHUNK_SIZE_WITH_PADDING);
                    let multi_chunk_buffer_index = crate::utils::offset_to_index(multi_chunk_texture_page_start+voxel_position, MULTI_CHUNK_TEXTURE_SIZE);
                    densities[j] = copy[multi_chunk_buffer_index];
                }

                if let Err(err) = meshing_request_tx.send(MeshingRequest { node, data: densities }) {
                    log::error!("send error: {err}");
                }
            }

            let a = std::time::Instant::now();
            let time_micros = (a-i).as_micros();

            // 0.5ms, that's pretty significant
            if time_micros > 500 {
                log::warn!("chunk density readback took {}us for {} chunks", time_micros, chunk_count);
            } 

        });

    }
}

unsafe fn dispatch_multi_chunk_compute(ctx: &GraphicsContext<'_>, scratchy: &mut buffer::ScratchBuffer, cmd: vk::CommandBuffer, data: &mut PerFrameChunkGenerationData, compute_pipelines: &HashMap<&'static str, crate::pipeline::GenericComputePipeline>, popped: Vec<Node>) {
    let full_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let voxel_readback_image_barrier = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_queue_family_index(ctx.queue_family_index)
        .dst_queue_family_index(ctx.queue_family_index)
        .image(data.readback_voxel_image.image)
        .subresource_range(full_subresource_range);
    let image_memory_barriers = [voxel_readback_image_barrier];

    let readback_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(data.readback_voxel_buffer.buffer)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::HOST_READ)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::HOST_READ)
        .src_stage_mask(vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::TRANSFER)
        .src_queue_family_index(ctx.queue_family_index)
        .dst_queue_family_index(ctx.queue_family_index)
        .size(vk::WHOLE_SIZE);
    let buffer_memory_barriers = [
        readback_buffer_barrier
    ];

    let dep = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers).image_memory_barriers(&image_memory_barriers);
    ctx.device.cmd_pipeline_barrier2(cmd, &dep);

    
    assert!(popped.len() <= NUM_MULTI_CHUNKS);
    
    // execute compute shader and do image->buffer copy
    ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute_pipelines[crate::renderer::COMPUTE_VOXELS].entry_points["main"].pipeline);
        
    let chunk_positions_and_scales = popped.iter().map(get_chunk_world_position_and_scale).collect::<Vec<_>>();

    let bytes = cast_slice(&chunk_positions_and_scales);
    buffer::write_with_scratch_buffer(ctx, cmd, scratchy, bytes, data.multi_chunk_buffer.buffer, 0);
    let multi_chunk_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(data.multi_chunk_buffer.buffer)
        .size(vk::WHOLE_SIZE)
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .src_queue_family_index(ctx.queue_family_index)
        .dst_queue_family_index(ctx.queue_family_index);
    let buffer_memory_barriers = [multi_chunk_buffer_barrier];
    ctx.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers));
            
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
            
    // TODO (OPTIMIZATION): we can be smart here and only dispatch the number of groups that we ACTUALLY need in the shader
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

    let image_barrier = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_queue_family_index(ctx.queue_family_index)
        .dst_queue_family_index(ctx.queue_family_index)
        .image(data.readback_voxel_image.image)
        .subresource_range(subresource_range);
    let image_memory_barriers = [image_barrier];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    ctx.device.cmd_pipeline_barrier2(cmd, &dep);
            
    // TODO (OPTIMIZATION): instead of doing one whole copy for the entire texture, copy each "chunk" inside the multi-chunk into its own contiguous part in the buffer
    // this should allow for a memcpy readback on the CPU, instead of having to fetch through the entire multi-chunk texture, which can lower cache hits because it is so big
    // we know exactly the number of chunks that will be read, so data.readback_voxel_buffer does not *need* to map to the entire multi-chunk texture, it coul be a 1D "array" of chunk densities 
    let region = vk::BufferImageCopy2::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D::default().depth(MULTI_CHUNK_TEXTURE_SIZE as u32).width(MULTI_CHUNK_TEXTURE_SIZE as u32).height(MULTI_CHUNK_TEXTURE_SIZE as u32))
        .image_subresource(subresource_layers);
    let regions = [region];
    let copy = vk::CopyImageToBufferInfo2::default()
        .dst_buffer(data.readback_voxel_buffer.buffer)
        .src_image(data.readback_voxel_image.image)
        .src_image_layout(vk::ImageLayout::GENERAL)
        .regions(&regions);
    ctx.device.cmd_copy_image_to_buffer2(cmd, &copy);

    let readback_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(data.readback_voxel_buffer.buffer)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::HOST_READ)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::HOST)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .src_queue_family_index(ctx.queue_family_index)
        .dst_queue_family_index(ctx.queue_family_index)
        .size(vk::WHOLE_SIZE);

    let buffer_barriers = [readback_buffer_barrier];
    let dep = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers).image_memory_barriers(&[]);
    ctx.device.cmd_pipeline_barrier2(cmd, &dep);

    data.pending_chunk_node_requests = Some(popped);
}