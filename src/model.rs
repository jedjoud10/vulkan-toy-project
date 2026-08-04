use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::Allocator;

use crate::{buffer, debug, material::Material, others, ray_tracing, renderer::GraphicsContext, texture};

pub struct Model {
    vertex_positions_buffer: buffer::Buffer,
    vertex_normals_buffer: buffer::Buffer,
    vertex_uvs_buffer: buffer::Buffer,

    index_buffer: buffer::Buffer,
    index_count: usize,

    pub blas: ray_tracing::AccelerationStructureData,
}

impl Model {
    pub unsafe fn new(name: &str, ctx: &mut GraphicsContext, cmd: vk::CommandBuffer, mut writer: &mut buffer::ScratchBuffer) -> Self {
        let obj_model_bytes = others::load_model(name).unwrap();
        let obj = obj::load_obj::<obj::Position, &[u8], u32>(&obj_model_bytes).unwrap();

        let mut positions = Vec::<vek::Vec3<f32>>::new();
        let mut normals = Vec::<vek::Vec3<f32>>::new();
        let mut uvs = Vec::<vek::Vec2<f32>>::new();

        let mut indices: Vec<u32> = obj.indices;
        let vertex_count = obj.vertices.len();
        let index_count = indices.len();
        for vertex in obj.vertices {
            positions.push(vek::Vec3::<f32>::from(vertex.position));
            // normals.push(vek::Vec3::<f32>::from(vertex.normal));
            // uvs.push(vek::Vec2::<f32>::from_slice(&vertex.texture[0..2]));
            normals.push(vek::Vec3::<f32>::default());
            uvs.push(vek::Vec2::<f32>::default());
        
        }


        meshopt::optimize_vertex_cache_in_place(&mut indices, vertex_count);

        let vertex_positions_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, &mut writer, cast_slice(positions.as_slice()), "vertex positions buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        
        let vertex_normals_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, &mut writer, cast_slice(normals.as_slice()), "vertex normals buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        let vertex_uvs_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, &mut writer, cast_slice(uvs.as_slice()), "vertex uvs buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        let index_buffer = buffer::create_buffer_write_with_scratch_buffer(ctx, cmd, &mut writer, cast_slice(indices.as_slice()), "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(vertex_positions_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::VERTEX_SHADER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ)
            .size(vertex_positions_buffer.size as u64)
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

        let blas = ray_tracing::create_blas(ctx, cmd, ray_tracing::BlasGeometry::Triangles {
            vertex_count,
            vertex_offset: 0,
            vertex_stride:  size_of::<vek::Vec3<f32>>(),
            index_count,
            index_offset: 0,
            index_stride: size_of::<u32>(),
            vertex_buffer_address: vertex_positions_buffer.address,
            index_buffer_address: index_buffer.address
        });

        Self {
            vertex_positions_buffer,
            vertex_normals_buffer,
            vertex_uvs_buffer,
            index_buffer,
            index_count,
            blas,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) -> ray_tracing::AccelerationStructureData {
        self.index_buffer.destroy(device, allocator);
        self.vertex_positions_buffer.destroy(device, allocator);
        self.vertex_normals_buffer.destroy(device, allocator);
        self.vertex_uvs_buffer.destroy(device, allocator);
        
        self.blas
    }
}