use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::Allocator;

use crate::{buffer, debug, material::Material, others, ray_tracing, renderer::GraphicsContext, texture};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuModelMetadata {
    pub material_base_index: u32,
    pub storage_buffers_base_index: u32,
}

// pretty inefficient as there's no batching or culling of any kind
pub struct Model {
    vertex_positions_buffer: buffer::Buffer,
    vertex_normals_buffer: buffer::Buffer,
    vertex_uvs_buffer: buffer::Buffer,

    index_buffer: buffer::Buffer,
    index_count: usize,
    object_to_world: vek::Mat4<f32>,
    blas: ray_tracing::AccelerationStructureData,
    pub instance: vk::AccelerationStructureInstanceKHR,
    position: vek::Vec3<f32>,
    pub base_index: u32,

    pub material_index: u32,
}

impl Model {
    pub unsafe fn new(position: vek::Vec3<f32>, name: &str, ctx: &mut GraphicsContext, material_index: u32, cmd: vk::CommandBuffer, mut writer: &mut buffer::BufferWriter) -> Self {
        let obj_model_bytes = others::load_model(name).unwrap();
        let obj = obj::load_obj::<obj::TexturedVertex, &[u8], u32>(&obj_model_bytes).unwrap();

        let mut positions = Vec::<vek::Vec3<f32>>::new();
        let mut normals = Vec::<vek::Vec3<f32>>::new();
        let mut uvs = Vec::<vek::Vec2<f32>>::new();

        let mut indices: Vec<u32> = obj.indices;
        let vertex_count = obj.vertices.len();
        let index_count = indices.len();
        for vertex in obj.vertices {
            positions.push(vek::Vec3::<f32>::from(vertex.position));
            normals.push(vek::Vec3::<f32>::from(vertex.normal));
            uvs.push(vek::Vec2::<f32>::from_slice(&vertex.texture[0..2]));
        }


        meshopt::optimize_vertex_cache_in_place(&mut indices, vertex_count);

        let vertex_positions_buffer = buffer::create_buffer_write_with_staging_buffer(ctx, cmd, &mut writer, cast_slice(positions.as_slice()), "vertex positions buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        
        let vertex_normals_buffer = buffer::create_buffer_write_with_staging_buffer(ctx, cmd, &mut writer, cast_slice(normals.as_slice()), "vertex normals buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        let vertex_uvs_buffer = buffer::create_buffer_write_with_staging_buffer(ctx, cmd, &mut writer, cast_slice(uvs.as_slice()), "vertex uvs buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        let index_buffer = buffer::create_buffer_write_with_staging_buffer(ctx, cmd, &mut writer, cast_slice(indices.as_slice()), "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

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

        let (blas, instance) = ray_tracing::create_blas(
            ctx,
            cmd,
            vertex_count,
            0,
            size_of::<vek::Vec3<f32>>(),
            index_count,
            0,
            size_of::<u32>(),
            &vertex_positions_buffer,
            &index_buffer,
            0
        );

        Self {
            vertex_positions_buffer,
            vertex_normals_buffer,
            vertex_uvs_buffer,
            index_buffer,
            index_count,
            object_to_world: vek::Mat4::identity(),
            blas,
            instance,
            position,
            base_index: 0,
            material_index,
        }
    }

    pub fn add_per_frame_backing_storage_buffers(&mut self, storage_buffers: &mut Vec<vk::DescriptorBufferInfo>) {
        self.base_index = storage_buffers.len() as u32;

        storage_buffers.push(vk::DescriptorBufferInfo::default().buffer(self.index_buffer.buffer).range(vk::WHOLE_SIZE));
        storage_buffers.push(vk::DescriptorBufferInfo::default().buffer(self.vertex_normals_buffer.buffer).range(vk::WHOLE_SIZE));
        storage_buffers.push(vk::DescriptorBufferInfo::default().buffer(self.vertex_uvs_buffer.buffer).range(vk::WHOLE_SIZE));
    }

    pub fn update(&mut self, elapsed: f32, movement: &crate::movement::Movement) {
        let position = self.position;
        let rotation = vek::Quaternion::identity();
        
        /*
        let position = self.position + vek::Vec3::unit_y() * elapsed.sin() * 0.2f32;
        let rotation = vek::Quaternion::rotation_x(elapsed * 0.2f32);
        */
        let scale = 3f32;

        let matrix = vek::Mat4::from(rotation).scaled_3d(scale).translated_3d(position);

        self.object_to_world = matrix;

        let row_arrays = &matrix.into_row_arrays()[0..3];
        let matrix: [f32; 12] = cast_slice::<[f32;4],f32>(row_arrays).try_into().unwrap();
        self.instance.transform = vk::TransformMatrixKHR { matrix } 
    }

    pub unsafe fn render(&self, cmd: vk::CommandBuffer, device: &ash::Device, pipeline_layout: vk::PipelineLayout, materials: &[Material]) {
        #[derive(Clone, Copy, Pod, Zeroable)]
        #[repr(C)]
        struct PushConstant {
            object_to_world: vek::Mat4<f32>,
            albedo_sampled_image_index: u32,
        }

        let pc = PushConstant {
            object_to_world: self.object_to_world,
            albedo_sampled_image_index: materials[self.material_index as usize].base_index,
        };

        device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_positions_buffer.buffer, self.vertex_normals_buffer.buffer, self.vertex_uvs_buffer.buffer], &[0, 0, 0]);
        device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
        device.cmd_push_constants(cmd, pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytes_of(&pc));
        device.cmd_draw_indexed(cmd, self.index_count as u32, 1, 0, 0, 0);
    }

    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        self.index_buffer.destroy(device, allocator);
        self.vertex_positions_buffer.destroy(device, allocator);
        self.vertex_normals_buffer.destroy(device, allocator);
        self.vertex_uvs_buffer.destroy(device, allocator);
        
        self.blas.destroy(acceleration_structure_device, device, allocator);
    }
}