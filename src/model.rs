use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::Allocator;

use crate::{buffer, debug, material::Material, others, ray_tracing, renderer::GraphicsContext, texture};

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
}

impl Model {
    pub unsafe fn new(position: vek::Vec3<f32>, obj_model_bytes: &[u8], ctx: &mut GraphicsContext) -> Self {
        let obj = obj::load_obj::<obj::TexturedVertex, &[u8], u32>(obj_model_bytes).unwrap();

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

        let vertex_positions_buffer = buffer::create_buffer_with(ctx, cast_slice(positions.as_slice()), "vertex positions buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        
        let vertex_normals_buffer = buffer::create_buffer_with(ctx, cast_slice(normals.as_slice()), "vertex normals buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        let vertex_uvs_buffer = buffer::create_buffer_with(ctx, cast_slice(uvs.as_slice()), "vertex uvs buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        let index_buffer = buffer::create_buffer_with(ctx, cast_slice(indices.as_slice()), "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        
        let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(ctx.pool);
        let cmd = ctx.device
            .allocate_command_buffers(&cmd_buffer_create_info)
            .unwrap()[0];
        ctx.device.begin_command_buffer(cmd, &Default::default()).unwrap();

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
            &index_buffer
        );

        ctx.device.end_command_buffer(cmd).unwrap();

        let buffers = [cmd];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&buffers);

        ctx.device.queue_submit(ctx.queue, & [submit], vk::Fence::null()).unwrap();

        // TODO: optimize and use the same command buffer throughout initialization
        ctx.device.device_wait_idle().unwrap();

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
        }
    }

    pub fn update(&mut self, elapsed: f32, movement: &crate::movement::Movement) {
        //let position = movement.position + movement.forward() * 2f32;
        let position = self.position + vek::Vec3::unit_y() * elapsed.sin() * 0.2f32;
        let rotation = vek::Quaternion::rotation_x(elapsed * 0.2f32);
        let scale = 3f32;

        let matrix = vek::Mat4::from(rotation).scaled_3d(scale).translated_3d(position);

        self.object_to_world = matrix;

        let row_arrays = &matrix.into_row_arrays()[0..3];
        let matrix: [f32; 12] = cast_slice::<[f32;4],f32>(row_arrays).try_into().unwrap();
        self.instance.transform = vk::TransformMatrixKHR { matrix } 
    }

    pub unsafe fn render(&self, cmd: vk::CommandBuffer, device: &ash::Device, pipeline_layout: vk::PipelineLayout, material: &Material) {
        #[derive(Clone, Copy, Pod, Zeroable)]
        #[repr(C)]
        struct PushConstant {
            object_to_world: vek::Mat4<f32>,
            albedo_sampled_image_index: usize,
        }

        let pc = PushConstant {
            object_to_world: self.object_to_world,
            albedo_sampled_image_index: material.albedo_index,
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