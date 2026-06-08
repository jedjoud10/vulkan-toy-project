use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of};
use gpu_allocator::vulkan::Allocator;

use crate::{buffer, debug, material::Material, others, ray_tracing};

// pretty inefficient as there's no batching or culling of any kind
pub struct Model {
    vertex_buffer: buffer::Buffer,
    index_buffer: buffer::Buffer,
    index_count: usize,
    object_to_world: vek::Mat4<f32>,
    blas: ray_tracing::AccelerationStructureData,
    pub instance: vk::AccelerationStructureInstanceKHR,
}

impl Model {
    pub unsafe fn new(obj_model_bytes: &[u8], device: &ash::Device, acceleration_structure_device: &ash::khr::acceleration_structure::Device, mut allocator: &mut Allocator, debug_marker: &debug::DebugMarker, pool: vk::CommandPool, queue: vk::Queue) -> Self {
        let obj = obj::load_obj::<obj::Position, &[u8], u32>(obj_model_bytes).unwrap();

        let vertices: Vec<vek::Vec3<f32>> = obj.vertices.into_iter().map(|x: obj::Position| vek::Vec3::<f32>::from(x.position)).collect::<Vec<_>>();
        let mut indices: Vec<u32> = obj.indices;

        meshopt::optimize_vertex_cache_in_place(&mut indices, vertices.len());

        let vertex_count = vertices.len();
        let vertex_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<vek::Vec3::<f32>>() * vertices.len(), &debug_marker, "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        buffer::write_to_buffer(&device, pool, queue, vertex_buffer.buffer, &mut allocator, bytemuck::cast_slice(vertices.as_slice()));

        let index_count = indices.len();
        let index_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<u32>()  * indices.len(), &debug_marker, "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        buffer::write_to_buffer(&device, pool, queue, index_buffer.buffer, &mut allocator, bytemuck::cast_slice(indices.as_slice()));

        let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(pool);
        let cmd = device
            .allocate_command_buffers(&cmd_buffer_create_info)
            .unwrap()[0];
        device.begin_command_buffer(cmd, &Default::default()).unwrap();

        let (blas, instance) = ray_tracing::create_blas(
            device,
            cmd,
            acceleration_structure_device,
            allocator,
            debug_marker,
            vertex_count,
            0,
            size_of::<vek::Vec3<f32>>(),
            index_count,
            0,
            size_of::<u32>(),
            &vertex_buffer,
            &index_buffer
        );

        device.end_command_buffer(cmd).unwrap();

        let buffers = [cmd];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&buffers);

        device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();

        // TODO: optimize and use the same command buffer throughout initialization
        device.device_wait_idle().unwrap();

        Self {
            vertex_buffer,
            index_buffer,
            index_count,
            object_to_world: vek::Mat4::identity(),
            blas,
            instance,
        }
    }

    pub fn update(&mut self, elapsed: f32, movement: &crate::movement::Movement) {
        //let position = movement.position + movement.forward() * 2f32;
        let position = vek::Vec3::new(0f32, 10f32, 0f32); 
        let matrix = vek::Mat4::<f32>::translation_3d(position);

        self.object_to_world = matrix;

        let three_by_four = &matrix.into_row_arrays()[0..3];
        let k: &[f32] = bytemuck::cast_slice(three_by_four);
        let m: [f32; 12] = k.try_into().unwrap();

        self.instance.transform = vk::TransformMatrixKHR { matrix: m } 
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

        device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
        device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
        device.cmd_push_constants(cmd, pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytes_of(&pc));
        device.cmd_draw_indexed(cmd, self.index_count as u32, 1, 0, 0, 0);
    }

    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, mut allocator: &mut Allocator) {
        self.index_buffer.destroy(device, allocator);
        self.vertex_buffer.destroy(device, allocator);
        self.blas.destroy(acceleration_structure_device, device, allocator);
    }
}