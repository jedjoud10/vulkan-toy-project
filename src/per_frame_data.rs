use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use crate::{buffer, others, pipeline::{self, PerFrameUniformData}, ray_tracing, renderer::GraphicsContext};

pub const FRAMES_IN_FLIGHT: usize = 3;
pub const SCRATCH_BUFFER_SIZE: usize = 1 << 13;


pub struct ScratchBuffer {
    pub buffer: buffer::Buffer, 
    pub bytes_written: usize,
    queue_family_index: u32,
}

pub struct ScratchBufferBarrierInfo {
    pub src_stage_mask: vk::PipelineStageFlags2,
    pub dst_stage_mask: vk::PipelineStageFlags2,
    pub src_access_mask: vk::AccessFlags2,
    pub dst_access_mask: vk::AccessFlags2,

} 

impl ScratchBufferBarrierInfo {
    pub fn full() -> Self {
        Self {
            src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            src_access_mask: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            dst_access_mask: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        }
    }
}

impl ScratchBuffer {
    pub unsafe fn begin_of_cmd_recording(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        self.bytes_written = 0;
        let barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index);
        let buffer_memory_barriers = [barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);
    }

    /// Returns the GPU buffer start address range of the written data  
    pub unsafe fn write_bytes(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, bytes: &[u8], barrier_info: Option<ScratchBufferBarrierInfo>) -> u64 {
        assert!(bytes.len() + self.bytes_written < SCRATCH_BUFFER_SIZE, "scratch buffer overrun");

        device.cmd_update_buffer(cmd, self.buffer.buffer, self.bytes_written as u64, bytes);
    
        let barrier_info = barrier_info.unwrap_or_else(ScratchBufferBarrierInfo::full);
        
        let barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.buffer.buffer)
            .src_stage_mask(barrier_info.src_stage_mask)
            .dst_stage_mask(barrier_info.dst_stage_mask)
            .src_access_mask(barrier_info.src_access_mask)
            .dst_access_mask(barrier_info.dst_access_mask)
            .size(bytes.len() as u64)
            .offset(self.bytes_written as u64)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index);
        let buffer_memory_barriers = [barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);

        let prev = self.buffer.address + self.bytes_written as u64;
        self.bytes_written += bytes.len();
        prev
    }
}

pub struct PerFrameData {
    pub main_descriptor_set: vk::DescriptorSet,
    pub present_complete_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,    
    pub query_pool: vk::QueryPool,
    pub pipeline_statistics_query_pool: vk::QueryPool,
    pub scratch_buffer: ScratchBuffer,
}

impl PerFrameData {
    pub unsafe fn create_per_frame_data(
        ctx: &mut GraphicsContext,
    ) -> Self {
        let present_complete_semaphore = ctx.device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let end_fence = ctx.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).unwrap();
        log::info!("created semaphore and fence");

        let layouts = [ctx.main_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(ctx.descriptor_pool)
            .set_layouts(&layouts);
        
        let main_descriptor_set = ctx.device.allocate_descriptor_sets(&allocate_info).unwrap()[0];
        crate::debug::set_object_name(main_descriptor_set, ctx.debug_marker, "main descriptor set");
        log::info!("created bindless descriptor set");

        let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(ctx.pool);
        let cmd = ctx.device
            .allocate_command_buffers(&cmd_buffer_create_info)
            .unwrap()[0];

        
        let query_pool = others::create_query_pool(&ctx.device);
        let pipeline_statistics_query_pool = others::create_pipeline_stats_pool(&ctx.device);

        let scratch_buffer_buffer = buffer::create_buffer(ctx, SCRATCH_BUFFER_SIZE, "per frame scratch buffer", vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);

        
        Self {
            present_complete_semaphore,
            end_fence,
            cmd,
            main_descriptor_set,
            query_pool,
            pipeline_statistics_query_pool,
            scratch_buffer: ScratchBuffer { buffer: scratch_buffer_buffer, bytes_written: 0, queue_family_index: ctx.queue_family_index },
        }
    }
    
    pub unsafe fn destroy_everything(self, device: &ash::Device, cmd_pool: vk::CommandPool, allocator: &mut Allocator) {
        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_fence(self.end_fence, None);
        log::info!("destroyed semaphore and fences frame data");            

        device.free_command_buffers(cmd_pool, &[self.cmd]);
        log::info!("destroyed cmd buffer frame data");          

        device.destroy_query_pool(self.query_pool, None);
        device.destroy_query_pool(self.pipeline_statistics_query_pool, None);
        log::info!("destroyed query pools frame data");

        self.scratch_buffer.buffer.destroy(device, allocator);
    }
}