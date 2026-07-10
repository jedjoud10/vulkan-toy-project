use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use crate::{buffer, others, pipeline::{self, PerFrameUniformData}, ray_tracing, renderer::GraphicsContext};

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct PerFrameData {
    pub main_descriptor_set: vk::DescriptorSet,
    pub present_complete_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,    
    pub query_pool: vk::QueryPool,
    pub pipeline_statistics_query_pool: vk::QueryPool,
    pub scratch_buffer: buffer::ScratchBuffer,
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
        
        Self {
            present_complete_semaphore,
            end_fence,
            cmd,
            main_descriptor_set,
            query_pool,
            pipeline_statistics_query_pool,
            scratch_buffer: buffer::ScratchBuffer::new(ctx),
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

        self.scratch_buffer.destroy(device, allocator);
    }
}