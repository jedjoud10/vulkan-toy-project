use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use crate::pipeline::{self, PerFrameUniformData};

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct PerFrameData {
    pub present_complete_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,    
    pub uniform_buffer: crate::buffer::Buffer,
}

impl PerFrameData {
    pub unsafe fn create_per_frame_data(
        device: &ash::Device,
        pool: vk::CommandPool,
        allocator: &mut Allocator,
        binder: &Option<ash::ext::debug_utils::Device>,
    ) -> Self {
        let present_complete_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let render_finished_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let end_fence = device.create_fence(&Default::default(), None).unwrap();
        log::info!("created semaphores and fence");

        let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(pool);
        let cmd = device
            .allocate_command_buffers(&cmd_buffer_create_info)
            .unwrap()[0];

        let uniform_buffer = crate::buffer::create_buffer(device, allocator, size_of::<PerFrameUniformData>(), binder, "per frame uniform buffer", vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);

        Self {
            present_complete_semaphore,
            render_finished_semaphore,
            end_fence,
            cmd,
            uniform_buffer,
        }
    }
    
    pub unsafe fn destroy_everything(self, device: &ash::Device, cmd_pool: vk::CommandPool, allocator: &mut Allocator) {
        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_semaphore(self.render_finished_semaphore, None);
        device.destroy_fence(self.end_fence, None);
        log::info!("destroyed semaphores and fences frame data");            

        device.free_command_buffers(cmd_pool, &[self.cmd]);
        log::info!("destroyed cmd buffer frame data");       

        self.uniform_buffer.destroy(device, allocator);
        log::info!("destroyed per frame uniform buffer");       
    }
}