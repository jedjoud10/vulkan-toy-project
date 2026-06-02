use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use crate::pipeline::{self, PerFrameUniformData};

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct PerFrameData {
    pub main_descriptor_set: vk::DescriptorSet,
    pub present_complete_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,    
}

impl PerFrameData {
    pub unsafe fn create_per_frame_data(
        device: &ash::Device,
        pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        allocator: &mut Allocator,
        descriptor_set_layout: vk::DescriptorSetLayout,
        binder: &Option<ash::ext::debug_utils::Device>,
    ) -> Self {
        let present_complete_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let end_fence = device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).unwrap();
        log::info!("created semaphore and fence");

        let layouts = [descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        
        let main_descriptor_set = device.allocate_descriptor_sets(&allocate_info).unwrap()[0];
        crate::debug::set_object_name(main_descriptor_set, binder, "main descriptor set");
        log::info!("created bindless descriptor set");

        let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(pool);
        let cmd = device
            .allocate_command_buffers(&cmd_buffer_create_info)
            .unwrap()[0];

        
        Self {
            present_complete_semaphore,
            end_fence,
            cmd,
            main_descriptor_set,
        }
    }
    
    pub unsafe fn destroy_everything(self, device: &ash::Device, cmd_pool: vk::CommandPool, allocator: &mut Allocator) {
        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_fence(self.end_fence, None);
        log::info!("destroyed semaphore and fences frame data");            

        device.free_command_buffers(cmd_pool, &[self.cmd]);
        log::info!("destroyed cmd buffer frame data");          
    }
}