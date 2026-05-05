use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::pipeline;

pub struct PerFrameDescriptorSets {
    pub main_render_per_frame: vk::DescriptorSet,
}

pub struct PerFrameData {
    pub swapchain_image: vk::Image,
    pub rt_image: vk::Image,
    pub rt_image_allocation: Option<Allocation>,
    pub present_complete_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,
    pub per_frame_descriptor_sets: PerFrameDescriptorSets,
    pub src_image_view: vk::ImageView,
    pub dst_image_view: vk::ImageView,
}

impl PerFrameData {
    pub unsafe fn create_per_frame_data(
        device: &ash::Device,
        pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        render_compute_pipeline: &pipeline::RenderPipeline,
        swapchain_image: vk::Image
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

        let per_frame_descriptor_set_layouts = [render_compute_pipeline.descriptor_set_layout[0]];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&per_frame_descriptor_set_layouts);
        let all_descriptor_sets_for_frame = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();

        Self {
            swapchain_image,
            rt_image: vk::Image::null(),
            rt_image_allocation: None,
            present_complete_semaphore,
            render_finished_semaphore,
            end_fence,
            cmd,
            per_frame_descriptor_sets: PerFrameDescriptorSets {
                main_render_per_frame: all_descriptor_sets_for_frame[0],
            },
            src_image_view: vk::ImageView::null(),
            dst_image_view: vk::ImageView::null(),
        }
    }

    pub unsafe fn recreate_rt_images_and_image_views_and_update_descriptor_sets(
        &mut self,
        device: &ash::Device,
        swapchain_format: vk::Format,
        swapchain_image: vk::Image,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        queue_family_index: u32,
        extent: vk::Extent2D,
        binder: &Option<ash::ext::debug_utils::Device>,
        scaling_factor: u32,
    ) {
        let queue_family_indices = [queue_family_index];
        let rt_image_create_info = vk::ImageCreateInfo::default()
            .extent(vk::Extent3D {
                width: extent.width / scaling_factor,
                height: extent.height / scaling_factor,
                depth: 1,
            })
            .format(swapchain_format)
            .image_type(vk::ImageType::TYPE_2D)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .mip_levels(1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .samples(vk::SampleCountFlags::TYPE_1)
            .queue_family_indices(&queue_family_indices)
            .tiling(vk::ImageTiling::OPTIMAL)
            .array_layers(1);
        let rt_image = device.create_image(&rt_image_create_info, None).unwrap();
        let requirements = device.get_image_memory_requirements(rt_image);

        let rt_image_allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Render Texture Image Allocation",
                requirements,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                location: gpu_allocator::MemoryLocation::GpuOnly,
            })
            .unwrap();

        device
            .bind_image_memory(rt_image, rt_image_allocation.memory(), rt_image_allocation.offset())
            .unwrap();

        crate::debug::set_object_name(rt_image, binder, "Render Texture Image");

        self.rt_image = rt_image;
        self.rt_image_allocation = Some(rt_image_allocation);
        self.swapchain_image = swapchain_image;

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let src_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(swapchain_format)
            .image(rt_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let dst_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(swapchain_format)
            .image(self.swapchain_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        self.src_image_view = device
            .create_image_view(&src_image_view_create_info, None)
            .unwrap();
        self.dst_image_view = device
            .create_image_view(&dst_image_view_create_info, None)
            .unwrap();

        let descriptor_rt_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.src_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_image_infos = [descriptor_rt_image_info];

        let image_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(descriptor_image_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(self.per_frame_descriptor_sets.main_render_per_frame)
            .image_info(&descriptor_image_infos);
        
        device.update_descriptor_sets(&[image_descriptor_write], &[]);

    }
}

pub unsafe fn transfer_rt_images(
    device: &ash::Device,
    queue_family_index: u32,
    per_frame_data: &[PerFrameData],
    pool: vk::CommandPool,
    queue: vk::Queue,
) {
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];

    let cmd_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device
        .begin_command_buffer(cmd, &cmd_buffer_begin_info)
        .unwrap();

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let barriers = per_frame_data.iter().map(|data| {
        vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(data.rt_image)
            .subresource_range(subresource_range)
    });

    let barriers = barriers.collect::<Vec<_>>();
    let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    device.end_command_buffer(cmd).unwrap();

    let cmds = [cmd];
    //let wait_masks = [vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::ALL_GRAPHICS];
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&cmds)
        .wait_semaphores(&[])
        .signal_semaphores(&[])
        .wait_dst_stage_mask(&[]);
    device.queue_submit(queue, &[submit_info], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();
    device.free_command_buffers(pool, &[cmd]);
}
