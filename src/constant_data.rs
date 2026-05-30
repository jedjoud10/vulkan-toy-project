use ash::vk;
use gpu_allocator::vulkan::Allocation;

pub struct ConstantData {    
    pub rendered_image: vk::Image,
    pub rendered_image_allocation: Option<Allocation>,
    pub rendered_depth_image: vk::Image,
    pub rendered_depth_image_allocation: Option<Allocation>,
    
    pub bloom_image: vk::Image,
    pub bloom_image_allocation: Option<Allocation>,
    pub bloom_mip_image_views: Vec<vk::ImageView>,

    pub rendered_image_view: vk::ImageView,
    pub rendered_depth_image_image_view: vk::ImageView,
    pub entire_bloom_image_view: vk::ImageView,
}

impl ConstantData {
    pub unsafe fn create_constant_descriptor_sets(
    ) -> Self {
        Self {
            rendered_image_view: vk::ImageView::null(),
            bloom_image: vk::Image::null(),
            bloom_image_allocation: None,
            entire_bloom_image_view: vk::ImageView::null(),
            bloom_mip_image_views: Default::default(),
            rendered_image: vk::Image::null(),
            rendered_image_allocation: None,
            rendered_depth_image: vk::Image::null(),
            rendered_depth_image_allocation: None,
            rendered_depth_image_image_view: vk::ImageView::null(),
        }
    }
    
    pub unsafe fn recreate_rt_images_and_image_views_and_update_descriptor_sets(
        &mut self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        queue_family_index: u32,
        extent: vk::Extent2D,
        binder: &Option<ash::ext::debug_utils::Device>,
        scaling_factor: u32,
    ) {
        log::debug!("recreate images & descriptor set stuff for per-frame-data...");

        let rendered_image_format = vk::Format::R16G16B16A16_SFLOAT;
        let (rendered_image, rendered_image_allocation) = create_image(device, rendered_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Tmp Rendered Texture (pre-process)", None, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED);

        let depth_image_format = vk::Format::D32_SFLOAT;
        let (depth_image, depth_image_allocation) = create_image(device, depth_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Depth Texture", None, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);

        let scaled = vek::Vec2::new(extent.width, extent.height) / scaling_factor;
        let bloom_mip_levels = scaled.map(|x| u32::ilog2(x)).reduce_min() - 2;

        let (bloom_image, bloom_image_allocation) = create_image(device, rendered_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Bloom Texture", Some(bloom_mip_levels), vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);
        
        self.rendered_image = rendered_image;
        self.rendered_image_allocation = Some(rendered_image_allocation);

        self.bloom_image = bloom_image;
        self.bloom_image_allocation = Some(bloom_image_allocation);

        self.rendered_depth_image = depth_image;
        self.rendered_depth_image_allocation = Some(depth_image_allocation);


        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let detph_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .level_count(1)
            .layer_count(1);
        let entire_bloom_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(1);

        let rendered_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(rendered_image_format)
            .image(self.rendered_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let rendered_depth_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(depth_image_format)
            .image(self.rendered_depth_image)
            .subresource_range(detph_subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let entire_bloom_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(rendered_image_format)
            .image(self.bloom_image)
            .subresource_range(entire_bloom_subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        self.rendered_image_view = device
            .create_image_view(&rendered_image_view_create_info, None)
            .unwrap();
        self.entire_bloom_image_view = device
            .create_image_view(&entire_bloom_image_view_create_info, None)
            .unwrap();
        self.rendered_depth_image_image_view = device
            .create_image_view(&rendered_depth_image_view_create_info, None)
            .unwrap();

        self.bloom_mip_image_views.clear();

        // create bloom image views
        log::debug!("creating bloom image views...");
        for mip_level in 0..bloom_mip_levels {
            let bloom_image_view_create_info = vk::ImageViewCreateInfo::default()
                .components(vk::ComponentMapping::default())
                .flags(vk::ImageViewCreateFlags::empty())
                .format(rendered_image_format)
                .image(self.bloom_image)
                .subresource_range(vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(mip_level)
                    .level_count(1)
                )
                .view_type(vk::ImageViewType::TYPE_2D);


            let image_view = device
                .create_image_view(&bloom_image_view_create_info, None)
                .unwrap();
            self.bloom_mip_image_views.push(image_view);
        }
    }
    
    pub unsafe fn destroy_rt_images_and_image_views(&mut self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        device.destroy_image_view(self.rendered_image_view, None);
        device.destroy_image_view(self.entire_bloom_image_view, None);
        device.destroy_image_view(self.rendered_depth_image_image_view, None);
        log::info!("destroyed image views");

        for image_view in self.bloom_mip_image_views.iter() {
            device.destroy_image_view(*image_view, None);
            log::info!("destroyed bloom image view");
        }
        self.bloom_mip_image_views.clear();
    
        device.destroy_image(self.rendered_image, None);
        allocator.free(self.rendered_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed rendered image");

        device.destroy_image(self.bloom_image, None);
        allocator.free(self.bloom_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed bloom image");

        device.destroy_image(self.rendered_depth_image, None);
        allocator.free(self.rendered_depth_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed depth image");
    }
}


unsafe fn create_image(
    device: &ash::Device,
    format: vk::Format,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    queue_family_index: u32,
    extent: vk::Extent2D,
    binder: &Option<ash::ext::debug_utils::Device>,
    scaling_factor: u32,
    name: &str,
    mip_levels: Option<u32>,
    usage: vk::ImageUsageFlags,
) -> (vk::Image, Allocation) {
    let queue_family_indices = [queue_family_index];
    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: extent.width / scaling_factor,
            height: extent.height / scaling_factor,
            depth: 1,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(mip_levels.unwrap_or(1))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(usage)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let image = device.create_image(&image_create_info, None).unwrap();
    let requirements = device.get_image_memory_requirements(image);

    let image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &format!("{name} Image Allocation"),
            requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    device
        .bind_image_memory(image, image_allocation.memory(), image_allocation.offset())
        .unwrap();

    crate::debug::set_object_name(image, binder, name);
    (image, image_allocation)
}

pub unsafe fn transfer_layout_for_images(
    device: &ash::Device,
    queue_family_index: u32,
    const_data: &ConstantData,
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
    let depth_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH)
        .level_count(1)
        .layer_count(1);

    let rendered_image_transition = vk::ImageMemoryBarrier2::default()
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
        .image(const_data.rendered_image)
        .subresource_range(subresource_range);

    let depth_image_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(const_data.rendered_depth_image)
        .subresource_range(depth_subresource_range);

    let bloom_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .layer_count(1);
    let bloom_image_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(
            vk::AccessFlags2::TRANSFER_READ
                | vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::SHADER_SAMPLED_READ,
        )
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(const_data.bloom_image)
        .subresource_range(bloom_subresource_range);

    let barriers = [rendered_image_transition, bloom_image_transition, depth_image_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    
    device.end_command_buffer(cmd).unwrap();
    let cmds = [cmd];
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&cmds);
    device.queue_submit(queue, &[submit_info], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();
    device.free_command_buffers(pool, &[cmd]);
}
