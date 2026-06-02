use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
pub struct VoxelTexture3D {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
}

impl VoxelTexture3D {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}

const SIZE: u32 = 66;
const IMAGE_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

pub unsafe fn create_voxel_texture(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    queue_family_index: u32,
) -> VoxelTexture3D {
    let queue_family_indices = [queue_family_index];
    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: SIZE,
            height: SIZE,
            depth: SIZE,
        })
        .format(IMAGE_FORMAT)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let image = device.create_image(&image_create_info, None).unwrap();
    crate::debug::set_object_name(image, binder, "Voxel Texture");

    
    let image_requirements = device.get_image_memory_requirements(image);
    let image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Image Allocation",
            requirements: image_requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(image, image_allocation.memory(), image_allocation.offset()).unwrap();

    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .level_count(1)
        .base_array_layer(0)
        .base_mip_level(0);

    let image_layout_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::empty())
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(image)
        .subresource_range(image_subresource_range);
    let image_memory_barriers = [image_layout_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    // end command buffer and submit
    device.end_command_buffer(cmd).unwrap();
    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);
    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .format(IMAGE_FORMAT)
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .subresource_range(image_subresource_range);
    let image_view = device.create_image_view(&image_view_create_info, None).unwrap();


    VoxelTexture3D {
        image,
        image_view,
        allocation: image_allocation,
    }
}