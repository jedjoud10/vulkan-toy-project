use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::renderer::GraphicsContext;


pub struct Texture {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
}

impl Texture {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}

pub unsafe fn create_texture(
    ctx: &mut GraphicsContext,
    image_file_bytes: Option<&[u8]>,
    size: u32,
) -> Texture {
    let GraphicsContext {
        device,
        queue_family_index,
        host_image_copy_device,
        allocator,
        debug_marker,
        ..
    } = ctx;

    let opt_image_buffer = image_file_bytes.map(|image_file_bytes| {
        let dynamic_image = image::load_from_memory(image_file_bytes).unwrap();
        let dynamic_image = dynamic_image.resize_exact(size, size, image::imageops::FilterType::Nearest);
        dynamic_image.into_rgba8()
    });


    let queue_family_indices = [*queue_family_index];
    let format = vk::Format::R8G8B8A8_UNORM;

    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::empty())
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::HOST_TRANSFER_EXT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let image = device.create_image(&image_create_info, None).unwrap();
    crate::debug::set_object_name(image, debug_marker, "Texture");

    let requirements = device.get_image_memory_requirements(image);
    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(image, allocation.memory(), allocation.offset()).unwrap();

    let image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .base_mip_level(0)
        .level_count(1);

    let image_subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .layer_count(1)
        .base_array_layer(0);

    let transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(image_subresource_range);

    host_image_copy_device.transition_image_layout(&[transition]);

    if let Some(image_buffer) = opt_image_buffer {
        let bytes = image_buffer.as_raw();

        let region = vk::MemoryToImageCopyEXT::default()
            .host_pointer(bytes.as_ptr() as *const _)
            .image_extent(vk::Extent3D::default().height(size).width(size).depth(1))
            .image_subresource(image_subresource_layers);
        let regions = [region];

        let copy_memory_to_image_info = vk::CopyMemoryToImageInfoEXT::default()
            .dst_image(image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .flags(vk::HostImageCopyFlagsEXT::empty())
            .regions(&regions);

        host_image_copy_device.copy_memory_to_image(&copy_memory_to_image_info);
    }
    

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(format)
        .image(image)
        .subresource_range(image_subresource_range)
        .view_type(vk::ImageViewType::TYPE_2D);
    let image_view = device
        .create_image_view(&image_view_create_info, None)
        .unwrap();

    Texture {
        image,
        image_view,
        allocation,
    }
}