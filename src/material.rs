use std::io::Read;

use ash::vk;
use bytemuck::{bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::buffer;

pub struct Material {
    pub albedo_image: vk::Image,
    pub albedo_image_view: vk::ImageView,
    pub albedo_allocation: Allocation,
    pub albedo_index: usize,
}


impl Material {
    pub unsafe fn new(
        abledo_image: &[u8],
        device: &ash::Device,
        host_image_copy_device: &ash::ext::host_image_copy::Device,
        allocator: &mut Allocator,
        binder: &Option<ash::ext::debug_utils::Device>,
        queue: vk::Queue,
        pool: vk::CommandPool,
        queue_family_index: u32,
    ) -> Self {
        let (image, image_allocation, image_view) = create_texture(abledo_image, device, host_image_copy_device, allocator, binder, queue_family_index);

        Self {
            albedo_image: image,
            albedo_image_view: image_view,
            albedo_allocation: image_allocation,
            albedo_index: 0,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.albedo_image_view, None);
        device.destroy_image(self.albedo_image, None);
        allocator.free(self.albedo_allocation).unwrap();
    }
}

unsafe fn create_texture(
    image_file_bytes: &[u8],
    device: &ash::Device,
    host_image_copy_device: &ash::ext::host_image_copy::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue_family_index: u32
) -> (vk::Image, Allocation, vk::ImageView) {
    let size = 256;

    let mut dynamic_image = image::load_from_memory(image_file_bytes).unwrap();
    let dynamic_image = dynamic_image.resize_exact(size, size, image::imageops::FilterType::Nearest);
    let pixels = dynamic_image.into_rgba8();

    let queue_family_indices = [queue_family_index];
    let format = vk::Format::R8G8B8A8_UNORM;

    let mut bytes = pixels.as_raw();
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
    crate::debug::set_object_name(image, binder, "Albedo Texture");

    let requirements = device.get_image_memory_requirements(image);
    let image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(image, image_allocation.memory(), image_allocation.offset()).unwrap();

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

    let transitions = [transition];
    host_image_copy_device.transition_image_layout(&[transition]);

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
    (image, image_allocation, image_view)
}