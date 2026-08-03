use ash::vk;
use bytemuck::cast_slice;
use gpu_allocator::vulkan::{Allocation, Allocator};
use half::f16;

use crate::{renderer::GraphicsContext, voxel::offset_to_index};


pub const SIZE: u32 = 32;
pub const _SIZE: usize = SIZE as usize;
pub const FORMAT: vk::Format = vk::Format::R16_SFLOAT;

// from https://github.com/jedjoud10/vulkan-toy-project/blob/d3ae7315d94f54a213fa6a757dd69f45cb8eb8b2/src/voxel.rs
pub unsafe fn create_voxel_image(
    ctx: &mut GraphicsContext
) -> SdfImage {
    let GraphicsContext {
        device,
        queue_family_index,
        host_image_copy_device,
        allocator,
        debug_marker,
        ..
    } = ctx;

    let queue_family_indices = [*queue_family_index];

    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: SIZE,
            height: SIZE,
            depth: SIZE,
        })
        .format(FORMAT)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::empty())
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::HOST_TRANSFER_EXT)
        .queue_family_indices(&queue_family_indices)
        .samples(vk::SampleCountFlags::TYPE_1)
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

    let transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(image_subresource_range);

    host_image_copy_device.transition_image_layout(&[transition]).unwrap();


    let image_subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .layer_count(1)
        .base_array_layer(0);
    
    let mut texels = vec![f16::ZERO; (SIZE*SIZE*SIZE) as usize];

    for x in 0..SIZE {
        for y in 0..SIZE {
            for z in 0..SIZE {
                let p = vek::Vec3::new(x,y,z).as_::<usize>();
                let fp = p.as_::<f32>() - (SIZE as f32 * 0.5);
                let distance = (fp + vek::Vec3::new(0f32, -10f32, 0f32)).magnitude() - 5.0f32;
                texels[offset_to_index(p, SIZE as usize)] = f16::from_f32(distance);
            }
        }    
    }

    let bytes = cast_slice::<f16, u8>(&texels);
    
    let region = vk::MemoryToImageCopyEXT::default()
        .host_pointer(bytes.as_ptr() as *const _)
        .image_extent(vk::Extent3D::default().height(SIZE).width(SIZE).depth(SIZE))
        .image_subresource(image_subresource_layers);
    let regions = [region];
    
    let copy_memory_to_image_info = vk::CopyMemoryToImageInfoEXT::default()
        .dst_image(image)
        .dst_image_layout(vk::ImageLayout::GENERAL)
        .flags(vk::HostImageCopyFlagsEXT::empty())
        .regions(&regions);
    
    host_image_copy_device.copy_memory_to_image(&copy_memory_to_image_info).unwrap();

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(FORMAT)
        .image(image)
        .subresource_range(image_subresource_range)
        .view_type(vk::ImageViewType::TYPE_3D);
    let image_view = device
        .create_image_view(&image_view_create_info, None)
        .unwrap();

    let transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(image)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .subresource_range(image_subresource_range);

    host_image_copy_device.transition_image_layout(&[transition]).unwrap();

    SdfImage {
        image,
        allocation,
        image_view,
    }
}

pub struct SdfImage {
    pub image: vk::Image,
    pub allocation: Allocation,
    pub image_view: vk::ImageView,
}

impl SdfImage {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}
