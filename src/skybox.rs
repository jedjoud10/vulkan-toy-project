use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::renderer::GraphicsContext;
pub struct Skybox {
    pub skybox_image: vk::Image,
    pub skybox_image_view: vk::ImageView,
    pub skybox_array_image_view: vk::ImageView,
    
    pub ambient_skybox_image: vk::Image,
    pub ambient_skybox_image_view: vk::ImageView,
    pub ambient_skybox_array_image_view: vk::ImageView,
    
    pub clouds_image: vk::Image,
    pub clouds_image_view: vk::ImageView,
    
    pub ambient_skybox_image_allocation: Allocation,
    pub skybox_image_allocation: Allocation,
    pub clouds_image_allocation: Allocation,
}

impl Skybox {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.skybox_image_view, None);
        device.destroy_image_view(self.skybox_array_image_view, None);
        device.destroy_image(self.skybox_image, None);

        device.destroy_image_view(self.ambient_skybox_image_view, None);
        device.destroy_image_view(self.ambient_skybox_array_image_view, None);
        device.destroy_image(self.ambient_skybox_image, None);
                
        device.destroy_image(self.clouds_image, None);
        device.destroy_image_view(self.clouds_image_view, None);
        
        
        allocator.free(self.ambient_skybox_image_allocation).unwrap();
        allocator.free(self.skybox_image_allocation).unwrap();
        allocator.free(self.clouds_image_allocation).unwrap();
        
    }
}

pub const AMBIENT_SKYBOX_RESOLUTION: u32 = 16;
pub const SKYBOX_RESOLUTION: u32 = 256;
pub const CLOUDS_RESOLUTION: u32 = 512;
const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;



pub unsafe fn create_skybox(
    ctx: &mut GraphicsContext,
) -> Skybox {
    let GraphicsContext {
        device,
        queue_family_index,
        host_image_copy_device,
        ref mut allocator,
        debug_marker,
        ..
    } = *ctx;

    let queue_family_indices = [queue_family_index];
    
    let skybox_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: SKYBOX_RESOLUTION,
            height: SKYBOX_RESOLUTION,
            depth: 1,
        })
        .format(FORMAT)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::HOST_TRANSFER_EXT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(6);
    let skybox_image = device.create_image(&skybox_image_create_info, None).unwrap();
    crate::debug::set_object_name(skybox_image, debug_marker, "Skybox Texture");

    let requirements = device.get_image_memory_requirements(skybox_image);
    let skybox_image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(skybox_image, skybox_image_allocation.memory(), skybox_image_allocation.offset()).unwrap();
    
    let clouds_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: CLOUDS_RESOLUTION,
            height: CLOUDS_RESOLUTION,
            depth: 1,
        })
        .format(FORMAT)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::HOST_TRANSFER_EXT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let clouds_image = device.create_image(&clouds_image_create_info, None).unwrap();
    crate::debug::set_object_name(clouds_image, debug_marker, "Clouds Texture");

    let requirements = device.get_image_memory_requirements(clouds_image);
    let clouds_image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(clouds_image, clouds_image_allocation.memory(), clouds_image_allocation.offset()).unwrap();

    
    let ambient_skybox_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: AMBIENT_SKYBOX_RESOLUTION,
            height: AMBIENT_SKYBOX_RESOLUTION,
            depth: 1,
        })
        .format(FORMAT)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::HOST_TRANSFER_EXT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(6);
    let ambient_skybox_image = device.create_image(&ambient_skybox_image_create_info, None).unwrap();
    crate::debug::set_object_name(ambient_skybox_image, debug_marker, "Ambient Skybox Texture");

    let requirements = device.get_image_memory_requirements(ambient_skybox_image);
    let ambient_skybox_image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(ambient_skybox_image, ambient_skybox_image_allocation.memory(), ambient_skybox_image_allocation.offset()).unwrap();

    let skybox_image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(6)
        .base_mip_level(0)
        .level_count(1);
    let clouds_image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .base_mip_level(0)
        .level_count(1);

    let skybox_image_layout_transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(skybox_image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(skybox_image_subresource_range);
    let ambient_skybox_image_layout_transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(ambient_skybox_image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(skybox_image_subresource_range);
    let clouds_image_layout_transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(clouds_image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(clouds_image_subresource_range);

    host_image_copy_device.transition_image_layout(&[skybox_image_layout_transition, ambient_skybox_image_layout_transition, clouds_image_layout_transition]).unwrap();

    let skybox_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(FORMAT)
        .image(skybox_image)
        .subresource_range(skybox_image_subresource_range)
        .view_type(vk::ImageViewType::CUBE);
    let skybox_image_view = device
        .create_image_view(&skybox_image_view_create_info, None)
        .unwrap();

    
    let ambient_skybox_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(FORMAT)
        .image(ambient_skybox_image)
        .subresource_range(skybox_image_subresource_range)
        .view_type(vk::ImageViewType::CUBE);
    let ambient_skybox_image_view = device
        .create_image_view(&ambient_skybox_image_view_create_info, None)
        .unwrap();

    let skybox_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(FORMAT)
        .image(skybox_image)
        .subresource_range(skybox_image_subresource_range)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY);
    let skybox_array_image_view = device
        .create_image_view(&skybox_image_view_create_info, None)
        .unwrap();

    let ambient_skybox_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(FORMAT)
        .image(ambient_skybox_image)
        .subresource_range(skybox_image_subresource_range)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY);
    let ambient_skybox_array_image_view = device
        .create_image_view(&ambient_skybox_image_view_create_info, None)
        .unwrap();


    let clouds_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(FORMAT)
        .image(clouds_image)
        .subresource_range(clouds_image_subresource_range)
        .view_type(vk::ImageViewType::TYPE_2D);
    let clouds_image_view = device
        .create_image_view(&clouds_image_view_create_info, None)
        .unwrap();

    Skybox {
        skybox_image,
        skybox_image_view,
        skybox_image_allocation,
        skybox_array_image_view,
        clouds_image,
        clouds_image_view,
        clouds_image_allocation,
        ambient_skybox_image,
        ambient_skybox_image_view,
        ambient_skybox_array_image_view,
        ambient_skybox_image_allocation,
    }
}