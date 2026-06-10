use std::io::Read;

use ash::vk;
use bytemuck::{bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use image::EncodableLayout;

use crate::{buffer, renderer::GraphicsContext, texture::{Texture, create_texture}};


pub struct Material {
    pub albedo_texture: Texture,
    pub roughness_texture: Texture,
    pub metal_texture: Texture,
    pub normal_texture: Texture,
    
    pub albedo_index: usize,
}

unsafe fn load_image_and_create_texture(
    ctx: &mut GraphicsContext,
    image_file_bytes: &[u8],
    size: u32,
) -> Texture {
    let dynamic_image = image::load_from_memory(image_file_bytes).unwrap();

    let dynamic_image = if size != dynamic_image.width() {
        dynamic_image.resize_exact(size, size, image::imageops::FilterType::Nearest)
    } else {
        dynamic_image
    };

    let img = dynamic_image.to_rgba8();
    create_texture(ctx, Some(img.as_bytes()), size)
}

impl Material {
    pub unsafe fn new(
        ctx: &mut GraphicsContext,
    ) -> Self {
        // TODO: do some channel packing here
        // TODO: implement compressed textures using DXT / BC formats
        let size = 256;
        let albedo_texture = load_image_and_create_texture(ctx, include_bytes!("../materials/metal/metal_0077_color_1k.jpg"), size);
        let roughness_texture = load_image_and_create_texture(ctx, include_bytes!("../materials/metal/metal_0077_roughness_1k.jpg"), size);
        let metal_texture = load_image_and_create_texture(ctx, include_bytes!("../materials/metal/metal_0077_metallic_1k.jpg"), size);
        let normal_texture = load_image_and_create_texture(ctx, include_bytes!("../materials/metal/metal_0077_normal_opengl_1k.png"), size);


        Self {
            albedo_texture,
            roughness_texture,
            metal_texture,
            normal_texture,
            albedo_index: 0,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.albedo_texture.destroy(device, allocator);
        self.roughness_texture.destroy(device, allocator);
        self.metal_texture.destroy(device, allocator);
        self.normal_texture.destroy(device, allocator);
    }
    
    pub fn add_per_frame_sampled_images(&mut self, sampled_image_infos: &mut Vec<vk::DescriptorImageInfo>) {
        self.albedo_index = sampled_image_infos.len();
        sampled_image_infos.push(vk::DescriptorImageInfo::default()
            .image_view(self.albedo_texture.image_view)
            .image_layout(vk::ImageLayout::GENERAL));
        sampled_image_infos.push(vk::DescriptorImageInfo::default()
            .image_view(self.roughness_texture.image_view)
            .image_layout(vk::ImageLayout::GENERAL));
        sampled_image_infos.push(vk::DescriptorImageInfo::default()
            .image_view(self.metal_texture.image_view)
            .image_layout(vk::ImageLayout::GENERAL));
        sampled_image_infos.push(vk::DescriptorImageInfo::default()
            .image_view(self.normal_texture.image_view)
            .image_layout(vk::ImageLayout::GENERAL));
    }
}
