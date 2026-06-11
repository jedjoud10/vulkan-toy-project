use std::io::Read;

use ash::vk;
use bytemuck::{bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use image::{EncodableLayout, GenericImage, GenericImageView};

use crate::{buffer, renderer::GraphicsContext, texture::{Texture, create_texture}};

pub struct Material {
    pub albedo_texture: Texture,
    pub arm_texture: Texture,
    pub normal_texture: Texture,
    
    pub base_index: u32,
}

unsafe fn load_image_and_create_texture(
    ctx: &mut GraphicsContext,
    image_file_bytes: &[u8],
    size: u32,
    srgb: bool,
) -> Texture {
    let dynamic_image = image::load_from_memory(image_file_bytes).unwrap();

    let dynamic_image = if size != dynamic_image.width() {
        dynamic_image.resize_exact(size, size, image::imageops::FilterType::Nearest)
    } else {
        dynamic_image
    };

    let img = dynamic_image.to_rgba8();
    create_texture(ctx, Some(img.as_bytes()), size, srgb)
}

fn store_in_channel(
    arm_texture: &mut image::FlatSamples<Vec<u8>>, 
    single_channel_image: image::DynamicImage,
    channel: u32,
    size: u32
) {
    let single_channel_image = if size != single_channel_image.width() {
        single_channel_image.resize_exact(size, size, image::imageops::FilterType::Nearest)
    } else {
        single_channel_image
    };

    for x in 0..size {
        for y in 0..size {
            *arm_texture.get_mut_sample(channel as u8, x, y).unwrap() = single_channel_image.get_pixel(x, y).0[0];
        }
    }
}

impl Material {
    pub unsafe fn new(
        ctx: &mut GraphicsContext,
        name: &str,
        material_assets: &include_dir::Dir,
    ) -> Self {
        let load_file = |thing_in_middle: &str| -> Option<&[u8]> {
            material_assets.get_file(format!("{name}_{thing_in_middle}_1k.png")).or_else(|| material_assets.get_file(format!("{name}_{thing_in_middle}_1k.jpg"))).map(|img| img.contents())
        };

        // TODO: implement compressed textures using DXT / BC formats
        let size = 256;
        let albedo_texture = load_image_and_create_texture(ctx, load_file("color").unwrap(), size, true);
        let normal_texture = load_image_and_create_texture(ctx, load_file("normal_opengl").unwrap(), size, false);

        let mut arm_texture = image::RgbaImage::new(size, size).into_flat_samples();
        if let Some(ao_image_file_bytes) = load_file("ao") {
            let dynamic_image = image::load_from_memory(ao_image_file_bytes).unwrap();
            store_in_channel(&mut arm_texture, dynamic_image, 0, size);
        }

        if let Some(ao_image_file_bytes) = load_file("roughness") {
            let dynamic_image = image::load_from_memory(ao_image_file_bytes).unwrap();
            store_in_channel(&mut arm_texture, dynamic_image, 1, size);
        }

        if let Some(ao_image_file_bytes) = load_file("metallic") {
            let dynamic_image = image::load_from_memory(ao_image_file_bytes).unwrap();
            store_in_channel(&mut arm_texture, dynamic_image, 2, size);
        }

        let img = arm_texture.samples;
        let arm_texture = create_texture(ctx, Some(&img), size, false);


        Self {
            albedo_texture,
            normal_texture,
            arm_texture,
            base_index: 0,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.albedo_texture.destroy(device, allocator);
        self.arm_texture.destroy(device, allocator);
        self.normal_texture.destroy(device, allocator);
    }
    
    pub fn add_per_frame_sampled_images(&mut self, sampled_image_infos: &mut Vec<vk::DescriptorImageInfo>) {
        self.base_index = sampled_image_infos.len() as u32;
        sampled_image_infos.push(vk::DescriptorImageInfo::default()
            .image_view(self.albedo_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        sampled_image_infos.push(vk::DescriptorImageInfo::default()
            .image_view(self.arm_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        sampled_image_infos.push(vk::DescriptorImageInfo::default()
            .image_view(self.normal_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
    }
}
