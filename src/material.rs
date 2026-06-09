use std::io::Read;

use ash::vk;
use bytemuck::{bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::{buffer, renderer::GraphicsContext, texture::{Texture, create_texture}};


pub struct Material {
    pub albedo_texture: Texture,
    pub albedo_index: usize,
}


impl Material {
    pub unsafe fn new(
        ctx: &mut GraphicsContext,
        abledo_image: &[u8],
    ) -> Self {
        // TODO: do some channel packing here
        // TODO: implement compressed textures using DXT / BC formats
        let albedo_texture = create_texture(ctx, Some(abledo_image), 256);

        Self {
            albedo_texture,
            albedo_index: 0,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.albedo_texture.destroy(device, allocator);
    }
}
