use std::{borrow::Cow, io::{BufRead, Read}};

use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};
use image::{EncodableLayout, GenericImage, GenericImageView};

use crate::{buffer, others, per_frame_data, renderer::GraphicsContext, texture::{self, Texture, create_texture}};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuMaterialInfo {
    pub base_index: u32,
}

pub struct Material {
    pub albedo_texture: Texture,
    pub arm_texture: Texture,
    pub normal_texture: Texture,
    
    pub base_index: u32,
}

const CONST_TEXTURE_RESOLUTION: u32 = 1024; 
const RESIZE_IMAGE_FILTER: image::imageops::FilterType = image::imageops::FilterType::CatmullRom;


unsafe fn load_image_and_create_texture(
    ctx: &GraphicsContext,
    image_file_bytes: &[u8],
    size: u32,
    texture_type: crate::texture::TextureType,
) -> Texture {
    let dynamic_image = image::load_from_memory(image_file_bytes).unwrap();

    let dynamic_image = if size != dynamic_image.width() {
        dynamic_image.resize_exact(size, size, RESIZE_IMAGE_FILTER)
    } else {
        dynamic_image
    };

    let img = dynamic_image.to_rgba8();
    create_texture(ctx, Some(img.as_bytes()), size, texture_type)
}

fn store_in_channel(
    arm_texture: &mut image::FlatSamples<Vec<u8>>, 
    single_channel_image: image::DynamicImage,
    channel: u32,
    size: u32
) {
    let single_channel_image = if size != single_channel_image.width() {
        single_channel_image.resize_exact(size, size, RESIZE_IMAGE_FILTER)
    } else {
        single_channel_image
    };

    for x in 0..size {
        for y in 0..size {
            *arm_texture.get_mut_sample(channel as u8, x, y).unwrap() = single_channel_image.get_pixel(x, y).0[0];
        }
    }
}


// {name}_arm_1k.jpg
// {name}_diff_1k.jpg
// {name}_nor_dx_1k.png
// {name}_nor_gl_1k.png
fn fun_name<R: std::io::Seek + std::io::Read>(material_name: &str, map_names: &[&str], stream: &mut zip::ZipArchive<R>) -> Option<Vec<u8>> {
    // either called `diff` or `col` it seems?
    let mut albedo_data = None;
    for map_name in map_names {
        // could either be jpg or png, favour png first, then fallback to jpg
        for file_extension in ["png", "jpg"] {
            
            // TODO: wow this is horrible. fix this bs
            let s = format!("/textures/{material_name}_{map_name}_1k.{file_extension}");

            let p = std::path::Path::new(&s);
            log::debug!("checking zipped path: {:?}", &p);
            if let Ok(mut albedo_file) = stream.by_path(p) {
                log::debug!("file exists, loading zipped data");
                let mut data = vec![0; albedo_file.size() as usize];
                albedo_file.read_exact(&mut data).unwrap();
                albedo_data = Some(data);
                break;
            }
        }
    };
    albedo_data
}

impl Material {
    pub unsafe fn new_packed_zip(
        ctx: &GraphicsContext,
        name: &str,
    ) -> Self {
        let zipped = others::load_material_texture(&format!("{name}_1k.zip")).unwrap();

        let reader = std::io::Cursor::new(zipped);
        let mut stream = zip::read::ZipArchive::new(reader).unwrap();

        let albedo_data = fun_name(name, &["diff", "col"], &mut stream).expect("could not read albedo map");
        let normal_data = fun_name(name, &["nor_gl"], &mut stream).expect("could not read normal map");
        let arm_data = fun_name(name, &["arm"], &mut stream).expect("could not read arm map");
        

        let size = CONST_TEXTURE_RESOLUTION;
        let albedo_texture = load_image_and_create_texture(ctx, &albedo_data, size, texture::TextureType::Albedo);
        let normal_texture = load_image_and_create_texture(ctx, &normal_data, size, texture::TextureType::NormalMap);
        let arm_texture = load_image_and_create_texture(ctx, &arm_data, size, texture::TextureType::Arm);
        
        Self {
            albedo_texture,
            normal_texture,
            arm_texture,
            base_index: 0,
        }
    }

    pub unsafe fn new(
        ctx: &GraphicsContext,
        name: &str,
    ) -> Self {
        let load_file = |thing_in_middle: &str| -> Option<Cow<'static, [u8]>> {
            others::load_material_texture(&format!("{name}_{thing_in_middle}_1k.png")).or_else(|| others::load_material_texture(&format!("{name}_{thing_in_middle}_1k.jpg")))
        };

        let size = CONST_TEXTURE_RESOLUTION;
        let albedo_texture = load_image_and_create_texture(ctx, &load_file("color").unwrap(), size, texture::TextureType::Albedo);
        let normal_texture = load_image_and_create_texture(ctx, &load_file("normal_opengl").unwrap(), size, texture::TextureType::NormalMap);

        let mut arm_texture = image::RgbaImage::new(size, size).into_flat_samples();
        if let Some(ao_image_file_bytes) = load_file("ao") {
            let dynamic_image = image::load_from_memory(&ao_image_file_bytes).unwrap();
            store_in_channel(&mut arm_texture, dynamic_image, 0, size);
        }

        if let Some(ao_image_file_bytes) = load_file("roughness") {
            let dynamic_image = image::load_from_memory(&ao_image_file_bytes).unwrap();
            store_in_channel(&mut arm_texture, dynamic_image, 1, size);
        }

        if let Some(ao_image_file_bytes) = load_file("metallic") {
            let dynamic_image = image::load_from_memory(&ao_image_file_bytes).unwrap();
            store_in_channel(&mut arm_texture, dynamic_image, 2, size);
        }

        let img = arm_texture.samples;
        let arm_texture = create_texture(ctx, Some(&img), size, texture::TextureType::Arm);


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
    
    pub fn add_per_frame_sampled_images(&mut self, sampled_image_infos: &mut per_frame_data::PerFrameAllocator) {
        self.base_index = sampled_image_infos.current() as u32;

        sampled_image_infos.push(self.albedo_texture.image_view);
        sampled_image_infos.push(self.arm_texture.image_view);
        sampled_image_infos.push(self.normal_texture.image_view);
    }
}