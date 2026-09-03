use ash::vk;
use bytemuck::cast_slice;
use gpu_allocator::vulkan::{Allocation, Allocator};
use half::f16;
use noise::NoiseFn;
use rand::{RngExt, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{renderer::GraphicsContext, utils::{index_to_offset, offset_to_index}};


// from https://github.com/jedjoud10/vulkan-toy-project/blob/d3ae7315d94f54a213fa6a757dd69f45cb8eb8b2/src/voxel.rs
pub unsafe fn create_texture_3d(
    ctx: &mut GraphicsContext,
    size: vek::Extent3<u32>,
    format: vk::Format,
    mips: Option<u32>,
    name: &'static str
) -> Texture3D {
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
            width: size.w,
            height: size.h,
            depth: size.d,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(mips.unwrap_or(1))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::empty())
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::HOST_TRANSFER_EXT)
        .queue_family_indices(&queue_family_indices)
        .samples(vk::SampleCountFlags::TYPE_1)
        .array_layers(1);

    let image = device.create_image(&image_create_info, None).unwrap();
    crate::debug::set_object_name(image, debug_marker, name);

    let requirements = device.get_image_memory_requirements(image);
    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: name,
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
        .level_count(mips.unwrap_or(1));

    let transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(image_subresource_range);

    host_image_copy_device.transition_image_layout(&[transition]).unwrap();

    // generate_write_cpu_sdf_to_image(host_image_copy_device, image);

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(format)
        .image(image)
        .subresource_range(image_subresource_range)
        .view_type(vk::ImageViewType::TYPE_3D);
    let image_view = device
        .create_image_view(&image_view_create_info, None)
        .unwrap();

    let specific_mip_image_views = mips.map(|mip_count| {
        (0..mip_count).into_iter().map(|mip| {
            let image_subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .base_mip_level(mip)
                .level_count(1);

            let transition = vk::HostImageLayoutTransitionInfoEXT::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .subresource_range(image_subresource_range);

            host_image_copy_device.transition_image_layout(&[transition]).unwrap();

            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .components(vk::ComponentMapping::default())
                .flags(vk::ImageViewCreateFlags::empty())
                .format(format)
                .image(image)
                .subresource_range(image_subresource_range)
                .view_type(vk::ImageViewType::TYPE_3D);
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        }).collect::<Vec<vk::ImageView>>()
    });

    /*
    let transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(image)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .subresource_range(image_subresource_range);

    host_image_copy_device.transition_image_layout(&[transition]).unwrap();
    */

    Texture3D {
        image,
        allocation,
        image_view,
        size,
        specific_mip_image_views,
    }
}

pub unsafe fn generate_write_cpu_sdf_to_image(host_image_copy_device: &mut &ash::ext::host_image_copy::Device, image: vk::Image, size: u32) {
    let image_subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .layer_count(1)
        .base_array_layer(0);
        
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1234);
    
    // use an iterative process to generate smaller spheres on larger spheres (resembling fBm)
    let mut src_points = Vec::<(vek::Vec4::<f32>)>::new();
    let mut stack = Vec::<(usize, vek::Vec4::<f32>)>::new();

    // create seeds
    for _ in 0..5 {
        stack.push((0, vek::Vec4::new(
            rng.random_range(0f32..(size as f32)),
            rng.random_range(0f32..(size as f32)),
            rng.random_range(0f32..(size as f32)),
            rng.random_range(15f32..40f32)        
        )));
    }

    /*
    log::info!("generating points using iteration");

    while let Some((depth, node)) = stack.pop() {
        src_points.push(node);

        if depth < 1 {
            // pick new random seed spheres on the surface of this sphere
            for _ in 0..12 {
                let phi = rng.random_range(0f32..std::f32::consts::TAU);
                //let theta = rng.random_range(0f32..std::f32::consts::PI);
                let theta = std::f32::consts::PI * 0.5f32;
            
                let new_point = vek::Vec3::new(phi.sin() * theta.sin() , theta.cos(), phi.cos() * theta.sin()) * node.w + node.xyz();
                stack.push((depth + 1, new_point.with_w(node.w * rng.random_range(0.1f32..0.3f32))));
            }
        }
    }
    */

    //stack.push((0, vek::Vec4::new(4f32, 9f32, 3f32, 16f32)));

    
    let mut src_points = vec![vek::Vec4::<f32>::default(); 32];
    // simple random spheres naive implementation
    for point in src_points.iter_mut() {
        *point = vek::Vec4::new(
            rng.random_range(0f32..(size as f32)),
            rng.random_range(0f32..(size as f32)),
            rng.random_range(0f32..(size as f32)),
            rng.random_range(4f32..10f32)        
        );
    }
    

    log::info!("tilings points in 3D");
    
    // tiled in 3D (26 neighbours)
    let mut points = Vec::<vek::Vec4::<f32>>::with_capacity(src_points.len() * 27);
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                points.extend(src_points.iter().map(|point| point + vek::Vec4::new(x as f32 * size as f32, y as f32 * size as f32, z as f32 * size as f32, 0f32)));
            }
        }
    }

    log::info!("calculating SDF");

    let texels = (0..(size*size*size)).into_par_iter().map(|i| {
        let p = index_to_offset(i as usize, size as usize);
        let fp = p.as_::<f32>();
    
        //let distance = (fp + vek::Vec3::new(0f32, -10f32, 0f32)).magnitude() - 5.0f32;
        let mut distance = 100000f32;

        for point in points.iter() {
            distance = distance.min((fp - point.xyz()).magnitude() - point.w); 
        }

        f16::from_f32(distance)
    }).collect::<Vec<_>>();

    let bytes = cast_slice::<f16, u8>(&texels);
    
    let region = vk::MemoryToImageCopyEXT::default()
        .host_pointer(bytes.as_ptr() as *const _)
        .image_extent(vk::Extent3D::default().height(size).width(size).depth(size))
        .image_subresource(image_subresource_layers);
    let regions = [region];
    
    let copy_memory_to_image_info = vk::CopyMemoryToImageInfoEXT::default()
        .dst_image(image)
        .dst_image_layout(vk::ImageLayout::GENERAL)
        .flags(vk::HostImageCopyFlagsEXT::empty())
        .regions(&regions);
    
    host_image_copy_device.copy_memory_to_image(&copy_memory_to_image_info).unwrap();
}

// assumed to be 64x64x64
fn jump_flood(pixels: Vec<f32>, flip: bool) -> Vec<f32> {
    log::info!("calculating flood fill. flip:{flip}");
    let sizes = [32, 16, 8, 4, 2, 1, 2, 1];

    let threshold = 3.0;
    let mut colours_and_seeds = pixels.par_iter().enumerate().map(|(i, density)| {
        let is_set = if flip {
            *density > threshold
        } else {
            *density < -threshold
        };

        is_set.then_some(index_to_offset(i, 64).as_::<f32>())
    }).collect::<Vec<_>>();

    for size in sizes {
        let back_buffer = colours_and_seeds.clone();
        pixels.par_iter().enumerate().zip(colours_and_seeds.par_iter_mut()).for_each(|((p, _), curr)| {
            let p2 = index_to_offset(p, 64).as_::<i32>();
            
            for neighbour_offset_index in 0..27 {
                let neighbour_offset = index_to_offset(neighbour_offset_index, 3).as_::<i32>() - 1;

                let q2 = p2 + neighbour_offset * size;

                if q2.cmpge(&vek::Vec3::broadcast(0)).reduce_and() && q2.cmplt(&vek::Vec3::broadcast(64)).reduce_and() {
                    let q = offset_to_index(q2.as_::<usize>(), 64);
                    
                    if curr.is_none() && back_buffer[q].is_some() {
                        *curr = back_buffer[q];
                    }

                    if let Some((p_seed, q_seed)) = curr.zip(back_buffer[q]) {
                        let dist_p_s = p_seed.distance(p2.as_::<f32>());
                        let dist_p_s_prime = q_seed.distance(p2.as_::<f32>());
                        
                        if dist_p_s > dist_p_s_prime {
                            *curr = back_buffer[q];
                        }
                    }
                }
            }
        });
    }

    colours_and_seeds.iter().enumerate().map(|(index, seed)| {
        let p = index_to_offset(index, 64).as_::<f32>();
        let seed = seed.unwrap();
        
        if flip {
            -p.distance(seed)
        } else {
            p.distance(seed)
        }
        
    }).collect::<Vec<_>>()
}

fn jump_flood_w_negative(pixels: Vec<f32>) -> Vec<f32> {
    let a = jump_flood(pixels.clone(), false);
    let mut b = jump_flood(pixels, true);
    
    for (dst, src) in b.iter_mut().zip(a.into_iter()) {
        *dst += src;
    }

    b
}

fn d(p: vek::Vec3<f32>) -> f32 {
    let mut d = p.y;
    let scale = 40.0f32;
    return p.y - 5.0 + (p.x.sin() + p.z.cos()) * 10.0;
}

pub fn generate_terrain_chunk_data(offset: vek::Vec3<i32>, size: u32) -> Vec<f16> {
    log::info!("calculating SDF");

    let texels = (0..(size*size*size)).into_par_iter().map(|i| {
        let p = index_to_offset(i as usize, size as usize);
        let fp = p.as_::<f32>() * 0.5 + offset.as_::<f32>() * 32f32;
        f16::from_f32(d(fp) / 2.0)
    }).collect::<Vec<_>>();

    texels
}

pub fn generate_terrain_chunk_data2(offset: vek::Vec3<i32>, size: u32) -> Vec<f16> {
    log::info!("calculating SDF");

    let noise = noise::Simplex::new(1234);

    let seeds = (0..(size*size*size)).into_par_iter().map(|i| {
        let p = index_to_offset(i as usize, size as usize);
        let fp = p.as_::<f32>() * 0.5 + offset.as_::<f32>() * 31f32;

        let mut density = fp.y - 5f32;

        density += noise.get([fp.x as f64 * 0.05, fp.z as f64 * 0.05]) as f32 * 4f32;

        density
    }).collect::<Vec<_>>();

    let texels = jump_flood_w_negative(seeds).into_iter().map(|x| f16::from_f32(x * 0.25)).collect::<Vec<_>>();

    texels
}

pub unsafe fn write_image_data_host_image_copy(host_image_copy_device: &mut &ash::ext::host_image_copy::Device, bytes: &[u8], image: vk::Image, size: u32) {
    let image_subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .layer_count(1)
        .base_array_layer(0);
        
    
    let region = vk::MemoryToImageCopyEXT::default()
        .host_pointer(bytes.as_ptr() as *const _)
        .image_extent(vk::Extent3D::default().height(size).width(size).depth(size))
        .image_subresource(image_subresource_layers);
    let regions = [region];
    
    let copy_memory_to_image_info = vk::CopyMemoryToImageInfoEXT::default()
        .dst_image(image)
        .dst_image_layout(vk::ImageLayout::GENERAL)
        .flags(vk::HostImageCopyFlagsEXT::empty())
        .regions(&regions);
    
    host_image_copy_device.copy_memory_to_image(&copy_memory_to_image_info).unwrap();
}


pub unsafe fn write_image_data_scratch_buffer(ctx: &mut GraphicsContext, cmd: vk::CommandBuffer, scratch_buffer: &mut crate::buffer::ScratchBuffer, bytes: &[u8], image: vk::Image, size: u32) {
    let image_subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .layer_count(1)
        .base_array_layer(0);

    let full_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);


    // TODO: batch pipeline barriers
    let chunk_lookup_image_barrier = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(ctx.queue_family_index)
        .dst_queue_family_index(ctx.queue_family_index)
        .image(image)
        .subresource_range(full_subresource_range);
    let image_memory_barriers = [chunk_lookup_image_barrier];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    ctx.device.cmd_pipeline_barrier2(cmd, &dep);



    let written_bytes = scratch_buffer.write_bytes(bytes);
    let extent = vk::Extent3D::default().depth(size).height(size).width(size);
        
    let regions = [vk::BufferImageCopy2::default().buffer_image_height(0).buffer_row_length(0).buffer_offset(written_bytes.buffer_offset_start).image_extent(extent).image_subresource(image_subresource_layers)];
    let copy_info = vk::CopyBufferToImageInfo2::default()
        .dst_image(image)
        .dst_image_layout(vk::ImageLayout::GENERAL)
        .src_buffer(scratch_buffer.buffer)
        .regions(&regions);
    ctx.device.cmd_copy_buffer_to_image2(cmd, &copy_info);



    // TODO: batch pipeline barriers
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    ctx.device.cmd_pipeline_barrier2(cmd, &dep);
}


pub struct Texture3D {
    pub image: vk::Image,
    pub allocation: Allocation,
    pub image_view: vk::ImageView,
    pub size: vek::Extent3<u32>,

    pub specific_mip_image_views: Option<Vec<vk::ImageView>>,
}

impl Texture3D {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(mips) =  self.specific_mip_image_views {
            for mip in mips {
                device.destroy_image_view(mip, None);
            }
        }

        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}