use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use rand::{RngExt, SeedableRng};

use crate::{buffer, others, ray_tracing, renderer::GraphicsContext, sdf_texture::{self, SdfImage}};

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Aabb {
    pub min: vek::Vec3<f32>,
    pub max: vek::Vec3<f32>,
}

pub const IDENTBOX: Aabb = Aabb { min: vek::Vec3::broadcast(-1f32), max: vek::Vec3::broadcast(1f32) };

// some primitives use a local SDF
// others use (and thus contribute) to the global SDF
// primitives of different types
// primitives can have different geometries

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Primitive {
    pub aabb_blas_primitives_offset: u32,
}

pub const INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK: u32 = 1 << 20;

pub const VXGI_TEXTURE_SIZE: u32 = 128;

pub const SPAWN_TREES: bool = false;

#[derive(Clone, Copy)]
pub struct Prefab {
    aabb_start_index: usize,
    blas_index: usize,
    full_aabb: Aabb
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Node {
    pub position: vek::Vec3<f32>,
    pub next_node_index: u32,
}

pub struct Scene {
    // TLAS will store all primitives in the world, not bound to any chunks
    pub tlas: ray_tracing::TopLevelAccelerationStructure,
    
    // primitive BLASes are unique to avoid duplicating them
    pub blases: Vec<ray_tracing::AccelerationStructureData>,
    pub blases_instances: Vec<vk::AccelerationStructureInstanceKHR>,

    pub lookup_texture: SdfImage,
    pub lookup_texture_r32_cpu: Vec<u32>,
    pub primitive_flat_list: Vec<Node>,
    pub primitive_flat_buffer: buffer::Buffer,    

    // to be able to advance the ray during a HWRT possible intersection, we need to do a software ray-AABB test and get tmin value
    // for that, we need to store the AABBs of the geometries
    pub aabbs_buffer: buffer::Buffer,
    pub aabbs: Vec<Aabb>,

    pub texture: SdfImage,
    pub texture2: SdfImage,
    pub vxgi_texture: SdfImage,
    pub texture4: SdfImage,
    
    pub prefabs: Vec<Prefab>,
}

impl Scene {
    pub unsafe fn new(mut ctx: &mut GraphicsContext) -> Self {  
        let texture = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(256), vk::Format::R16_SFLOAT, None);
        let texture2 = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(64), vk::Format::R16G16_SFLOAT, None);

        // for some reason, using R8G8B8A8_UNORM actually harms the render time instead of improving it... wut? 
        let vxgi_texture = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(VXGI_TEXTURE_SIZE), vk::Format::R16G16B16A16_SFLOAT, Some(6));
        let texture4 = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(128), vk::Format::R16G16B16A16_SFLOAT, None);     
        
        let tlas = ray_tracing::pre_create_tlas(&mut ctx);
        
        let aabbs_buffer = buffer::create_buffer_default_flags(&mut ctx, size_of::<Aabb>() * 100, "scene BLAS AABBs buffer");
        let aabbs = vec![];

        let blases = Vec::new();
        let blases_instances = Vec::new();

        let lookup_texture = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(128), vk::Format::R32_UINT, None);
        let primitive_flat_list = Vec::<Node>::new();
        let primitive_flat_buffer = buffer::create_buffer_default_flags(ctx, size_of::<Node>() * 1000, "a");     
        let lookup_texture_r32_cpu = vec![0u32; 128*128*128];


        let mut this = Self {
            tlas,
            blases,
            blases_instances,
            aabbs_buffer,
            aabbs,
            texture,
            texture2,
            vxgi_texture,
            texture4,
            lookup_texture,
            primitive_flat_list,
            primitive_flat_buffer,
            lookup_texture_r32_cpu,
            prefabs: Vec::new(),
        };

        let prefab = this.create_primitive_prefab(ctx, &[IDENTBOX]);
        this.create_primitive(-vek::Vec3::unit_y(), vek::Quaternion::identity(), vek::Vec3::new(1000f32, 1f32, 1000f32), false, prefab);

        let tree_prefab = this.create_primitive_prefab(ctx, &[Aabb {
            min: vek::Vec3::new(-3f32, -5f32, -3f32),
            max: vek::Vec3::new(3f32, 5f32, 3f32),
        }]);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(432);
        for x in -1..1 {
            for z in -1..1 {
                this.create_primitive(
                    vek::Vec3::new(rng.random_range(-40f32..40f32), 1f32, rng.random_range(-40f32..40f32)),
                    vek::Quaternion::default(),
                    vek::Vec3::new(1f32, 1f32, 1f32),
                    false, 
                    prefab,
                );
            }
        }



        
        /*
        if SPAWN_TREES {
            for x in -50..50 {
                for z in -50..50 {
                    
                    this.create_primitive(
                        vek::Vec3::new(rng.random_range(-400f32..400f32), 4f32, rng.random_range(-400f32..400f32)),
                        vek::Quaternion::rotation_y(rng.random_range(0f32..std::f32::consts::TAU)),
                        vek::Vec3::new(1f32, 1f32, 1f32),
                        true, 
                        tree_prefab,
                    );
                }
            }
        }
        */


        this
    }


    pub unsafe fn create_primitive_prefab(&mut self, mut ctx: &mut GraphicsContext, aabbs: &[Aabb]) -> Prefab {
        let cmd = others::begin_recording(&mut ctx);
        let mut writer = buffer::begin_buffer_writer(&mut ctx);
        
        let s = self.aabbs.len();
        self.aabbs.extend_from_slice(aabbs);
        let written = writer.write_bytes(cast_slice(&aabbs));

        let geometry = ray_tracing::BlasGeometry::AABBs {
            aabb_buffer_address: written.buffer_device_address_start,
            max_count: aabbs.len() as u32
        };

        let blas1 = ray_tracing::create_blas(&mut ctx, cmd, geometry);

        others::end_recording_and_submit(&mut ctx, cmd);
        buffer::end_buffer_writer(&mut ctx, writer);

        self.blases.push(blas1);

        let prefab = Prefab {
            blas_index: s,
            aabb_start_index: self.blases.len()-1,
            full_aabb: aabbs.iter().copied().reduce(|a, b| Aabb { min: vek::Vec3::partial_min(a.min, b.min), max: vek::Vec3::partial_min(a.max, b.max) }).unwrap()
        };

        self.prefabs.push(prefab);
        prefab
    }

    pub unsafe fn create_primitive(&mut self, position: vek::Vec3<f32>, rotation: vek::Quaternion<f32>, scale: vek::Vec3<f32>, is_local: bool, prefab: Prefab) {
        if is_local {
            assert!((scale.partial_cmpeq(&vek::Vec3::one())).reduce_and(), "scale must be one when dealing with local SDF prims");
        }

        let local_sdf_bit = if is_local {
            INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK
        } else {
            0
        };

        self.blases_instances.push(ray_tracing::instantiate_blas(
            rotation,
            position,
            scale,
            &self.blases[prefab.blas_index],
            prefab.aabb_start_index as u32 | local_sdf_bit,
            0xFF,
        ));

        log::info!("{}", self.primitive_flat_list.len());

        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    
                    let offset = (position.floor() + 64f32).as_::<i32>() + vek::Vec3::new(x,y,z);
                    let index = crate::utils::offset_to_index(offset.as_::<usize>(), 128);
                    
                    let previous = &mut self.lookup_texture_r32_cpu[index];
                    
                    let current = self.primitive_flat_list.len() as u32;
                    assert!(current < u8::MAX as u32);
                    let current = current as u8;

                    let mut bytes = previous.to_ne_bytes();

                    for (i, k) in bytes.iter().enumerate() {
                        if *k == 0 {
                            bytes[i] = current;
                            break;
                        }
                    }
                    
                    self.lookup_texture_r32_cpu[index] = u32::from_ne_bytes(bytes);
                }
            }
        }



        self.primitive_flat_list.push(Node {
            position,
            next_node_index: 0,
        })
    }

    pub unsafe fn destroy(self, device: &ash::Device, acceleration_structure_device: &ash::khr::acceleration_structure::Device, mut allocator: &mut gpu_allocator::vulkan::Allocator) {
        // self.texture.destroy(&device, &mut allocator);
        // self.texture2.destroy(&device, &mut allocator);
        // self.texture3.destroy(&device, &mut allocator);
        
        log::info!("destroyed sdf texture");

        
        for x in self.blases {
            x.destroy(&acceleration_structure_device, &device, &mut allocator);
        }
        log::info!("destroyed BLASes");

        self.tlas.destroy(&acceleration_structure_device, &device, &mut allocator);
        log::info!("destroyed TLAS");

        self.aabbs_buffer.destroy(&device, &mut allocator);
        // self.primitives_buffer.destroy(device, &mut allocator);
        log::info!("destroyed gpu repr");
    }
}