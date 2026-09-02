use std::collections::{HashMap, HashSet};

use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use half::f16;
use rand::{RngExt, SeedableRng};
use vek::Clamp;

use crate::{buffer, others, ray_tracing::{self, calculate_matrix}, renderer::GraphicsContext, sdf_texture::{self, SdfImage}, utils::{index_to_offset, offset_to_index}};

#[derive(Default, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Aabb {
    pub min: vek::Vec3<f32>,
    pub max: vek::Vec3<f32>,
}
impl Aabb {
    fn to_gpu_format(&self) -> GpuAabb {
        let min = self.min;
        let max = self.max;

        // TODO: why is `vec3::with_w` not const??? :(
        GpuAabb { min: vek::Vec4::new(min.x, min.y, min.z, 0f32).map(|x| half::f16::from_f32(x)), max: vek::Vec4::new(max.x, max.y, max.z, 0f32).map(|x| half::f16::from_f32(x)) }
    }
}

#[derive(Default, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct GpuAabb {
    pub min: vek::Vec4<half::f16>,
    pub max: vek::Vec4<half::f16>,
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

// size of chunk's SDF texture
pub const CHUNK_SDF_TEXTURE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;        
pub const CHUNK_LOGICAL_SIZE: u32 = 64;
pub const CHUNK_PHYSICAL_SIZE: u32 = 4;
pub const NUM_CHUNKS_POOL: i32 = 500;
pub const CHUNK_LOOKUP_TEXTURE_SIZE: u32 = 128;
pub const CHUNK_LOOKUP_TEXTURE_HALF_SIZE: u32 = CHUNK_LOOKUP_TEXTURE_SIZE / 2;


pub const SPAWN_TREES: bool = false;
pub const SPAWN_PRIMITIVES: bool = true;

#[derive(Default, Clone, Copy)]
pub struct Prefab {
    aabb_start_index: usize,
    blas_index: usize,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Node {
    pub transform_index: u32,
    pub sdf_type: u32,
}

pub struct Transform {
    pub position: vek::Vec3<f32>,
    pub rotation: vek::Quaternion<f32>,
    pub scale: vek::Vec3<f32>,
}
impl Transform {
    pub(crate) fn transform(&self) -> vek::Mat4<f32> {
        crate::ray_tracing::calculate_matrix(self.rotation, self.position, self.scale)
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PackedTransform {
    pub matrix: [f32; 12]
}

pub struct Chunk {
    pub position: vek::Vec3<i32>,
    pub texture: sdf_texture::SdfImage,
    pub used: bool,
}

pub struct Scene {
    // TLAS will store all primitives in the world, not bound to any chunks
    pub tlas: ray_tracing::TopLevelAccelerationStructure,
    
    // primitive BLASes are unique to avoid duplicating them
    pub blases: Vec<ray_tracing::AccelerationStructureData>,
    pub blases_instances: Vec<vk::AccelerationStructureInstanceKHR>,

    // transforms for all primitives. added whenever we instantiate a primitive
    pub transforms: Vec<Transform>,
    pub inverse_transforms_buffer: buffer::Buffer,

    // lookup texture to see what primitives intersect the same voxels
    pub lookup_texture: SdfImage,
    pub lookup_texture_r32_cpu: Vec<u32>,

    // primitive nodes (that apply to the global SDF)
    pub primitive_flat_list: Vec<Node>,
    pub primitive_flat_buffer: buffer::Buffer,    

    // to be able to advance the ray during a HWRT possible intersection, we need to do a software ray-AABB test and get tmin value
    // for that, we need to store the AABBs of the geometries
    pub gpu_packed_aabbs_buffer: buffer::Buffer,
    pub gpu_packed_aabbs: Vec<GpuAabb>,

    pub texture: SdfImage,
    pub texture2: SdfImage,

    pub vxgi_texture: SdfImage,
    

    pub chunks: Vec<Chunk>,
    pub chunk_buffer_lookup: buffer::Buffer,

    pub chunk_lookup_texture_bruh: SdfImage,
    pub chunk_lookup_texture_r32_cpu: Vec<u32>,
    

    pub identity_prefab: Prefab,
    pub tree_prefab: Prefab,

    pub bvh_nodes: buffer::Buffer,   
    pub bvh_primitive_indices_lookup: buffer::Buffer,    

    pub bvh: obvhs::bvh2::Bvh2,

    pub dirty_chunks: Vec<u32>,
    pub chunk_positions_to_indices: HashMap<vek::Vec3<i32>, u32>,
}

impl Scene {
    pub unsafe fn new(mut ctx: &mut GraphicsContext) -> Self {  
        let texture = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(256), vk::Format::R16_SFLOAT, None);
        let texture2 = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(64), vk::Format::R16G16_SFLOAT, None);

        // for some reason, using R8G8B8A8_UNORM actually harms the render time instead of improving it... wut? 
        let vxgi_texture = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(VXGI_TEXTURE_SIZE), vk::Format::R16G16B16A16_SFLOAT, Some(6));
        
        let tlas = ray_tracing::pre_create_tlas(&mut ctx);
        
        let gpu_packed_aabbs_buffer = buffer::create_buffer_default_flags(&mut ctx, size_of::<GpuAabb>() * 1000, "scene BLAS AABBs buffer");
        let aabbs = vec![];

        let blases = Vec::new();
        let blases_instances = Vec::new();

        let lookup_texture = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(128), vk::Format::R32_UINT, None);
        let primitive_flat_list = Vec::<Node>::new();
        let primitive_flat_buffer = buffer::create_buffer_default_flags(ctx, size_of::<Node>() * 1000, "primitive flat buffer");     
        let lookup_texture_r32_cpu = vec![0u32; 128*128*128];

        let transforms = Vec::<Transform>::new();
        let inverse_transforms_buffer = buffer::create_buffer_default_flags(ctx, size_of::<PackedTransform>() * 1000, "transforms buffer");     

        let chunk_buffer_lookup = buffer::create_buffer_default_flags(ctx, size_of::<u64>() * 1000, "chunk buffer");     
        let chunks = Vec::<Chunk>::new();


        let chunk_lookup_texture_bruh = sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(128), vk::Format::R32_UINT, None);
        let chunk_lookup_texture_r32_cpu = vec![0u32; 128*128*128];

        let bvh_nodes = buffer::create_buffer_default_flags(ctx, size_of::<obvhs::bvh2::node::Bvh2Node>() * 1000, "chunksdfgsdfg");     
        let bvh_primitive_indices_lookup = buffer::create_buffer_default_flags(ctx, size_of::<u32>() * 1000, "chunksdfgsdfg");     

        let mut this = Self {
            tlas,
            bvh_primitive_indices_lookup,
            blases,
            blases_instances,
            gpu_packed_aabbs_buffer,
            gpu_packed_aabbs: aabbs,
            texture,
            texture2,
            bvh_nodes,
            chunk_buffer_lookup,
            vxgi_texture,
            chunks,
            lookup_texture,
            primitive_flat_list,
            primitive_flat_buffer,
            lookup_texture_r32_cpu,
            identity_prefab: Prefab::default(),
            tree_prefab: Prefab::default(),
            transforms,
            inverse_transforms_buffer,
            chunk_lookup_texture_bruh,
            chunk_lookup_texture_r32_cpu,
            dirty_chunks: Default::default(),
            bvh: obvhs::bvh2::Bvh2::default(),
            chunk_positions_to_indices: Default::default()
        };

        this.chunk_lookup_texture_r32_cpu.fill(u32::MAX);
        for _ in 0..NUM_CHUNKS_POOL {
            // TODO: separate normals from SDF
            // normals don't need to be floating point values, they can be R8G8B8_SNORM instead
            // also, we might want to store some metadata alongside the normals (like material type or material properties)
            let img =  sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(CHUNK_LOGICAL_SIZE), CHUNK_SDF_TEXTURE_FORMAT, None);

            this.chunks.push(Chunk {
                position: vek::Vec3::zero(),
                texture: img,
                used: false,
            });
        }
        

        this.create_ghost_primitive(-vek::Vec3::unit_y(), vek::Quaternion::identity(), vek::Vec3::new(20f32, 1f32, 20f32));

        let mut rng = rand::rngs::SmallRng::seed_from_u64(432);
        if SPAWN_PRIMITIVES {
            for x in -4..4 {
                for z in -4..4 {
                    this.create_primitive(
                        vek::Vec3::new(rng.random_range(-20f32..20f32), rng.random_range(-0f32..3f32), rng.random_range(-20f32..20f32)),
                        vek::Quaternion::default(),
                        1f32,
                        rng.random_range(0u32..=2u32),
                    );
                }
            }
        }


        this
    }

    pub fn update(&mut self, elapsed: f32) {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(elapsed.floor() as u64);
        /*
        for (index, transform) in self.transforms.iter_mut().enumerate().skip(1) {
            if (self.blases_instances[index].instance_custom_index_and_mask.low_24() & INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK) == 0 {
                /*
                // global sdf
                transform.rotation = transform.rotation.rotated_x(rng.random_range(-1f32..1f32) * 0.01);
                transform.rotation = transform.rotation.rotated_y(rng.random_range(-1f32..1f32) * 0.01);
                transform.rotation = transform.rotation.rotated_z(rng.random_range(-1f32..1f32) * 0.01);

                transform.position.y += rng.random_range(-1f32..1f32) * 0.01;
                */ 
            }
        }
        */

        let mut core_build_time = std::time::Duration::default();
        struct TestPrimitive<'a> {
            transform: &'a Transform
        }
        impl<'a> obvhs::Boundable for TestPrimitive<'a> {
            fn aabb(&self) -> obvhs::aabb::Aabb {
                let mut aabb = obvhs::aabb::Aabb::from_point(glam::Vec3A::from_array((self.transform.position-1f32).into_array()));
                aabb.extend(glam::Vec3A::from_array((self.transform.position + 1.0).into_array()));
                aabb
            }
        } 
        let primitives = self.primitive_flat_list.iter().map(|node| TestPrimitive { transform: &self.transforms[node.transform_index as usize] }).collect::<Vec<_>>();
        self.bvh = obvhs::bvh2::builder::build_bvh2(&primitives, obvhs::BvhBuildParams::very_slow_build(), &mut core_build_time);
    }
    
    pub unsafe fn create_ghost_primitive(&mut self, position: vek::Vec3<f32>, rotation: vek::Quaternion<f32>, scale: vek::Vec3<f32>) -> usize {
        self.transforms.push(Transform {
            position,
            rotation,
            scale,
        });

        self.primitive_flat_list.push(Node {
            transform_index: (self.transforms.len() - 1) as u32,
            sdf_type: u32::MAX,
        });


        self.blases_instances.len() - 1
    }

    pub unsafe fn create_primitive(&mut self, position: vek::Vec3<f32>, rotation: vek::Quaternion<f32>, scale: f32, sdf_type: u32) -> usize {
        self.transforms.push(Transform {
            position,
            rotation,
            scale: vek::Vec3::broadcast(scale),
        });

        self.primitive_flat_list.push(Node {
            transform_index: (self.transforms.len() - 1) as u32,
            sdf_type,
        });


        self.blases_instances.len() - 1
    }

    pub unsafe fn destroy(self, device: &ash::Device, acceleration_structure_device: &ash::khr::acceleration_structure::Device, mut allocator: &mut gpu_allocator::vulkan::Allocator) {
        for chunk in self.chunks {
            chunk.texture.destroy(device, allocator);
        }
        log::info!("destroyed chunks");

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

        self.gpu_packed_aabbs_buffer.destroy(device, allocator);
        self.chunk_lookup_texture_bruh.destroy(device, allocator);
        self.primitive_flat_buffer.destroy(device, allocator);
        self.lookup_texture.destroy(device, allocator);
        self.inverse_transforms_buffer.destroy(device, allocator);
        self.chunk_buffer_lookup.destroy(device, allocator);
        self.texture2.destroy(device, allocator);
        self.texture.destroy(device, allocator);
        self.vxgi_texture.destroy(device, allocator);
        self.bvh_nodes.destroy(device, allocator);
        self.bvh_primitive_indices_lookup.destroy(device, allocator);
        // self.primitives_buffer.destroy(device, &mut allocator);
        log::info!("destroyed gpu repr");
    }
}