use std::collections::{HashMap, HashSet};

use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use half::f16;
use rand::{RngExt, SeedableRng};
use vek::Clamp;

use crate::{buffer, others, ray_tracing::{self, calculate_matrix}, renderer::GraphicsContext, sdf_texture::{self, Texture3D}, utils::{index_to_offset, offset_to_index}};

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

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Primitive {
    pub aabb_blas_primitives_offset: u32,
}


pub const VXGI_TEXTURE_SIZE: u32 = 128;

// size of chunk's SDF texture
pub const CHUNK_SDF_TEXTURE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;        
pub const CHUNK_LOGICAL_SIZE: u32 = 64;
pub const CHUNK_PHYSICAL_SIZE: u32 = 4;

pub const CHUNK_LOOKUP_TEXTURE_SIZE: u32 = 128;
pub const CHUNK_LOOKUP_TEXTURE_VOLUME: u32 = CHUNK_LOOKUP_TEXTURE_SIZE * CHUNK_LOOKUP_TEXTURE_SIZE * CHUNK_LOOKUP_TEXTURE_SIZE;
pub const CHUNK_LOOKUP_TEXTURE_HALF_SIZE: u32 = CHUNK_LOOKUP_TEXTURE_SIZE / 2;



pub const MAX_CHUNKS: usize = 500;
pub const MAX_BVH_NODES: usize = 500;
pub const MAX_PRIMITIVES: usize = 1000;



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
    
    pub(crate) fn aabb(&self) -> vek::Aabb<f32> {
        let transform = self.transform();
        let vertices = crate::utils::ZERO_TO_ONE_CUBE_VERTICES.map(|vertex| {
            let negative_to_one = vertex * 2.0 - 1.0;
            transform.mul_point(negative_to_one)
        });

        let min = vertices.iter().copied().reduce(|x, y| vek::Vec3::partial_min(x, y)).unwrap();
        let max = vertices.iter().copied().reduce(|x, y| vek::Vec3::partial_max(x, y)).unwrap();
        let offset = 0.2f32;

        vek::Aabb {
            // add a little offset since we might do smooth blending and stuff
            min: min-offset,
            max: max+offset
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PackedTransform {
    pub matrix: [f32; 12]
}

pub struct Chunk {
    pub position: vek::Vec3<i32>,
    pub texture: sdf_texture::Texture3D,
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

    // primitive nodes (that apply to the global SDF)
    pub primitive_flat_list: Vec<Node>,
    pub primitive_flat_buffer: buffer::Buffer,    

    // to be able to advance the ray during a HWRT possible intersection, we need to do a software ray-AABB test and get tmin value
    // for that, we need to store the AABBs of the geometries
    pub gpu_packed_aabbs_buffer: buffer::Buffer,
    pub gpu_packed_aabbs: Vec<GpuAabb>,

    pub texture: Texture3D,
    pub texture2: Texture3D,

    pub vxgi_texture: Texture3D,
    

    pub chunks: Vec<Chunk>,
    pub chunk_buffer_lookup: buffer::Buffer,

    pub chunk_lookup_texture: Texture3D,
    pub chunk_lookup_texture_r32_cpu: Vec<u32>,
    

    pub identity_prefab: Prefab,
    pub tree_prefab: Prefab,

    pub bvh_nodes_buffer: buffer::Buffer,   
    pub bvh_primitive_indices_lookup_buffer: buffer::Buffer,    

    pub bvh: obvhs::bvh2::Bvh2,

    pub dirty_chunks: Vec<u32>,
    pub chunk_positions_to_indices: HashMap<vek::Vec3<i32>, u32>,
}

impl Scene {
    pub unsafe fn new(mut ctx: &mut GraphicsContext) -> Self {  
        let texture = sdf_texture::create_texture_3d(ctx, vek::Extent3::broadcast(256), vk::Format::R16_SFLOAT, None, "");
        let texture2 = sdf_texture::create_texture_3d(ctx, vek::Extent3::broadcast(64), vk::Format::R16G16_SFLOAT, None, "");

        // for some reason, using R8G8B8A8_UNORM actually harms the render time instead of improving it... wut? 
        let vxgi_texture = sdf_texture::create_texture_3d(ctx, vek::Extent3::broadcast(VXGI_TEXTURE_SIZE), vk::Format::R16G16B16A16_SFLOAT, Some(6), "VXGI texture");
        
        let tlas = ray_tracing::pre_create_tlas(&mut ctx);
        
        let gpu_packed_aabbs_buffer = buffer::create_buffer_default_flags(&mut ctx, size_of::<GpuAabb>() * 1000, "scene BLAS AABBs buffer");
        let aabbs = vec![];

        let blases = Vec::new();
        let blases_instances = Vec::new();

        // create the buffer that contains the primitives
        // primitives are just an enum (representing their SDF type) and an index which points to the transforms buffer
        let primitive_flat_list = Vec::<Node>::new();
        let primitive_flat_buffer = buffer::create_buffer_default_flags(ctx, size_of::<Node>() * MAX_BVH_NODES, "primitive flat buffer");     

        // transforms buffer of pre-computed inverted f32 mat3x4 matrices
        let transforms = Vec::<Transform>::new();
        let inverse_transforms_buffer = buffer::create_buffer_default_flags(ctx, size_of::<PackedTransform>() * MAX_PRIMITIVES, "transforms buffer");     

        // chunk buffer simply stores u32s as an indirection step to the target sampler image texture index
        let chunk_buffer_lookup = buffer::create_buffer_default_flags(ctx, size_of::<u32>() * MAX_CHUNKS, "chunk buffer");     

        // mega lookup texture
        let chunk_lookup_texture_bruh = sdf_texture::create_texture_3d(ctx, vek::Extent3::broadcast(CHUNK_LOOKUP_TEXTURE_SIZE), vk::Format::R32_UINT, None, "chunk lookup texture");
        let chunk_lookup_texture_r32_cpu = vec![0u32; CHUNK_LOOKUP_TEXTURE_VOLUME as usize];
        
        // bvh buffer (I <3 obvh crate)
        let bvh_nodes = buffer::create_buffer_default_flags(ctx, size_of::<obvhs::bvh2::node::Bvh2Node>() * MAX_BVH_NODES, "packed BVH nodes buffer");     
        let bvh_primitive_indices_lookup = buffer::create_buffer_default_flags(ctx, size_of::<u32>() * MAX_PRIMITIVES, "BVH index lookup buffer");     
        
        let chunks = Vec::<Chunk>::new();
        
        let mut this = Self {
            tlas,
            bvh_primitive_indices_lookup_buffer: bvh_primitive_indices_lookup,
            blases,
            blases_instances,
            gpu_packed_aabbs_buffer,
            gpu_packed_aabbs: aabbs,
            texture,
            texture2,
            bvh_nodes_buffer: bvh_nodes,
            chunk_buffer_lookup,
            vxgi_texture,
            chunks,
            primitive_flat_list,
            primitive_flat_buffer,
            identity_prefab: Prefab::default(),
            tree_prefab: Prefab::default(),
            transforms,
            inverse_transforms_buffer,
            chunk_lookup_texture: chunk_lookup_texture_bruh,
            chunk_lookup_texture_r32_cpu,
            dirty_chunks: Default::default(),
            bvh: obvhs::bvh2::Bvh2::default(),
            chunk_positions_to_indices: Default::default()
        };

        this.chunk_lookup_texture_r32_cpu.fill(u32::MAX);

        // TODO: we don't need to allocate all the chunks up-front, we can dynamically create and destroy them as needed
        for _ in 0..MAX_CHUNKS {
            // TODO: separate normals from SDF
            // normals don't need to be floating point values, they can be R8G8B8_SNORM instead
            // also, we might want to store some metadata alongside the normals (like material type or material properties)
            let img =  sdf_texture::create_texture_3d(ctx, vek::Extent3::broadcast(CHUNK_LOGICAL_SIZE), CHUNK_SDF_TEXTURE_FORMAT, None, "chunk texture");

            this.chunks.push(Chunk {
                position: vek::Vec3::zero(),
                texture: img,
                used: false,
            });
        }
        

        this.create_ghost_primitive(-vek::Vec3::unit_y(), vek::Quaternion::identity(), vek::Vec3::new(20f32, 1f32,20f32));

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
    }

    pub fn rebuild_bvh(&mut self) {
        let mut core_build_time = std::time::Duration::default();
        struct TestPrimitive<'a> {
            transform: &'a Transform
        }
        impl<'a> obvhs::Boundable for TestPrimitive<'a> {
            fn aabb(&self) -> obvhs::aabb::Aabb {
                let aabb = self.transform.aabb();

                obvhs::aabb::Aabb {
                    min: glam::Vec3A::from_array(aabb.min.into_array()),
                    max: glam::Vec3A::from_array(aabb.max.into_array()),
                }
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

        
        for x in self.blases {
            x.destroy(&acceleration_structure_device, &device, &mut allocator);
        }
        log::info!("destroyed BLASes");

        self.tlas.destroy(&acceleration_structure_device, &device, &mut allocator);
        log::info!("destroyed TLAS");

        self.gpu_packed_aabbs_buffer.destroy(device, allocator);
        self.chunk_lookup_texture.destroy(device, allocator);
        self.primitive_flat_buffer.destroy(device, allocator);
        self.inverse_transforms_buffer.destroy(device, allocator);
        self.chunk_buffer_lookup.destroy(device, allocator);
        self.texture2.destroy(device, allocator);
        self.texture.destroy(device, allocator);
        self.vxgi_texture.destroy(device, allocator);
        self.bvh_nodes_buffer.destroy(device, allocator);
        self.bvh_primitive_indices_lookup_buffer.destroy(device, allocator);
        log::info!("destroyed gpu repr");
    }
}