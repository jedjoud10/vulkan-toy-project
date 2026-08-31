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

pub const NUM_CHUNKS_XZ: i32 = 10;
pub const SPAWN_TREES: bool = false;
pub const SPAWN_CHUNKS: bool = true;
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

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PackedTransform {
    pub matrix: [f32; 12]
}

pub struct Chunk {
    pub position: vek::Vec3<i32>,
    pub texture: sdf_texture::SdfImage,
    pub blas_instance_index: usize,
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

    pub wow: buffer::Buffer,   
    pub wow2: buffer::Buffer,    

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

        let wow = buffer::create_buffer_default_flags(ctx, size_of::<obvhs::bvh2::node::Bvh2Node>() * 1000, "chunksdfgsdfg");     
        let wow2 = buffer::create_buffer_default_flags(ctx, size_of::<u32>() * 1000, "chunksdfgsdfg");     

        let mut this = Self {
            tlas,
            wow2,
            blases,
            blases_instances,
            gpu_packed_aabbs_buffer,
            gpu_packed_aabbs: aabbs,
            texture,
            texture2,
            wow,
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


        /*
        if SPAWN_CHUNKS {
            let mut chunk = 0;
            for x in -2..2 {
                for y in 0..1 {
                    for z in -2..2 {
                        let chunk_position = vek::Vec3::new(x,y,z);
                        let texels = sdf_texture::generate_terrain_chunk_data2(chunk_position, 64);


                        let mut chunk_aabb = Aabb {
                            min: vek::Vec3::broadcast(32f32),
                            max: vek::Vec3::zero(),
                        };

                        for (k, s) in texels.iter().enumerate() {
                            let pos = index_to_offset(k, 64).as_::<f32>();
                        
                            if f16::to_f32(*s) < 1f32 {
                                chunk_aabb.min = vek::Vec3::partial_min(chunk_aabb.min, pos);
                                chunk_aabb.max = vek::Vec3::partial_max(chunk_aabb.max, pos);
                            }
                        }

                        chunk_aabb.min -= 1.0f32;
                        chunk_aabb.max += 1.0f32;

                        chunk_aabb.max *= 0.5f32;
                        chunk_aabb.min *= 0.5f32;

                        chunk_aabb.max = chunk_aabb.max.clamped(vek::Vec3::zero(), vek::Vec3::broadcast(32f32));
                        chunk_aabb.min = chunk_aabb.min.clamped(vek::Vec3::zero(), vek::Vec3::broadcast(32f32));
                        let prefab = this.create_primitive_prefab(ctx, &[chunk_aabb]);                    


                        let blas_instance_index = this.create_primitive2(chunk_position.as_::<f32>() * 32f32, prefab);

                        let mut img =  sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(64), vk::Format::R16_SFLOAT, None);

                        sdf_texture::write_cpu_sdf_to_image2(&mut ctx.host_image_copy_device, cast_slice(&texels), img.image, 64);

                        this.chunks.push(Chunk {
                            position: chunk_position,
                            texture: img,
                            blas_instance_index,
                        });
                        chunk += 1;

                        let texel_position = (chunk_position + 64).as_::<usize>();
                        let texel_index = offset_to_index(texel_position, 128);
                        // this.chunk_lookup_texture_r32_cpu[texel_index] = chunk;
                    }
                }
            }
        }
        */

        if SPAWN_CHUNKS {
            let mut chunk = 0;

            this.chunk_lookup_texture_r32_cpu.fill(u32::MAX);
            for x in -NUM_CHUNKS_XZ..NUM_CHUNKS_XZ {
                for y in 0..1 {
                    for z in -NUM_CHUNKS_XZ..NUM_CHUNKS_XZ {
                        let chunk_position = vek::Vec3::new(x,y,z);
                        let mut img =  sdf_texture::create_voxel_image(ctx, vek::Extent3::broadcast(64), vk::Format::R16G16B16A16_SFLOAT, None);

                        this.chunks.push(Chunk {
                            position: chunk_position,
                            texture: img,
                            blas_instance_index: 0,
                        });

                        let texel_position = (chunk_position + 64).as_::<usize>();
                        let texel_index = offset_to_index(texel_position, 128);
                        this.chunk_lookup_texture_r32_cpu[texel_index] = chunk;

                        this.chunk_positions_to_indices.insert(chunk_position, chunk);

                        chunk += 1;
                    }
                }
            }
        }

        

        this.identity_prefab = this.create_primitive_prefab(ctx, &[IDENTBOX]);
        this.create_primitive(-vek::Vec3::unit_y(), vek::Quaternion::identity(), vek::Vec3::new(1000f32, 1f32, 1000f32), false, None, 0, this.identity_prefab);

        this.tree_prefab = this.create_primitive_prefab(ctx, &[Aabb {
            min: vek::Vec3::new(-3f32, -5f32, -3f32),
            max: vek::Vec3::new(3f32, 5f32, 3f32),
        }]);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(432);
        if SPAWN_PRIMITIVES {
            for x in -4..4 {
                for z in -4..4 {
                    this.create_primitive(
                        vek::Vec3::new(rng.random_range(-20f32..20f32), rng.random_range(-0f32..3f32), rng.random_range(-20f32..20f32)),
                        vek::Quaternion::default(),
                        vek::Vec3::new(1f32, 1f32, 1f32),
                        false, 
                        None,
                        rng.random_range(0u32..=2u32),
                        this.identity_prefab,
                    );
                }
            }
        }


        if SPAWN_TREES {
            for x in -5..5 {
                for z in -5..5 {
                    
                    this.create_primitive(
                        vek::Vec3::new(rng.random_range(-400f32..400f32), 4f32, rng.random_range(-400f32..400f32)),
                        vek::Quaternion::rotation_y(rng.random_range(0f32..std::f32::consts::TAU)),
                        vek::Vec3::new(1f32, 1f32, 1f32),
                        true, 
                        None,
                        0,
                        this.tree_prefab,
                    );
                }
            }
        }


        this
    }

    pub fn update(&mut self, elapsed: f32) {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(elapsed.floor() as u64);
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

    pub unsafe fn create_primitive_prefab(&mut self, mut ctx: &mut GraphicsContext, aabbs: &[Aabb]) -> Prefab {
        let cmd = others::begin_recording(&mut ctx);
        let mut writer = buffer::begin_buffer_writer(&mut ctx);
        
        self.gpu_packed_aabbs.extend(aabbs.iter().map(|x| x.to_gpu_format()));
        let written = writer.write_bytes(cast_slice(&aabbs));
        
        let geometry = ray_tracing::BlasGeometry::AABBs {
            aabb_buffer_address: written.buffer_device_address_start,
            max_count: aabbs.len() as u32
        };
        
        let blas1 = ray_tracing::create_blas(&mut ctx, cmd, geometry);
        
        others::end_recording_and_submit(&mut ctx, cmd);
        buffer::end_buffer_writer(&mut ctx, writer);
        
        let s = self.blases.len();
        self.blases.push(blas1);

        Prefab {
            blas_index: s,
            aabb_start_index: self.blases.len()-1,
        }
    }


    
    pub unsafe fn create_primitive(&mut self, position: vek::Vec3<f32>, rotation: vek::Quaternion<f32>, scale: vek::Vec3<f32>, is_local: bool, chunk_index: Option<u32>, sdf_type: u32, prefab: Prefab) -> usize {
        if is_local {
            assert!((scale.partial_cmpeq(&vek::Vec3::one())).reduce_and(), "scale must be one when dealing with local SDF prims");
        }

        let local_sdf_bit = if is_local {
            INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK
        } else {
            0
        };

        let chunk_index = chunk_index.map(|x| x << 5).unwrap_or_default();

        self.blases_instances.push(ray_tracing::instantiate_blas(
            rotation,
            position,
            scale,
            &self.blases[prefab.blas_index],
            chunk_index | prefab.aabb_start_index as u32 | local_sdf_bit,
            0xFF,
        ));

        self.transforms.push(Transform {
            position,
            rotation,
            scale,
        });

        /*
        if !is_local {
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
                transform_index: (self.transforms.len() - 1) as u32,
                sdf_type,
            })
        }
        */

        self.primitive_flat_list.push(Node {
            transform_index: (self.transforms.len() - 1) as u32,
            sdf_type,
        });


        self.blases_instances.len() - 1

    }


    
    pub unsafe fn create_primitive2(&mut self, position: vek::Vec3<f32>, prefab: Prefab) -> usize {
        let scale = vek::Vec3::one();
        let rotation = vek::Quaternion::identity();

        self.blases_instances.push(ray_tracing::instantiate_blas(
            rotation,
            position,
            scale,
            &self.blases[prefab.blas_index],
            prefab.aabb_start_index as u32,
            0xFF,
        ));

        self.transforms.push(Transform {
            position,
            rotation,
            scale,
        });

        self.blases_instances.len() - 1

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

        self.gpu_packed_aabbs_buffer.destroy(&device, &mut allocator);
        // self.primitives_buffer.destroy(device, &mut allocator);
        log::info!("destroyed gpu repr");
    }
}