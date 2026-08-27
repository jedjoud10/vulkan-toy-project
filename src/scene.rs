use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use rand::{RngExt, SeedableRng};

use crate::{buffer, others, ray_tracing, renderer::GraphicsContext, sdf_texture};

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

pub const INSTANCE_CUSTOM_INDEX_AABB_LOOKUP_INDEX_MASK: u32 = 15;
pub const INSTANCE_CUSTOM_INDEX_LOOKUP_AABB_USING_PRIMITIVE_INDEX_FLAG_MASK: u32 = 32;

pub const INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK: u32 = 1 << 20;

pub const GI_TEXTURE_SIZE: u32 = 256;

pub const SPAWN_TREES: bool = true;

pub struct Chunk {
    pub position: vek::Vec3<i32>,

    // chunk needs to store a 3D lookup texture to check which primitives intersect the voxel grid
    // texture stores a u8 index into the primitives intersection buffer, which itself stores a linked list of pointers and an index into the global primitives buffer
    // pub primitives_voxel_lookup_texture: sdf_texture::SdfImage,
    // pub primitives_linked_list_buffer: buffer::Buffer,

    // each chunk will have a BLAS for the dynamic terrain part
    pub dynamic_blas: ray_tracing::AccelerationStructureData,
}

pub struct Scene {
    // chunks
    // pub chunks: Vec<Chunk>,

    // TLAS will store all primitives in the world, not bound to any chunks
    pub tlas: ray_tracing::TopLevelAccelerationStructure,
    
    // primitive BLASes are unique to avoid duplicating them
    pub blases: Vec<ray_tracing::AccelerationStructureData>,
    pub blases_instances: Vec<vk::AccelerationStructureInstanceKHR>,

    // chunk blases
    // pub chunk_blases_instances: Vec<vk::AccelerationStructureInstanceKHR>,

    // to be able to advance the ray during a HWRT possible intersection, we need to do a software ray-AABB test and get tmin value
    // for that, we need to store the AABBs of the geometries
    pub aabbs_buffer: buffer::Buffer,
    pub aabbs: Vec<Aabb>,

    /*
    // primitives are geometric shapes stored in the world
    // we trace against their BLASes to accelerate traversal
    pub primitives_buffer: buffer::Buffer,
    pub primitives: Vec<Primitive>,
    */
}

impl Scene {
    pub unsafe fn new(mut ctx: &mut GraphicsContext) -> Self {       
        
        let tlas = ray_tracing::pre_create_tlas(&mut ctx);
        
        let aabbs_buffer = buffer::create_buffer_default_flags(&mut ctx, size_of::<Aabb>() * 100, "scene BLAS AABBs buffer");
        let aabbs = vec![];

        let blases = Vec::new();
        let blases_instances = Vec::new();


        let mut this = Self {
            tlas,
            blases,
            blases_instances,
            aabbs_buffer,
            aabbs,
        };

        let prefab = this.create_primitive_prefab(ctx, &[IDENTBOX]);
        this.create_primitive(-vek::Vec3::unit_y(), vek::Quaternion::identity(), vek::Vec3::new(1000f32, 1f32, 1000f32), false, prefab);
        // this.create_primitive(ctx, vek::Vec3::unit_y() * 2.0f32, vek::Quaternion::identity(), vek::Vec3::new(1f32, 1f32, 1f32), &[IDENTBOX], true);

        if SPAWN_TREES {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(432);
            for x in -20..20 {
                for z in -20..20 {
                    
                    this.create_primitive(
                        vek::Vec3::new(rng.random_range(-400f32..400f32), 4f32, rng.random_range(-400f32..400f32)),
                        vek::Quaternion::rotation_y(rng.random_range(0f32..std::f32::consts::TAU)),
                        vek::Vec3::new(1f32, 1f32, 1f32),
                        true, 
                        prefab,
                    );
                
                    /*
                    let pos = vek::Vec3::new(rng.random_range(-600f32..600f32), 8f32, rng.random_range(-600f32..600f32));
                    blases_instances.push(ray_tracing::instantiate_blas(vek::Quaternion::identity(), pos, vek::Vec3::one(), &blases[0], 0, 0xFF));
                    primitives.push(Primitive { position: pos, flags: 1 });
                    */
                }
            }
        }


        this
    }


    pub unsafe fn create_primitive_prefab(&mut self, mut ctx: &mut GraphicsContext, aabbs: &[Aabb]) -> (usize, usize) {
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
        (s, self.blases.len()-1)
    }

    pub unsafe fn create_primitive(&mut self, position: vek::Vec3<f32>, rotation: vek::Quaternion<f32>, scale: vek::Vec3<f32>, is_local: bool, (aabb_start_index, blas): (usize, usize)) {
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
            &self.blases[blas],
            aabb_start_index as u32 | local_sdf_bit,
            0xFF,
        ));
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