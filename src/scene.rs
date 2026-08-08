use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use rand::{RngExt, SeedableRng};

use crate::{buffer, others, ray_tracing, renderer::GraphicsContext, sdf_texture};

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Aabb {
    min: vek::Vec3<f32>,
    max: vek::Vec3<f32>,
}

// some primitives use a local SDF
// others use (and thus contribute) to the global SDF
// primitives of different types
// primitives can have different geometries

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Primitive {
    pub position: vek::Vec3<f32>,
    pub flags: u32,
}

const INSTANCE_CUSTOM_INDEX_AABB_LOOKUP_INDEX_MASK: u32 = 15;
const INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK: u32 = 16;


pub struct Scene {
    pub tlas: ray_tracing::TopLevelAccelerationStructure,
    
    pub blases: Vec<ray_tracing::AccelerationStructureData>,
    pub blases_instances: Vec<vk::AccelerationStructureInstanceKHR>,

    pub scene_representation_for_sdf_buffer: buffer::Buffer,
    pub scene_representation_for_sdf: Vec<Aabb>,

    pub primitives_buffer: buffer::Buffer,
    pub primitives: Vec<Primitive>,

    pub texture: sdf_texture::SdfImage,
}

impl Scene {
    pub unsafe fn new(mut ctx: &mut GraphicsContext) -> Self {
        let cmd = others::begin_recording(&mut ctx);
        let mut writer = buffer::begin_buffer_writer(&mut ctx);
        
        let aabbs = [ray_tracing::AccelerationStructureAabb {
            min: -vek::Vec3::one(),
            max: vek::Vec3::one(),
        }];

        let written = writer.write_bytes(cast_slice(&aabbs));

        let geometry = ray_tracing::BlasGeometry::AABBs {
            aabb_buffer_address: written.buffer_device_address_start,
            max_count: 1
        };

        let blas1 = ray_tracing::create_blas(&mut ctx, cmd, geometry);



        let aabbs = [ray_tracing::AccelerationStructureAabb {
            min: -vek::Vec3::broadcast(5f32),
            max: vek::Vec3::broadcast(5f32),
        }];

        let written = writer.write_bytes(cast_slice(&aabbs));

        let geometry = ray_tracing::BlasGeometry::AABBs {
            aabb_buffer_address: written.buffer_device_address_start,
            max_count: 1
        };

        let blas2 = ray_tracing::create_blas(&mut ctx, cmd, geometry);


        let blases = vec![blas1, blas2];

        others::end_recording_and_submit(&mut ctx, cmd);
        buffer::end_buffer_writer(&mut ctx, writer);
        
        let tlas = ray_tracing::pre_create_tlas(&mut ctx);
        
        let scene_representation_for_sdf_buffer = buffer::create_buffer_default_flags(&mut ctx, size_of::<Aabb>() * 100, "a");
        let scene_representation_for_sdf = vec![Aabb { min: -vek::Vec3::one(), max: vek::Vec3::one() }, Aabb { min: -vek::Vec3::broadcast(5f32), max: vek::Vec3::broadcast(5f32) }];

        let primitives_buffer = buffer::create_buffer_default_flags(&mut ctx, size_of::<Primitive>() * 100, "a");
        let mut primitives = Vec::<Primitive>::new();

        let mut blases_instances = Vec::new();

        // plane
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::identity(),
            -vek::Vec3::unit_y(),
            vek::Vec3::new(1000.0, 1.0, 1000.0),
            &blases[0],
            0,
            0xFF,
        ));

        // sphere
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::identity(),
            vek::Vec3::unit_y(),
            vek::Vec3::broadcast(2f32),
            &blases[0],
            0,
            0xFF,
        ));

        // hex
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::default(),
            vek::Vec3::new(10f32, 2f32, 0f32),
            vek::Vec3::new(2.6, 2.1, 4.1),
            &blases[0],
            0,
            0xFF,
        ));

        // torus
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::rotation_3d(2.4f32, vek::Vec3::<f32>::one().normalized()),
            vek::Vec3::new(2f32, 2f32, 5f32),
            vek::Vec3::new(7f32, 2f32, 7f32),
            &blases[0],
            0,
            0xFF,
        ));

        // cylinder
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::default(),
            vek::Vec3::new(-10f32,0f32, -0f32),
            vek::Vec3::new(2f32, 400f32, 2f32),
            &blases[0],
            0,
            0xFF,
        ));

        // sphere 2
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::default(),
            vek::Vec3::new(0f32,10f32, 0f32),
            vek::Vec3::new(6f32, 6f32, 6f32),
            &blases[0],
            0,
            0xFF,
        ));

        
        let mut rng = rand::rngs::SmallRng::seed_from_u64(432);
        for x in -20..20 {
            for z in -20..20 {
                let aabb_index = 1 & INSTANCE_CUSTOM_INDEX_AABB_LOOKUP_INDEX_MASK;
                let is_local_sdf = INSTANCE_CUSTOM_INDEX_LOCAL_SDF_FLAG_MASK;

                blases_instances.push(ray_tracing::instantiate_blas(
                    vek::Quaternion::rotation_y(rng.random_range(0f32..std::f32::consts::TAU)),
                    vek::Vec3::new(rng.random_range(-200f32..200f32), 2f32, rng.random_range(-200f32..200f32)),
                    vek::Vec3::one(), // cannot do non-uniform scale! cannot do scale in general unless we account for it in the shader side!
                    &blases[1],
                    aabb_index | is_local_sdf,
                    0xFF
                ));

                /*
                let pos = vek::Vec3::new(rng.random_range(-600f32..600f32), 8f32, rng.random_range(-600f32..600f32));
                blases_instances.push(ray_tracing::instantiate_blas(vek::Quaternion::identity(), pos, vek::Vec3::one(), &blases[0], 0, 0xFF));
                primitives.push(Primitive { position: pos, flags: 1 });
                */
            }
        }

        let texture = sdf_texture::create_voxel_image(ctx);
    
        Self {
            scene_representation_for_sdf,
            scene_representation_for_sdf_buffer,
            tlas,
            blases,
            blases_instances,
            texture,
            primitives,
            primitives_buffer,
        }
    }

    pub unsafe fn add_primitive(&mut self, ctx: &mut GraphicsContext, pos: vek::Vec3<f32>, add: bool) {
        self.blases_instances.push(ray_tracing::instantiate_blas(vek::Quaternion::identity(), pos, vek::Vec3::one(), &self.blases[0], 0, 0xFF));
        self.primitives.push(Primitive { position: pos, flags: if add { 1 } else { 0 } });
    }

    pub unsafe fn destroy(self, device: &ash::Device, acceleration_structure_device: &ash::khr::acceleration_structure::Device, mut allocator: &mut gpu_allocator::vulkan::Allocator) {
        self.texture.destroy(&device, &mut allocator);
        log::info!("destroyed sdf texture");

        
        for x in self.blases {
            x.destroy(&acceleration_structure_device, &device, &mut allocator);
        }
        log::info!("destroyed BLASes");

        self.tlas.destroy(&acceleration_structure_device, &device, &mut allocator);
        log::info!("destroyed TLAS");

        self.scene_representation_for_sdf_buffer.destroy(&device, &mut allocator);
        self.primitives_buffer.destroy(device, &mut allocator);
        log::info!("destroyed gpu repr");
    }
}