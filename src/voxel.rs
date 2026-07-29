use std::io::Read;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use noise::NoiseFn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use crate::{buffer::{self, ScratchBuffer}, renderer::GraphicsContext};

mod sparse;
mod util;
mod chunk;
mod testing;

pub use sparse::*;
pub use util::*;
pub use testing::*;
pub use util::{TOTAL_SIZE};


pub unsafe fn create_sparse_structures(
    ctx: &mut GraphicsContext,
    cmd: vk::CommandBuffer,
    scratch_buffer: &mut ScratchBuffer,
    force_regenerate: bool,
) -> SparseVoxelOctree {
    let mut svo = SparseVoxelOctree::new_with_root_node(ctx);

    let cached_folder_path = dirs::data_dir()
        .map(|data_dir| data_dir.join("nodlemanstuff").join("vulkanvoxelraytracer"));
    let cached_file_path = cached_folder_path
        .as_ref()
        .cloned()
        .map(|data_dir| data_dir.join("map.data"));

    
    let cached_file = cached_file_path.as_ref().and_then(|path| std::fs::File::open(path).ok());

    let res = if let Some(cached_file) = cached_file && !force_regenerate {
        // load data from file
        log::warn!("{}", cached_file_path.unwrap().as_os_str().to_str().unwrap());
        log::info!("cache file found, loading chunks...");
        log::info!("cache file size (compressed): {}", bytesize::ByteSize::b(cached_file.metadata().unwrap().len() as u64).display());

        let mut zlib_decoder = flate2::read::ZlibDecoder::new(cached_file);
        let mut vec = Vec::<u8>::new();
        zlib_decoder.read_to_end(&mut vec).unwrap();

        log::info!("data size (uncompressed): {}", bytesize::ByteSize::b(vec.len() as u64).display());
        
        minicbor::decode::<SparseVoxelTreeBuildResultGpuBuffers>(&vec).unwrap()
    } else {
        // regenerate chunks and save to file
        log::warn!("cache file not found (or was forced to regenerate), regenerating chunks...");
        let mut fbm = noise::Fbm::<noise::Perlin>::new(0); 
        fbm.octaves = 3;
        fbm.frequency = 0.001;
        
        let mut extra = noise::Fbm::<noise::Billow::<noise::Simplex>>::new(0); 
        extra.octaves = 3;
        extra.frequency = 0.01;
        
        const NUM_CHUNKS: usize = (util::TOTAL_SIZE as usize / 64);

        let mut chunk_positions = Vec::<vek::Vec3<usize>>::new();

        for x in 0..256 {
            for z in 0..256 {
                for y in 0..2 {
                    chunk_positions.push(vek::Vec3::new(x,y + 2,z));
                }
            }
        }

        let chunks = chunk_positions.into_par_iter().map(|chunk_position: vek::Vec3<usize>| {
            let mut voxel_bit_set = fixedbitset::FixedBitSet::with_capacity(64*64*64);
                        
            for index in 0..(64*64*64)  {
                let local_position = util::index_to_offset(index, 64);
                let world_position = local_position + chunk_position * 64;
                let pos = world_position.as_::<f64>();

                let height = fbm.get([pos.x, pos.z]) * 300.0f64 - 160.0f64;

                let density = height + pos.y;

                voxel_bit_set.set(index, density < 0f64);
            }

            let mut chunk = chunk::Chunk::new(chunk_position.as_::<u32>(), voxel_bit_set);
            chunk.rebuild();
            log::info!("generated chunk at {chunk_position}");
            chunk
        }).collect::<Vec<_>>();

        for chunk in chunks {
            svo.register_chunk(chunk);
        }

        let res: SparseVoxelTreeBuildResultGpuBuffers = convert_to_buffers(&svo);
        

        if let Some(path) = cached_file_path {
            log::warn!("{}", path.as_os_str().to_str().unwrap());
            std::fs::create_dir_all(cached_folder_path.unwrap()).unwrap();
            let file = std::fs::File::create(&path).unwrap();
            let zlib_encoder = flate2::write::ZlibEncoder::new(file, flate2::Compression::fast());
            let mut writer = minicbor::encode::write::Writer::new(zlib_encoder);
            minicbor::encode(res.clone(), &mut writer).unwrap();
            log::debug!("wrote cached serialized data to file");
        } else {
            log::error!("cached file path could not be found");
        }

        res
    };

    svo.apply_update_gpu_buffers(ctx, cmd, scratch_buffer, &res);
    log::info!("created & updated sparse voxel tree buffers");

    for k in res.indices.iter().take(32) {
        log::debug!("index:{}", k);
    }

    for k in res.bitmasks.iter().take(32) {
        log::debug!("bitmask:{}", k);
    }


    svo
}


pub unsafe fn create_sparse_structures2(
    ctx: &mut GraphicsContext,
    cmd: vk::CommandBuffer,
    scratch_buffer: &mut ScratchBuffer,
    force_regenerate: bool,
) -> TestingStructure {
    let mut svo = TestingStructure::new(ctx);

    svo.rebuild(ctx, cmd, scratch_buffer);
    
    svo
}
