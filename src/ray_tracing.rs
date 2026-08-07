use std::ptr::slice_from_raw_parts;

use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use gpu_allocator::vulkan::Allocator;
use crate::{buffer, renderer::GraphicsContext};


#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct AccelerationStructureAabb {
    pub min: vek::Vec3<f32>,
    pub max: vek::Vec3<f32>,
}

pub struct AccelerationStructureData {
    pub backing_buffer: buffer::Buffer,
    pub scratch_buffer: buffer::Buffer,
    pub acceleration_structure: vk::AccelerationStructureKHR,
    pub acceleration_structure_address: u64,
}

impl AccelerationStructureData {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        acceleration_structure_device.destroy_acceleration_structure(self.acceleration_structure, None);
        self.scratch_buffer.destroy(device, allocator);
        self.backing_buffer.destroy(device, allocator);            
    } 
}

pub enum BlasGeometry {
    Triangles {
        vertex_count: usize,
        vertex_offset: usize,
        vertex_stride: usize,
        index_count: usize,
        index_offset: usize,
        index_stride: usize,
        vertex_buffer_address: u64,
        index_buffer_address: u64,
    },
    AABBs {
        aabb_buffer_address: u64,
        max_count: u32,
    }
}


pub unsafe fn rebuild_blas(
    ctx: &mut GraphicsContext,
    cmd: vk::CommandBuffer,
    geometry: BlasGeometry,
    blas: &AccelerationStructureData,
) {
    let (geometries, max_primitive_counts) = match geometry {
        BlasGeometry::Triangles { vertex_count, vertex_offset, vertex_stride, index_count, index_offset, index_stride, vertex_buffer_address, index_buffer_address } => {
            let vertex_data_device_address = vertex_buffer_address;
            let index_data_device_address = index_buffer_address;

            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .index_type(vk::IndexType::UINT32)
                .max_vertex(vertex_count as u32)
                .vertex_stride(vertex_stride as u64)
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_data_device_address + (vertex_stride * vertex_offset) as u64 })
                .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_data_device_address + (index_stride * index_offset) as u64 });
            let geometry_tmp = vk::AccelerationStructureGeometryDataKHR { triangles: triangles };

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(geometry_tmp)
                .flags(vk::GeometryFlagsKHR::OPAQUE);
            
            ([geometry], [index_count as u32 / 3])    
        },
        BlasGeometry::AABBs { aabb_buffer_address, max_count } => {
            let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                .stride(size_of::<AccelerationStructureAabb>() as u64)
                .data(vk::DeviceOrHostAddressConstKHR { device_address: aabb_buffer_address });
            let geometry_tmp = vk::AccelerationStructureGeometryDataKHR { aabbs: aabbs };

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::AABBS)
                .geometry(geometry_tmp)
                .flags(vk::GeometryFlagsKHR::OPAQUE);
            
            ([geometry], [max_count])   
        },
    };

    let mut acceleration_structure_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .geometries(&geometries);

    let tmp = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .first_vertex(0)
        .primitive_count(max_primitive_counts[0])
        .primitive_offset(0)
        .transform_offset(0);
    let tmp2 = &[tmp];
    let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[tmp2];

    acceleration_structure_build_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: blas.scratch_buffer.address };
    acceleration_structure_build_geometry_info.dst_acceleration_structure = blas.acceleration_structure;

    let queue_family_index = ctx.queue_family_index;

    let backing_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(blas.backing_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_READ)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR )
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let scratch_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(blas.scratch_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_READ)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let buffer_memory_barriers = [backing_buffer_barrier, scratch_buffer_barrier];
    let dep = vk::DependencyInfo::default()
        .buffer_memory_barriers(&buffer_memory_barriers);
    ctx.device.cmd_pipeline_barrier2(cmd, &dep);

    ctx.acceleration_structure_device.cmd_build_acceleration_structures(cmd, &[acceleration_structure_build_geometry_info], build_range_infos);

    let backing_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(blas.backing_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let scratch_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(blas.scratch_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let buffer_memory_barriers = [backing_buffer_barrier, scratch_buffer_barrier];
    let dep = vk::DependencyInfo::default()
        .buffer_memory_barriers(&buffer_memory_barriers);
    ctx.device.cmd_pipeline_barrier2(cmd, &dep);

}

pub unsafe fn create_blas(
    ctx: &mut GraphicsContext,
    cmd: vk::CommandBuffer,
    geometry: BlasGeometry,
) -> AccelerationStructureData {
    log::debug!("creating & building BLAS");

    let (geometries, max_primitive_counts) = match geometry {
        BlasGeometry::Triangles { vertex_count, vertex_offset, vertex_stride, index_count, index_offset, index_stride, vertex_buffer_address, index_buffer_address } => {
            let vertex_data_device_address = vertex_buffer_address;
            let index_data_device_address = index_buffer_address;

            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .index_type(vk::IndexType::UINT32)
                .max_vertex(vertex_count as u32)
                .vertex_stride(vertex_stride as u64)
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_data_device_address + (vertex_stride * vertex_offset) as u64 })
                .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_data_device_address + (index_stride * index_offset) as u64 });
            let geometry_tmp = vk::AccelerationStructureGeometryDataKHR { triangles: triangles };

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(geometry_tmp)
                .flags(vk::GeometryFlagsKHR::OPAQUE);
            
            ([geometry], [index_count as u32 / 3])    
        },
        BlasGeometry::AABBs { aabb_buffer_address, max_count } => {
            let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                .stride(size_of::<AccelerationStructureAabb>() as u64)
                .data(vk::DeviceOrHostAddressConstKHR { device_address: aabb_buffer_address });
            let geometry_tmp = vk::AccelerationStructureGeometryDataKHR { aabbs: aabbs };

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::AABBS)
                .geometry(geometry_tmp)
                .flags(vk::GeometryFlagsKHR::OPAQUE);
            
            ([geometry], [max_count])   
        },
    };

    let mut acceleration_structure_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .geometries(&geometries);

    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();

    ctx.acceleration_structure_device.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &acceleration_structure_build_geometry_info, &max_primitive_counts, &mut sizes);
    

    let backing_buffer = buffer::create_buffer(ctx, sizes.acceleration_structure_size as usize, "AS backing buffer", vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR);
    let scratch_buffer = buffer::create_buffer(ctx, sizes.build_scratch_size as usize, "AS scratch buffer", vk::BufferUsageFlags::empty());

    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(backing_buffer.buffer)
        .size(sizes.acceleration_structure_size)
        .offset(0)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    let acceleration_structure = ctx.acceleration_structure_device.create_acceleration_structure(&create_info, None).unwrap();

    let tmp = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .first_vertex(0)
        .primitive_count(max_primitive_counts[0])
        .primitive_offset(0)
        .transform_offset(0);
    let tmp2 = &[tmp];
    let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[tmp2];

    acceleration_structure_build_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_buffer.address };
    acceleration_structure_build_geometry_info.dst_acceleration_structure = acceleration_structure;

    ctx.acceleration_structure_device.cmd_build_acceleration_structures(cmd, &[acceleration_structure_build_geometry_info], build_range_infos);
    
    let acceleration_structure_address = ctx.acceleration_structure_device.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(acceleration_structure));

    AccelerationStructureData {
        backing_buffer,
        scratch_buffer,
        acceleration_structure,
        acceleration_structure_address,
    }
}

pub fn instantiate_blas(rotation: vek::Quaternion<f32>, position: vek::Vec3<f32>, scale: vek::Vec3<f32>, data: &AccelerationStructureData, instance_index_low_24: u32, mask: u8) -> vk::AccelerationStructureInstanceKHR {
    let matrix: vek::Mat4::<f32> = vek::Mat4::<f32>::translation_3d(position) * vek::Mat4::from(rotation) * vek::Mat4::<f32>::scaling_3d(scale);
    let row_arrays = &matrix.into_row_arrays()[0..3];
    let matrix: [f32; 12] = cast_slice::<[f32;4],f32>(row_arrays).try_into().unwrap();

    vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR { matrix },
        instance_custom_index_and_mask: vk::Packed24_8::new(instance_index_low_24, mask),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: data.acceleration_structure_address, },
    }
}

pub struct TopLevelAccelerationStructure {
    pub data: AccelerationStructureData,
    
}

impl TopLevelAccelerationStructure {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        self.data.destroy(acceleration_structure_device, device, allocator);
    }
}

pub const TLAS_MAX_INSTANCES: u32 = 50_000;

pub unsafe fn pre_create_tlas(
    ctx: &mut GraphicsContext,
) -> TopLevelAccelerationStructure {
    log::debug!("precreating TLAS");
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false);
    let geometry_tmp = vk::AccelerationStructureGeometryDataKHR { instances: instances };
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(geometry_tmp)
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    let geometries = [geometry];
    let acceleration_structure_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS)
        .geometries(&geometries);

    let max_primitive_counts = [TLAS_MAX_INSTANCES];

    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    ctx.acceleration_structure_device.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &acceleration_structure_build_geometry_info, &max_primitive_counts, &mut sizes);

    let backing_buffer = buffer::create_buffer(ctx, sizes.acceleration_structure_size as usize, "TLAS backing buffer", vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR);
    let scratch_buffer = buffer::create_buffer(ctx, sizes.build_scratch_size as usize, "TLAS scratch buffer", vk::BufferUsageFlags::empty());

    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(backing_buffer.buffer)
        .size(sizes.acceleration_structure_size)
        .offset(0)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
    
    let acceleration_structure = ctx.acceleration_structure_device.create_acceleration_structure(&create_info, None).unwrap();

    TopLevelAccelerationStructure {
        data: AccelerationStructureData { backing_buffer, scratch_buffer, acceleration_structure, acceleration_structure_address: 0 }, // address is only needed for blas instances
    }
}

pub unsafe fn rebuild_tlas(
    instances: impl Iterator<Item = vk::AccelerationStructureInstanceKHR>,
    tlas: &TopLevelAccelerationStructure,
    ctx: &mut GraphicsContext,
    cmd: vk::CommandBuffer,
    per_frame_scratch_buffer: &mut crate::buffer::ScratchBuffer,
) {
    let instances = instances.collect::<Vec<_>>();
    let blases: &[vk::AccelerationStructureInstanceKHR] = instances.as_slice();

    if blases.is_empty() {
        return;
    }

    // the ONLY reason we are doing an unsafe `slice_from_raw_parts` is because vk::AccelerationStructureInstanceKHR does not implement bytemuck Pod/Zeroable
    let bytes = blases.len() * size_of::<vk::AccelerationStructureInstanceKHR>();
    let ptr = blases.as_ptr() as *const u8;
    let data = &*slice_from_raw_parts(ptr, bytes);


    let written_address = per_frame_scratch_buffer.write_bytes_aligned(data).buffer_device_address_start;

    /* 
    
    Some(ScratchBufferBarrierInfo {
        src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
        dst_stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::SHADER_READ,
    })
    
     */

    let queue_family_index = ctx.queue_family_index;
    let acceleration_structure_device = ctx.acceleration_structure_device;
    let device = ctx.device;
    
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR { device_address: written_address });
    let geometry_tmp = vk::AccelerationStructureGeometryDataKHR { instances: instances };

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(geometry_tmp)
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let geometries = [geometry];

    let mut acceleration_structure_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(&geometries);

    let tmp = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .first_vertex(0)
        .primitive_count(blases.len() as u32)
        .primitive_offset(0)
        .transform_offset(0);
    let tmp2 = &[tmp];
    let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[tmp2];

    let backing_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(tlas.data.backing_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_READ)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR )
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let scratch_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(tlas.data.scratch_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_READ)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let buffer_memory_barriers = [backing_buffer_barrier, scratch_buffer_barrier];
    let dep = vk::DependencyInfo::default()
        .buffer_memory_barriers(&buffer_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    acceleration_structure_build_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: tlas.data.scratch_buffer.address };
    acceleration_structure_build_geometry_info.dst_acceleration_structure = tlas.data.acceleration_structure;
    acceleration_structure_device.cmd_build_acceleration_structures(cmd, &[acceleration_structure_build_geometry_info], build_range_infos);

    let backing_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(tlas.data.backing_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let scratch_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(tlas.data.scratch_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let buffer_memory_barriers = [backing_buffer_barrier, scratch_buffer_barrier];
    let dep = vk::DependencyInfo::default()
        .buffer_memory_barriers(&buffer_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);
}