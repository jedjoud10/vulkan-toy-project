use std::ptr::slice_from_raw_parts;

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use crate::{buffer, renderer::GraphicsContext};

pub struct AccelerationStructureData {
    pub backing_buffer: buffer::Buffer,
    pub scratch_buffer: buffer::Buffer,
    pub acceleration_structure: vk::AccelerationStructureKHR,
}

impl AccelerationStructureData {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        acceleration_structure_device.destroy_acceleration_structure(self.acceleration_structure, None);
        self.scratch_buffer.destroy(device, allocator);
        self.backing_buffer.destroy(device, allocator);            
    } 
}



pub unsafe fn create_blas(
    ctx: &mut GraphicsContext,
    cmd: vk::CommandBuffer,
    vertex_count: usize,
    vertex_offset: usize,
    vertex_stride: usize,
    index_count: usize,
    index_offset: usize,
    index_stride: usize,
    vertex_buffer: &buffer::Buffer,
    index_buffer: &buffer::Buffer,
) -> (AccelerationStructureData, vk::AccelerationStructureInstanceKHR) {
    let vertex_data_device_address = vertex_buffer.address;
    let index_data_device_address = index_buffer.address;
    
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

    let geometries = [geometry];

    let mut acceleration_structure_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .geometries(&geometries);

    let max_primitive_counts = [index_count as u32 / 3];

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
        .primitive_count(index_count as u32 / 3)
        .primitive_offset(0)
        .transform_offset(0);
    let tmp2 = &[tmp];
    let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[tmp2];

    acceleration_structure_build_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_buffer.address };
    acceleration_structure_build_geometry_info.dst_acceleration_structure = acceleration_structure;

    
    ctx.acceleration_structure_device.cmd_build_acceleration_structures(cmd, &[acceleration_structure_build_geometry_info], build_range_infos);
    
    let acceleration_structure_address = ctx.acceleration_structure_device.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(acceleration_structure));
    
    let identity_matrix = [
        1f32, 0f32, 0f32, 0f32,
        0f32, 1f32, 0f32, 0f32,
        0f32, 0f32, 1f32, 0f32,
    ];

    (AccelerationStructureData {
        backing_buffer,
        scratch_buffer,
        acceleration_structure,
    }, vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR { matrix: identity_matrix },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: acceleration_structure_address, },
    })
}

pub struct TopLevelAccelerationStructure {
    pub data: Option<AccelerationStructureData>,
    
}

impl TopLevelAccelerationStructure {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(data) = self.data {
            data.destroy(acceleration_structure_device, device, allocator);
        }
    }
}

pub const TLAS_MAX_INSTANCES: u32 = 1000;

pub unsafe fn pre_create_tlas(
    ctx: &mut GraphicsContext,
) -> TopLevelAccelerationStructure {

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
        data: Some(AccelerationStructureData { backing_buffer, scratch_buffer, acceleration_structure }),
    }
}

pub unsafe fn rebuild_tlas(
    static_instances: &[vk::AccelerationStructureInstanceKHR],
    dynamic_instances: &[vk::AccelerationStructureInstanceKHR],
    tlas: &TopLevelAccelerationStructure,
    cmd: vk::CommandBuffer,
    acceleration_structure_device: &ash::khr::acceleration_structure::Device,
    device: &ash::Device,
    mut allocator: &mut Allocator,
    debug_marker: &Option<ash::ext::debug_utils::Device>,
    queue_family_index: u32,
    scratch_buffer: &mut crate::per_frame_data::ScratchBuffer,
) {
    let prev = tlas.data.as_ref().unwrap();

    let instances = static_instances.iter().chain(dynamic_instances.iter()).copied().collect::<Vec<_>>();
    let blases = instances.as_slice();
    let bytes = blases.len() * size_of::<vk::AccelerationStructureInstanceKHR>();
    let ptr = blases.as_ptr() as *const u8;
    let data = &*slice_from_raw_parts(ptr, bytes);
    let written_address = scratch_buffer.write_bytes(device, cmd, data, queue_family_index);
    
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

    acceleration_structure_build_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: prev.scratch_buffer.address };
    acceleration_structure_build_geometry_info.dst_acceleration_structure = prev.acceleration_structure;
    acceleration_structure_device.cmd_build_acceleration_structures(cmd, &[acceleration_structure_build_geometry_info], build_range_infos);

    let backing_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(prev.backing_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let scratch_buffer_barrier = vk::BufferMemoryBarrier2::default()
        .buffer(prev.scratch_buffer.buffer)
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .size(vk::WHOLE_SIZE)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index);
    let buffer_memory_barriers = [backing_buffer_barrier, scratch_buffer_barrier];
    let dep = vk::DependencyInfo::default()
        .buffer_memory_barriers(&buffer_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);
}