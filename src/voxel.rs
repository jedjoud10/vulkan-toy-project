use std::ptr::slice_from_raw_parts;

use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::buffer;

pub const VERTICES_PER_CHUNK: usize = 1 << 18;
pub const TRIANGLES_PER_CHUNK: usize = 1 << 18;
pub const VERTEX_STRIDE: usize = size_of::<vek::Vec3::<f32>>();
pub const INDEX_STRIDE: usize = size_of::<u32>();

const PADDING: u32 = 2;
const SIZE: u32 = 64;
const IMAGE_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
        

pub struct MultipleChunks {
    pub voxel_texture: VoxelTexture3D,
    pub vertex_buffer: buffer::Buffer,
    pub index_buffer: buffer::Buffer,
    pub vertex_counter: buffer::Buffer,
    pub index_counter: buffer::Buffer,
    pub indirect_draw_buffer: buffer::Buffer,
    pub tlas: Option<AccelerationStructureData>,
    pub blases: Vec<vk::AccelerationStructureInstanceKHR>,
    pub total_num_chunks: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

impl MultipleChunks {
    pub unsafe fn create(
        device: &ash::Device,
        allocator: &mut Allocator,
        debug_marker: &Option<ash::ext::debug_utils::Device>,
        queue: vk::Queue,
        pool: vk::CommandPool,
        queue_family_index: u32,
        total_num_chunks: usize,
    ) -> Self {
        
        let vertex_buffer = buffer::create_buffer(&device, allocator, VERTEX_STRIDE * VERTICES_PER_CHUNK * total_num_chunks, &debug_marker, "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let index_buffer = buffer::create_buffer(&device, allocator, INDEX_STRIDE * 3 * TRIANGLES_PER_CHUNK * total_num_chunks, &debug_marker, "index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR);
        let indirect_draw_buffer = buffer::create_buffer(&device, allocator, size_of::<DrawIndexedIndirectCommand>() * total_num_chunks, &debug_marker, "indirect buffer", vk::BufferUsageFlags::INDIRECT_BUFFER);

        
        let arr = (0..total_num_chunks).into_iter().map(|i| DrawIndexedIndirectCommand {
            index_count: 0,
            instance_count: 1,
            first_index: (3 * TRIANGLES_PER_CHUNK * i) as u32,
            vertex_offset: (VERTICES_PER_CHUNK * i) as i32,
            first_instance: 0,
        }).collect::<Vec<_>>();
        buffer::write_to_buffer_with_offset(device, pool, queue, indirect_draw_buffer.buffer, allocator, cast_slice(&arr), 0);

        let vertex_counter = buffer::create_counter_buffer(&device, allocator, &debug_marker, "vertex counter");
        let index_counter = buffer::create_counter_buffer(&device, allocator, &debug_marker, "index counter");
        let voxel_texture = create_voxel_texture(device, allocator, &debug_marker, queue, pool, queue_family_index);

        Self {
            voxel_texture,
            vertex_buffer,
            index_buffer,
            vertex_counter,
            index_counter,
            indirect_draw_buffer,
            tlas: None,
            blases: Vec::new(),
            total_num_chunks
        }
    }

    
    pub unsafe fn do_sum_shi(
        &mut self,
        chunk_index: usize,
        chunk_offset: vek::Vec3<i32>,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        density_pipeline: vk::Pipeline,
        surface_generation_pipeline: vk::Pipeline,
        queue_family_index: u32
    ) {
        let groups = 16;

        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            density_pipeline,
        );

        let constants = bytemuck::bytes_of(&chunk_offset); 
        device.cmd_push_constants(cmd, pipeline_layout, vk::ShaderStageFlags::ALL, 0, constants);

        device.cmd_dispatch(cmd, groups+1, groups+1, groups+1);

        let zero = 0u32;
        device.cmd_update_buffer(cmd, self.vertex_counter.buffer, 0, bytemuck::bytes_of(&zero));
        device.cmd_update_buffer(cmd, self.index_counter.buffer, 0, bytemuck::bytes_of(&zero));

        let vertex_counter_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_counter.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let index_counter_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_counter.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let voxelize_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(self.voxel_texture.image)
            .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_array_layer(0).base_mip_level(0).layer_count(1).level_count(1));
        let index_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let buffer_memory_barriers = [vertex_counter_barrier, index_counter_barrier, index_buffer_barrier, vertex_buffer_barrier];
        let image_memory_barriers = [voxelize_image_barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers)
            .image_memory_barriers(&image_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);        
        
        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            surface_generation_pipeline,
        );

        device.cmd_dispatch(cmd, groups, groups, groups);

        let index_counter_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_counter.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.vertex_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let index_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.index_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let buffer_memory_barriers = [index_counter_barrier, vertex_buffer_barrier, index_buffer_barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);

        let regions = [vk::BufferCopy::default().size(size_of::<u32>() as u64).dst_offset((size_of::<DrawIndexedIndirectCommand>() * chunk_index) as u64).src_offset(0)];
        device.cmd_copy_buffer(cmd, self.index_counter.buffer, self.indirect_draw_buffer.buffer, &regions);

        let indirect_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.indirect_draw_buffer.buffer)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .size(vk::WHOLE_SIZE)
            .offset(0)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index);
        let buffer_memory_barriers = [indirect_buffer_barrier];
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);
    }

    pub unsafe fn create_blas(
        &mut self,
        chunk_index: usize,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        acceleration_structure_device: &ash::khr::acceleration_structure::Device,
        mut allocator: &mut Allocator,
        debug_marker: &Option<ash::ext::debug_utils::Device>
    ) -> AccelerationStructureData {
        let vertex_data_device_address = self.vertex_buffer.address;
        let index_data_device_address = self.index_buffer.address;
        
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .index_type(vk::IndexType::UINT32)
            .max_vertex(VERTICES_PER_CHUNK as u32)
            .vertex_stride(VERTEX_STRIDE as u64)
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_data_device_address + (VERTEX_STRIDE * VERTICES_PER_CHUNK * chunk_index) as u64 })
            .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_data_device_address + (3 * INDEX_STRIDE * TRIANGLES_PER_CHUNK * chunk_index) as u64 });
        let geometry_tmp = vk::AccelerationStructureGeometryDataKHR { triangles: triangles };
    
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(geometry_tmp)
            .flags(vk::GeometryFlagsKHR::OPAQUE);
    
        let geometries = [geometry];
    
        let mut acceleration_structure_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .geometries(&geometries);
    
        let max_primitive_counts = [TRIANGLES_PER_CHUNK as u32];
    
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    
        acceleration_structure_device.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &acceleration_structure_build_geometry_info, &max_primitive_counts, &mut sizes);
    
        
    
        let backing_buffer = buffer::create_buffer(&device, &mut allocator, sizes.acceleration_structure_size as usize, &debug_marker, "AS backing buffer", vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR);
        let scratch_buffer = buffer::create_buffer(&device, &mut allocator, sizes.build_scratch_size as usize, &debug_marker, "AS scratch buffer", vk::BufferUsageFlags::empty());
    
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(backing_buffer.buffer)
            .size(sizes.acceleration_structure_size)
            .offset(0)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        let acceleration_structure = acceleration_structure_device.create_acceleration_structure(&create_info, None).unwrap();
    
        let tmp = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(TRIANGLES_PER_CHUNK as u32)
            .primitive_offset(0)
            .transform_offset(0);
        let tmp2 = &[tmp];
        let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[tmp2];
    
        acceleration_structure_build_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_buffer.address };
        acceleration_structure_build_geometry_info.dst_acceleration_structure = acceleration_structure;
    
        
        acceleration_structure_device.cmd_build_acceleration_structures(cmd, &[acceleration_structure_build_geometry_info], build_range_infos);
        
        let acceleration_structure_address = acceleration_structure_device.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(acceleration_structure));
        
        let identity_matrix = [
            1f32, 0f32, 0f32, 0f32,
            0f32, 1f32, 0f32, 0f32,
            0f32, 0f32, 1f32, 0f32,
        ];
    
        self.blases.push(vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR { matrix: identity_matrix },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: acceleration_structure_address, },
        });

        AccelerationStructureData {
            backing_buffer,
            scratch_buffer,
            acceleration_structure,
        }
    }
    
    pub unsafe fn create_tlas(
        &mut self,
        cmd: vk::CommandBuffer,
        acceleration_structure_device: &ash::khr::acceleration_structure::Device,
        device: &ash::Device,
        mut allocator: &mut Allocator,
        debug_marker: &Option<ash::ext::debug_utils::Device>,
        queue_family_index: u32,
        scratch_buffer: &mut crate::per_frame_data::ScratchBuffer,
    ) {
        if self.tlas.is_some() {
            return;
        }

        /*
        if let Some(accel_struct) = self.tlas.take() {
            accel_struct.destroy(acceleration_structure_device, device, allocator);          
        }
        */

        let bytes = self.blases.len() * size_of::<vk::AccelerationStructureInstanceKHR>();
        let ptr = self.blases.as_ptr() as *const u8;
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
    
        let max_primitive_counts = [self.blases.len() as u32];
    
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    
        acceleration_structure_device.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &acceleration_structure_build_geometry_info, &max_primitive_counts, &mut sizes);
    
        let backing_buffer = buffer::create_buffer(&device, &mut allocator, sizes.acceleration_structure_size as usize, &debug_marker, "TLAS backing buffer", vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR);
        let scratch_buffer = buffer::create_buffer(&device, &mut allocator, sizes.build_scratch_size as usize, &debug_marker, "TLAS scratch buffer", vk::BufferUsageFlags::empty());
    
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(backing_buffer.buffer)
            .size(sizes.acceleration_structure_size)
            .offset(0)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        let acceleration_structure = acceleration_structure_device.create_acceleration_structure(&create_info, None).unwrap();
    
        let tmp = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(self.total_num_chunks as u32)
            .primitive_offset(0)
            .transform_offset(0);
        let tmp2 = &[tmp];
        let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[tmp2];
    
        acceleration_structure_build_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_buffer.address };
        acceleration_structure_build_geometry_info.dst_acceleration_structure = acceleration_structure;
    
            
        acceleration_structure_device.cmd_build_acceleration_structures(cmd, &[acceleration_structure_build_geometry_info], build_range_infos);
    
        self.tlas = Some(AccelerationStructureData {
            backing_buffer,
            scratch_buffer,
            acceleration_structure,
        });
    }
    
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(accel_struct) = self.tlas {
            accel_struct.destroy(acceleration_structure_device, device, allocator);          
        }

        self.vertex_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
        self.vertex_counter.destroy(device, allocator);
        self.index_counter.destroy(device, allocator);
        self.voxel_texture.destroy(device, allocator);
        self.indirect_draw_buffer.destroy(device, allocator);
    } 
}

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

pub struct Chunk {
    pub chunk_index: usize,
    pub chunk_offset: vek::Vec3<i32>,
    pub built: bool,
    pub vertex_buffer_start_offset: usize,
    pub index_buffer_start_offset: usize,
    pub accel_structure: Option<AccelerationStructureData>,
}

impl Chunk {
    pub unsafe fn destroy(self, acceleration_structure_device: &ash::khr::acceleration_structure::Device, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(accel_struct) = self.accel_structure {
            accel_struct.destroy(acceleration_structure_device, device, allocator);          
        }
    } 
}

pub struct VoxelTexture3D {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
}

impl VoxelTexture3D {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}

pub unsafe fn create_voxel_texture(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    queue_family_index: u32,
) -> VoxelTexture3D {
    let queue_family_indices = [queue_family_index];
    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: SIZE+PADDING,
            height: SIZE+PADDING,
            depth: SIZE+PADDING,
        })
        .format(IMAGE_FORMAT)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let image = device.create_image(&image_create_info, None).unwrap();
    crate::debug::set_object_name(image, binder, "Voxel Texture");

    
    let image_requirements = device.get_image_memory_requirements(image);
    let image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Image Allocation",
            requirements: image_requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(image, image_allocation.memory(), image_allocation.offset()).unwrap();

    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .level_count(1)
        .base_array_layer(0)
        .base_mip_level(0);

    let image_layout_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::empty())
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(image)
        .subresource_range(image_subresource_range);
    let image_memory_barriers = [image_layout_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    // end command buffer and submit
    device.end_command_buffer(cmd).unwrap();
    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);
    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .format(IMAGE_FORMAT)
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .subresource_range(image_subresource_range);
    let image_view = device.create_image_view(&image_view_create_info, None).unwrap();


    VoxelTexture3D {
        image,
        image_view,
        allocation: image_allocation,
    }
}