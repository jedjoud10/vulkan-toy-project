
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::renderer::GraphicsContext;

// acceleration structure scratch buffer requires alignment to be at least 256
// of course we can pass a flag to state that we are allocating an acceleration structure scratch buffer
// but it is easier to simply set this as the min alignment for all buffers
const MIN_BUFFER_ALIGNMENT: u64 = 256;

// `write_to_buffer` calls that update more than these amount of bytes will revert to using the staging buffer implementation
const BUFFER_WRITE_INLINE_MAX_BYTES_THRESHOLD: usize = 65536; // vulkan spec states that data size must be less than this 

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub address: u64,
    pub size: usize,
}

impl Buffer {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_buffer(self.buffer, None);
        allocator.free(self.allocation).unwrap();
    }
}


pub unsafe fn create_buffer(
    ctx: &mut GraphicsContext,
    size: usize,
    name: &str,
    flags: vk::BufferUsageFlags,
) -> Buffer {
    let bytes_formatted = bytesize::ByteSize::b(size as u64);
    log::debug!("creating buffer '{}' ({})", name, bytes_formatted.display().si());
    let buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::DEVICE_ADDRESS_CAPTURE_REPLAY)
        .usage(flags | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(size as u64);

    // TODO: use a user-side sub-allocator for buffers to avoid doing the expensive API call
    let buffer = ctx.device.create_buffer(&buffer_create_info, None).unwrap();

    let mut requirements = ctx.device.get_buffer_memory_requirements(buffer);
    requirements.alignment = requirements.alignment.max(MIN_BUFFER_ALIGNMENT); 

    let allocation = ctx.allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &format!("{name} allocation"),
            requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
        })
        .unwrap();

    crate::debug::set_object_name(buffer, ctx.debug_marker, name);
    
    let device_memory = allocation.memory();
    ctx.device.bind_buffer_memory(buffer, device_memory, allocation.offset()).unwrap();
    
    log::debug!("created buffer '{}' of size {}", name, bytes_formatted.display().si());

    
    let info = vk::BufferDeviceAddressInfo::default()
        .buffer(buffer);
    let address = ctx.device.get_buffer_device_address(&info);

    Buffer {
        buffer, allocation, address, size
    }
}


pub unsafe fn create_buffer_with(
    ctx: &mut GraphicsContext,
    bytes: &[u8],
    name: &str,
    flags: vk::BufferUsageFlags,
) -> Buffer {
    let buffer = create_buffer(ctx, bytes.len(), name, flags);
    write_to_buffer(ctx, buffer.buffer, bytes);
    buffer
}

// this either creates a staging buffer write or writes to the buffer through cmd_update_buffer
// switches between both impls depending on the amount of data to write
pub unsafe fn write_to_buffer(
    ctx: &mut GraphicsContext,
    dst_buffer: vk::Buffer,
    bytes: &[u8]
) {
    write_to_buffer_with_offset(ctx, dst_buffer, bytes, 0);
}

pub unsafe fn write_to_buffer_with_offset(
    ctx: &mut GraphicsContext,
    dst_buffer: vk::Buffer,
    bytes: &[u8],
    dst_offset: u64,
) {
    if bytes.is_empty() {
        return;
    }

    let device = ctx.device;
    let pool = ctx.pool;
    let queue = ctx.queue;
    let allocator = &mut ctx.allocator;

    let start = std::time::Instant::now();
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let bytes_formatted = bytesize::ByteSize::b(bytes.len() as u64);
    let staging_buffer_opt = if bytes.len() < BUFFER_WRITE_INLINE_MAX_BYTES_THRESHOLD {
        // inline (command buffer write) impl
        log::info!("writing {} to buffer, using inline path", bytes_formatted.display().si());
        device.cmd_update_buffer(cmd, dst_buffer, dst_offset, bytes);
        None
    } else {
        log::info!("writing {} to buffer, using staging buffer path", bytes_formatted.display().si());
        let (staging_buffer, allocation) = create_staging_buffer(device, *allocator, bytes);

        let region = vk::BufferCopy2::default()
            .dst_offset(dst_offset)
            .size(bytes.len() as u64)
            .src_offset(0);

        let regions = [region];
        let copy_staging_buffer_to_buffer = vk::CopyBufferInfo2::default()
            .dst_buffer(dst_buffer)
            .regions(&regions)
            .src_buffer(staging_buffer);

        device.cmd_copy_buffer2(cmd, &copy_staging_buffer_to_buffer);
        Some((staging_buffer, allocation))
    };

    
    device.end_command_buffer(cmd).unwrap();

    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);

    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();

    // TODO: definitely don't do this if we want to optimize, but for now it's ok
    device.device_wait_idle().unwrap();

    // destroy staging buffer if we used it
    if let Some((staging_buffer, allocation)) = staging_buffer_opt {
        allocator.free(allocation).unwrap();
        device.destroy_buffer(staging_buffer, None);
    }

    let end = std::time::Instant::now();
    log::debug!("buffer write took {}μs", (end-start).as_micros());
}

pub unsafe fn create_staging_buffer(device: &ash::Device, allocator: &mut Allocator, bytes: &[u8]) -> (vk::Buffer, Allocation) {
    log::trace!("created staging buffer for {}", bytesize::ByteSize::b(bytes.len() as u64).display().si());
    let staging_buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(bytes.len() as u64);

    let staging_buffer = device.create_buffer(&staging_buffer_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(staging_buffer);
    let mut allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Staging Buffer",
            requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
        })
        .unwrap();

    device.bind_buffer_memory(staging_buffer, allocation.memory(), allocation.offset()).unwrap();
        
    let dst_slice = allocation.mapped_slice_mut().unwrap();

    // FIXME: for some reason on nvidia the slice has different size? shouldn't gpu_allocator handle this type of stuff...
    dst_slice[..(bytes.len())].copy_from_slice(bytes);
    (staging_buffer, allocation)
}

pub unsafe fn create_counter_buffer(
    ctx: &mut GraphicsContext,
    name: &str,
) -> Buffer {
    create_buffer(ctx, size_of::<u32>(), name, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
}


/*
pub unsafe fn create_staging_buffer2<'a>(device: &'a ash::Device, allocator: &'a mut Allocator, bytes: &[u8]) -> TemporaryStagingBuffer<'a> {
    let (buffer, allocation) = create_staging_buffer(device, allocator, bytes);
    TemporaryStagingBuffer { buffer, allocation: Some(allocation), device, allocator }
}

pub struct TemporaryStagingBuffer<'a> {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub device: &'a ash::Device,
    pub allocator: &'a mut Allocator,
}

impl<'a> Drop for TemporaryStagingBuffer<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_buffer(self.buffer, None);
            self.allocator.free(self.allocation.take().unwrap()).unwrap();
        }
    }
}
*/

/*
// TODO: finish impl when needed. not needed rn since there's no bottleneck lol
pub struct BufferWriter<'a> {
    pub buffer: vk::Buffer,
    pub temp: Vec<u8>,
    pub writes: Vec<(usize, usize)>,
    pub device: &'a ash::Device,
    pub allocator: &'a mut Allocator,
}

impl<'a> BufferWriter<'a> {
    pub fn new(device: &'a ash::Device, allocator: &'a mut Allocator, buffer: vk::Buffer) -> Self {
        BufferWriter { buffer, temp: Vec::new(), writes: Vec::new(), device, allocator }
    }

    pub fn write(&mut self, data: &[u8], dst_offset: u64) {
        let offset = self.temp.len();
        let length = data.len();
        self.writes.push((offset, length));
        self.temp.extend_from_slice(data);
    }
}
*/