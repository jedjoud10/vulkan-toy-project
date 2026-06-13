use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use ash::vk;
use smallvec::SmallVec;

use crate::{debug::DebugMarker, renderer::GraphicsContext};

pub const STORAGE_IMAGE_COUNT: u32 = 160;
pub const STORAGE_BUFFER_COUNT: u32 = 160;
pub const UNIFORM_BUFFER_COUNT: u32 = 60;
pub const SAMPLED_IMAGE_COUNT: u32 = 40;
pub const SAMPLER_COUNT: u32 = 2;
pub const ACCELERATION_STRUCTURES_COUNT: u32 = 1;
pub const MAX_DESCRIPTOR_SETS: u32 = crate::per_frame_data::FRAMES_IN_FLIGHT as u32;


pub unsafe fn create_descriptor_pool_and_bindless_descriptor_set(device: &ash::Device, debug_marker: &DebugMarker) -> (vk::DescriptorPool, vk::DescriptorSetLayout) {
    let images = vk::DescriptorPoolSize::default()
        .descriptor_count(STORAGE_IMAGE_COUNT*MAX_DESCRIPTOR_SETS)
        .ty(vk::DescriptorType::STORAGE_IMAGE);
    let storage_buffers = vk::DescriptorPoolSize::default()
        .descriptor_count(STORAGE_BUFFER_COUNT*MAX_DESCRIPTOR_SETS)
        .ty(vk::DescriptorType::STORAGE_BUFFER);
    let dynamic_buffers = vk::DescriptorPoolSize::default()
        .descriptor_count(UNIFORM_BUFFER_COUNT*MAX_DESCRIPTOR_SETS)
        .ty(vk::DescriptorType::UNIFORM_BUFFER);
    let sampled_images = vk::DescriptorPoolSize::default()
        .descriptor_count(SAMPLED_IMAGE_COUNT*MAX_DESCRIPTOR_SETS)
        .ty(vk::DescriptorType::SAMPLED_IMAGE);
    let samplers = vk::DescriptorPoolSize::default()
        .descriptor_count(SAMPLER_COUNT*MAX_DESCRIPTOR_SETS)
        .ty(vk::DescriptorType::SAMPLER);
    let acceleration_structures = vk::DescriptorPoolSize::default()
        .descriptor_count(ACCELERATION_STRUCTURES_COUNT*MAX_DESCRIPTOR_SETS)
        .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
    let descriptor_pool_sizes = [images, storage_buffers, dynamic_buffers, sampled_images, samplers, acceleration_structures];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .max_sets(MAX_DESCRIPTOR_SETS)
        .pool_sizes(&descriptor_pool_sizes);
    let descriptor_pool = device
        .create_descriptor_pool(&descriptor_pool_create_info, None)
        .unwrap();
    crate::debug::set_object_name(descriptor_pool, debug_marker, "descriptor pool");
    log::info!("created descriptor pool");


    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(STORAGE_IMAGE_COUNT)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(STORAGE_BUFFER_COUNT)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::ALL),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_count(SAMPLED_IMAGE_COUNT)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_count(SAMPLER_COUNT)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .stage_flags(vk::ShaderStageFlags::ALL),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_count(ACCELERATION_STRUCTURES_COUNT)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .stage_flags(vk::ShaderStageFlags::ALL),
    ];

    let binding_flags = std::iter::repeat(vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND).take(bindings.len()).collect::<SmallVec<[_; 10]>>();
    let mut descriptor_set_layout_binding_flags_create_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
        .binding_flags(binding_flags.as_slice());

    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .bindings(&bindings)
        .push_next(&mut descriptor_set_layout_binding_flags_create_info);
    let descriptor_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        .unwrap();
    crate::debug::set_object_name(descriptor_set_layout, debug_marker, "main descriptor set layout");
    log::info!("created bindless descriptor set layout");

    (descriptor_pool, descriptor_set_layout)
}


pub unsafe fn create_query_pool(
    device: &ash::Device
) -> vk::QueryPool {
    let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    let query = device.create_query_pool(&create_info, None).unwrap();
    device.reset_query_pool(query, 0, 2);
    query
}

pub unsafe fn create_pipeline_stats_pool(
    device: &ash::Device
) -> vk::QueryPool {
    let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::PIPELINE_STATISTICS)
        .pipeline_statistics(vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS)
        .query_count(1);
    let query = device.create_query_pool(&create_info, None).unwrap();
    device.reset_query_pool(query, 0, 1);

    query
}

pub unsafe fn find_appropriate_queue_family_index(
    physical_device: vk::PhysicalDevice,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> usize {
    queue_family_properties
        .iter()
        .enumerate()
        .position(|(i, props)| {
            let present = surface_loader
                .get_physical_device_surface_support(physical_device, i as u32, surface_khr)
                .unwrap();
            let graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let compute = props.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let has_timestamps = props.timestamp_valid_bits > 0;
            present && graphics && compute && has_timestamps
        })
        .unwrap()
}

pub unsafe fn create_surface(
    instance: &ash::Instance,
    entry: &ash::Entry,
    window: &winit::window::Window,
) -> (ash::khr::surface::Instance, vk::SurfaceKHR) {
    let surface = ash_window::create_surface(
        entry,
        instance,
        window.display_handle().unwrap().as_raw(),
        window.window_handle().unwrap().as_raw(),
        None,
    )
    .unwrap();
    let surface_loader = ash::khr::surface::Instance::new(entry, instance);
    (surface_loader, surface)
}

pub unsafe fn begin_recording(ctx: &mut GraphicsContext) -> vk::CommandBuffer {
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(ctx.pool);
    let cmd = ctx.device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    ctx.device.begin_command_buffer(cmd, &Default::default()).unwrap();

    cmd
}

pub unsafe fn end_recording_and_submit(ctx: &mut GraphicsContext, cmd: vk::CommandBuffer) {
    ctx.device.end_command_buffer(cmd).unwrap();

    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);

    // TODO: batch submits perhaps?
    ctx.device.queue_submit(ctx.queue, & [submit], vk::Fence::null()).unwrap();
    ctx.device.device_wait_idle().unwrap();
}

#[cfg(debug_assertions)]
mod dynamically_loaded {
    use std::{borrow::Cow, io::Read, path::PathBuf};

    fn load_from_folder(folder: &str, value: &str) -> Option<Cow<'static, [u8]>> {
        log::debug!("load '{}' dynamically", value);
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let mut path = PathBuf::from(manifest_dir);
        path.push(folder);
        path.push(value);
        
        let mut file = std::fs::File::open(path).ok()?;
        
        let mut vec = Vec::<u8>::new();
        file.read_to_end(&mut vec).ok()?;
        Some(Cow::Owned(vec))
    }


    pub fn load_compiled_shader(value: &str) -> Option<Cow<'static, [u8]>> {
        load_from_folder("compiled_shaders", value)
    }

    pub fn load_material_texture(value: &str) -> Option<Cow<'static, [u8]>> {
        load_from_folder("materials", value)
    }

    pub fn load_model(value: &str) -> Option<Cow<'static, [u8]>> {
        load_from_folder("models", value)
    }
}

#[cfg(not(debug_assertions))]
mod statically_loaded {
    use std::{borrow::Cow};

    static COMPILED_SHADERS: include_dir::Dir = include_dir::include_dir!("$CARGO_MANIFEST_DIR/compiled_shaders");
    static MATERIALS: include_dir::Dir = include_dir::include_dir!("$CARGO_MANIFEST_DIR/materials");
    static MODELS: include_dir::Dir = include_dir::include_dir!("$CARGO_MANIFEST_DIR/models");

    pub fn load_compiled_shader(value: &str) -> Option<Cow<'static, [u8]>> {
        log::debug!("load '{}' statically", value);
        COMPILED_SHADERS.get_file(value).map(|x| Cow::Borrowed(x.contents()))
    }

    pub fn load_material_texture(value: &str) -> Option<Cow<'static, [u8]>> {
        log::debug!("load '{}' statically", value);
        MATERIALS.get_file(value).map(|x| Cow::Borrowed(x.contents()))
    }

    pub fn load_model(value: &str) -> Option<Cow<'static, [u8]>> {
        log::debug!("load '{}' statically", value);
        MODELS.get_file(value).map(|x| Cow::Borrowed(x.contents()))
    }
}

#[cfg(debug_assertions)]
pub use dynamically_loaded::*;


#[cfg(not(debug_assertions))]
pub use statically_loaded::*;