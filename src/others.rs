use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use ash::vk;
use smallvec::SmallVec;

pub const STORAGE_IMAGE_COUNT: u32 = 160;
pub const STORAGE_BUFFER_COUNT: u32 = 60;
pub const UNIFORM_BUFFER_COUNT: u32 = 60;
pub const COMBINED_IMAGE_SAMPLER_COUNT: u32 = 40;


pub unsafe fn create_descriptor_pool_and_bindless_descriptor_set(device: &ash::Device, binder: &Option<ash::ext::debug_utils::Device>) -> (vk::DescriptorPool, vk::DescriptorSetLayout, vk::DescriptorSet) {
    let images = vk::DescriptorPoolSize::default()
        .descriptor_count(STORAGE_IMAGE_COUNT)
        .ty(vk::DescriptorType::STORAGE_IMAGE);
    let storage_buffers = vk::DescriptorPoolSize::default()
        .descriptor_count(STORAGE_BUFFER_COUNT)
        .ty(vk::DescriptorType::STORAGE_BUFFER);
    let dynamic_buffers = vk::DescriptorPoolSize::default()
        .descriptor_count(UNIFORM_BUFFER_COUNT)
        .ty(vk::DescriptorType::UNIFORM_BUFFER);
    let combined_image_samplers = vk::DescriptorPoolSize::default()
        .descriptor_count(COMBINED_IMAGE_SAMPLER_COUNT)
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
    let descriptor_pool_sizes = [images, storage_buffers, dynamic_buffers, combined_image_samplers];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET | vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .max_sets(1)
        .pool_sizes(&descriptor_pool_sizes);
    let descriptor_pool = device
        .create_descriptor_pool(&descriptor_pool_create_info, None)
        .unwrap();
    crate::debug::set_object_name(descriptor_pool, binder, "descriptor pool");
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
            .descriptor_count(COMBINED_IMAGE_SAMPLER_COUNT)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
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
    crate::debug::set_object_name(descriptor_set_layout, binder, "main descriptor set layout");
    log::info!("created bindless descriptor set layout");


    let layouts = [descriptor_set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    let descriptor_set = device.allocate_descriptor_sets(&allocate_info).unwrap()[0];
    crate::debug::set_object_name(descriptor_set, binder, "main descriptor set");
    log::info!("created bindless descriptor set");

    (descriptor_pool, descriptor_set_layout, descriptor_set)
}


pub unsafe fn create_query_pool(
    device: &ash::Device
) -> vk::QueryPool {
    let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    let query = device.create_query_pool(&create_info, None).unwrap();

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

pub unsafe fn find_async_compute_queue(
    _physical_device: vk::PhysicalDevice,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
) -> usize {
    queue_family_properties
        .iter()
        .enumerate()
        .position(|(_i, props)| {
            let graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let compute = props.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let transfer = props.queue_flags.contains(vk::QueueFlags::TRANSFER);

            !graphics & compute & !transfer
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