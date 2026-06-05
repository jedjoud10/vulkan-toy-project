
use ash::vk;

use crate::others;

pub unsafe fn create_device_and_queue(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> (ash::Device, u32, vk::Queue) {
    let queue_family_properties =
        instance.get_physical_device_queue_family_properties(physical_device);
    let queue_family_index = others::find_appropriate_queue_family_index(
        physical_device,
        queue_family_properties,
        surface_loader,
        surface_khr,
    ) as u32;

    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_priorities(&[1.0])
        .queue_family_index(queue_family_index);
    let queue_create_infos = [queue_create_info];

    let mut compute_derivatives = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
        .compute_derivative_group_quads(true)
        .compute_derivative_group_linear(true);
    let mut shader_clock = vk::PhysicalDeviceShaderClockFeaturesKHR::default()
        .shader_device_clock(true)
        .shader_subgroup_clock(true); 
    let mut atomics = vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default()
        .shader_image_int64_atomics(true); 
    let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
        .mesh_shader(true)
        .task_shader(true);
    let mut extended_state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
        .extended_dynamic_state3_polygon_mode(true);
    let mut acceleration_structure = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
        .acceleration_structure(true)
        .acceleration_structure_indirect_build(true);
    let device_features_base = vk::PhysicalDeviceFeatures::default()
        .multi_draw_indirect(true)
        .shader_int16(true)
        .shader_int64(true)
        .sparse_binding(true)
        .sparse_residency_image3_d(true)
        .fill_mode_non_solid(true)
        .pipeline_statistics_query(true);
    let mut device_features_11 = vk::PhysicalDeviceVulkan11Features::default()
        .uniform_and_storage_buffer16_bit_access(true);
    let mut device_features_12 = vk::PhysicalDeviceVulkan12Features::default()
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true)
        .shader_float16(true)
        .shader_int8(true)
        .host_query_reset(true)
        .timeline_semaphore(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .shader_storage_image_array_non_uniform_indexing(true)
        .shader_storage_buffer_array_non_uniform_indexing(true)
        .shader_storage_texel_buffer_array_non_uniform_indexing(true)
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_storage_image_update_after_bind(true)
        .descriptor_binding_uniform_buffer_update_after_bind(true)
        .descriptor_binding_variable_descriptor_count(true)
        .runtime_descriptor_array(true);
    let mut device_features_13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(true)
        .maintenance4(true)
        .dynamic_rendering(true);

    let device_extension_names = [
        ash::khr::swapchain::NAME,
        ash::khr::shader_float16_int8::NAME,
        ash::khr::shader_atomic_int64::NAME,
        ash::khr::shader_clock::NAME,
        ash::ext::shader_image_atomic_int64::NAME,
        ash::khr::shader_draw_parameters::NAME,
        ash::khr::dynamic_rendering::NAME,
        ash::ext::host_image_copy::NAME,
        ash::ext::host_query_reset::NAME,
        ash::khr::timeline_semaphore::NAME,
        ash::khr::buffer_device_address::NAME,
        ash::ext::mesh_shader::NAME,
        ash::khr::deferred_host_operations::NAME,
        ash::ext::extended_dynamic_state3::NAME,
        ash::khr::acceleration_structure::NAME,
        ash::khr::ray_query::NAME,
        
        
        
        // TODO: remove when ash vk1.4
        c"VK_KHR_compute_shader_derivatives",
    ];

    let device_extension_names_ptrs = device_extension_names
        .iter()
        .map(|cstr| cstr.as_ptr())
        .collect::<Vec<_>>();

    let device_create_info = vk::DeviceCreateInfo::default()
        .enabled_extension_names(&device_extension_names_ptrs)
        .enabled_features(&device_features_base)
        .queue_create_infos(&queue_create_infos)
        .push_next(&mut device_features_13)
        .push_next(&mut device_features_12)
        .push_next(&mut device_features_11)
        .push_next(&mut atomics)
        .push_next(&mut shader_clock)
        .push_next(&mut compute_derivatives)
        .push_next(&mut mesh_shader)
        .push_next(&mut extended_state3)
        .push_next(&mut acceleration_structure);

    let device = instance
        .create_device(physical_device, &device_create_info, None)
        .unwrap();
    log::info!("created device");

    let queue = device.get_device_queue(queue_family_index, 0);
    log::info!("fetched queue");

    (device, queue_family_index, queue)
}
