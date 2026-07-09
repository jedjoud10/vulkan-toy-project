
use std::{ffi::c_void, marker::PhantomData};

use ash::vk::{self, Bool32, ExtendsDeviceCreateInfo, ExtendsPhysicalDeviceFeatures2, StructureType, TaggedStructure};

use crate::others;
use crate::physical_device_vulkan_14_features::*;
use vk::*;

macro_rules! impl_extended_feature {
    ($i:ident, $($required:ident)*, $($niceties:ident)*) => {
        impl ExtendedFeature for $i<'_> {
            fn clone(&self) -> Box<dyn ExtendedFeature> {
                Box::new(
                    $i::<'static> {
                        s_type: Self::STRUCTURE_TYPE,
                        p_next: ::core::ptr::null_mut(),
                        _marker: PhantomData,

                        $(
                            $required: self.$required,
                        )*
                    
                        $(
                            $niceties: self.$niceties,
                        )*

                        ..Default::default()
                    }
                )
            }

            fn check_required(&self) -> bool {
                $(
                    {
                        let feature_name = stringify!($required);
                        let super_feature_name = stringify!($i); // super feature, featuring the creature. larry.

                        if self.$required == 1 {
                            log::info!("required feature '{}' supported", feature_name);
                        } else {
                            log::error!("required feature '{}' of features '{}' not supported", feature_name, super_feature_name);
                            return false;
                        }
                    }

                )*

                true
            }

            fn check_niceties(&self) {
                $(
                    {
                        let feature_name = stringify!($niceties);
                        let super_feature_name = stringify!($i); // super feature, featuring the creature. larry.

                        if self.$niceties == 1 {
                            log::info!("nice-to-have feature '{}' supported", feature_name);
                        } else {
                            log::warn!("nice-to-have feature '{}' of features '{}' not supported", feature_name, super_feature_name);
                        }
                    }

                )*
            }
        }
    };
}

pub trait ExtendedFeature: ExtendsDeviceCreateInfo + ExtendsPhysicalDeviceFeatures2 {
    fn clone(&self) -> Box<dyn ExtendedFeature>;
    fn check_required(&self) -> bool;
    fn check_niceties(&self);
}


impl_extended_feature!(PhysicalDeviceComputeShaderDerivativesFeaturesNV, compute_derivative_group_quads compute_derivative_group_linear,);
impl_extended_feature!(PhysicalDeviceShaderClockFeaturesKHR, shader_subgroup_clock shader_device_clock,);
impl_extended_feature!(PhysicalDeviceShaderAtomicFloatFeaturesEXT, shader_shared_float32_atomics shader_shared_float32_atomic_add,);
impl_extended_feature!(PhysicalDeviceShaderImageAtomicInt64FeaturesEXT, shader_image_int64_atomics,);
impl_extended_feature!(PhysicalDeviceMeshShaderFeaturesEXT, mesh_shader task_shader,);
impl_extended_feature!(PhysicalDeviceExtendedDynamicState3FeaturesEXT, extended_dynamic_state3_polygon_mode,);
impl_extended_feature!(PhysicalDeviceAccelerationStructureFeaturesKHR, acceleration_structure acceleration_structure_indirect_build descriptor_binding_acceleration_structure_update_after_bind,);
impl_extended_feature!(PhysicalDeviceRayQueryFeaturesKHR, ray_query,);
impl_extended_feature!(PhysicalDeviceRayTracingPositionFetchFeaturesKHR, ray_tracing_position_fetch,);
impl_extended_feature!(PhysicalDeviceVulkan11Features, uniform_and_storage_buffer16_bit_access shader_draw_parameters,);
impl_extended_feature!(PhysicalDeviceVulkan12Features, storage_buffer8_bit_access uniform_and_storage_buffer8_bit_access shader_float16 shader_int8 host_query_reset timeline_semaphore shader_sampled_image_array_non_uniform_indexing shader_storage_image_array_non_uniform_indexing shader_storage_buffer_array_non_uniform_indexing shader_storage_texel_buffer_array_non_uniform_indexing buffer_device_address buffer_device_address_capture_replay descriptor_indexing descriptor_binding_partially_bound descriptor_binding_sampled_image_update_after_bind descriptor_binding_storage_buffer_update_after_bind descriptor_binding_storage_image_update_after_bind descriptor_binding_uniform_buffer_update_after_bind descriptor_binding_variable_descriptor_count runtime_descriptor_array,);
impl_extended_feature!(PhysicalDeviceVulkan13Features, synchronization2 maintenance4 dynamic_rendering,);
impl_extended_feature!(PhysicalDeviceVulkan14Features, , host_image_copy);

pub struct Intermediates {
    pub features: Vec<Box<dyn ExtendedFeature>>,
}

pub unsafe fn get_physical_device_features_and_check(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<Intermediates> {
    let mut physical_device_features = vk::PhysicalDeviceFeatures2::default();
    
    let mut features = Vec::<Box<dyn ExtendedFeature>>::new();

    features.push(Box::new(vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()));
    features.push(Box::new(vk::PhysicalDeviceShaderClockFeaturesKHR::default()));
    features.push(Box::new(vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()));
    features.push(Box::new(vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default()));
    features.push(Box::new(vk::PhysicalDeviceMeshShaderFeaturesEXT::default()));
    features.push(Box::new(vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()));
    features.push(Box::new(vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()));
    features.push(Box::new(vk::PhysicalDeviceRayQueryFeaturesKHR::default()));
    features.push(Box::new(vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()));
    features.push(Box::new(vk::PhysicalDeviceVulkan11Features::default()));
    features.push(Box::new(vk::PhysicalDeviceVulkan12Features::default()));
    features.push(Box::new(vk::PhysicalDeviceVulkan13Features::default()));
    features.push(Box::new(PhysicalDeviceVulkan14Features::default()));

    for feature in features.iter_mut() {
        physical_device_features = physical_device_features.push_next(feature.as_mut());
    }

    instance.get_physical_device_features2(physical_device, &mut physical_device_features);

    let physical_device_base_features_supported = 
        physical_device_features.features.shader_int16 == 1 &&
        physical_device_features.features.multi_draw_indirect == 1 &&
        physical_device_features.features.shader_int64 == 1 &&
        physical_device_features.features.sparse_binding == 1 &&
        physical_device_features.features.fill_mode_non_solid == 1 &&
        physical_device_features.features.sparse_residency_image3_d == 1 &&
        physical_device_features.features.pipeline_statistics_query == 1;

    // we are running on a modern version of vulkan, these should be enabled by default by now...
    assert!(physical_device_base_features_supported);

    for feature in features.iter() {
        if !feature.check_required() {
            return None;
        }

        feature.check_niceties();
    }

    Some(Intermediates { features })
}

pub unsafe fn create_device_and_queue(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    intermediates: Intermediates,
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

    let device_features_base = vk::PhysicalDeviceFeatures::default()
        .multi_draw_indirect(true)
        .shader_int16(true)
        .shader_int64(true)
        .sparse_binding(true)
        .sparse_residency_image3_d(true)
        .fill_mode_non_solid(true)
        .pipeline_statistics_query(true);

    let device_extension_names = [
        ash::khr::swapchain::NAME,
        
        // host / device
        ash::ext::host_query_reset::NAME,
        ash::ext::host_image_copy::NAME,
        ash::khr::buffer_device_address::NAME,
        ash::khr::timeline_semaphore::NAME,
        ash::khr::deferred_host_operations::NAME,

        // extra shader / pipeline extensions
        ash::khr::dynamic_rendering::NAME,
        ash::ext::mesh_shader::NAME,
        ash::khr::shader_float16_int8::NAME,
        ash::khr::shader_atomic_int64::NAME,
        ash::khr::shader_clock::NAME,
        ash::ext::shader_image_atomic_int64::NAME,
        ash::ext::shader_atomic_float::NAME,
        ash::khr::shader_draw_parameters::NAME,
        ash::ext::extended_dynamic_state3::NAME,
        c"VK_KHR_compute_shader_derivatives",         // TODO: remove when ash vk1.4

        // ray tracing extensions
        ash::khr::ray_tracing_position_fetch::NAME,
        ash::khr::acceleration_structure::NAME,
        ash::khr::ray_query::NAME,
    ];

    let device_extension_names_ptrs = device_extension_names
        .iter()
        .map(|cstr| cstr.as_ptr())
        .collect::<Vec<_>>();

    let mut device_create_info = vk::DeviceCreateInfo::default()
        .enabled_extension_names(&device_extension_names_ptrs)
        .enabled_features(&device_features_base)
        .queue_create_infos(&queue_create_infos);

    
    let mut cloned = intermediates.features.into_iter().map(|x| x.clone()).collect::<Vec<_>>();
    for feature in cloned.iter_mut() {
        device_create_info = device_create_info.push_next(feature.as_mut());
    }

    let device = instance
        .create_device(physical_device, &device_create_info, None)
        .unwrap();
    log::info!("created device");

    let queue = device.get_device_queue(queue_family_index, 0);
    log::info!("fetched queue");

    (device, queue_family_index, queue)
}
