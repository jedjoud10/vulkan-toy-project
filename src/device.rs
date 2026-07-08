
use std::{ffi::c_void, marker::PhantomData};

use ash::vk::{self, Bool32, ExtendsDeviceCreateInfo, ExtendsPhysicalDeviceFeatures2, StructureType, TaggedStructure};

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
    let mut float_atomics = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
        .shader_shared_float32_atomics(true)
        .shader_shared_float32_atomic_add(true); 
    let mut image_atomics = vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default()
        .shader_image_int64_atomics(true); 
    let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
        .mesh_shader(true)
        .task_shader(true);
    let mut extended_state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
        .extended_dynamic_state3_polygon_mode(true);
    let mut acceleration_structure = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
        .acceleration_structure(true)
        .acceleration_structure_indirect_build(true)
        .descriptor_binding_acceleration_structure_update_after_bind(true);
    let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default()
        .ray_query(true);
    let mut ray_tracing_position_fetch = vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()
        .ray_tracing_position_fetch(true);
    let device_features_base = vk::PhysicalDeviceFeatures::default()
        .multi_draw_indirect(true)
        .shader_int16(true)
        .shader_int64(true)
        .sparse_binding(true)
        .sparse_residency_image3_d(true)
        .fill_mode_non_solid(true)
        .pipeline_statistics_query(true);
    let mut device_features_11 = vk::PhysicalDeviceVulkan11Features::default()
        .uniform_and_storage_buffer16_bit_access(true)
        .shader_draw_parameters(true);
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
        .buffer_device_address_capture_replay(true)
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
    let mut device_features_14 = PhysicalDeviceVulkan14Features::default()
        .host_image_copy(true);

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

    let device_create_info = vk::DeviceCreateInfo::default()
        .enabled_extension_names(&device_extension_names_ptrs)
        .enabled_features(&device_features_base)
        .queue_create_infos(&queue_create_infos)
        .push_next(&mut device_features_14)
        .push_next(&mut device_features_13)
        .push_next(&mut device_features_12)
        .push_next(&mut device_features_11)
        .push_next(&mut image_atomics)
        .push_next(&mut float_atomics)
        .push_next(&mut shader_clock)
        .push_next(&mut compute_derivatives)
        .push_next(&mut mesh_shader)
        .push_next(&mut extended_state3)
        .push_next(&mut acceleration_structure)
        .push_next(&mut ray_query_features)
        .push_next(&mut ray_tracing_position_fetch);

    let device = instance
        .create_device(physical_device, &device_create_info, None)
        .unwrap();
    log::info!("created device");

    let queue = device.get_device_queue(queue_family_index, 0);
    log::info!("fetched queue");

    (device, queue_family_index, queue)
}

// TODO: remove when ash reaches vk1.4, probably post generator rewrite
// from https://raw.githubusercontent.com/ash-rs/ash/refs/heads/master/ash/src/vk/definitions.rs
#[repr(C)]
#[derive(Copy, Clone)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceVulkan14Features.html>"]
#[must_use]
pub struct PhysicalDeviceVulkan14Features<'a> {
    pub s_type: StructureType,
    pub p_next: *mut c_void,
    pub global_priority_query: Bool32,
    pub shader_subgroup_rotate: Bool32,
    pub shader_subgroup_rotate_clustered: Bool32,
    pub shader_float_controls2: Bool32,
    pub shader_expect_assume: Bool32,
    pub rectangular_lines: Bool32,
    pub bresenham_lines: Bool32,
    pub smooth_lines: Bool32,
    pub stippled_rectangular_lines: Bool32,
    pub stippled_bresenham_lines: Bool32,
    pub stippled_smooth_lines: Bool32,
    pub vertex_attribute_instance_rate_divisor: Bool32,
    pub vertex_attribute_instance_rate_zero_divisor: Bool32,
    pub index_type_uint8: Bool32,
    pub dynamic_rendering_local_read: Bool32,
    pub maintenance5: Bool32,
    pub maintenance6: Bool32,
    pub pipeline_protected_access: Bool32,
    pub pipeline_robustness: Bool32,
    pub host_image_copy: Bool32,
    pub push_descriptor: Bool32,
    pub _marker: PhantomData<&'a ()>,
}
unsafe impl Send for PhysicalDeviceVulkan14Features<'_> {}
unsafe impl Sync for PhysicalDeviceVulkan14Features<'_> {}
impl ::core::default::Default for PhysicalDeviceVulkan14Features<'_> {
    #[inline]
    fn default() -> Self {
        Self {
            s_type: Self::STRUCTURE_TYPE,
            p_next: ::core::ptr::null_mut(),
            global_priority_query: Bool32::default(),
            shader_subgroup_rotate: Bool32::default(),
            shader_subgroup_rotate_clustered: Bool32::default(),
            shader_float_controls2: Bool32::default(),
            shader_expect_assume: Bool32::default(),
            rectangular_lines: Bool32::default(),
            bresenham_lines: Bool32::default(),
            smooth_lines: Bool32::default(),
            stippled_rectangular_lines: Bool32::default(),
            stippled_bresenham_lines: Bool32::default(),
            stippled_smooth_lines: Bool32::default(),
            vertex_attribute_instance_rate_divisor: Bool32::default(),
            vertex_attribute_instance_rate_zero_divisor: Bool32::default(),
            index_type_uint8: Bool32::default(),
            dynamic_rendering_local_read: Bool32::default(),
            maintenance5: Bool32::default(),
            maintenance6: Bool32::default(),
            pipeline_protected_access: Bool32::default(),
            pipeline_robustness: Bool32::default(),
            host_image_copy: Bool32::default(),
            push_descriptor: Bool32::default(),
            _marker: PhantomData,
        }
    }
}

unsafe impl<'a> TaggedStructure for PhysicalDeviceVulkan14Features<'a> {
    // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES = 55
    const STRUCTURE_TYPE: StructureType = StructureType::from_raw(55);
}
unsafe impl ExtendsPhysicalDeviceFeatures2 for PhysicalDeviceVulkan14Features<'_> {}
unsafe impl ExtendsDeviceCreateInfo for PhysicalDeviceVulkan14Features<'_> {}
impl<'a> PhysicalDeviceVulkan14Features<'a> {
    #[inline]
    pub fn global_priority_query(mut self, global_priority_query: bool) -> Self {
        self.global_priority_query = global_priority_query.into();
        self
    }
    #[inline]
    pub fn shader_subgroup_rotate(mut self, shader_subgroup_rotate: bool) -> Self {
        self.shader_subgroup_rotate = shader_subgroup_rotate.into();
        self
    }
    #[inline]
    pub fn shader_subgroup_rotate_clustered(
        mut self,
        shader_subgroup_rotate_clustered: bool,
    ) -> Self {
        self.shader_subgroup_rotate_clustered = shader_subgroup_rotate_clustered.into();
        self
    }
    #[inline]
    pub fn shader_float_controls2(mut self, shader_float_controls2: bool) -> Self {
        self.shader_float_controls2 = shader_float_controls2.into();
        self
    }
    #[inline]
    pub fn shader_expect_assume(mut self, shader_expect_assume: bool) -> Self {
        self.shader_expect_assume = shader_expect_assume.into();
        self
    }
    #[inline]
    pub fn rectangular_lines(mut self, rectangular_lines: bool) -> Self {
        self.rectangular_lines = rectangular_lines.into();
        self
    }
    #[inline]
    pub fn bresenham_lines(mut self, bresenham_lines: bool) -> Self {
        self.bresenham_lines = bresenham_lines.into();
        self
    }
    #[inline]
    pub fn smooth_lines(mut self, smooth_lines: bool) -> Self {
        self.smooth_lines = smooth_lines.into();
        self
    }
    #[inline]
    pub fn stippled_rectangular_lines(mut self, stippled_rectangular_lines: bool) -> Self {
        self.stippled_rectangular_lines = stippled_rectangular_lines.into();
        self
    }
    #[inline]
    pub fn stippled_bresenham_lines(mut self, stippled_bresenham_lines: bool) -> Self {
        self.stippled_bresenham_lines = stippled_bresenham_lines.into();
        self
    }
    #[inline]
    pub fn stippled_smooth_lines(mut self, stippled_smooth_lines: bool) -> Self {
        self.stippled_smooth_lines = stippled_smooth_lines.into();
        self
    }
    #[inline]
    pub fn vertex_attribute_instance_rate_divisor(
        mut self,
        vertex_attribute_instance_rate_divisor: bool,
    ) -> Self {
        self.vertex_attribute_instance_rate_divisor = vertex_attribute_instance_rate_divisor.into();
        self
    }
    #[inline]
    pub fn vertex_attribute_instance_rate_zero_divisor(
        mut self,
        vertex_attribute_instance_rate_zero_divisor: bool,
    ) -> Self {
        self.vertex_attribute_instance_rate_zero_divisor =
            vertex_attribute_instance_rate_zero_divisor.into();
        self
    }
    #[inline]
    pub fn index_type_uint8(mut self, index_type_uint8: bool) -> Self {
        self.index_type_uint8 = index_type_uint8.into();
        self
    }
    #[inline]
    pub fn dynamic_rendering_local_read(mut self, dynamic_rendering_local_read: bool) -> Self {
        self.dynamic_rendering_local_read = dynamic_rendering_local_read.into();
        self
    }
    #[inline]
    pub fn maintenance5(mut self, maintenance5: bool) -> Self {
        self.maintenance5 = maintenance5.into();
        self
    }
    #[inline]
    pub fn maintenance6(mut self, maintenance6: bool) -> Self {
        self.maintenance6 = maintenance6.into();
        self
    }
    #[inline]
    pub fn pipeline_protected_access(mut self, pipeline_protected_access: bool) -> Self {
        self.pipeline_protected_access = pipeline_protected_access.into();
        self
    }
    #[inline]
    pub fn pipeline_robustness(mut self, pipeline_robustness: bool) -> Self {
        self.pipeline_robustness = pipeline_robustness.into();
        self
    }
    #[inline]
    pub fn host_image_copy(mut self, host_image_copy: bool) -> Self {
        self.host_image_copy = host_image_copy.into();
        self
    }
    #[inline]
    pub fn push_descriptor(mut self, push_descriptor: bool) -> Self {
        self.push_descriptor = push_descriptor.into();
        self
    }
}