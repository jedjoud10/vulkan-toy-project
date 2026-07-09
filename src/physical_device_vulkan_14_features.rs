use std::{ffi::c_void, marker::PhantomData};
use ash::vk::{self, Bool32, ExtendsDeviceCreateInfo, ExtendsPhysicalDeviceFeatures2, StructureType, TaggedStructure};

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