use std::ffi::{CStr, c_char};

use ash::vk;
use raw_window_handle::RawDisplayHandle;

const DEBUG_INSTANCE_EXTENSIONS: &[&CStr] =
    &[
        ash::ext::debug_utils::NAME,
        ash::ext::validation_features::NAME,
    ];


const REQUIRED_INSTANCE_EXTENSIONS: &[&CStr] =
    &[
        ash::khr::surface::NAME,
    ];

const DEBUG_INSTANCE_VALIDATION_LAYERS: &[&CStr] = &[
    c"VK_LAYER_KHRONOS_validation",
];

pub unsafe fn create_instance(
    entry: &ash::Entry,
    raw_display_handle: RawDisplayHandle,
    debug_stuff: bool,
) -> ash::Instance {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan Experiments")
        .api_version(vk::API_VERSION_1_3)
        .application_version(0)
        .engine_version(0)
        .engine_name(c"Unnamed");

    let mut extension_names_ptrs = ash_window::enumerate_required_extensions(raw_display_handle)
        .unwrap()
        .to_vec();

    extension_names_ptrs.extend(REQUIRED_INSTANCE_EXTENSIONS.iter().map(|s| s.as_ptr()));

    if debug_stuff {
        extension_names_ptrs.extend(DEBUG_INSTANCE_EXTENSIONS.iter().map(|s| s.as_ptr()));
    }
    
    let mut validation_ptrs: Vec<*const c_char> = Vec::new();
    let mut enabled_validation_features: Vec<vk::ValidationFeatureEnableEXT> = Vec::new();

    if debug_stuff {
        validation_ptrs.extend(DEBUG_INSTANCE_VALIDATION_LAYERS
            .iter()
            .map(|cstr| cstr.as_ptr()));
        enabled_validation_features.extend([
            vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            vk::ValidationFeatureEnableEXT::BEST_PRACTICES
        ]);
    }

    let mut validation_features = ash::vk::ValidationFeaturesEXT::default()
        .enabled_validation_features(&enabled_validation_features);

    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&validation_ptrs)
        .enabled_extension_names(&extension_names_ptrs)
        .push_next(&mut validation_features);
    entry.create_instance(&instance_create_info, None).unwrap()
}
