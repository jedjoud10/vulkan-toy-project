use std::ffi::{CStr, CString};
use ash::vk;

pub const FRAMES_IN_FLIGHT: u32 = 2;

pub unsafe fn create_swapchain(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    extent: vk::Extent2D,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> (
    ash::khr::swapchain::Device,
    vk::SwapchainKHR,
    Vec<vk::Image>,
    vk::Format,
) {
    let surface_capabilities = surface_loader
        .get_physical_device_surface_capabilities(physical_device, surface_khr)
        .unwrap();

    let mut frames_in_flight = FRAMES_IN_FLIGHT;
    if frames_in_flight < surface_capabilities.min_image_count || frames_in_flight > surface_capabilities.max_image_count {
        log::error!("could not use specific frame in flight count: {FRAMES_IN_FLIGHT}, reverting to swapchain minimum of {}", surface_capabilities.min_image_count);
        frames_in_flight = surface_capabilities.min_image_count;
    }

    let present_modes: Vec<vk::PresentModeKHR> = surface_loader
        .get_physical_device_surface_present_modes(physical_device, surface_khr)
        .unwrap();
    let surface_formats: Vec<vk::SurfaceFormatKHR> = surface_loader
        .get_physical_device_surface_formats(physical_device, surface_khr)
        .unwrap();
    let _present = present_modes
        .iter()
        .copied()
        .find(|&x| x == vk::PresentModeKHR::IMMEDIATE || x == vk::PresentModeKHR::MAILBOX)
        .unwrap();
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface_khr)
        .min_image_count(frames_in_flight)
        .image_format(surface_formats[0].format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(extent)
        .image_array_layers(1)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .image_usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE,
        )
        .clipped(true)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .old_swapchain(vk::SwapchainKHR::null())
        .present_mode(vk::PresentModeKHR::IMMEDIATE);

    let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
    let swapchain = swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();
    let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();

    if let Some(binder) = binder {
        for (i, image) in images.iter().enumerate() {
            let name = CString::new(format!("swapchain image {i}")).unwrap();
            let marker = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(*image)
                .object_name(&name);
            binder.set_debug_utils_object_name(&marker).unwrap();
        }
    }

    (swapchain_loader, swapchain, images, surface_formats[0].format)
}