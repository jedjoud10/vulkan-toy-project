use ash::vk;
use bytemuck::Pod;
use bytemuck::Zeroable;
use bytemuck::bytes_of;
use bytemuck::cast_slice;
use bytesize::ByteSize;
use rand::RngExt;
use rand::SeedableRng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use smallvec::SmallVec;
use crate::material::GpuMaterialInfo;
use crate::ray_tracing;
use crate::sdf_texture;
use crate::query_pool_statistics;
use crate::debug_text;
use crate::input::Button;
use crate::input::Input;
use crate::material::Material;
use crate::movement::Movement;
use crate::per_frame_data;
use crate::physical_device::PhysicalDeviceAndScore;
use crate::samplers;
use crate::query_pool_statistics::QueryPoolStatistics;
use crate::shader_compiler;
use crate::voxel;
use crate::voxel::SparseVoxelOctree;
use crate::voxel::TestingStructure;
use winit::event::MouseButton;
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::time::Duration;
use std::time::Instant;
use std::fmt::Write;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::Window;

use crate::swapchain;
use crate::pipeline;
use crate::skybox;
use crate::buffer;
use crate::instance;
use crate::physical_device;
use crate::device;
use crate::debug;
use crate::others;
use crate::per_frame_data::PerFrameData;
use crate::render_targets_data::RenderTargetsData;

const COMPUTE_POST_PROCESS: &'static str = "compute_post_process";
const BLOOM_UPSAMPLE_ENTRY_POINT: &'static str = "bloom_upsample";
const BLOOM_DOWNSAMPLE_ENTRY_POINT: &'static str = "bloom_downsample";
const WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT: &'static str = "write_swapchain_image";
const COMPUTE_SKY: &str = "compute_sky";
const WRITE_CLOUDS_ENTRY_POINT: &str = "write_clouds";
const WRITE_SKYBOX_ENTRY_POINT: &str = "write_skybox";
const BLUR_AMBIENT_SKYBOX_ENTRY_POINT: &str = "blur_skybox_ambient";

const COMPUTE_FULLSCREEN: &str = "fullscreen";

pub struct GraphicsContext<'a> {
    pub device: &'a ash::Device,
    pub pool: vk::CommandPool,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub mesh_shader_device: &'a ash::ext::mesh_shader::Device,
    pub extended_dynamic_state3_device: &'a ash::ext::extended_dynamic_state3::Device,
    pub acceleration_structure_device: &'a ash::khr::acceleration_structure::Device,
    pub host_image_copy_device: &'a ash::ext::host_image_copy::Device,

    // TODO: hide this behind mutex or rwlock so that we can share GraphicsContext across threads safely
    pub allocator: &'a mut gpu_allocator::vulkan::Allocator,
    pub debug_marker: &'a debug::DebugMarker,
    pub main_descriptor_set_layout: vk::DescriptorSetLayout,
    pub main_pipeline_layout: vk::PipelineLayout,
    pub descriptor_pool: vk::DescriptorPool,
}

macro_rules! dbgtext_writeln {
    ($dst:expr, $($arg:tt)*) => {
        writeln!($dst, $($arg)*).unwrap();
    };
}


pub struct InternalApp {
    // entry, physical device, logical device
    entry: ash::Entry,
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    
    // debug stuff
    debug: Option<(
        ash::ext::debug_utils::Instance,
        vk::DebugUtilsMessengerEXT
    )>,
    debug_marker: Option<ash::ext::debug_utils::Device>,
    
    // surface & swapchain
    surface_loader: ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    swapchain_format: vk::Format,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    
    // queue
    queue: vk::Queue,
    queue_family_index: u32,
    
    // cmd buffs
    pool: vk::CommandPool,
    
    // pipelines
    graphics_pipelines: HashMap<&'static str, pipeline::GenericGraphicsPipeline>,
    compute_pipelines: HashMap<&'static str, pipeline::GenericComputePipeline>,

    // extra devices
    mesh_shader_device: ash::ext::mesh_shader::Device,
    extended_dynamic_state3_device: ash::ext::extended_dynamic_state3::Device,
    acceleration_structure_device: ash::khr::acceleration_structure::Device,
    host_image_copy_device: ash::ext::host_image_copy::Device,
    // TODO: when using ash rewrite; use KHR_copy_memory_indirect since it was promoted from NV_copy_memory_indirect

    // descriptors & frames in flight
    main_descriptor_set_layout: vk::DescriptorSetLayout,
    main_pipeline_layout: vk::PipelineLayout,
    frames_in_flight: SmallVec<[PerFrameData; per_frame_data::FRAMES_IN_FLIGHT]>,
    render_finished_semaphores: SmallVec<[vk::Semaphore; swapchain::SWAPCHAIN_IMAGES]>,
    descriptor_pool: vk::DescriptorPool,
    render_targets_data: RenderTargetsData,
            
    // important too
    allocator: gpu_allocator::vulkan::Allocator,
    

    materials: Vec<Material>,
    materials_buffer: buffer::Buffer,
    // voxels: SparseVoxelOctree,
    // voxels2: TestingStructure,


    tlas: crate::ray_tracing::TopLevelAccelerationStructure,
    blases: Vec<crate::ray_tracing::AccelerationStructureData>,
    blases_instances: Vec<vk::AccelerationStructureInstanceKHR>,

    texture: sdf_texture::SdfImage,

    scene_representation_for_sdf: Vec<vek::Vec3<f32>>,
    scene_representation_for_sdf_buffer: buffer::Buffer,

    timestamp_period: f32,
    skybox: skybox::Skybox,
    samplers: samplers::Samplers,
    uniform_buffer: buffer::Buffer,

    // debug settings
    debug_type: u32,
    wireframe: bool,
    toggles_bitmask: u32,
    debug_text: debug_text::DebugText,

    // other CPU stuff
    pub was_resized: bool,
    pub window: Window,
    pub input: Input,    
    last_frame_cpu_cmd_record_duration: Duration,
    movement: Movement,
    frame_count: u64,
    sun: vek::Vec3<f32>,
    args: crate::Args,
    stats: QueryPoolStatistics,
}

impl InternalApp {
    pub unsafe fn new(event_loop: &ActiveEventLoop, args: crate::Args) -> Self {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        if args.fullscreen {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }

        window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap();
        window.set_cursor_visible(false);
        let raw_display_handle = window.display_handle().unwrap().as_raw();
        let entry = ash::Entry::load().unwrap();

        #[cfg(debug_assertions)]
        let running_cfg_debug_assertions = true;

        #[cfg(not(debug_assertions))]
        let running_cfg_debug_assertions = false;

        let debug_stuff = args.validate || running_cfg_debug_assertions;
        let instance = instance::create_instance(&entry, raw_display_handle, debug_stuff);
        log::info!("created instance");
        let debug_messenger = debug::create_debug_messenger(&entry, &instance, debug_stuff).inspect(|_x| {
            log::info!("created debug utils messenger");
        });

        let (surface_loader, surface_khr) = others::create_surface(&instance, &entry, &window);
        log::info!("created surface");

        let mut physical_device_candidates = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| {
                physical_device::get_physical_device_score(
                    physical_device,
                    &instance,
                    &surface_loader,
                    surface_khr,
                )
            })
            .collect::<Vec<PhysicalDeviceAndScore>>();
        physical_device_candidates.sort_by_key(|tmp| tmp.score);

        if physical_device_candidates.is_empty() {
            log::error!("no physical device was chosen!");
            panic!();
        }
        
        let PhysicalDeviceAndScore {
            physical_device,
            intermediates,
            ..
        } = physical_device_candidates.pop().unwrap();

        let mut physical_device_properties = vk::PhysicalDeviceProperties2::default();
        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);
        let physical_device_name = physical_device_properties.properties.device_name_as_c_str().unwrap().to_str().unwrap();

        log::info!("selected physical device \"{}\"", physical_device_name);

        let (device, queue_family_index, queue) = device::create_device_and_queue(
            &instance,
            physical_device,
            &surface_loader,
            surface_khr,
            intermediates
        );
        log::info!("created device and fetched main queue");

        let mesh_shader_device = ash::ext::mesh_shader::Device::new(&instance, &device);
        let extended_dynamic_state3_device = ash::ext::extended_dynamic_state3::Device::new(&instance, &device);
        let acceleration_structure_device = ash::khr::acceleration_structure::Device::new(&instance, &device);
        let host_image_copy_device = ash::ext::host_image_copy::Device::new(&instance, &device);

        let debug_marker = debug_messenger.is_some().then(|| {
            let device = debug::create_debug_marker(&instance, &device);
            log::info!("created debug marker object");
            device
        });

        let mut allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings {
                    log_leaks_on_shutdown: running_cfg_debug_assertions,
                    log_frees: false,
                    ..Default::default()
                },
                buffer_device_address: true,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            })
            .unwrap();
        log::info!("created gpu allocator");

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = device.create_command_pool(&pool_create_info, None).unwrap();
        log::info!("created cmd pool");

        let mut extent = vk::Extent2D {
            width: 800,
            height: 600,
        };

        if args.fullscreen {
            extent = vk::Extent2D {
                width: window.inner_size().width,
                height: window.inner_size().height,
            }
        }

        let (swapchain_loader, swapchain, swapchain_images, swapchain_image_views, swapchain_format) = swapchain::create_swapchain(
            &instance,
            &surface_loader,
            surface_khr,
            physical_device,
            &device,
            extent,
            &debug_marker,
            None,
        );
        log::info!("created swapchain with {} images", swapchain_images.len());

        let (descriptor_pool, main_descriptor_set_layout) = others::create_descriptor_pool_and_bindless_descriptor_set(&device, &debug_marker);

        let main_pipeline_layout = pipeline::create_bindless_pipeline_layout(&device, &debug_marker, main_descriptor_set_layout);
        log::info!("created bindless pipeline layout");

        let mut graphics_pipelines = HashMap::<&'static str, pipeline::GenericGraphicsPipeline>::new();
        let mut compute_pipelines = HashMap::<&'static str, pipeline::GenericComputePipeline>::new();

        let spec_constants_bitflags = if args.readback_performance_queries { 1u32 } else { 0 };  

        let spec_constants = [
            spec_constants_bitflags,
            skybox::SKYBOX_RESOLUTION, skybox::CLOUDS_RESOLUTION, skybox::AMBIENT_SKYBOX_RESOLUTION,
            args.downscale_factor,
        ];

        let settings = [pipeline::PipelineCreateSettings {
            pipeline_debug_name: "post process compute pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &[WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT, BLOOM_DOWNSAMPLE_ENTRY_POINT, BLOOM_UPSAMPLE_ENTRY_POINT] },
            spec_constants: Some(&spec_constants),
            file_name_without_extension: COMPUTE_POST_PROCESS,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "sky compute pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &[WRITE_SKYBOX_ENTRY_POINT, WRITE_CLOUDS_ENTRY_POINT, BLUR_AMBIENT_SKYBOX_ENTRY_POINT] },
            spec_constants: Some(&spec_constants),
            file_name_without_extension: COMPUTE_SKY,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "compute fullscreen shader",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &["main"] },
            spec_constants: Some(&spec_constants),
            file_name_without_extension: COMPUTE_FULLSCREEN,
        }];

        let mut compiled = HashMap::<String, Cow<[u8]>>::default(); 
        shader_compiler::compile_all_shaders(&mut compiled);

        // compile the pipelines in parallel
        // ouug shii :eyes:
        log::info!("creating pipelines...");
        let generic_pipelines = settings.into_par_iter().map(|setting| {
            let spv_file_name = setting.file_name_without_extension;
            let raw_bytes = &compiled[spv_file_name];
            
            let len = raw_bytes.len();
            assert!(len.is_multiple_of(4));

            // align to word sizes
            // previous assertion upholds that the number of bytes is a multiple of word size
            let mut vec = vec![0u32; len / 4];
            let dst_slice = bytemuck::cast_slice_mut::<u32, u8>(vec.as_mut_slice());
            dst_slice.copy_from_slice(&raw_bytes);

            let raw_words = &vec;
            let pipeline = pipeline::create_generic_pipeline(raw_words, &device, &debug_marker, main_pipeline_layout, setting);
            (spv_file_name, pipeline)
        }).collect::<Vec<_>>();

        for (spv_file_name, pipeline) in generic_pipelines {
            match pipeline {
                pipeline::GenericPipeline::Graphics(generic_graphics_pipeline) => { graphics_pipelines.insert(spv_file_name, generic_graphics_pipeline); },
                pipeline::GenericPipeline::Compute(generic_compute_pipeline) => { compute_pipelines.insert(spv_file_name, generic_compute_pipeline); },
            }
        }

        let samplers = samplers::Samplers::create_samplers(&device);
        log::info!("created samplers");        

        let mut ctx = GraphicsContext {
            device: &device,
            pool,
            queue,
            queue_family_index,
            mesh_shader_device: &mesh_shader_device,
            extended_dynamic_state3_device: &extended_dynamic_state3_device,
            acceleration_structure_device: &acceleration_structure_device,
            host_image_copy_device: &host_image_copy_device,
            allocator: &mut allocator,
            debug_marker: &debug_marker,
            main_descriptor_set_layout,
            main_pipeline_layout,
            descriptor_pool,
        };

        let skybox = skybox::create_skybox(&mut ctx);
        log::info!("created skybox");

        let frames_in_flight = (0..per_frame_data::FRAMES_IN_FLIGHT).into_iter().map(|_| {
            PerFrameData::create_per_frame_data(&mut ctx)
        }).collect::<SmallVec<[PerFrameData; per_frame_data::FRAMES_IN_FLIGHT]>>();
        log::info!("created frames in flight structures");

        let mut render_targets_data = RenderTargetsData::create_constant_descriptor_sets();
        render_targets_data.recreate_rt_images_and_image_views_and_update_descriptor_sets(&mut ctx, extent, args.downscale_factor);
        log::info!("created constant descriptor sets");

        let timestamp_period = physical_device_properties.properties.limits.timestamp_period;

        let cmd = others::begin_recording(&mut ctx);
        let mut writer = buffer::begin_buffer_writer(&mut ctx);

        let materials = vec![
            Material::new(&mut ctx, "metal/metal_0077"),
            Material::new(&mut ctx, "ground/ground_0029"),
            Material::new(&mut ctx, "metal_2/metal_0066"),
            Material::new(&mut ctx, "ground_2/ground_0019"),
        ];

        let materials_buffer = buffer::create_buffer(&mut ctx, materials.len() * size_of::<GpuMaterialInfo>(), "gpu materials buffer", vk::BufferUsageFlags::TRANSFER_DST);

        //let voxels = voxel::create_sparse_structures(&mut ctx, cmd, &mut writer, false);
        //let voxels2 = voxel::create_sparse_structures2(&mut ctx, cmd, &mut writer, false);


        
        let aabbs = [ray_tracing::AccelerationStructureAabb {
            min: -vek::Vec3::one(),
            max: vek::Vec3::one(),
        }];

        let written = writer.write_bytes(cast_slice(&aabbs));

        let geometry = ray_tracing::BlasGeometry::AABBs {
            aabb_buffer_address: written.buffer_device_address_start,
            max_count: 1
        };

        let blas1 = ray_tracing::create_blas(&mut ctx, cmd, geometry);

        let aabbs = [ray_tracing::AccelerationStructureAabb {
            min: -vek::Vec3::new(4f32, 10f32,4f32),
            max: vek::Vec3::new(4f32, 10f32, 4f32),
        }];

        let written = writer.write_bytes(cast_slice(&aabbs));

        let geometry = ray_tracing::BlasGeometry::AABBs {
            aabb_buffer_address: written.buffer_device_address_start,
            max_count: 1
        };

        let blas2 = ray_tracing::create_blas(&mut ctx, cmd, geometry);

        let blases = vec![blas1, blas2];

        others::end_recording_and_submit(&mut ctx, cmd);
        buffer::end_buffer_writer(&mut ctx, writer);

        let debug_text = debug_text::DebugText::new(&mut ctx);

        let render_finished_semaphores: SmallVec<[vk::Semaphore; swapchain::SWAPCHAIN_IMAGES]> = (0..swapchain::SWAPCHAIN_IMAGES).into_iter().map(|_| {
            device.create_semaphore(&Default::default(), None).unwrap()
        }).collect::<SmallVec<[vk::Semaphore; swapchain::SWAPCHAIN_IMAGES]>>();

        let uniform_buffer = buffer::create_buffer(
            &mut ctx,
            size_of::<pipeline::PerFrameUniformData>(),
            "per frame uniform buffer",
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
        );

        let tlas = ray_tracing::pre_create_tlas(&mut ctx);
        let scene_representation_for_sdf_buffer = buffer::create_buffer_default_flags(&mut ctx, size_of::<u32>() + size_of::<vek::Vec3<f32>>() * 100, "a");
        let scene_representation_for_sdf = vec![vek::Vec3::<f32>::one(), vek::Vec3::new(4f32, 10f32, 4f32)];

        let mut blases_instances = Vec::new();

        // plane
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::identity(),
            -vek::Vec3::unit_y(),
            vek::Vec3::new(1000.0, 1.0, 1000.0),
            &blases[0],
            0,
            0xFF,
        ));

        // sphere
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::identity(),
            vek::Vec3::unit_y(),
            vek::Vec3::broadcast(2f32),
            &blases[0],
            0,
            0xFF,
        ));

        // hex
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::default(),
            vek::Vec3::new(10f32, 2f32, 0f32),
            vek::Vec3::new(2.6, 2.1, 4.1),
            &blases[0],
            0,
            0xFF,
        ));

        // torus
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::rotation_3d(2.4f32, vek::Vec3::<f32>::one().normalized()),
            vek::Vec3::new(2f32, 2f32, 5f32),
            vek::Vec3::new(7f32, 2f32, 7f32),
            &blases[0],
            0,
            0xFF,
        ));

        // cylinder
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::default(),
            vek::Vec3::new(-10f32,0f32, -0f32),
            vek::Vec3::new(2f32, 400f32, 2f32),
            &blases[0],
            0,
            0xFF,
        ));

        // sphere 2
        blases_instances.push(ray_tracing::instantiate_blas(
            vek::Quaternion::default(),
            vek::Vec3::new(0f32,10f32, 0f32),
            vek::Vec3::new(6f32, 6f32, 6f32),
            &blases[0],
            0,
            0xFF,
        ));

        let texture = sdf_texture::create_voxel_image(&mut ctx);



        let mut rng = rand::rngs::SmallRng::seed_from_u64(432);
        for x in -20..20 {
            for z in -20..20 {
                blases_instances.push(ray_tracing::instantiate_blas(
                    vek::Quaternion::rotation_y(rng.random_range(0f32..std::f32::consts::TAU)),
                    vek::Vec3::new(rng.random_range(-300f32..300f32), 8f32, rng.random_range(-300f32..300f32)),
                    vek::Vec3::one(), // cannot do non-uniform scale! cannot do scale in general unless we account for it in the shader side!
                    &blases[1],
                    1,
                    0xFF
                ));
            }
        }


        Self {
            materials_buffer,
            materials,
            tlas, 
            blases,
            blases_instances,

            scene_representation_for_sdf,
            scene_representation_for_sdf_buffer,

            texture,
            last_frame_cpu_cmd_record_duration: Default::default(),
            frame_count: 0,
            input: Default::default(),
            movement: Movement::new(),
            window,
            instance,
            entry,
            device,
            physical_device,
            surface_loader,
            surface_khr,
            debug: debug_messenger,
            debug_marker,
            swapchain_loader,
            swapchain_format,
            swapchain,
            queue_family_index,
            queue,
            pool,
            render_targets_data,
            descriptor_pool,
            timestamp_period,
            allocator,
            skybox,
            was_resized: false,
            frames_in_flight,
            sun: vek::Vec3::new(1f32, 0.3f32,0.5f32).normalized(),
            debug_type: 0,
            stats: query_pool_statistics::QueryPoolStatistics::new(),
            args,
            samplers,
            swapchain_images,
            swapchain_image_views,
            main_descriptor_set_layout,
            main_pipeline_layout,
            mesh_shader_device,
            extended_dynamic_state3_device,
            graphics_pipelines,
            compute_pipelines,
            wireframe: false,
            toggles_bitmask: 1u32 | 2u32,
            debug_text,
            render_finished_semaphores,
            uniform_buffer,
            acceleration_structure_device,
            host_image_copy_device,
        }
    }

    pub unsafe fn click(&mut self, add: bool) {
        let position = self.movement.forward() * 5.0f32 + self.movement.position;

        let mut ctx = GraphicsContext {
            device: &self.device,
            pool: self.pool,
            queue: self.queue,
            queue_family_index: self.queue_family_index,
            mesh_shader_device: &self.mesh_shader_device,
            extended_dynamic_state3_device: &self.extended_dynamic_state3_device,
            acceleration_structure_device: &self.acceleration_structure_device,
            host_image_copy_device: &self.host_image_copy_device,
            allocator: &mut self.allocator,
            debug_marker: &self.debug_marker,
            main_descriptor_set_layout: self.main_descriptor_set_layout,
            main_pipeline_layout: self.main_pipeline_layout,
            descriptor_pool: self.descriptor_pool,
        };

        if add {
            self.blases_instances.push(ray_tracing::instantiate_blas(vek::Quaternion::identity(), position, vek::Vec3::one(), &self.blases[1], 1, 0xFF));
            //self.scene_representation_for_sdf.push(position);
        }
    }

    pub unsafe fn recreate_swapchain(&mut self) {
        log::warn!("recreating swapchain");
        self.was_resized = false;
        self.device.device_wait_idle().unwrap();

        let width = self.window.inner_size().width;
        let height = self.window.inner_size().height;
        
        let extent = vk::Extent2D { width, height };

        // recreate swapchain (pass in old swapchain as well) 
        let (swapchain_loader, swapchain, swapchain_images, swapchain_image_views, swapchain_format) = swapchain::create_swapchain(
            &self.instance,
            &self.surface_loader,
            self.surface_khr,
            self.physical_device,
            &self.device,
            extent,
            &self.debug_marker,
            Some(self.swapchain)
        );

        // destroy old swapchain and image views...
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        for swapchain_image_view in self.swapchain_image_views.iter() {
            self.device.destroy_image_view(*swapchain_image_view, None);
        }

        self.swapchain_loader = swapchain_loader;
        self.swapchain_format = swapchain_format;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain = swapchain;

        self.render_targets_data.destroy_rt_images_and_image_views(&self.device, &mut self.allocator);

        let mut ctx = GraphicsContext {
            device: &self.device,
            pool: self.pool,
            queue: self.queue,
            queue_family_index: self.queue_family_index,
            mesh_shader_device: &self.mesh_shader_device,
            extended_dynamic_state3_device: &self.extended_dynamic_state3_device,
            acceleration_structure_device: &self.acceleration_structure_device,
            host_image_copy_device: &self.host_image_copy_device,
            allocator: &mut self.allocator,
            debug_marker: &self.debug_marker,
            main_descriptor_set_layout: self.main_descriptor_set_layout,
            descriptor_pool: self.descriptor_pool,
            main_pipeline_layout: self.main_pipeline_layout,
        };

        self.render_targets_data.recreate_rt_images_and_image_views_and_update_descriptor_sets(&mut ctx, extent, self.args.downscale_factor);


        for frame in self.frames_in_flight.iter_mut() {
            self.device.destroy_semaphore(frame.present_complete_semaphore, None);
            frame.present_complete_semaphore = self.device.create_semaphore(&Default::default(), None).unwrap();
        }

        for render_finished_semaphore in self.render_finished_semaphores.iter_mut() {
            self.device.destroy_semaphore(*render_finished_semaphore, None);
            *render_finished_semaphore = self.device.create_semaphore(&Default::default(), None).unwrap();
        }


        self.device.device_wait_idle().unwrap();
    }

    pub unsafe fn pre_render(&mut self, delta: f32) -> ControlFlow<()> {
        let size = self.window.inner_size().cast::<f32>();
        self.movement.update(&self.input, size.width / size.height, delta);
        if self.input.get_button(KeyCode::F5).pressed() {
            if self.window.fullscreen().is_none() {
                self
                    .window
                    .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            } else {
                self.window.set_fullscreen(None);
            }
        }
        let left = self.input.get_button(Button::Mouse(MouseButton::Left)).pressed();
        let right = self.input.get_button(Button::Mouse(MouseButton::Right)).pressed();
        if left || right {
            self.click(left);
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyH)).pressed() {
            self.debug_type = (self.debug_type as i32 + 1).rem_euclid(8) as u32;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyG)).pressed() {
            self.debug_type = (self.debug_type as i32 - 1).rem_euclid(8) as u32;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyT)).pressed() {
            self.wireframe = !self.wireframe; 
        }

        // TODO: make this better rofl
        if self.input.get_button(Button::Keyboard(KeyCode::Digit1)).pressed() {
            self.toggles_bitmask ^= 1;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::Digit2)).pressed() {
            self.toggles_bitmask ^= 2;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::Digit3)).pressed() {
            self.toggles_bitmask ^= 4;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::Digit4)).pressed() {
            self.toggles_bitmask ^= 8;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::Digit5)).pressed() {
            self.toggles_bitmask ^= 16;
        }


        if self.input.get_button(Button::Keyboard(KeyCode::KeyJ)).pressed() {
            let report = self.allocator.generate_report();
            log::debug!("{:?}", report);
        }
        if self.input.get_button(Button::Mouse(MouseButton::Middle)).held() {
            self.sun = self.movement.forward();
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyQ)).pressed() {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }

    pub unsafe fn render(&mut self, delta: f32, elapsed: f32) {
        let frame_in_flight_index = self.frame_count % (self.frames_in_flight.len() as u64);
        let render_targets = &self.render_targets_data;
        let &mut PerFrameData {
            present_complete_semaphore,
            end_fence,
            cmd,
            main_descriptor_set,
            query_pool,
            pipeline_statistics_query_pool,
            ref mut scratch_buffer,
            ref mut readback_buffer,
            ..
        } = &mut self.frames_in_flight[frame_in_flight_index as usize];


        let present_complete_semaphores = [present_complete_semaphore];

        let pre_wait_for_fence = Instant::now();
        if let Err(err) = self.device.wait_for_fences(&[end_fence], true, u64::MAX) {
            log::error!("wait on fence err: {:?}", err);
        } else {
            self.stats.import_data(self.frame_count, &self.device, query_pool, self.timestamp_period);
        }

        let post_wait_for_fence = Instant::now();

        let pre_acquire_swapchain = Instant::now();
        let (acquired_swapchain_image_index, suboptimal) = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null(),
            )
            .unwrap();
        let post_acquire_swapchain = Instant::now();

        let swapchain_image = self.swapchain_images[acquired_swapchain_image_index as usize]; // then compose onto this...
        let swapchain_image_view = self.swapchain_image_views[acquired_swapchain_image_index as usize];

        
        if suboptimal || self.was_resized {
            log::debug!("suboptimal: {suboptimal}");
            log::debug!("was resized: {}", self.was_resized);
            
            self.recreate_swapchain();
            self.was_resized = false;
            return;
        }

        self.device.reset_fences(&[end_fence]).unwrap();
        let render_finished_semaphore = [self.render_finished_semaphores[acquired_swapchain_image_index as usize]];

        // create bindless descriptor write for storage images        
        let swapchain_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(swapchain_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let rendered_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(render_targets.rendered_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let skybox_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.skybox_array_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let ambient_skybox_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.ambient_skybox_array_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let clouds_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.clouds_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let voxel_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.texture.image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut storage_image_infos = vec![
            swapchain_image_view_descriptor_image_info,
            rendered_image_view_descriptor_image_info,
            skybox_image_view_descriptor_image_info,
            ambient_skybox_image_view_descriptor_image_info,
            clouds_image_view_descriptor_image_info,
            voxel_image_view_descriptor_image_info
        ];
        
        // add bloom storage image views
        for bloom_storage_image_view in render_targets.bloom_mip_image_views.iter() {
            storage_image_infos.push(vk::DescriptorImageInfo::default()
                .image_view(*bloom_storage_image_view)
                .image_layout(vk::ImageLayout::GENERAL)
            );
        }
        
        let storage_image_write = vk::WriteDescriptorSet::default()
            .descriptor_count(storage_image_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_array_element(0)
            .dst_set(main_descriptor_set)
            .image_info(&storage_image_infos);      

        // create bindless descriptor write for storage buffers
        let storage_buffer_infos = vec![
            vk::DescriptorBufferInfo::default()
                .buffer(self.uniform_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(self.debug_text.buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(readback_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(self.materials_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(self.scene_representation_for_sdf_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
        ];

        let storage_buffer_write = vk::WriteDescriptorSet::default()
            .descriptor_count(storage_buffer_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_binding(1)
            .dst_set(main_descriptor_set)
            .buffer_info(&storage_buffer_infos);

        // create bindless descriptor write for image samplers
        let skybox_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.skybox_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let ambient_skybox_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.ambient_skybox_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let clouds_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.clouds_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let rendered_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(render_targets.rendered_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let entire_bloom_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(render_targets.entire_bloom_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let voxel_sampled_image_view_descriptor_write_info = vk::DescriptorImageInfo::default()
            .image_view(self.texture.image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut sampled_image_infos = vec![
            skybox_sampled_image_view_descriptor_image_info,
            ambient_skybox_sampled_image_view_descriptor_image_info,
            clouds_sampled_image_view_descriptor_image_info,
            rendered_sampled_image_view_descriptor_image_info,
            voxel_sampled_image_view_descriptor_write_info,
            entire_bloom_sampled_image_view_descriptor_image_info,
        ];

        // add bloom sampled image views
        for bloom_sampled_image_view in render_targets.bloom_mip_image_views.iter() {
            sampled_image_infos.push(vk::DescriptorImageInfo::default()
                .image_view(*bloom_sampled_image_view)
                .image_layout(vk::ImageLayout::GENERAL)
            );
        }

        // add material sampled image views
        for material in self.materials.iter_mut() {
            material.add_per_frame_sampled_images(&mut sampled_image_infos);
        }

        let sampled_image_write = vk::WriteDescriptorSet::default()
            .descriptor_count(sampled_image_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(2)
            .dst_set(main_descriptor_set)
            .image_info(&sampled_image_infos);

        let samplers = [
            vk::DescriptorImageInfo::default()
                .sampler(self.samplers.nearest),
            vk::DescriptorImageInfo::default()
                .sampler(self.samplers.linear_unclamped),
            vk::DescriptorImageInfo::default()
                .sampler(self.samplers.linear_clamped)
        ];
        let sampler_states_write = vk::WriteDescriptorSet::default()
            .descriptor_count(samplers.len() as u32)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .dst_binding(3)
            .dst_set(main_descriptor_set)
            .image_info(&samplers);

        let tlases = [self.tlas.data.acceleration_structure];
        let mut acceleration_structure_write_tmp = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&tlases);

        let acceleration_structure_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .dst_set(main_descriptor_set)
            .dst_binding(4)
            .push_next(&mut acceleration_structure_write_tmp);

        self.device.update_descriptor_sets(&[storage_image_write, storage_buffer_write, sampled_image_write, sampler_states_write, acceleration_structure_write], &[]);

        // TODO: ideally, these would:
        // 1. be dynamically allocated using some sort of per-frame arena with indexing
        // 2. be passed to the shader either using a uniform buffer (since these are constant anyways)
        const BLOOM_MIPS_STORAGE_IMAGE_START_IDX: u32 = 6; // bloom needs to be last since it is dynamically allocated (can have a dynamic number of bloom mips, depending on screen res)
        const RENDERED_SAMPLER_IMAGE_IDX: u32 = 3;
        const BLOOM_MIPS_SAMPLED_IMAGE_START_IDX: u32 = 6; // bloom needs to be last since it is dynamically allocated (can have a dynamic number of bloom mips, depending on screen res)

        let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device
            .begin_command_buffer(cmd, &cmd_buffer_begin_info)
            .unwrap();
        let cpu_cmd_record_start = Instant::now();
        self.device.cmd_reset_query_pool(cmd, query_pool, 0, query_pool_statistics::NUM_TIMESTAMP_QUERIES as u32);
        self.device.cmd_reset_query_pool(cmd, pipeline_statistics_query_pool, 0, 1);
                
        scratch_buffer.begin_of_cmd_recording(self.queue_family_index, &self.device, cmd);
        
        let mut ctx = GraphicsContext {
            device: &self.device,
            pool: self.pool,
            queue: self.queue,
            queue_family_index: self.queue_family_index,
            mesh_shader_device: &self.mesh_shader_device,
            extended_dynamic_state3_device: &self.extended_dynamic_state3_device,
            acceleration_structure_device: &self.acceleration_structure_device,
            host_image_copy_device: &self.host_image_copy_device,
            allocator: &mut self.allocator,
            debug_marker: &self.debug_marker,
            main_descriptor_set_layout: self.main_descriptor_set_layout,
            main_pipeline_layout: self.main_pipeline_layout,
            descriptor_pool: self.descriptor_pool,
        };
        
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        dbgtext_writeln!(&mut self.debug_text, "CPU delta: {:.2}ms", delta*1000f32);
        dbgtext_writeln!(&mut self.debug_text, "CPU command buffer record duration: {:.2}ms", self.last_frame_cpu_cmd_record_duration.as_micros() as f32 / 1000.0f32);
        dbgtext_writeln!(&mut self.debug_text, "CPU fence wait duration: {:.2}ms", (post_wait_for_fence - pre_wait_for_fence).as_micros() as f32 / 1000.0f32);
        dbgtext_writeln!(&mut self.debug_text, "CPU fence acquire swapchain duration: {:.2}ms", (post_acquire_swapchain - pre_acquire_swapchain).as_micros() as f32 / 1000.0f32);

        self.stats.add_to_debug_text(&mut self. debug_text);

        dbgtext_writeln!(&mut self.debug_text, "pos: {:.2}", self.movement.position);
        dbgtext_writeln!(&mut self.debug_text, "debug type: {}", self.debug_type);
        dbgtext_writeln!(&mut self.debug_text, "toggles bitmask: {:#032b}", self.toggles_bitmask);
        dbgtext_writeln!(&mut self.debug_text, "wireframe: {}", self.wireframe);
        dbgtext_writeln!(&mut self.debug_text, "updating frustum: {}", self.movement.update_frustum);

        let report = ctx.allocator.generate_report();
        let reserved_bytes = ByteSize::b(report.total_reserved_bytes).display().iec();
        let allocated_bytes = ByteSize::b(report.total_reserved_bytes).display().iec();
        dbgtext_writeln!(&mut self.debug_text, "reserved bytes: {}", reserved_bytes);
        dbgtext_writeln!(&mut self.debug_text, "allocated bytes: {}", allocated_bytes);

        if self.args.readback_performance_queries {
            let readback_buffer_readback_barrier = vk::BufferMemoryBarrier2::default()
                .buffer(readback_buffer.buffer)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
                .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(self.queue_family_index)
                .size(vk::WHOLE_SIZE);
            let buffer_memory_barriers = [readback_buffer_readback_barrier];
            let dep = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);
            self.device.cmd_pipeline_barrier2(cmd, &dep);

            let data = readback_buffer.allocation.mapped_slice_mut().unwrap();
            let readback_debug_buffer_data = cast_slice::<u8, u32>(data); 
            
            let million_sdf_calls = readback_debug_buffer_data[0] as f64 / 1_000_000f64;
            dbgtext_writeln!(&mut self.debug_text, "SDF calls (millions): {:.2}", million_sdf_calls);
            
            let million_traced_rays = readback_debug_buffer_data[1] as f64 / 1_000_000f64;
            dbgtext_writeln!(&mut self.debug_text, "rays traced (millions): {:.2}", million_traced_rays);

            dbgtext_writeln!(&mut self.debug_text, "million rays per second: {:.2}", million_traced_rays / (self.stats.get_compute_region_duration()));

            // clear data for next frame
            data.fill(0);
        }


        
        self.debug_text.update_debug_text(&ctx.device, cmd);        

        let gpu_material_data = self.materials.iter().map(|x| GpuMaterialInfo { base_index: x.base_index }).collect::<Vec<_>>();
        buffer::write_with_scratch_buffer(&mut ctx, cmd, scratch_buffer, cast_slice(&gpu_material_data), self.materials_buffer.buffer, 0);

        /*
        // write count
        let cnt = self.scene_representation_for_sdf.len() as u32;
        buffer::write_with_scratch_buffer(&mut ctx, cmd, scratch_buffer, bytes_of(&cnt), self.scene_representation_for_sdf_buffer.buffer, 0);

        // write actual scene repr data
        buffer::write_with_scratch_buffer(&mut ctx, cmd, scratch_buffer, cast_slice(&self.scene_representation_for_sdf), self.scene_representation_for_sdf_buffer.buffer, size_of::<u32>() as u64);
        */

        buffer::write_with_scratch_buffer(&mut ctx, cmd, scratch_buffer, cast_slice(&self.scene_representation_for_sdf), self.scene_representation_for_sdf_buffer.buffer, 0);
        

        // rebuild TLAS
        ray_tracing::rebuild_tlas(
            self.blases_instances.iter().copied(),
            &self.tlas,
            &mut ctx,
            cmd,
            scratch_buffer,
        );


        // bind the descriptor set for subsequent pipelines
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.main_pipeline_layout,
            0,
            &[main_descriptor_set],
            &[],
        );
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.main_pipeline_layout,
            0,
            &[main_descriptor_set],
            &[],
        );

        let size = self.window.inner_size();
        let window_size_no_downscale = vek::Vec2::<u32>::new(size.width, size.height);
        let size = vek::Vec2::<u32>::new(size.width, size.height) / self.args.downscale_factor;

        let size_f32 = size.map(|x| x as f32);

        

        let uniform_per_frame_data = pipeline::PerFrameUniformData {
            view_matrix: self.movement.view_matrix,
            projection_matrix: self.movement.proj_matrix,
            view_projection_matrix: self.movement.proj_matrix * self.movement.view_matrix,
            inv_view_matrix: self.movement.view_matrix.inverted(),
            inv_projection_matrix: self.movement.proj_matrix.inverted(),
            screen_resolution: size_f32,
            position: self.movement.position.with_w(0f32),
            forward: self.movement.forward().with_w(0f32),
            
            sun: self.sun.normalized().with_w(0f32),
            camera_frustum_planes: self.movement.camera_frustum_planes,
            debug_type: self.debug_type,
            time: elapsed,
            toggles_bitmask: self.toggles_bitmask,

            frame_count: self.frame_count,

            _padding: Default::default(),
            _padding2: Default::default(),
            
        };

        self.device.cmd_update_buffer(cmd, self.uniform_buffer.buffer, 0, bytemuck::bytes_of(&uniform_per_frame_data));


        let uniform_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.uniform_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let debug_text_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.debug_text.buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let materials_gpu_buffer = vk::BufferMemoryBarrier2::default()
            .buffer(self.materials_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let scene_repr_gpu_buffer = vk::BufferMemoryBarrier2::default()
            .buffer(self.scene_representation_for_sdf_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let buffer_memory_barriers = [uniform_buffer_barrier, debug_text_buffer_barrier, materials_gpu_buffer, scene_repr_gpu_buffer];
        let dep = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        self.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, query_pool, 0);

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_SKY][WRITE_CLOUDS_ENTRY_POINT],
        );

        self.device.cmd_dispatch(cmd, skybox::CLOUDS_RESOLUTION.div_ceil(8), skybox::CLOUDS_RESOLUTION.div_ceil(8), 1);

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_SKY][WRITE_SKYBOX_ENTRY_POINT]
        );

        self.device.cmd_dispatch(cmd, skybox::SKYBOX_RESOLUTION.div_ceil(8), skybox::SKYBOX_RESOLUTION.div_ceil(8), 6);

        let skybox_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(6);
        let clouds_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let skybox_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.skybox_image)
            .subresource_range(skybox_subresource_range);
        let clouds_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.clouds_image)
            .subresource_range(clouds_subresource_range);
        let image_memory_barriers = [skybox_image_barrier, clouds_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_SKY][BLUR_AMBIENT_SKYBOX_ENTRY_POINT]
        );

        self.device.cmd_dispatch(cmd, skybox::AMBIENT_SKYBOX_RESOLUTION, skybox::AMBIENT_SKYBOX_RESOLUTION, 6);
        
        
        let skybox_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(6);
        let clouds_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let skybox_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.skybox_image)
            .subresource_range(skybox_subresource_range);
        let clouds_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.clouds_image)
            .subresource_range(clouds_subresource_range);
        let ambient_clouds_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.ambient_skybox_image)
            .subresource_range(clouds_subresource_range);
        let rendered_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(render_targets.rendered_image)
            .subresource_range(subresource_range);
        let depth_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(render_targets.rendered_depth_image)
            .subresource_range(vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .layer_count(1));
        let image_memory_barriers = [skybox_image_barrier, clouds_image_barrier, rendered_image_barrier, ambient_clouds_image_barrier, depth_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_FULLSCREEN]["main"],
        );

        self.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, query_pool, 1);

        self.device.cmd_dispatch(cmd, size.x.div_ceil(8), size.y.div_ceil(8), 1);

        self.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, query_pool, 2);

        // transition rendered image from color attachment to sampled shader read (for bloom passes)
        let rendered_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(render_targets.rendered_image)
            .subresource_range(subresource_range);
        let full_passes_bloom = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(render_targets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let image_memory_barriers = [rendered_image_barrier, full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);   


        let image_memory_barriers = [full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        
        #[derive(Debug, Clone, Copy, Pod, Zeroable)]
        #[repr(C)]
        struct BloomPushConstantData {
            previous_bloom_size: vek::Vec2<f32>,
            src_sampled_img_idx: u32,
            dst_storage_img_idx: u32,
        }

        // execute bloom downsample passes
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_POST_PROCESS][BLOOM_DOWNSAMPLE_ENTRY_POINT],
        );

        // there is no need to go down to the largest mip since we will be sampling from a smaller mip anyways
        let minimum_upsampling_mip = 2; 

        for mip in 0..(render_targets.bloom_mip_image_views.len() as u32-1) {
            // no need to pipeline barrier for the first pass, as we just waited for the render texture image to finish right before this
            if mip > 0 {
                // wait on previous mip level to be done
                let previous_mip_level_subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(mip)
                    .level_count(1);
                let previous_mip_image_memory_barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(render_targets.bloom_image)
                    .subresource_range(previous_mip_level_subresource_range);
                let barriers = [previous_mip_image_memory_barrier];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device.cmd_pipeline_barrier2(cmd, &dep);
            }
            
            
            let previous_mip_size = size / (1 << (mip)); // larger mip
            let next_mip_size = size / (1 << (mip+1)); // smaller mip

            let downsample_dispatch_push_constants = BloomPushConstantData {
                previous_bloom_size: previous_mip_size.as_::<f32>(),
                src_sampled_img_idx: if mip == 0 { RENDERED_SAMPLER_IMAGE_IDX } else { mip + BLOOM_MIPS_SAMPLED_IMAGE_START_IDX },
                dst_storage_img_idx: mip + BLOOM_MIPS_STORAGE_IMAGE_START_IDX + 1,
            };

            //log::info!("{:?}", downsample_dispatch_push_constants);

            self.device.cmd_push_constants(cmd, self.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&downsample_dispatch_push_constants));
            self.device.cmd_dispatch(cmd, next_mip_size.x.div_ceil(8), next_mip_size.y.div_ceil(8), 1);
        }

        let full_passes_bloom = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(render_targets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let image_memory_barriers = [full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute bloom upsample passes
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_POST_PROCESS][BLOOM_UPSAMPLE_ENTRY_POINT],
        );

        for mip in (minimum_upsampling_mip..(render_targets.bloom_mip_image_views.len() as u32 - 1)).rev() {
            // no need to pipeline barrier for the very first pass (we did a full pipeline barrier for the entire bloom image right before this)
            if mip != render_targets.bloom_mip_image_views.len() as u32 - 2 {
                // wait on previous mip level to be done
                let previous_mip_level_subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(mip+1)
                    .level_count(1);
                let previous_mip_image_memory_barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(render_targets.bloom_image)
                    .subresource_range(previous_mip_level_subresource_range);
                let barriers = [previous_mip_image_memory_barrier];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device.cmd_pipeline_barrier2(cmd, &dep);
            }
            
            
            let previous_mip_size = size / (1 << (mip+1)); // smaller mip
            let next_mip_size = size / (1 << (mip)); // larger mip

            let upsample_dispatch_push_constants = BloomPushConstantData {
                previous_bloom_size: previous_mip_size.as_::<f32>(),
                src_sampled_img_idx: mip + BLOOM_MIPS_SAMPLED_IMAGE_START_IDX + 1,
                dst_storage_img_idx: mip + BLOOM_MIPS_STORAGE_IMAGE_START_IDX,
            };

            //log::info!("{:?}", upsample_dispatch_push_constants);

            self.device.cmd_push_constants(cmd, self.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&upsample_dispatch_push_constants));
            self.device.cmd_dispatch(cmd, next_mip_size.x.div_ceil(8), next_mip_size.y.div_ceil(8), 1);
        }

        let entire_bloom_image = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(render_targets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let swapchain_image_undefined_to_blit_dst_layout_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(swapchain_image)
            .subresource_range(subresource_range);
        let image_memory_barriers = [entire_bloom_image, swapchain_image_undefined_to_blit_dst_layout_transition];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);     

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_POST_PROCESS][WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT],
        );

        self.device.cmd_dispatch(cmd, window_size_no_downscale.x.div_ceil(8), window_size_no_downscale.y.div_ceil(8), 1);

        let blit_dst_to_present_layout_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(swapchain_image)
            .subresource_range(subresource_range);

        let image_memory_barriers = [blit_dst_to_present_layout_transition];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        self.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, query_pool, 3);
        self.device.end_command_buffer(cmd).unwrap();
        let now = Instant::now();
        self.last_frame_cpu_cmd_record_duration = now - cpu_cmd_record_start;

        let cmds = [cmd];
        let wait_masks = [vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::COMPUTE_SHADER];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(&render_finished_semaphore)
            .wait_dst_stage_mask(&wait_masks)
            .wait_semaphores(&present_complete_semaphores);

        self.device
            .queue_submit(self.queue, &[submit_info], end_fence)
            .unwrap();

        let swapchains = [self.swapchain];
        let indices = [acquired_swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(&render_finished_semaphore);

        let suboptimal = self.swapchain_loader
            .queue_present(self.queue, &present_info)
            .unwrap();

        self.frame_count += 1;
        if suboptimal {
            self.recreate_swapchain();
        }
    }

    pub unsafe fn destroy(mut self) {
        self.device.device_wait_idle().unwrap();

        self.texture.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed sdf texture");

        self.materials_buffer.destroy(&self.device, &mut self.allocator);
        for material in self.materials {
            material.destroy(&self.device, &mut self.allocator);
        }
        
        
        for (_, graphic_pipeline) in self.graphics_pipelines {
            graphic_pipeline.destroy(&self.device);
        }
        log::info!("destroyed graphic pipelines");

        for (_, compute_pipeline) in self.compute_pipelines {
            compute_pipeline.destroy(&self.device);
        }
        log::info!("destroyed compute pipelines");
                
        self.skybox.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed skybox");

        self.uniform_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed per frame uniform buffer");    
        
        self.debug_text.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed debug text buffer");

        for x in self.blases {
            x.destroy(&self.acceleration_structure_device, &self.device, &mut self.allocator);
        }
        log::info!("destroyed BLASes");

        self.tlas.destroy(&self.acceleration_structure_device, &self.device, &mut self.allocator);
        log::info!("destroyed TLAS");

        self.scene_representation_for_sdf_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed gpu repr");

        log::info!("waiting for all frame in flight fences...");
        let fences = self.frames_in_flight.iter().map(|x| x.end_fence).collect::<Vec<_>>();
        self.device
            .wait_for_fences(&fences, true, u64::MAX)
            .unwrap();
        for frame in self.frames_in_flight.into_iter() {
            frame.destroy_everything(&self.device, self.pool, &mut self.allocator);
        }
        for sem in self.render_finished_semaphores {
            self.device.destroy_semaphore(sem, None);
        }

        self.render_targets_data.destroy_rt_images_and_image_views(&self.device, &mut self.allocator);
        log::info!("destroyed const descriptor sets");

        for swapchain_image_view in self.swapchain_image_views {
            self.device.destroy_image_view(swapchain_image_view, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        log::info!("destroyed swapchain");


        self.surface_loader.destroy_surface(self.surface_khr, None);
        log::info!("destroyed surface");

        self.samplers.destroy_samplers(&self.device);
        log::info!("destroyed samplers");

        self.device.destroy_command_pool(self.pool, None);
        log::info!("destroyed cmd pool");
        
        self.device.destroy_descriptor_set_layout(self.main_descriptor_set_layout, None);
        log::info!("destroyed bindless descriptor set layout");
        
        self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        log::info!("destroyed descriptor pool");

        self.device.destroy_pipeline_layout(self.main_pipeline_layout, None);
        log::info!("destroyed bindless pipeline layout");
        

        drop(self.allocator);
        self.device.destroy_device(None);
        log::info!("destroyed device");

        if let Some((inst, debug_messenger)) = self.debug {
            inst.destroy_debug_utils_messenger(debug_messenger, None);
            log::info!("destroyed debug utils messenger");
        }

        self.instance.destroy_instance(None);
        log::info!("destroyed instance");

        drop(self.entry); // DO NOT REMOVE ENTRY FROM STRUCT. NEEDED!!!
        log::info!("everything is done!");
    }
}
