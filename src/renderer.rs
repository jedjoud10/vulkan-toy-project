use ash::vk;
use bytemuck::Pod;
use bytemuck::Zeroable;
use bytesize::ByteSize;
use include_dir::Dir;
use include_dir::include_dir;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use smallvec::SmallVec;
use crate::input::Button;
use crate::input::Input;
use crate::material::Material;
use crate::model;
use crate::movement::Movement;
use crate::per_frame_data;
use crate::physics;
use crate::ray_tracing;
use crate::samplers;
use crate::tesselation;
use crate::voxel;
use winit::event::MouseButton;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::time::Duration;
use std::time::Instant;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::Window;
use crate::statistics::Statistics;

use crate::swapchain;
use crate::ticker;
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

const COMPUTE_POST_PROCESS_SPV: &'static str = "compute_post_process.spv";
const BLOOM_UPSAMPLE_ENTRY_POINT: &'static str = "bloom_upsample";
const BLOOM_DOWNSAMPLE_ENTRY_POINT: &'static str = "bloom_downsample";
const WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT: &'static str = "write_swapchain_image";
const COMPUTE_SKY_SPV: &str = "compute_sky.spv";
const WRITE_CLOUDS_ENTRY_POINT: &str = "write_clouds";
const WRITE_SKYBOX_ENTRY_POINT: &str = "write_skybox";
const BLUR_AMBIENT_SKYBOX_ENTRY_POINT: &str = "blur_skybox_ambient";

const VOXELIZE_SURFACE_ENTRY_POINT: &str = "voxelize_surface";
const CALCULATE_DENSITY_ENTRY_POINT: &str = "calculate_density";

const RASTERIZED_SPV: &str = "rasterized.spv";
const RASTERIZED_CHUNK_SPV: &str = "rasterized_chunk.spv";
const COMPUTE_SURFACE_SPV: &str = "compute_surface.spv";
const RASTERIZED_MS_PASSTHROUGH_SPV: &str = "rasterized_ms_passthrough.spv";
const RASTERIZED_MS_TESSELALTION_SPV: &str = "rasterized_ms_tesselation.spv";
const RASTERIZED_MS_GENERATED_1_SPV: &str = "rasterized_ms_generated_1.spv";
const RASTERIZED_MS_GENERATED_GRASS_SPV: &str = "rasterized_ms_generated_grass.spv";
const RASTERIZED_BACKGROUND_SPV: &str = "rasterized_background.spv";
        

const VERTEX_ATTRIBUTE_DESCRIPTIONS: &'static [vk::VertexInputAttributeDescription] = &[vk::VertexInputAttributeDescription {
    binding: 0,
    format: vk::Format::R32G32B32_SFLOAT,
    location: 0,
    offset: 0
}, vk::VertexInputAttributeDescription {
    binding: 1,
    format: vk::Format::R32G32B32_SFLOAT,
    location: 1,
    offset: 0
}, vk::VertexInputAttributeDescription {
    binding: 2,
    format: vk::Format::R32G32_SFLOAT,
    location: 2,
    offset: 0
}];
const VERTEX_BINDING_DESCRIPTIONS: &'static [vk::VertexInputBindingDescription] = &[vk::VertexInputBindingDescription {
    binding: 0,
    stride: size_of::<vek::Vec3<f32>>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
}, vk::VertexInputBindingDescription {
    binding: 1,
    stride: size_of::<vek::Vec3<f32>>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
}, vk::VertexInputBindingDescription {
    binding: 2,
    stride: size_of::<vek::Vec2<f32>>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
}];



const VERTEX_ATTRIBUTE_DESCRIPTIONS_CHUNK: &'static [vk::VertexInputAttributeDescription] = &[vk::VertexInputAttributeDescription {
    binding: 0,
    format: vk::Format::R32G32B32_SFLOAT,
    location: 0,
    offset: 0
}];
const VERTEX_BINDING_DESCRIPTIONS_CHUNK: &'static [vk::VertexInputBindingDescription] = &[vk::VertexInputBindingDescription {
    binding: 0,
    stride: 4 * 3 as u32,
    input_rate: vk::VertexInputRate::VERTEX,
}];

pub struct GraphicsContext<'a> {
    pub device: &'a ash::Device,
    pub pool: vk::CommandPool,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub mesh_shader_device: &'a ash::ext::mesh_shader::Device,
    pub extended_dynamic_state3_device: &'a ash::ext::extended_dynamic_state3::Device,
    pub acceleration_structure_device: &'a ash::khr::acceleration_structure::Device,
    pub host_image_copy_device: &'a ash::ext::host_image_copy::Device,
    pub allocator: &'a mut gpu_allocator::vulkan::Allocator,
    pub debug_marker: &'a debug::DebugMarker,
    pub main_descriptor_set_layout: vk::DescriptorSetLayout,
    pub main_pipeline_layout: vk::PipelineLayout,
    pub descriptor_pool: vk::DescriptorPool,
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

    // materials
    materials: Vec<Material>,
    
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
    const_descriptor_sets: RenderTargetsData,
            
    // important too
    allocator: gpu_allocator::vulkan::Allocator,
    
    // other GPU stuff
    multiple_chunks: voxel::MultipleChunks,
    chunks: Vec<voxel::Chunk>,
    models: Vec<model::Model>,
    tlas: ray_tracing::TopLevelAccelerationStructure,
    static_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    dynamic_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    
    timestamp_period: f32,
    skybox: skybox::Skybox,
    lights_buffer: buffer::Buffer,
    lights: Vec<vek::Vec4<f32>>,
    velocities: Vec<vek::Vec4<f32>>,
    samplers: samplers::Samplers,
    tesselation_buffer: buffer::Buffer,
    uniform_buffer: buffer::Buffer,

    // debug settings
    debug_type: u32,
    wireframe: bool,
    toggles_bitmask: u32,
    debug_text_buffer: buffer::Buffer,

    // other CPU stuff
    physics: physics::Physics,
    pub was_resized: bool,
    pub window: Window,
    pub input: Input,    
    last_frame_cpu_cmd_record_duration: Duration,
    movement: Movement,
    frame_count: u64,
    ticker: ticker::Ticker,
    sun: vek::Vec3<f32>,
    args: crate::Args,
    stats: Statistics,
}

static COMPILED_SHADERS: Dir = include_dir!("$CARGO_MANIFEST_DIR/compiled_shaders");

impl InternalApp {
    pub unsafe fn new(event_loop: &ActiveEventLoop, args: crate::Args) -> Self {
        let mut assets = HashMap::<&str, Vec<u32>>::new();
        for file in COMPILED_SHADERS.files() {
            let len = file.contents().len();
            assert!(len.is_multiple_of(4));

            let mut vec = vec![0u32; len / 4];
            let dst_slice = bytemuck::cast_slice_mut::<u32, u8>(vec.as_mut_slice());

            dst_slice.copy_from_slice(file.contents());

            let file_name = file.path().file_name().unwrap().to_str().unwrap();
            log::debug!("added shader '{file_name}' to assets");
            assets.insert(file_name, vec);
        }

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

        let debug_stuff = args.enable_debug_stuff || running_cfg_debug_assertions;
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
            .map(|physical_device| {
                let score = physical_device::get_physical_device_score(
                    physical_device,
                    &instance,
                    &surface_loader,
                    surface_khr,
                );
                (physical_device, score)
            })
            .filter_map(|(a, b)| b.map(|val| (a, val)))
            .collect::<Vec<(vk::PhysicalDevice, u32)>>();
        physical_device_candidates.sort_by_key(|(_, score)| *score);

        if physical_device_candidates.is_empty() {
            log::error!("no physical device was chosen!");
            panic!();
        }

        let physical_device = physical_device_candidates.last().unwrap().0;
        let mut physical_device_properties = vk::PhysicalDeviceProperties2::default();
        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);
        let physical_device_name = physical_device_properties.properties.device_name_as_c_str().unwrap().to_str().unwrap();

        log::info!("selected physical device \"{}\"", physical_device_name);

        let (device, queue_family_index, queue) = device::create_device_and_queue(
            &instance,
            physical_device,
            &surface_loader,
            surface_khr,
        );
        log::info!("created device and fetched main queue");

        let mesh_shader_device = ash::ext::mesh_shader::Device::new(&instance, &device);
        let extended_dynamic_state3_device = ash::ext::extended_dynamic_state3::Device::new(&instance, &device);
        let acceleration_structure_device = ash::khr::acceleration_structure::Device::new(&instance, &device);
        let host_image_copy_device = ash::ext::host_image_copy::Device::new(&instance, &device);

        let debug_marker = debug_messenger.is_some().then(|| {
            let device = debug::create_debug_marker(&instance, &device);
            log::info!("created debug marker object names binder");
            device
        });

        let mut allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings {
                    log_leaks_on_shutdown: true,
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

        let settings = [pipeline::PipelineCreateSettings {
            pipeline_debug_name: "post process compute pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &[WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT, BLOOM_DOWNSAMPLE_ENTRY_POINT, BLOOM_UPSAMPLE_ENTRY_POINT] },
            spec_constants: Some(&[args.downscale_factor]),
            spv_file_name: COMPUTE_POST_PROCESS_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "sky compute pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &[WRITE_SKYBOX_ENTRY_POINT, WRITE_CLOUDS_ENTRY_POINT, BLUR_AMBIENT_SKYBOX_ENTRY_POINT] },
            spec_constants: Some(&[skybox::SKYBOX_RESOLUTION, skybox::CLOUDS_RESOLUTION, skybox::AMBIENT_SKYBOX_RESOLUTION]),
            spv_file_name: COMPUTE_SKY_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "main render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::GraphicsMeshShader { face_culling: true, task_shader: false },
            spec_constants: None,
            spv_file_name: RASTERIZED_MS_PASSTHROUGH_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "tesselation render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::GraphicsMeshShader { face_culling: true, task_shader: true },
            spec_constants: None,
            spv_file_name: RASTERIZED_MS_TESSELALTION_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "background sky pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Graphics { face_culling: false, vertex_input: vk::PipelineVertexInputStateCreateInfo::default() },
            spec_constants: None,
            spv_file_name: RASTERIZED_BACKGROUND_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "generated render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::GraphicsMeshShader { face_culling: true, task_shader: true },
            spec_constants: None,
            spv_file_name: RASTERIZED_MS_GENERATED_1_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "grass render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::GraphicsMeshShader { face_culling: false, task_shader: true },
            spec_constants: None,
            spv_file_name: RASTERIZED_MS_GENERATED_GRASS_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "rasterized render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Graphics {
                face_culling: true,
                vertex_input: vk::PipelineVertexInputStateCreateInfo::default().vertex_attribute_descriptions(VERTEX_ATTRIBUTE_DESCRIPTIONS).vertex_binding_descriptions(VERTEX_BINDING_DESCRIPTIONS),
            },
            spec_constants: None,
            spv_file_name: RASTERIZED_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "rasterized chunk render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Graphics {
                face_culling: true,
                vertex_input: vk::PipelineVertexInputStateCreateInfo::default().vertex_attribute_descriptions(VERTEX_ATTRIBUTE_DESCRIPTIONS_CHUNK).vertex_binding_descriptions(VERTEX_BINDING_DESCRIPTIONS_CHUNK),
            },
            spec_constants: None,
            spv_file_name: RASTERIZED_CHUNK_SPV,
        }, pipeline::PipelineCreateSettings {
            pipeline_debug_name: "compute surface shader",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &[CALCULATE_DENSITY_ENTRY_POINT, VOXELIZE_SURFACE_ENTRY_POINT] },
            spec_constants: None,
            spv_file_name: COMPUTE_SURFACE_SPV,
        }];

        // compile the pipelines in parallel
        // ouug shii :eyes:
        log::info!("creating pipelines...");
        let generic_pipelines = settings.into_par_iter().map(|setting| {
            let spv_file_name = setting.spv_file_name;
            let raw = &assets[spv_file_name];
            let pipeline = pipeline::create_generic_pipeline(raw, &device, &debug_marker, main_pipeline_layout, setting);
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

        let skybox = skybox::create_skybox(
            &device,
            &mut allocator,
            &debug_marker,
            queue,
            pool,
            queue_family_index
        );
        log::info!("created skybox");

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

        let frames_in_flight = (0..per_frame_data::FRAMES_IN_FLIGHT).into_iter().map(|_| {
            PerFrameData::create_per_frame_data(&mut ctx)
        }).collect::<SmallVec<[PerFrameData; per_frame_data::FRAMES_IN_FLIGHT]>>();
        log::info!("created frames in flight structures");

        const NUM_LIGHTS: usize = 1;

        let lights_buffer = buffer::create_buffer(&mut ctx, size_of::<vek::Vec4<f32>>() * NUM_LIGHTS, "lights buffer", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let mut lights = Vec::<vek::Vec4<f32>>::new();

        for _i in 0..NUM_LIGHTS {
            let x = rand::random_range(-10f32..10f32);
            let y = rand::random_range(-10f32..10f32);
            let z = rand::random_range(-10f32..10f32);
            lights.push(vek::Vec4::new(x,y,z, 1.0));
        }

        let velocities = vec![vek::Vec4::<f32>::zero(); NUM_LIGHTS];

        buffer::write_to_buffer(&mut ctx, lights_buffer.buffer, bytemuck::cast_slice(lights.as_slice()));
        log::info!("created lights buffer");

        let mut const_descriptor_sets = RenderTargetsData::create_constant_descriptor_sets();
        const_descriptor_sets.recreate_rt_images_and_image_views_and_update_descriptor_sets(&mut ctx, extent, args.downscale_factor);
        crate::render_targets_data::transfer_layout_for_images(&device, queue_family_index, &const_descriptor_sets, pool, queue);
        log::info!("created constant descriptor sets");

        let timestamp_period = physical_device_properties.properties.limits.timestamp_period;

        let mut chunks = Vec::<voxel::Chunk>::new();
        let chunk_render_distance = 1;
        let mut index = 0;

        for x in -chunk_render_distance..=chunk_render_distance {
            for y in -1..1 {
                for z in -chunk_render_distance..=chunk_render_distance {
                    let chunk_offset = vek::Vec3::new(x, y, z);
                    chunks.push(voxel::Chunk {
                        chunk_index: index,
                        chunk_offset,
                        built: false,
                        vertex_buffer_start_offset: (voxel::VERTICES_PER_CHUNK * index),
                        index_buffer_start_offset: (3 * voxel::TRIANGLES_PER_CHUNK * index),
                        accel_structure: None,
                    });
                    index += 1;
                }
            }
        }

        let multiple_chunks = voxel::MultipleChunks::create(&mut ctx, index);
        
        let models = vec![
            //model::Model::new(vek::Vec3::new(0f32, 20f32, 0f32), include_bytes!("../models/sphere.obj"), &mut ctx),
            //model::Model::new(vek::Vec3::new(10f32, 20f32, 0f32), include_bytes!("../models/not_so_sphere.obj"), &mut ctx),
            //model::Model::new(vek::Vec3::new(-10f32, 20f32, 0f32), include_bytes!("../models/modular_industrial_pipes_01_1k.obj"), &mut ctx),
            //model::Model::new(vek::Vec3::new(-30f32, 20f32, 0f32), include_bytes!("../models/namaqualand_boulder_02_1k.obj"), &mut ctx),
        ];

        let tlas = ray_tracing::pre_create_tlas(&mut ctx);

        let tesselation_buffer = tesselation::precompute_tesselation_buffer(&mut ctx);
        let debug_text_buffer = buffer::create_buffer(&mut ctx, 1024, "debug text", vk::BufferUsageFlags::STORAGE_BUFFER);

        let render_finished_semaphores: SmallVec<[vk::Semaphore; swapchain::SWAPCHAIN_IMAGES]> = (0..swapchain::SWAPCHAIN_IMAGES).into_iter().map(|_| {
            device.create_semaphore(&Default::default(), None).unwrap()
        }).collect::<SmallVec<[vk::Semaphore; swapchain::SWAPCHAIN_IMAGES]>>();

        let uniform_buffer = buffer::create_buffer(
            &mut ctx,
            size_of::<pipeline::PerFrameUniformData>(),
            "per frame uniform buffer",
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
        );

        let materials = vec![
            Material::new(&mut ctx)
        ];

        let physics = physics::Physics {
            objects: vec![
                /*
                physics::DynamicObject {
                    vertices: vec![
                        physics::Vertex { inv_mass: 0f32, position: vek::Vec3::new(0f32, 20f32, 0f32), velocity: vek::Vec3::default() },
                        physics::Vertex { inv_mass: 1f32, position: vek::Vec3::new(1f32, 16f32, -1f32), velocity: vek::Vec3::default() },
                        physics::Vertex { inv_mass: 1f32, position: vek::Vec3::new(2f32, 14f32, 0f32), velocity: vek::Vec3::default() },
                        physics::Vertex { inv_mass: 1f32, position: vek::Vec3::new(4f32, 10f32, 2f32), velocity: vek::Vec3::default() },        
                    ],
                    constraints: vec![
                        physics::Constraint { cardinality: 2, function: Box::new(|data: &[vek::Vec3<f32>]| -> f32 {
                            (data[0] - data[1]).magnitude() - 2f32
                        }), indices: Vec::from([1, 0]), stiffness: 1.0f32, mode: physics::Mode::Equality },
                        physics::Constraint { cardinality: 2, function: Box::new(|data: &[vek::Vec3<f32>]| -> f32 {
                            (data[0] - data[1]).magnitude() - 2f32
                        }), indices: Vec::from([1, 2]), stiffness: 1.0f32, mode: physics::Mode::Equality },
                        physics::Constraint { cardinality: 2, function: Box::new(|data: &[vek::Vec3<f32>]| -> f32 {
                            (data[0] - data[1]).magnitude() - 2f32
                        }), indices: Vec::from([3, 2]), stiffness: 0.1f32, mode: physics::Mode::Equality },
                    ]
                }
                */
            ]
        };

        Self {
            multiple_chunks,
            last_frame_cpu_cmd_record_duration: Default::default(),
            frame_count: 0,
            input: Default::default(),
            movement: Movement::new(),
            window,
            instance,
            entry,
            device,
            physical_device,
            lights_buffer,
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
            const_descriptor_sets,
            descriptor_pool,
            timestamp_period,
            allocator,
            skybox,
            was_resized: false,
            frames_in_flight,
            ticker: ticker::Ticker { accumulator: 0f32, count: 0 },
            sun: vek::Vec3::new(1f32, 0.3f32,0.5f32).normalized(),
            debug_type: 0,
            stats: Default::default(),
            args,
            lights,
            samplers,
            swapchain_images,
            swapchain_image_views,
            main_descriptor_set_layout,
            main_pipeline_layout,
            mesh_shader_device,
            extended_dynamic_state3_device,
            tesselation_buffer,
            graphics_pipelines,
            compute_pipelines,
            velocities,
            wireframe: false,
            toggles_bitmask: 0,
            debug_text_buffer,
            render_finished_semaphores,
            uniform_buffer,
            chunks,
            acceleration_structure_device,
            models,
            static_instances: Vec::new(),
            dynamic_instances: Vec::new(),
            tlas,
            materials,
            host_image_copy_device,
            physics,
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

        let new_model = model::Model::new(position, include_bytes!("../models/sphere.obj"), &mut ctx);
        self.models.push(new_model);

        self.physics.objects.push(physics::DynamicObject { vertices: vec![physics::Vertex { inv_mass: if add { 1f32 } else { 0f32 }, position, velocity: vek::Vec3::zero() }], constraints: vec![] })
        /*
        let vcount = self.physics.objects[0].vertices.len();
        self.physics.objects[0].vertices.push(physics::Vertex { inv_mass: if add { 1f32 } else { 0f32 }, position, velocity: vek::Vec3::zero() });

        self.physics.objects[0].constraints.push(physics::Constraint { cardinality: 2, function: Box::new(|data: &[vek::Vec3<f32>]| -> f32 {
            (data[0] - data[1]).magnitude() - 2f32
        }), indices: Vec::from([vcount, vcount-1]), stiffness: 1.0f32, mode: physics::Mode::Equality });
        */
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

        self.const_descriptor_sets.destroy_rt_images_and_image_views(&self.device, &mut self.allocator);

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

        self.const_descriptor_sets.recreate_rt_images_and_image_views_and_update_descriptor_sets(&mut ctx, extent, self.args.downscale_factor);
        crate::render_targets_data::transfer_layout_for_images(&self.device, self.queue_family_index, &self.const_descriptor_sets, self.pool, self.queue);


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
        if self.input.get_button(Button::Keyboard(KeyCode::KeyP)).pressed() {
            let render_time_avg = self.stats.get_average_in_ms();
            let delta_ms = delta * 1000f32;
            log::info!("CPU delta: {delta_ms:.3}, Main Compute Render Time Average: {render_time_avg:.3}");
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyL)).pressed() {
            self.stats.start_benchmarking(self.frame_count);
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
        if self.input.get_button(Button::Keyboard(KeyCode::Digit1)).pressed() {
            self.toggles_bitmask ^= 1;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::Digit2)).pressed() {
            self.toggles_bitmask ^= 2;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::Digit3)).pressed() {
            self.toggles_bitmask ^= 4;
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
        if self.ticker.update(delta) {
            self.physics.tick();
        }
        let frame_in_flight_index = self.frame_count % (self.frames_in_flight.len() as u64);
        let const_data = &self.const_descriptor_sets;
        let &mut PerFrameData {
            present_complete_semaphore,
            end_fence,
            cmd,
            main_descriptor_set,
            query_pool,
            pipeline_statistics_query_pool,
            ref mut scratch_buffer,
            ..
        } = &mut self.frames_in_flight[frame_in_flight_index as usize];
        

        let present_complete_semaphores = [present_complete_semaphore];

        let pre_wait_for_fence = Instant::now();
        if let Err(err) = self.device.wait_for_fences(&[end_fence], true, u64::MAX) {
            log::error!("wait on fence err: {:?}", err);
            // return;
        } else {
            // wait for a few frames so that the queries get populated
            if self.frame_count > per_frame_data::FRAMES_IN_FLIGHT as u64 {
                // try to fetch timestamp queries
                let mut timestamps = [0u64; 2];
                let okay = self.device.get_query_pool_results(query_pool, 0, &mut timestamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).is_ok();
                if okay {
                    let delta_in_ms = ((timestamps[1].saturating_sub(timestamps[0])) as f64 * self.timestamp_period as f64) / 1000000.0f64;
                    self.stats.push_query_timings(delta_in_ms);
                }
            }


            /*
            // try to fetch pipeline statistics queriy
            let mut data = [0u64; 1];
            let okay = self.device.get_query_pool_results(self.pipeline_statistics_query_pool, 0, &mut data, vk::QueryResultFlags::TYPE_64).is_ok();
            if okay {
                dbg!(data);
            }
            */
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

        let num_chunks = self.chunks.len() as u64;
        let chunk = &mut self.chunks[(self.frame_count % (num_chunks)) as usize];
        let voxel::Chunk {
            chunk_index,
            chunk_offset,
            built,
            vertex_buffer_start_offset,
            index_buffer_start_offset,
            accel_structure,
        } = chunk;
        let voxel::MultipleChunks {
            voxel_texture,
            vertex_buffer,
            index_buffer,
            vertex_counter,
            index_counter,
            ..
        } = &mut self.multiple_chunks;

        // create bindless descriptor write for storage images        
        let swapchain_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(swapchain_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let rendered_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(const_data.rendered_image_view)
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
            .image_view(voxel_texture.image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut storage_image_infos = vec![swapchain_image_view_descriptor_image_info, rendered_image_view_descriptor_image_info, skybox_image_view_descriptor_image_info, ambient_skybox_image_view_descriptor_image_info, clouds_image_view_descriptor_image_info, voxel_image_view_descriptor_image_info];
        
        // add bloom storage image views
        for bloom_storage_image_view in const_data.bloom_mip_image_views.iter() {
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
        let storage_buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(self.uniform_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(vertex_buffer.buffer)
                .offset((*vertex_buffer_start_offset * voxel::VERTEX_STRIDE) as u64)
                .range((voxel::VERTICES_PER_CHUNK * voxel::VERTEX_STRIDE) as u64),
            vk::DescriptorBufferInfo::default()
                .buffer(index_buffer.buffer)
                .offset((*index_buffer_start_offset * voxel::INDEX_STRIDE) as u64)
                .range((3 * voxel::TRIANGLES_PER_CHUNK * voxel::INDEX_STRIDE) as u64),
            vk::DescriptorBufferInfo::default()
                .buffer(self.tesselation_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(self.lights_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(self.debug_text_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(vertex_counter.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(index_counter.buffer)
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
            .image_view(const_data.rendered_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let entire_bloom_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(const_data.entire_bloom_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut sampled_image_infos = vec![skybox_sampled_image_view_descriptor_image_info, ambient_skybox_sampled_image_view_descriptor_image_info, clouds_sampled_image_view_descriptor_image_info, rendered_sampled_image_view_descriptor_image_info, entire_bloom_sampled_image_view_descriptor_image_info];

        // add bloom sampled image views
        for bloom_sampled_image_view in const_data.bloom_mip_image_views.iter() {
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
                .sampler(self.samplers.linear)
        ];
        let sampler_states_write = vk::WriteDescriptorSet::default()
            .descriptor_count(samplers.len() as u32)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .dst_binding(3)
            .dst_set(main_descriptor_set)
            .image_info(&samplers);

        if let Some(tlas) = self.tlas.data.as_ref() {
            let wuh = [tlas.acceleration_structure];

            let mut acceleration_structure_write_tmp = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&wuh);

            let acceleration_structure_write = vk::WriteDescriptorSet::default()
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .dst_set(main_descriptor_set)
                .dst_binding(4)
                .push_next(&mut acceleration_structure_write_tmp);

            self.device.update_descriptor_sets(&[storage_image_write, storage_buffer_write, sampled_image_write, sampler_states_write, acceleration_structure_write], &[]);
        } else {
            self.device.update_descriptor_sets(&[storage_image_write, storage_buffer_write, sampled_image_write, sampler_states_write], &[]);
        }

        //self.device.update_descriptor_sets(&[storage_image_write, storage_buffer_write, sampled_image_write, sampler_states_write], &[]);

        // TODO: ideally, these would:
        // 1. be dynamically allocated using some sort of per-frame arena with indexing
        // 2. be passed to the shader either using a uniform buffer (since these are constant anyways)
        const BLOOM_MIPS_STORAGE_IMAGE_START_IDX: u32 = 6; // bloom needs to be last since it is dynamically allocated (can have a dynamic number of bloom mips, depending on screen res)
        const RENDERED_SAMPLER_IMAGE_IDX: u32 = 3;
        const BLOOM_MIPS_SAMPLED_IMAGE_START_IDX: u32 = 5; // bloom needs to be last since it is dynamically allocated (can have a dynamic number of bloom mips, depending on screen res)

        scratch_buffer.bytes_written = 0;
        let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device
            .begin_command_buffer(cmd, &cmd_buffer_begin_info)
            .unwrap();
        let cpu_cmd_record_start = Instant::now();
        self.device.cmd_reset_query_pool(cmd, query_pool, 0, 2);
        self.device.cmd_reset_query_pool(cmd, pipeline_statistics_query_pool, 0, 1);
        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, query_pool, 0);

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let mut text = String::new();
        text += &format!("CPU delta: {:.2}ms\n", delta*1000f32);
        text += &format!("CPU command buffer record duration: {:.2}ms\n", self.last_frame_cpu_cmd_record_duration.as_micros() as f32 / 1000.0f32);
        text += &format!("CPU fence wait duration: {:.2}ms\n", (post_wait_for_fence - pre_wait_for_fence).as_micros() as f32 / 1000.0f32);
        text += &format!("CPU fence acquire swapchain duration: {:.2}ms\n", (post_acquire_swapchain - pre_acquire_swapchain).as_micros() as f32 / 1000.0f32);
        text += &format!("GPU main frame: {:.2}ms\n", self.stats.get_average_in_ms());
        text += &format!("pos: {:.2}\n", self.movement.position);
        text += &format!("debug type: {}\n", self.debug_type);
        text += &format!("toggles bitmask: {:#032b}\n", self.toggles_bitmask);
        text += &format!("wireframe: {}\n", self.wireframe);
        text += &format!("updating frustum: {}\n", self.movement.update_frustum);

        let report = self.allocator.generate_report();
        let reserved_bytes = ByteSize::b(report.total_reserved_bytes).display().iec();
        let allocated_bytes = ByteSize::b(report.total_reserved_bytes).display().iec();
        text += &format!("reserved bytes: {}\n", reserved_bytes);
        text += &format!("allocated bytes: {}\n", allocated_bytes);
        
        
        #[derive(Clone, Copy, Pod, Zeroable)]
        #[repr(C)]
        struct DebugTextLineHeader {
            start_byte_offset: u32,
            char_count: u32,
        }
        
        // write total number of lines
        let total_num_lines = text.lines().count() as u32;
        let mut bytes = bytemuck::bytes_of(&total_num_lines).to_vec();

        // write headers for each line
        let mut prefix_sum_chars_only = 0u32;
        for line in text.lines() {
            // calculate the total size in bytes prior to the actual text data
            let mut total_size_prior = total_num_lines * size_of::<DebugTextLineHeader>() as u32;

            // plus also the u32 to indicate the line count
            total_size_prior += size_of::<u32>() as u32;

            let header_for_line = DebugTextLineHeader {
                start_byte_offset: total_size_prior + prefix_sum_chars_only, 
                char_count: line.as_bytes().len() as u32,
            };
            
            bytes.extend_from_slice(bytemuck::bytes_of(&header_for_line));
            prefix_sum_chars_only += line.as_bytes().len() as u32;
        }

        for line in text.lines() {
            bytes.extend_from_slice(line.as_bytes());
        }

        // wtf
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        self.device.cmd_update_buffer(cmd, self.debug_text_buffer.buffer, 0, &bytes);

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

            _padding: Default::default(),
        };

        self.device.cmd_update_buffer(cmd, self.uniform_buffer.buffer, 0, bytemuck::bytes_of(&uniform_per_frame_data));


        let uniform_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.uniform_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let debug_text_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.debug_text_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let buffer_memory_barriers = [uniform_buffer_barrier, debug_text_buffer_barrier];
        let dep = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        


        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_SKY_SPV][WRITE_CLOUDS_ENTRY_POINT],
        );

        self.device.cmd_dispatch(cmd, skybox::CLOUDS_RESOLUTION.div_ceil(8), skybox::CLOUDS_RESOLUTION.div_ceil(8), 1);

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_SKY_SPV][WRITE_SKYBOX_ENTRY_POINT]
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
            self.compute_pipelines[COMPUTE_SKY_SPV][BLUR_AMBIENT_SKYBOX_ENTRY_POINT]
        );

        self.device.cmd_dispatch(cmd, skybox::AMBIENT_SKYBOX_RESOLUTION, skybox::AMBIENT_SKYBOX_RESOLUTION, 6);

        if !*built {
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

            self.multiple_chunks.do_sum_shi(
                *chunk_index,
                *chunk_offset,
                &self.device,
                cmd, self.main_pipeline_layout,
                self.compute_pipelines[COMPUTE_SURFACE_SPV][CALCULATE_DENSITY_ENTRY_POINT],
                self.compute_pipelines[COMPUTE_SURFACE_SPV][VOXELIZE_SURFACE_ENTRY_POINT],
                self.queue_family_index
            );
            let (data, instance) = self.multiple_chunks.create_blas(&mut ctx, *chunk_index, cmd);
            accel_structure.replace(data);
            *built = true;
            self.static_instances.push(instance);
        }

        // update dynamic instances for TLAS
        self.dynamic_instances.clear();
        for (idx, model) in self.models.iter_mut().enumerate() {
            model.update(elapsed, self.physics.objects[idx].vertices[0].position, &self.movement);
            //model.update(elapsed, self.physics.objects[0].vertices[idx].position, &self.movement);
            self.dynamic_instances.push(model.instance);
        }

        // rebuild TLAS
        ray_tracing::rebuild_tlas(
            &self.static_instances,
            &self.dynamic_instances,
            &self.tlas,
            cmd,
            &self.acceleration_structure_device,
            &self.device,
            &mut self.allocator,
            &self.debug_marker,
            self.queue_family_index,
            scratch_buffer
        );
        
        
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
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(const_data.rendered_image)
            .subresource_range(subresource_range);
        let image_memory_barriers = [skybox_image_barrier, clouds_image_barrier, rendered_image_barrier, ambient_clouds_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(vk::Extent2D::default().width(size.x).height(size.y));
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0f32; 4] } })
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image_view(self.const_descriptor_sets.rendered_image_view);
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1f32, stencil: 0 } })
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image_view(self.const_descriptor_sets.rendered_depth_image_image_view);
        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment)
            .layer_count(1)
            .render_area(render_area)
            .view_mask(0);

        
        self.device.cmd_begin_query(cmd, pipeline_statistics_query_pool, 0, vk::QueryControlFlags::empty());

        let viewport = vk::Viewport::default().height(size.y as f32).width(size.x as f32).x(0f32).y(0f32).min_depth(0f32).max_depth(1f32);
        self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        self.device.cmd_set_scissor(cmd, 0, &[render_area]);
        
        self.device.cmd_begin_rendering(cmd, &rendering_info);

        // render background skybox and clouds
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_BACKGROUND_SPV]);
        self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, vk::PolygonMode::FILL);
        self.device.cmd_draw(cmd, 6, 1, 0, 0);

        /*
        // render objs using mesh shaders (passthrough) 
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_MS_PASSTHROUGH_SPV]);
        self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, if self.wireframe { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });
        let triangle_count = self.index_count / 3;
        self.device.cmd_push_constants(cmd, self.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&triangle_count));
        self.mesh_shader_device.cmd_draw_mesh_tasks(cmd,  triangle_count.div_ceil(32), 1, 1);
        */

        // render objs (tesselated)
        /*
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_MS_TESSELALTION_SPV]);
        self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, if self.wireframe { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });
        let triangle_count = self.index_count / 3;
        self.mesh_shader_device.cmd_draw_mesh_tasks(cmd,  triangle_count, 1, 1);
        */

        // render other objs
        /*
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_MS_GENERATED_1_SPV]);
        self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, if self.wireframe { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });
        for x in -1..1i32 {
            for y in -0..1i32 {
                for z in -1..1i32 {
                    #[derive(Clone, Copy, Pod, Zeroable)]
                    #[repr(C)]
                    struct PushConstantsStuff {
                        chunk_offset: vek::Vec3::<i32>,
                        lod: u32,
                    }
                    
                    let chunk_offset = vek::Vec3::<i32>::new(x, y, z);
                    /*
                    int3 group_offset = (int3)gid + chunk_offset_cs * 8;
                    float3 chunk_offset_ws = (float3)CHUNK_SIZE * group_offset; 
                    float3 chunk_center = (float3)CHUNK_SIZE * group_offset + (float3)CHUNK_SIZE * 0.5; 
                    let lod = floor(clamp(distance(chunk_center, uniforms.position.xyz) / 100, 0, 2));
                    */
                    let lod = 0;
                    let pc = PushConstantsStuff {
                        chunk_offset,
                        lod,
                    };
                    self.device.cmd_push_constants(cmd, self.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&pc));
                    self.mesh_shader_device.cmd_draw_mesh_tasks(cmd,  8, 8, 8);
                }
            }
        }
        */
        /*
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_MS_GENERATED_GRASS_SPV]);
        self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, if self.wireframe { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });
        self.mesh_shader_device.cmd_draw_mesh_tasks(cmd,  16, 16, 1);
        */
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_CHUNK_SPV]);
        self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, if self.wireframe { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });

        // render chunks
        self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.multiple_chunks.vertex_buffer.buffer], &[0]);
        self.device.cmd_bind_index_buffer(cmd, self.multiple_chunks.index_buffer.buffer, 0, vk::IndexType::UINT32);
        self.device.cmd_draw_indexed_indirect(cmd, self.multiple_chunks.indirect_draw_buffer.buffer, 0, self.chunks.len() as u32, size_of::<voxel::DrawIndexedIndirectCommand>() as u32);

        // render models
        for model in self.models.iter() {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_SPV]);
            self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, if self.wireframe { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });
            model.render(cmd, &self.device, self.main_pipeline_layout, &self.materials[0]);
            
        }

        

        self.device.cmd_end_rendering(cmd);
        self.device.cmd_end_query(cmd, pipeline_statistics_query_pool, 0);


        // transition rendered image from color attachment to sampled shader read (for bloom passes)
        let rendered_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(const_data.rendered_image)
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
            .image(const_data.bloom_image)
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
            self.compute_pipelines[COMPUTE_POST_PROCESS_SPV][BLOOM_DOWNSAMPLE_ENTRY_POINT],
        );

        // there is no need to go down to the largest mip since we will be sampling from a smaller mip anyways
        let minimum_upsampling_mip = 2; 

        for mip in 0..(const_data.bloom_mip_image_views.len() as u32-1) {
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
                    .image(const_data.bloom_image)
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
            .image(const_data.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let image_memory_barriers = [full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute bloom upsample passes
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[COMPUTE_POST_PROCESS_SPV][BLOOM_UPSAMPLE_ENTRY_POINT],
        );

        for mip in (minimum_upsampling_mip..(const_data.bloom_mip_image_views.len() as u32 - 1)).rev() {
            // no need to pipeline barrier for the very first pass (we did a full pipeline barrier for the entire bloom image right before this)
            if mip != const_data.bloom_mip_image_views.len() as u32 - 2 {
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
                    .image(const_data.bloom_image)
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
            .image(const_data.bloom_image)
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
            self.compute_pipelines[COMPUTE_POST_PROCESS_SPV][WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT],
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

        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query_pool, 1);
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


        self.stats.end_of_frame(self.frame_count);
        self.frame_count += 1;
        if suboptimal {
            self.recreate_swapchain();
        }
    }

    pub unsafe fn destroy(mut self) {
        self.device.device_wait_idle().unwrap();

        self.tlas.destroy(&self.acceleration_structure_device, &self.device, &mut self.allocator);

        for model in self.models {
            model.destroy(&self.acceleration_structure_device, &self.device, &mut self.allocator);
        }
        log::info!("destroyed models");

        for material in self.materials {
            material.destroy(&self.device, &mut self.allocator);
        }
        log::info!("destroyed materials");

        self.multiple_chunks.destroy(&self.device, &mut self.allocator);
        for chunk in self.chunks {
            chunk.destroy(&self.acceleration_structure_device, &self.device, &mut self.allocator);
        }
        log::info!("destroyed chunks");
        
        self.tesselation_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed tesselation buffer");
        
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

        self.lights_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed lights buffer");

        self.uniform_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed per frame uniform buffer");    
        
        self.debug_text_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed debug text buffer");

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

        self.const_descriptor_sets.destroy_rt_images_and_image_views(&self.device, &mut self.allocator);
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
