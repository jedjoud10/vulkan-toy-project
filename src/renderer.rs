use ash::vk;
use bytemuck::Pod;
use bytemuck::Zeroable;
use gpu_allocator::vulkan::Allocation;
use rand::RngExt;
use rand::SeedableRng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use smallvec::SmallVec;
use crate::input::Button;
use crate::input::Input;
use crate::movement::Movement;
use crate::samplers;
use winit::event::MouseButton;
use std::collections::HashMap;
use std::ops::ControlFlow;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::Window;
use crate::statistics::Statistics;
use crate::asset;

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
use crate::constant_data::ConstantData;

const COMPUTE_POST_PROCESS_SPV: &'static str = "compute_post_process.spv";
const BLOOM_UPSAMPLE_ENTRY_POINT: &'static str = "bloom_upsample";
const BLOOM_DOWNSAMPLE_ENTRY_POINT: &'static str = "bloom_downsample";
const WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT: &'static str = "write_swapchain_image";
const RASTERIZED_MS_WAVES_SPV: &str = "rasterized_ms_waves.spv";
const COMPUTE_SKY_SPV: &str = "compute_sky.spv";
const WRITE_CLOUDS_ENTRY_POINT: &str = "write_clouds";
const WRITE_SKYBOX_ENTRY_POINT: &str = "write_skybox";
const RASTERIZED_MS_PASSTHROUGH_SPV: &str = "rasterized_ms_passthrough.spv";
const RASTERIZED_MS_TESSELALTION_SPV: &str = "rasterized_ms_tesselation.spv";
const RASTERIZED_BACKGROUND_SPV: &str = "rasterized_background.spv";
        


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

    // descriptors & frames in flight
    main_descriptor_set: vk::DescriptorSet,
    main_descriptor_set_layout: vk::DescriptorSetLayout,
    main_pipeline_layout: vk::PipelineLayout,
    frames_in_flight: Vec<PerFrameData>,
    descriptor_pool: vk::DescriptorPool,
    const_descriptor_sets: ConstantData,
            
    // important too
    allocator: gpu_allocator::vulkan::Allocator,

    // vertex and index buffer shi
    vertex_buffer: buffer::Buffer,
    index_buffer: buffer::Buffer,
    index_count: u32,
    
    // other GPU stuff
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    skybox: skybox::Skybox,
    lights_buffer: buffer::Buffer,
    lights: Vec<vek::Vec4<f32>>,
    samplers: samplers::Samplers,
    tesselation_buffer: buffer::Buffer,
    
    // other CPU stuff
    pub was_resized: bool,
    pub window: Window,
    pub input: Input,    
    movement: Movement,
    frame_count: u64,
    ticker: ticker::Ticker,
    sun: vek::Vec3<f32>,
    debug_type: u32,
    args: crate::Args,
    stats: Statistics,
}

impl InternalApp {
    pub unsafe fn new(event_loop: &ActiveEventLoop, args: crate::Args) -> Self {
        let mut assets = HashMap::<&str, &[u32]>::new();
        //asset!("raytracer.spv", assets);
        asset!("compute_sky.spv", assets);
        asset!("compute_post_process.spv", assets);
        //asset!("voxel_interesting_compute.spv", assets);
        asset!("rasterized_ms_tesselation.spv", assets);
        asset!("rasterized_ms_passthrough.spv", assets);
        asset!("rasterized_ms_waves.spv", assets);
        asset!("rasterized_background.spv", assets);
        

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
                    log_leaks_on_shutdown: false,
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
        );
        log::info!("created swapchain with {} images", swapchain_images.len());

        let (descriptor_pool, main_descriptor_set_layout, main_descriptor_set) = others::create_descriptor_pool_and_bindless_descriptor_set(&device, &debug_marker);

        let main_pipeline_layout = pipeline::create_bindless_pipeline_layout(&device, &debug_marker, main_descriptor_set_layout);
        

        let mut graphics_pipelines = HashMap::<&'static str, pipeline::GenericGraphicsPipeline>::new();
        let mut compute_pipelines = HashMap::<&'static str, pipeline::GenericComputePipeline>::new();

        let settings = [pipeline::PipelineCreateSettings {
            shader_module_debug_name: "post process compute shader module",
            pipeline_debug_name: "post process compute pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &[WRITE_SWAPCHAIN_IMAGE_ENTRY_POINT, BLOOM_DOWNSAMPLE_ENTRY_POINT, BLOOM_UPSAMPLE_ENTRY_POINT] },
            spec_constants: Some(&[args.downscale_factor]),
            spv_file_name: COMPUTE_POST_PROCESS_SPV,
        }, pipeline::PipelineCreateSettings {
            shader_module_debug_name: "sky compute shader module",
            pipeline_debug_name: "sky compute pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Compute { entry_points: &[WRITE_SKYBOX_ENTRY_POINT, WRITE_CLOUDS_ENTRY_POINT] },
            spec_constants: Some(&[skybox::SKYBOX_RESOLUTION, skybox::CLOUDS_RESOLUTION]),
            spv_file_name: COMPUTE_SKY_SPV,
        }, pipeline::PipelineCreateSettings {
            shader_module_debug_name: "main render rasterization shader module",
            pipeline_debug_name: "main render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::GraphicsMeshShader { face_culling: true, task_shader: false },
            spec_constants: None,
            spv_file_name: RASTERIZED_MS_PASSTHROUGH_SPV,
        }, pipeline::PipelineCreateSettings {
            shader_module_debug_name: "tesselation render rasterization shader module",
            pipeline_debug_name: "tesselation render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::GraphicsMeshShader { face_culling: true, task_shader: true },
            spec_constants: None,
            spv_file_name: RASTERIZED_MS_TESSELALTION_SPV,
        }, pipeline::PipelineCreateSettings {
            shader_module_debug_name: "waves render rasterization shader module",
            pipeline_debug_name: "waves render pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::GraphicsMeshShader { face_culling: true, task_shader: false },
            spec_constants: None,
            spv_file_name: RASTERIZED_MS_WAVES_SPV,
        }, pipeline::PipelineCreateSettings {
            shader_module_debug_name: "background sky rasterization shader module",
            pipeline_debug_name: "background sky pipeline",
            wtf_kind_of_pipeline_is_this: pipeline::PipelineCreateType::Graphics { face_culling: false, vertex_input: vk::PipelineVertexInputStateCreateInfo::default() },
            spec_constants: None,
            spv_file_name: RASTERIZED_BACKGROUND_SPV,
        }];

        // compile the pipelines in parallel
        // ouug shii :eyes:
        let generic_pipelines = settings.into_par_iter().map(|setting| {
            let spv_file_name = setting.spv_file_name;
            let raw = assets[spv_file_name];
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

        let frames_in_flight = (0..crate::per_frame_data::FRAMES_IN_FLIGHT).into_iter().map(|_| {
            PerFrameData::create_per_frame_data(&device, pool, &mut allocator, &debug_marker)
        }).collect::<Vec<_>>();
        log::info!("created frames in flight structures");

        const NUM_LIGHTS: usize = 100;

        let lights_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<vek::Vec4<f32>>() * NUM_LIGHTS, &debug_marker, "lights buffer", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let mut lights = Vec::<vek::Vec4<f32>>::new();

        /*
        for _i in 0..NUM_LIGHTS {
            let x = rand::random_range((voxel::TOTAL_SIZE as f32 / 2.0f32 - 10f32)..(voxel::TOTAL_SIZE as f32 / 2.0f32 + 10f32));
            let y = rand::random_range(0f32..(voxel::TOTAL_SIZE as f32));
            let z = rand::random_range((voxel::TOTAL_SIZE as f32 / 2.0f32 - 10f32)..(voxel::TOTAL_SIZE as f32 / 2.0f32 + 10f32));
            lights.push(vek::Vec4::new(x,y,z, 1.0));
        }
        */

        //buffer::write_to_buffer(&device, pool, queue, lights_buffer.buffer, &mut allocator, bytemuck::cast_slice(lights.as_slice()));
        log::info!("created lights buffer");

        let mut const_descriptor_sets = ConstantData::create_constant_descriptor_sets();
        const_descriptor_sets.recreate_rt_images_and_image_views_and_update_descriptor_sets(&device, &mut allocator, queue_family_index, extent, &debug_marker, args.downscale_factor);
        crate::constant_data::transfer_layout_for_images(&device, queue_family_index, &const_descriptor_sets, pool, queue);
        log::info!("created constant descriptor sets");

        let query_pool = others::create_query_pool(&device);
        let timestamp_period = physical_device_properties.properties.limits.timestamp_period;

        let thingy = include_bytes!("../models/bunny_subdivided.obj");
        let obj = obj::load_obj::<obj::Position, &[u8], u32>(thingy).unwrap();

        // replace with vec reinterpret...
        let vertices: Vec<vek::Vec3<f32>> = obj.vertices.into_iter().map(|x| vek::Vec3::<f32>::from(x.position)).collect::<Vec<_>>();
        let mut indices: Vec<u32> = obj.indices;

        //indices.sort_by_key(|triangle| triangle.x + triangle.y + triangle.z);
        //meshopt::optimize_vertex_cache_in_place(&mut indices, vertices.len());

        let vertex_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<vek::Vec3::<f32>>() * vertices.len(), &debug_marker, "vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER);
        buffer::write_to_buffer(&device, pool, queue, vertex_buffer.buffer, &mut allocator, bytemuck::cast_slice(vertices.as_slice()));

        let index_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<u32>()  * indices.len(), &debug_marker, "index buffer", vk::BufferUsageFlags::INDEX_BUFFER);
        let index_count = indices.len() as u32;
        buffer::write_to_buffer(&device, pool, queue, index_buffer.buffer, &mut allocator, bytemuck::cast_slice(indices.as_slice()));

        let mesh_shader_device = ash::ext::mesh_shader::Device::new(&instance, &device);
        let extended_dynamic_state3_device = ash::ext::extended_dynamic_state3::Device::new(&instance, &device);

        let tesselation_buffer = crate::tesselation::precompute_tesselation_buffer(&device, &mut allocator, &debug_marker, pool, queue);

        Self {
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
            query_pool,
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
            main_descriptor_set,
            main_descriptor_set_layout,
            main_pipeline_layout,
            vertex_buffer,
            index_buffer,
            index_count,
            mesh_shader_device,
            extended_dynamic_state3_device,
            tesselation_buffer,
            graphics_pipelines,
            compute_pipelines,
        }
    }

    pub unsafe fn click(&mut self, add: bool) {
        let position = (self.movement.forward() * 5.0f32 + self.movement.position).floor().as_::<u32>();

        /*
        self.svo.set(position, add);
        self.svo.rebuild(&self.device, self.pool, self.queue, &mut self.allocator);
        */
    }

    pub unsafe fn recreate_swapchain(&mut self) {
        log::warn!("recreating swapchain");
        self.was_resized = false;
        self.device.device_wait_idle().unwrap();

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        for swapchain_image_view in self.swapchain_image_views.iter() {
            self.device.destroy_image_view(*swapchain_image_view, None);
        }

        let width = self.window.inner_size().width;
        let height = self.window.inner_size().height;
        
        let extent = vk::Extent2D { width, height };

        let (swapchain_loader, swapchain, swapchain_images, swapchain_image_views, swapchain_format) = swapchain::create_swapchain(
            &self.instance,
            &self.surface_loader,
            self.surface_khr,
            self.physical_device,
            &self.device,
            extent,
            &self.debug_marker,
        );

        self.swapchain_loader = swapchain_loader;
        self.swapchain_format = swapchain_format;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain = swapchain;

        self.const_descriptor_sets.destroy_rt_images_and_image_views(&self.device, self.descriptor_pool, &mut self.allocator);
        self.const_descriptor_sets.recreate_rt_images_and_image_views_and_update_descriptor_sets(&self.device, &mut self.allocator, self.queue_family_index, extent, &self.debug_marker, self.args.downscale_factor);
        crate::constant_data::transfer_layout_for_images(&self.device, self.queue_family_index, &self.const_descriptor_sets, self.pool, self.queue);
                
        for frame in self.frames_in_flight.iter_mut() {
            self.device.destroy_semaphore(frame.present_complete_semaphore, None);
            self.device.destroy_semaphore(frame.render_finished_semaphore, None);

            let create_info = vk::SemaphoreCreateInfo::default();
            frame.render_finished_semaphore = self.device.create_semaphore(&create_info, None).unwrap();
            frame.present_complete_semaphore = self.device.create_semaphore(&create_info, None).unwrap();
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
        //let frame_in_flight_index = 0;
        let frame_in_flight_index = self.frame_count % (self.frames_in_flight.len() as u64);
        let const_data = &self.const_descriptor_sets;
        let PerFrameData {
            present_complete_semaphore,
            end_fence,
            cmd,
            uniform_buffer,
            ..
        } = &self.frames_in_flight[frame_in_flight_index as usize];

        let cmd = *cmd;
        let present_complete_semaphores = [*present_complete_semaphore];
        let end_fence = *end_fence;

        if let Err(err) = self.device.wait_for_fences(&[end_fence], true, u64::MAX) {
            log::error!("wait on fence err: {:?}", err);
            // return;
        } else {
            /*
            let mut timestamps = [0u64; 2];
            let okay = self.device.get_query_pool_results(self.query_pool, 0, &mut timestamps, vk::QueryResultFlags::TYPE_64).is_ok();
            if okay {
                let delta_in_ms = ((timestamps[1].saturating_sub(timestamps[0])) as f64 * self.timestamp_period as f64) / 1000000.0f64;
                self.stats.push_query_timings(delta_in_ms);
            }
            */
        }

        let (acquired_swapchain_image_index, suboptimal) = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                *present_complete_semaphore,
                vk::Fence::null(),
            )
            .unwrap();

        let swapchain_image = self.swapchain_images[acquired_swapchain_image_index as usize]; // then compose onto this...
        let swapchain_image_view = self.swapchain_image_views[acquired_swapchain_image_index as usize];

               

        /*
        //log::debug!("frame in flight index: {frame_in_flight_index}, acquire swapchain image index: {acquired_swapchain_image_index}");

        let descriptor_swapchain_image_view_info = vk::DescriptorImageInfo::default()
            .image_view(swapchain_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());

        // rt image for compositor (write only)
        let composition_compute_descriptor_image_infos_1 = [descriptor_swapchain_image_view_info];
        let composition_compute_image_descriptor_write_1 = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(per_frame_descriptor_sets.compositor_per_frame)
            .image_info(&composition_compute_descriptor_image_infos_1);


        let descriptor_uniform_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let render_rasterization_per_frame_buffer_infos = [descriptor_uniform_buffer_info];
        let render_rasterization_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .dst_binding(0)
            .dst_set(per_frame_descriptor_sets.rasterizer_per_frame)
            .buffer_info(&render_rasterization_per_frame_buffer_infos);

        let background_rasterization_per_frame_buffer_infos = [descriptor_uniform_buffer_info];
        let background_rasterization_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .dst_binding(0)
            .dst_set(per_frame_descriptor_sets.background_rasterizer_per_frame)
            .buffer_info(&background_rasterization_per_frame_buffer_infos);

        // update per frame descriptor sets
        self.device.update_descriptor_sets(&[composition_compute_image_descriptor_write_1, render_rasterization_descriptor_write, background_rasterization_descriptor_write], &[]);
        */
        if suboptimal || self.was_resized {
            log::debug!("suboptimal: {suboptimal}");
            log::debug!("was resized: {}", self.was_resized);
            
            self.recreate_swapchain();
            self.was_resized = false;
            return;
        }

        self.device.reset_fences(&[end_fence]).unwrap();

        let render_finished_semaphore = [self.frames_in_flight[acquired_swapchain_image_index as usize].render_finished_semaphore];

        

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
        let clouds_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.clouds_image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut storage_image_infos = vec![swapchain_image_view_descriptor_image_info, rendered_image_view_descriptor_image_info, skybox_image_view_descriptor_image_info, clouds_image_view_descriptor_image_info];
        
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
            .dst_set(self.main_descriptor_set)
            .image_info(&storage_image_infos);       

        // create bindless descriptor write for storage buffers
        let descriptor_uniform_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let descriptor_vertex_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(self.vertex_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let descriptor_index_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(self.index_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let descriptor_tess_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(self.tesselation_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let storage_buffer_infos = [descriptor_uniform_buffer_info, descriptor_vertex_buffer_info, descriptor_index_buffer_info, descriptor_tess_buffer_info];
        let storage_buffer_write = vk::WriteDescriptorSet::default()
            .descriptor_count(storage_buffer_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_binding(1)
            .dst_set(self.main_descriptor_set)
            .buffer_info(&storage_buffer_infos);

        // create bindless descriptor write for combined image samplers
        let skybox_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.skybox_image_view)
            .sampler(self.samplers.skybox_sampler)
            .image_layout(vk::ImageLayout::GENERAL);
        let clouds_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.skybox.clouds_image_view)
            .sampler(self.samplers.skybox_sampler)
            .image_layout(vk::ImageLayout::GENERAL);
        let rendered_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(const_data.rendered_image_view)
            .sampler(self.samplers.bloom_sampler)
            .image_layout(vk::ImageLayout::GENERAL);
        let entire_bloom_sampled_image_view_descriptor_image_info = vk::DescriptorImageInfo::default()
            .image_view(const_data.entire_bloom_image_view)
            .sampler(self.samplers.bloom_sampler)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut sampled_image_infos = vec![skybox_sampled_image_view_descriptor_image_info, clouds_sampled_image_view_descriptor_image_info, rendered_sampled_image_view_descriptor_image_info, entire_bloom_sampled_image_view_descriptor_image_info];

        // add bloom sampled image views
        for bloom_sampled_image_view in const_data.bloom_mip_image_views.iter() {
            sampled_image_infos.push(vk::DescriptorImageInfo::default()
                .image_view(*bloom_sampled_image_view)
                .sampler(self.samplers.bloom_sampler)
                .image_layout(vk::ImageLayout::GENERAL)
            );
        }

        let sampled_image_write = vk::WriteDescriptorSet::default()
            .descriptor_count(sampled_image_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_binding(2)
            .dst_set(self.main_descriptor_set)
            .image_info(&sampled_image_infos);

        // update the bindless descriptor set
        self.device.update_descriptor_sets(&[storage_image_write, storage_buffer_write, sampled_image_write], &[]);

        // TODO: ideally, these would:
        // 1. be dynamically allocated using some sort of per-frame arena with indexing
        // 2. be passed to the shader either using a uniform buffer (since these are constant anyways)
        const SWAPCHAIN_STORAGE_IMAGE_IDX: u32 = 0;
        const RENDERED_STORAGE_IMAGE_IDX: u32 = 1;
        const SKYBOX_STORAGE_IMAGE_IDX: u32 = 2;
        const CLOUDS_STORAGE_IMAGE_IDX: u32 = 3;
        const BLOOM_MIPS_STORAGE_IMAGE_START_IDX: u32 = 4; // bloom needs to be last since it is dynamically allocated (can have a dynamic number of bloom mips, depending on screen res)
        
        const UNIFORM_BUFFER_THINGY_IDX: u32 = 0;
        
        const SKYBOX_SAMPLER_IMAGE_IDX: u32 = 0;
        const CLOUDS_SAMPLER_IMAGE_IDX: u32 = 1;
        const RENDERED_SAMPLER_IMAGE_IDX: u32 = 2;
        const ENTIRE_BLOOM_SAMPLER_IMAGE_IDX: u32 = 3;
        const BLOOM_MIPS_SAMPLED_IMAGE_START_IDX: u32 = 4; // bloom needs to be last since it is dynamically allocated (can have a dynamic number of bloom mips, depending on screen res)


        let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device
            .begin_command_buffer(cmd, &cmd_buffer_begin_info)
            .unwrap();
        self.device.cmd_reset_query_pool(cmd, self.query_pool, 0, 2);

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // bind the descriptor set for subsequent pipelines
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.main_pipeline_layout,
            0,
            &[self.main_descriptor_set],
            &[],
        );
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.main_pipeline_layout,
            0,
            &[self.main_descriptor_set],
            &[],
        );

        let size = self.window.inner_size();
        let window_size_no_downscale = vek::Vec2::<u32>::new(size.width, size.height);
        let size = vek::Vec2::<u32>::new(size.width, size.height) / self.args.downscale_factor;

        let size_f32 = size.map(|x| x as f32);

        // https://github.com/jedjoud10/cflake-engine/blob/3369199f0cfa8b220edc0363a76401b50c83fada/crates/math/src/bounds/frustum.rs#L47
        let camera_frustum_planes = {
            let columns = (self.movement.proj_matrix * self.movement.view_matrix).transposed().into_col_arrays();
            let columns = columns
                .into_iter()
                .map(vek::Vec4::from)
                .collect::<SmallVec<[vek::Vec4<f32>; 4]>>();

            // Magic from https://www.braynzarsoft.net/viewtutorial/q16390-34-aabb-cpu-side-frustum-culling
            // And also from https://gamedev.stackexchange.com/questions/156743/finding-the-normals-of-the-planes-of-a-view-frustum
            // YAY https://stackoverflow.com/questions/12836967/extracting-view-frustum-planes-gribb-hartmann-method
            let left = columns[3] + columns[0];
            let right = columns[3] - columns[0];
            let top = columns[3] - columns[1];
            let bottom = columns[3] + columns[1];
            let near = columns[3] + columns[2];
            let far = columns[3] - columns[2];
            [top, bottom, left, right, near, far]
        };

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
            camera_frustum_planes: camera_frustum_planes,
            debug_type: self.debug_type,
            time: elapsed,

            _padding: Default::default(),
        };

        self.device.cmd_update_buffer(cmd, uniform_buffer.buffer, 0, bytemuck::bytes_of(&uniform_per_frame_data));


        let uniform_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(uniform_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let buffer_memory_barriers = [uniform_buffer_barrier];
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
        let image_memory_barriers = [skybox_image_barrier, clouds_image_barrier, rendered_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        /*
        //self.sun = vek::Vec3::new((elapsed * 0.1f32).sin(), (elapsed * 0.05).sin(), (elapsed * 0.1f32).cos()).normalized();



        let lights_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.lights_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let uniform_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(uniform_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let image_memory_barriers = [];
        let buffer_memory_barriers = [lights_buffer_barrier, uniform_buffer_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers).buffer_memory_barriers(&buffer_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        



        let matrix = self.movement.proj_matrix.inverted() * self.movement.view_matrix;

        let push_constants = pipeline::PushConstants {
            screen_resolution: size_f32,
            _padding: Default::default(),
            matrix,
            position: self.movement.position.with_w(0f32),
            sun: self.sun.normalized().with_w(0f32),
            debug_type: self.debug_type,
            time: elapsed,
        };

        let raw = bytemuck::bytes_of(&push_constants);

        
        let uniform_per_frame_data = pipeline::PerFrameUniformData {
            view_matrix: self.movement.view_matrix,
            projection_matrix: self.movement.proj_matrix,
            inv_view_matrix: self.movement.view_matrix.inverted(),
            inv_projection_matrix: self.movement.proj_matrix.inverted(),
            screen_resolution: size_f32,
            position: self.movement.position.with_w(0f32),
            sun: self.sun.normalized().with_w(0f32),
            debug_type: self.debug_type,
            time: elapsed,

            _padding: Default::default(),
        };

        self.device.cmd_update_buffer(cmd, uniform_buffer.buffer, 0, bytemuck::bytes_of(&uniform_per_frame_data));
        
        self.device.cmd_update_buffer(cmd, self.lights_buffer.buffer, 0, bytemuck::cast_slice(&self.lights));

        if self.debug_type == 0 {
            let rendered_image_barrier = vk::ImageMemoryBarrier2::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(self.queue_family_index)
                .image(constant_descriptor_sets.rendered_image)
                .subresource_range(subresource_range);
            let image_memory_barriers = [rendered_image_barrier];
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

            let viewport = vk::Viewport::default().height(size.y as f32).width(size.x as f32).x(0f32).y(0f32).min_depth(0f32).max_depth(1f32);
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[render_area]);
            
            self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_GRAPHICS, self.query_pool, 0);
            self.device.cmd_begin_rendering(cmd, &rendering_info);


            // render background skybox and clouds
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.rasterization_background_pipeline.pipeline);
            self.device.cmd_draw(cmd, 6, 1, 0, 0);

            /*
            // render chunk meshes stuff
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.rasterization_pipeline.pipeline_layout, 0, &[per_frame_descriptor_sets.rasterizer_per_frame, constant_descriptor_sets.main_render_rasterization_render_pipeline_descriptor_set], &[]);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.rasterization_pipeline.pipeline);
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.meshed.vertex_buffer.buffer], &[0]);
            self.device.cmd_bind_index_buffer(cmd, self.meshed.index_buffer.buffer, 0, vk::IndexType::UINT32);
            for single_chunk in self.meshed.chunks.iter() {
                self.device.cmd_draw_indexed(cmd, single_chunk.index_count as u32, 1, single_chunk.first_index as u32, single_chunk.vertex_start_offset as i32, 0);
            }
            
            self.device.cmd_end_rendering(cmd);
            */
            self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_GRAPHICS, self.query_pool, 1);

            let rendered_image_barrier = vk::ImageMemoryBarrier2::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(self.queue_family_index)
                .image(constant_descriptor_sets.rendered_image)
                .subresource_range(subresource_range);
            let image_memory_barriers = [rendered_image_barrier];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
            self.device.cmd_pipeline_barrier2(cmd, &dep);
        } else {
            
        } 

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[0].pipeline_layout,
            0,
            &[constant_descriptor_sets.sky_compute_pipeline_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[0].pipeline,
        );

        let push_constants = pipeline::SkyComputePushConstants {
            sun: self.sun.normalized().with_w(elapsed),
        };
        self.device.cmd_push_constants(cmd, self.sky_compute_pipeline.entry_points[0].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push_constants));
        self.device.cmd_dispatch(cmd, skybox::CLOUDS_RESOLUTION.div_ceil(8), skybox::CLOUDS_RESOLUTION.div_ceil(8), 1);

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[1].pipeline_layout,
            0,
            &[constant_descriptor_sets.sky_compute_pipeline_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[1].pipeline,
        );

        self.device.cmd_push_constants(cmd, self.sky_compute_pipeline.entry_points[1].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push_constants));
        self.device.cmd_dispatch(cmd, skybox::SKYBOX_RESOLUTION.div_ceil(8), skybox::SKYBOX_RESOLUTION.div_ceil(8), 6);
        
        let src_shader_write_to_shader_read = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(constant_descriptor_sets.rendered_image)
            .subresource_range(subresource_range);
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
        let image_memory_barriers = [src_shader_write_to_shader_read, skybox_image_barrier, clouds_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        let full_passes_bloom = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(constant_descriptor_sets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let image_memory_barriers = [full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute bloom downsample passes
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[1].pipeline,
        );
        for mip in 0..(constant_descriptor_sets.bloom_mip_image_views.len() as u32-1) {
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
                    .image(constant_descriptor_sets.bloom_image)
                    .subresource_range(previous_mip_level_subresource_range);
                let barriers = [previous_mip_image_memory_barrier];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device.cmd_pipeline_barrier2(cmd, &dep);
            }
            
            
            let previous_mip_size = size / (1 << (mip)); // larger mip
            let next_mip_size = size / (1 << (mip+1)); // smaller mip

            
            
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post_process_compute_pipeline.entry_points[1].pipeline_layout,
                0,
                &[constant_descriptor_sets.compositor_downsample_bloom[mip as usize]],
                &[],
            );
            
            let downsample_dispatch_push_constants = previous_mip_size.as_::<f32>();

            self.device.cmd_push_constants(cmd, self.post_process_compute_pipeline.entry_points[1].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&downsample_dispatch_push_constants));
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
            .image(constant_descriptor_sets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let image_memory_barriers = [full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute bloom upsample passes
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[2].pipeline,
        );

        // there is no need to go down to the largest mip since we will be sampling from a smaller mip anyways
        let minimum_upsampling_mip = 2;

        for mip in (minimum_upsampling_mip..(constant_descriptor_sets.bloom_mip_image_views.len() as u32 - 1)).rev() {
            // no need to pipeline barrier for the very first pass (we did a full pipeline barrier for the entire bloom image right before this)
            if mip != constant_descriptor_sets.bloom_mip_image_views.len() as u32 - 2 {
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
                    .image(constant_descriptor_sets.bloom_image)
                    .subresource_range(previous_mip_level_subresource_range);
                let barriers = [previous_mip_image_memory_barrier];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device.cmd_pipeline_barrier2(cmd, &dep);
            }
            
            
            let previous_mip_size = size / (1 << (mip+1)); // smaller mip
            let next_mip_size = size / (1 << (mip)); // larger mip

            
            
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post_process_compute_pipeline.entry_points[2].pipeline_layout,
                0,
                &[constant_descriptor_sets.compositor_upsample_bloom[mip as usize]],
                &[],
            );
            
            let upsample_dispatch_push_constants = previous_mip_size.as_::<f32>();

            self.device.cmd_push_constants(cmd, self.post_process_compute_pipeline.entry_points[2].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&upsample_dispatch_push_constants));
            self.device.cmd_dispatch(cmd, next_mip_size.x.div_ceil(8), next_mip_size.y.div_ceil(8), 1);
        }

        let last_pass_bloom = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(constant_descriptor_sets.bloom_image)
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
        let image_memory_barriers = [last_pass_bloom, swapchain_image_undefined_to_blit_dst_layout_transition];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute composition pass
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[0].pipeline_layout,
            0,
            //&[per_frame_descriptor_sets.compositor_per_frame, constant_descriptor_sets.compositor_compute_pipeline_descriptor_set],
            &[constant_descriptor_sets.compositor, per_frame_descriptor_sets.compositor_per_frame],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[0].pipeline,
        );

        // TODO: remove duplicate push constants by saving to uniform buffer type shift
        self.device.cmd_push_constants(
            cmd,
            self.post_process_compute_pipeline.entry_points[0].pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            raw,
        );

        self.device.cmd_dispatch(cmd, window_size_no_downscale.x.div_ceil(8), window_size_no_downscale.y.div_ceil(8), 1);
        */

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

        let viewport = vk::Viewport::default().height(size.y as f32).width(size.x as f32).x(0f32).y(0f32).min_depth(0f32).max_depth(1f32);
        self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        self.device.cmd_set_scissor(cmd, 0, &[render_area]);
        
        //self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_GRAPHICS, self.query_pool, 0);
        self.device.cmd_begin_rendering(cmd, &rendering_info);

        // render background skybox and clouds
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_BACKGROUND_SPV]);
        self.device.cmd_draw(cmd, 6, 1, 0, 0);

        // render waves
        //self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.waves_rasterization_pipeline.pipeline);
        //self.mesh_shader_device.cmd_draw_mesh_tasks(cmd, 32, 32, 1);

        // render objs
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *self.graphics_pipelines[RASTERIZED_MS_PASSTHROUGH_SPV]);
        self.extended_dynamic_state3_device.cmd_set_polygon_mode(cmd, if self.debug_type < 4 { vk::PolygonMode::FILL } else { vk::PolygonMode::LINE });
        let triangle_count = self.index_count / 3;
        self.device.cmd_push_constants(cmd, self.main_pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&triangle_count));
        self.mesh_shader_device.cmd_draw_mesh_tasks(cmd,  triangle_count.div_ceil(32), 1, 1);
        
        // self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
        // self.device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
        // self.device.cmd_draw_indexed(cmd, self.index_count, 1, 0, 0, 0);
        //self.mesh_shader_device.cmd_draw_mesh_tasks(cmd, (self.index_count / 3).div_ceil(32), 1, 1);

        self.device.cmd_end_rendering(cmd);
        //self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_GRAPHICS, self.query_pool, 1);


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

        self.device.end_command_buffer(cmd).unwrap();

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

        let _start = std::time::Instant::now();
        let suboptimal = self.swapchain_loader
            .queue_present(self.queue, &present_info)
            .unwrap();
        let _end = std::time::Instant::now();


        self.stats.end_of_frame(self.frame_count);
        self.frame_count += 1;

        // THERES STILL SOMETHING WRONG WITH FRAMES IN FLIGHT
        // FUCK
        self.device.wait_for_fences(&[end_fence], true, u64::MAX).unwrap();

        //log::debug!("CPU thread took: {}us", (_end-_start).as_micros());

        if suboptimal {
            self.recreate_swapchain();
        }
    }

    pub unsafe fn destroy(mut self) {
        self.device.device_wait_idle().unwrap();

        self.index_buffer.destroy(&self.device, &mut self.allocator);
        self.vertex_buffer.destroy(&self.device, &mut self.allocator);
        self.tesselation_buffer.destroy(&self.device, &mut self.allocator);

        for (_, graphic_pipeline) in self.graphics_pipelines {
            graphic_pipeline.destroy(&self.device);
        }

        for (_, compute_pipeline) in self.compute_pipelines {
            compute_pipeline.destroy(&self.device);
        }
                
        self.skybox.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed skybox");

        self.lights_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed lights buffer");

        self.device.destroy_query_pool(self.query_pool, None);
        log::info!("destroyed query pool");

        log::info!("waiting for all frame in flight fences...");
        let fences = self.frames_in_flight.iter().map(|x| x.end_fence).collect::<Vec<_>>();
        self.device
            .wait_for_fences(&fences, true, u64::MAX)
            .unwrap();
        for frame in self.frames_in_flight.into_iter() {
            frame.destroy_everything(&self.device, self.pool, &mut self.allocator);
        }

        self.const_descriptor_sets.destroy_rt_images_and_image_views(&self.device, self.descriptor_pool, &mut self.allocator);
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
        
        self.device.free_descriptor_sets(self.descriptor_pool, &[self.main_descriptor_set]).unwrap();
        log::info!("freed bindless descriptor set");

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
