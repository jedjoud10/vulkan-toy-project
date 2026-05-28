use std::{collections::HashMap, ffi::CString, ops::{Deref, Index}, str::FromStr};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use smallvec::SmallVec;


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PerFrameUniformData {
    pub view_matrix: vek::Mat4<f32>,
    pub projection_matrix: vek::Mat4<f32>,
    pub inv_view_matrix: vek::Mat4<f32>,
    pub inv_projection_matrix: vek::Mat4<f32>,
    pub view_projection_matrix: vek::Mat4<f32>,
    pub screen_resolution: vek::Vec2<f32>,
    pub _padding: vek::Vec2<f32>,
    pub position: vek::Vec4<f32>,
    pub forward: vek::Vec4<f32>,
    pub sun: vek::Vec4<f32>,
    pub camera_frustum_planes: [vek::Vec4<f32>; 6],
    pub debug_type: u32,
    pub toggles_bitmask: u32,
    pub time: f32,
}

pub struct SingleEntryPointWrapper {
    pub pipeline: vk::Pipeline,
    pub debug_pipeline_name: &'static str,
}

pub enum GenericPipeline {
    Graphics(GenericGraphicsPipeline),
    Compute(GenericComputePipeline),
}

pub struct GenericGraphicsPipeline {
    pub debug_pipeline_name: &'static str,
    pub module: vk::ShaderModule,
    pub pipeline: vk::Pipeline, 
}

impl Deref for GenericGraphicsPipeline {
    type Target = vk::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

pub struct GenericComputePipeline {
    pub module: vk::ShaderModule,
    pub entry_points: HashMap<&'static str, SingleEntryPointWrapper>,
}

impl Index<&'static str> for GenericComputePipeline {
    type Output = vk::Pipeline;

    fn index(&self, index: &'static str) -> &Self::Output {
        &self.entry_points[index].pipeline
    }
}

impl GenericComputePipeline {
    pub unsafe fn destroy(self, device: &ash::Device) {
        for (_, single_entry_point_wrapper) in self.entry_points {
            device.destroy_pipeline(single_entry_point_wrapper.pipeline, None);
            log::info!("destroyed single entry point wrapper '{}' compute pipeline", single_entry_point_wrapper.debug_pipeline_name);
        }
        
        device.destroy_shader_module(self.module, None);
    }
}

impl GenericGraphicsPipeline {
    pub unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_shader_module(self.module, None);
        
        log::info!("destroyed '{}' graphic pipeline", self.debug_pipeline_name);
    }
}

pub struct PipelineCreateSettings<'a> {
    pub pipeline_debug_name: &'static str,
    pub spv_file_name: &'static str,
    pub wtf_kind_of_pipeline_is_this: PipelineCreateType<'a>,
    pub spec_constants: Option<&'a [u32]>
} 

pub enum PipelineCreateType<'a> {
    Graphics {
        face_culling: bool,
        vertex_input: vk::PipelineVertexInputStateCreateInfo<'a>
    },
    GraphicsMeshShader {
        face_culling: bool,
        task_shader: bool
    },
    Compute {
        entry_points: &'a [&'static str],
    }
}

pub unsafe fn create_generic_pipeline(
    raw: &[u32],
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
    pipeline_layout: vk::PipelineLayout,
    settings: PipelineCreateSettings,
) -> GenericPipeline {
    let shader_module = create_shader_module(raw, device, binder, settings.pipeline_debug_name);
    let spec_constants = settings.spec_constants;
    let pipeline_debug_name = settings.pipeline_debug_name;

    match settings.wtf_kind_of_pipeline_is_this {
        PipelineCreateType::Graphics { face_culling, vertex_input } => {
            let pipeline = create_graphics_pipeline(
                device,
                binder,
                shader_module,
                pipeline_layout,
                spec_constants,
                spec_constants,
                vertex_input,
                pipeline_debug_name,
                face_culling
            );

            log::info!("created '{}' graphics pipeline", pipeline_debug_name);

            GenericPipeline::Graphics(GenericGraphicsPipeline { module: shader_module, pipeline, debug_pipeline_name: pipeline_debug_name })
        },
        PipelineCreateType::GraphicsMeshShader { face_culling, task_shader } => {
            let pipeline = create_graphics_pipeline_mesh_shader(
                device,
                binder,
                shader_module,
                pipeline_layout,
                pipeline_debug_name,
                face_culling,
                task_shader,
            );

            log::info!("created '{}' graphics (mesh shader) pipeline", pipeline_debug_name);

            GenericPipeline::Graphics (GenericGraphicsPipeline { module: shader_module, pipeline, debug_pipeline_name: pipeline_debug_name })
        },
        PipelineCreateType::Compute { entry_points } => {
            log::info!("created '{}' compute pipeline", pipeline_debug_name);

            GenericPipeline::Compute(GenericComputePipeline {
                module: shader_module,
                entry_points: entry_points.into_iter().map(|entry_point_name| {
                    (*entry_point_name, create_single_entry_point_pipeline(device, binder, shader_module, *entry_point_name, pipeline_layout, spec_constants))
                }).collect::<HashMap::<&'static str, SingleEntryPointWrapper>>()
            })
        },
    }
}

unsafe fn create_shader_module(raw: &[u32], device: &ash::Device, binder: &Option<ash::ext::debug_utils::Device>, name: &str) -> vk::ShaderModule {
    log::debug!("creating shader module for '{name}'");
    let shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());

    let shader_module = device
        .create_shader_module(&shader_module_create_info, None)
        .unwrap();
    crate::debug::set_object_name(shader_module, binder, name);
    log::debug!("created shader module for '{name}'");
    shader_module
}


pub unsafe fn create_graphics_pipeline_mesh_shader(
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
    shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    name: &str,
    face_culling: bool,
    task_shader: bool,
) -> vk::Pipeline {
    let task_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"task_main")
        .stage(vk::ShaderStageFlags::TASK_EXT)
        .module(shader_module);


    let mesh_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"mesh_main")
        .stage(vk::ShaderStageFlags::MESH_EXT)
        .module(shader_module);

    let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"frag_main")
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(shader_module);
    let mut stages = SmallVec::<[vk::PipelineShaderStageCreateInfo; 3]>::from_slice(&[mesh_shader_stage_create_info, fragment_shader_stage_create_info]);

    if task_shader {
        stages.push(task_shader_stage_create_info);
    }

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR, vk::DynamicState::POLYGON_MODE_EXT];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&dynamic_states);

    let color_attachment_formats = [vk::Format::R16G16B16A16_SFLOAT];
    let mut next = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_attachment_formats)
        .depth_attachment_format(vk::Format::D32_SFLOAT);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default().scissor_count(1).viewport_count(1);
    

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .cull_mode(if face_culling { vk::CullModeFlags::BACK } else { vk::CullModeFlags::NONE })
        .polygon_mode(vk::PolygonMode::FILL)
        .rasterizer_discard_enable(false)
        .depth_clamp_enable(false)
        .depth_bias_enable(false)
        .line_width(1.0f32)
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false);
    
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let attachments = [attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&attachments);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .stencil_test_enable(false)
        .depth_write_enable(true)
        .depth_test_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .render_pass(vk::RenderPass::null())
        .dynamic_state(&dynamic_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .stages(&stages)
        .layout(pipeline_layout)
        .push_next(&mut next);

    let pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_pipeline_create_info], None).unwrap()[0];
    crate::debug::set_object_name(pipeline, binder, format!("'{name}' graphics pipeline"));

    pipeline
}

pub unsafe fn create_graphics_pipeline(
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
    shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    spec_constants_vert: Option<&[u32]>,
    spec_constants_frag: Option<&[u32]>,
    
    vertex_input: vk::PipelineVertexInputStateCreateInfo<'_>,
    name: &str,
    face_culling: bool
) -> vk::Pipeline {
    let (data, entries) = convert_spec_constants(spec_constants_vert);
    let vertex_shader_stage_specialization_info = vk::SpecializationInfo::default()
        .map_entries(&entries)
        .data(&data);
    let vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"vert_main")
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(shader_module)
        .specialization_info(&vertex_shader_stage_specialization_info);

    let (data, entries) = convert_spec_constants(spec_constants_frag);
    let fragment_shader_stage_specialization_info = vk::SpecializationInfo::default()
        .map_entries(&entries)
        .data(&data);
    let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"frag_main")
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(shader_module)
        .specialization_info(&fragment_shader_stage_specialization_info);
    let stages = [vertex_shader_stage_create_info, fragment_shader_stage_create_info];

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&dynamic_states);

    let color_attachment_formats = [vk::Format::R16G16B16A16_SFLOAT];
    let mut next = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_attachment_formats)
        .depth_attachment_format(vk::Format::D32_SFLOAT);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default().scissor_count(1).viewport_count(1);
    

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .cull_mode(if face_culling { vk::CullModeFlags::BACK } else { vk::CullModeFlags::NONE })
        .polygon_mode(vk::PolygonMode::FILL)
        .rasterizer_discard_enable(false)
        .depth_clamp_enable(false)
        .depth_bias_enable(false)
        .line_width(1.0f32)
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false);
    
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let attachments = [attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&attachments);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .stencil_test_enable(false)
        .depth_write_enable(true)
        .depth_test_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .render_pass(vk::RenderPass::null())
        .dynamic_state(&dynamic_state)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .stages(&stages)
        .layout(pipeline_layout)
        .push_next(&mut next);

    let pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_pipeline_create_info], None).unwrap()[0];
    crate::debug::set_object_name(pipeline, binder, format!("'{name}' graphics pipeline"));

    pipeline
}

pub unsafe fn create_single_entry_point_pipeline(
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
    compute_shader_module: vk::ShaderModule,
    entry_point_name: &'static str,
    pipeline_layout: vk::PipelineLayout,
    spec_constants: Option<&[u32]>
) -> SingleEntryPointWrapper {
    let string = CString::from_str(entry_point_name).unwrap();

    let (data, entries) = convert_spec_constants(spec_constants);
    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(&entries)
        .data(&data);

    log::info!("creating single entry point pipeline for {entry_point_name}");
    let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(string.as_c_str())
        .stage(vk::ShaderStageFlags::COMPUTE)
        .specialization_info(&specialization_info)
        .module(compute_shader_module);
            
    let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
        .layout(pipeline_layout)
        .stage(shader_stage_create_info);

    let compute_pipelines = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[compute_pipeline_create_info],
            None,
        )
        .unwrap();
    
    crate::debug::set_object_name(compute_pipelines[0], binder, format!("entry point '{entry_point_name}' compute pipeline"));

    SingleEntryPointWrapper { pipeline: compute_pipelines[0], debug_pipeline_name: entry_point_name }
}

fn convert_spec_constants(spec_constants: Option<&[u32]>) -> (Vec<u8>, Vec::<vk::SpecializationMapEntry>) {
    let mut data = Vec::<u8>::new();
    let mut specialization_entries = Vec::new();
    if let Some(spec_constants) = spec_constants {
        let mut last_offset = 0u32;
        for (i, spec) in spec_constants.iter().enumerate() {
            specialization_entries.push(vk::SpecializationMapEntry::default()
                .constant_id(i as u32)
                .offset(last_offset)
                .size(size_of::<u32>())
            );
            data.extend_from_slice(bytemuck::bytes_of(spec));
            last_offset += size_of::<u32>() as u32;
        }
    }

    (data, specialization_entries)
}

pub unsafe fn create_bindless_pipeline_layout(device: &ash::Device, binder: &Option<ash::ext::debug_utils::Device>, descriptor_set_layout: vk::DescriptorSetLayout) -> vk::PipelineLayout {
    let push_constant_range = vk::PushConstantRange::default()
        .offset(0)
        .size(128u32)
        .stage_flags(vk::ShaderStageFlags::ALL);
    let push_constant_ranges = [push_constant_range];
    
    let layouts = [descriptor_set_layout];
    let compute_pipeline_test_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(push_constant_ranges.as_slice())
        .flags(vk::PipelineLayoutCreateFlags::empty())
        .set_layouts(&layouts);
    
    let pipeline_layout = device
        .create_pipeline_layout(&compute_pipeline_test_layout_create_info, None)
        .unwrap();

    crate::debug::set_object_name(pipeline_layout, binder, format!("main bindless pipeline layout"));
    pipeline_layout
}

